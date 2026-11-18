"""PyBullet quadrotor environment (Iris-class airframe, physical rotor model).

Controllers:
  direct    - the policy commands the four motors (end-to-end);
  residual  - a copy of the ArduPilot cascade (training/pid.py) flies the vehicle and
              bounded corrections are applied through its residual inputs.
The task reward is not defined here: training.wrappers.TaskReward computes it from `info`.
Domain randomisation (mass, payload drop, motors, delays, sensor noise, PID gains,
disturbance forces, random tracks) is configured in the configs/*.json files.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb

from controllers.base import (EncoderConfig, ObservationEncoder,
                              OBSERVATION_SCHEMA_ID, COMMAND_SCHEMA_IDS)
from experiments.mission import Mission
from experiments.scenarios import force_N, scenario_spec, seeded_rng
from .physics import (MASS, G, INERTIA, MAX_FORCE, ROTOR_POSITIONS,
                      YAW_MOMENT_PER_THRUST_M, MOTOR_FORCE_LIMIT_N,
                      DirectMotorConfig, DirectMotorModel)
from .pid import CascadedPID

ENVIRONMENT_VERSION = 'quadrotor-learning-v3-shared-rotors'
CONTROLLERS = ('direct', 'residual')
# Loss-of-control distance from the current goal, metres (longer routes need more).
FAIL_DISTANCE_M = 4.


def load_config(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / 'configs' / 'compensator_eval.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'quadrotor-training-config-v1':
        raise ValueError('Training configuration schema differs')

    def finite(item):
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Nonfinite training configuration')

    finite(value)
    if value['control_dt_s'] != .02:
        raise ValueError('Current interfaces require 50 Hz control')
    if not isinstance(value.get('observation_schema'), str) or not value['observation_schema']:
        raise ValueError('Declare observation_schema in the training configuration')
    return value


def finite_segment_distance(position, previous_target, current_target):
    if previous_target is None:
        return 0.
    leg = np.asarray(current_target) - np.asarray(previous_target)
    squared = float(leg @ leg)
    if squared <= 1e-15:
        return 0.
    fraction = np.clip(float((position - previous_target) @ leg) / squared, 0., 1.)
    return float(np.linalg.norm(position - (previous_target + fraction * leg)))


class QuadrotorEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 50}

    def __init__(self, controller='direct', training=False, scenario='nominal',
                 mission=None, seconds=None, waypoint_radius=.40,
                 render_mode=None, config_path=None, encoder_config=None,
                 observation_encoder=None, reward=None, reward_factory=None,
                 residual_limit_pwm=30.):
        super().__init__()
        if controller not in CONTROLLERS:
            raise ValueError('Supported controller: direct or residual')
        if mission not in (None, 'hover', 'route'):
            raise ValueError('Mission must be hover or route')
        if seconds is not None and (not math.isfinite(float(seconds)) or float(seconds) <= 0):
            raise ValueError('Positive finite episode duration required')
        if render_mode not in (None, 'human', 'rgb_array'):
            raise ValueError('Unknown PyBullet render mode')
        # Validate the mission and radius before opening a physical connection.
        Mission('hover', waypoint_radius)
        self.controller = controller
        self.architecture = controller
        self.physical_motors = controller in ('direct', 'residual')
        if not math.isfinite(residual_limit_pwm) or not 0 < residual_limit_pwm <= 100:
            raise ValueError('Residual PWM limit must be finite and in (0,100]')
        self.residual_limit_pwm = float(residual_limit_pwm)
        self.training, self.scenario = bool(training), scenario
        self.mission_override = mission
        self.seconds = None if seconds is None else float(seconds)
        self.waypoint_radius = float(waypoint_radius)
        self.render_mode = render_mode
        self.config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / 'configs' / 'compensator_eval.json'
        self.config = load_config(self.config_path)
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.control_dt = .02
        self.dt = .005 if self.physical_motors else .0025
        self.decimation = round(self.control_dt / self.dt)
        self.plant_config = self.config['plant']
        self.command_schema = COMMAND_SCHEMA_IDS[self.architecture]
        if self.config.get('command_schemas', {}).get(self.architecture) != self.command_schema:
            raise ValueError('Configured command schema differs from the physical interface')
        if observation_encoder is not None and encoder_config is not None:
            raise ValueError('Supply an encoder instance or encoder configuration, not both')
        if observation_encoder is None:
            if self.config['observation_schema'] != OBSERVATION_SCHEMA_ID:
                raise ValueError('A nonstandard observation schema requires an explicit encoder')
            cfg = encoder_config if encoder_config is not None else self.config.get('observation', {})
            if not isinstance(cfg, EncoderConfig):
                cfg = EncoderConfig(**cfg)
            observation_encoder = ObservationEncoder(cfg)
        self.encoder = observation_encoder
        self.encoder_config = self.encoder.config
        self.observation_schema = getattr(self.encoder, 'schema_id', None)
        if not isinstance(self.observation_schema, str) or not self.observation_schema:
            raise ValueError('Observation encoder must declare a nonempty schema_id')
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.encoder_config.size,), np.float32)
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        if reward is not None and reward_factory is not None:
            raise ValueError('Supply a reward instance or factory, not both')
        # Optional in-environment reward; by default the reward is computed by a wrapper.
        self.reward_tracker = reward if reward is not None else (
            reward_factory(dict(self.config.get('reward', {}))) if reward_factory is not None else None)
        self.rotors = ROTOR_POSITIONS.copy()
        self.spin = np.array([-1., -1., 1., 1.])
        self.yaw_coeff = YAW_MOMENT_PER_THRUST_M
        self.mix_inv = np.linalg.inv(np.vstack([np.ones(4), self.rotors[:, 1],
                                               -self.rotors[:, 0], self.yaw_coeff * self.spin]))
        self.cid = pb.connect(pb.GUI if render_mode == 'human' else pb.DIRECT)
        if self.cid < 0:
            raise RuntimeError('PyBullet connection failed')
        pb.setGravity(0, 0, -G, physicsClientId=self.cid)
        pb.setTimeStep(self.dt, physicsClientId=self.cid)
        collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[.13, .10, .035], physicsClientId=self.cid)
        visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[.13, .10, .035],
                                     rgbaColor=[.15, .35, .85, 1], physicsClientId=self.cid)
        self.body = pb.createMultiBody(MASS, collision, visual, basePosition=[0, 0, 3], physicsClientId=self.cid)
        plane = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=self.cid)
        pb.createMultiBody(0, plane, physicsClientId=self.cid)
        self._ready = False
        self._episode_done = False
        self.measured_state = None

    def state(self):
        position, quaternion = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        velocity, omega = pb.getBaseVelocity(self.body, physicsClientId=self.cid)
        rotation = np.asarray(pb.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return np.asarray(position), np.asarray(velocity), rotation, rotation.T @ np.asarray(omega)

    def observation(self, state=None):
        position, velocity, rotation, rates = self.state() if state is None else state
        position = position + self.sensor_rng.normal(0., self.parameters.get('noise_position', 0.), 3)
        velocity = velocity + self.sensor_rng.normal(0., self.parameters.get('noise_velocity', 0.), 3)
        rates = rates + self.sensor_rng.normal(0., self.parameters.get('noise_rates', 0.), 3)
        perturbation = self.sensor_rng.normal(0., self.parameters.get('noise_orientation', 0.), 3)
        noise_rotation = np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(perturbation))).reshape(3, 3)
        rotation = rotation @ noise_rotation
        self.measured_state = dict(position=position.copy(), velocity=velocity.copy(),
                                   rotation=rotation.copy(), body_rates=rates.copy())
        observation = np.asarray(self.encoder.observe(position, velocity, rotation, rates,
                                  self.target, self.previous_action, self.control_dt), dtype=np.float32)
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise ValueError('Encoder must return a finite observation matching config.size')
        return observation

    def _training_parameters(self, episode_seed):
        rng = seeded_rng(episode_seed, 450)
        cfg = self.config['training_distribution']
        probabilities = [cfg[key] for key in ('nominal_probability', 'constant_probability', 'pulse_probability')]
        if min(probabilities) < 0 or not math.isclose(sum(probabilities), 1.):
            raise ValueError('Training regime probabilities must sum to one')
        draw = rng.random()
        regime = 'nominal' if draw < probabilities[0] else 'constant' if draw < sum(probabilities[:2]) else 'pulse'
        spec = dict(name='training', regime=regime, force_events=[])
        if regime != 'nominal':
            magnitude = float(rng.uniform(cfg['force_min_N'], cfg['force_max_N']))
            angle = float(rng.uniform(-np.pi, np.pi))
            maximum_on=(cfg.get('route_on_max_s',min(6.,cfg['on_max_s'])) if self.mission.kind=='route' else cfg['on_max_s'])
            on = float(rng.uniform(cfg['on_min_s'],maximum_on))
            off = self.max_steps * self.control_dt + 1. if regime == 'constant' else on + float(rng.uniform(
                cfg['pulse_duration_min_s'], cfg['pulse_duration_max_s']))
            spec['force_events'] = [dict(on_s=on, off_s=off,
                force_enu_N=[float(magnitude * np.cos(angle)), float(magnitude * np.sin(angle)), 0.])]
            extra = int(cfg.get('extra_pulses_max', 0))
            if extra:
                # Sequences of gusts: extra pulses with random direction (incl. reversals) and duration.
                horizon = self.max_steps * self.control_dt
                for _ in range(int(rng.integers(0, extra + 1))):
                    m2 = float(rng.uniform(cfg['force_min_N'], cfg['force_max_N']))
                    a2 = float(rng.uniform(-np.pi, np.pi))
                    on2 = float(rng.uniform(cfg['on_min_s'], max(cfg['on_min_s'] + .1, horizon - 1.)))
                    off2 = on2 + float(rng.uniform(cfg.get('extra_pulse_min_s', .5), cfg.get('extra_pulse_max_s', 5.)))
                    spec['force_events'].append(dict(on_s=on2, off_s=off2,
                        force_enu_N=[float(m2 * np.cos(a2)), float(m2 * np.sin(a2)), 0.]))
        randomization = self.config.get('randomization', {})
        def sample(name, bounds):
            low, high = randomization.get(name,bounds)
            if not math.isfinite(low+high) or low>high:
                raise ValueError('Invalid randomization interval: '+name)
            return float(rng.uniform(low,high))
        delay_low,delay_high=randomization.get('delay_steps_range',(0,1))
        if not (type(delay_low) is int and type(delay_high) is int and 0<=delay_low<=delay_high):
            raise ValueError('Delay steps must be a nonnegative integer interval')
        spec.update(mass_scale=sample('mass_scale_range',(1.,1.25)),
            motor_scale=sample('motor_scale_range',(.95,1.05)),
            motor_torque_scale=sample('torque_scale_range',(.95,1.05)),
            pid_gain_scale=sample('pid_scale_range',(.95,1.05)),
            noise_position=sample('noise_position_range',(0.,.01)),
            noise_velocity=sample('noise_velocity_range',(0.,.02)),
            noise_orientation=sample('noise_orientation_range',(0.,.003)),
            noise_rates=sample('noise_rates_range',(0.,.01)),
            delay_steps=int(rng.integers(delay_low,delay_high+1)))
        for key in ('velocity_xy_limit', 'acceleration_xy_limit'):
            if 'pid_' + key + '_range' in randomization:
                spec['pid_' + key] = sample('pid_' + key + '_range', None)
        if rng.random()<float(randomization.get('payload_probability',.5)):
            horizon=self.max_steps*self.control_dt
            requested=sample('payload_fraction_range',(.05,.2))
            # Remove only carried payload; do not silently remove dry airframe mass.
            available=max(0.,(spec['mass_scale']-1.)/spec['mass_scale'])
            fraction=min(requested,available)
            if fraction>1e-8:
                time_key=('route_payload_drop_time_fraction_range' if self.mission.kind=='route'
                          else 'payload_drop_time_fraction_range')
                default_time=(.05,.2) if self.mission.kind=='route' else (.25,.65)
                spec.update(payload_drop_s=horizon*sample(time_key,default_time),
                            payload_fraction=fraction)
        return spec

    def _track(self, kind, episode_seed):
        """Optional configured track: random in training, fixed in evaluation.
        Returns Mission keyword arguments, or {} for the original fixed route."""
        spec = self.config.get('track')
        if kind != 'route' or not spec:
            return {}
        if not self.training:
            fixed = spec['fixed']
            return dict(points=fixed['points'], dwell_s=fixed['dwell_s'], radii_m=fixed['radii_m'])
        r = spec['random']; rng = seeded_rng(episode_seed, 470)
        n = int(rng.integers(r['legs'][0], r['legs'][1] + 1))
        points = [np.array([0., 0., float(rng.uniform(*r['altitude_m']))])]
        heading = float(rng.uniform(-np.pi, np.pi))
        for i in range(n):
            if i: heading += float(rng.uniform(-1, 1)) * np.radians(r['max_turn_deg'])
            length = float(rng.uniform(*r['leg_m']))
            step = np.array([np.cos(heading), np.sin(heading), 0.]) * length
            nxt = points[-1] + step
            nxt[2] = float(rng.uniform(*r['altitude_m']))
            points.append(nxt)
        dwell = [.5] + [float(rng.uniform(*r['stop_dwell_s'])) if rng.random() < r['stop_probability'] else 0.
                        for _ in range(n - 1)] + [r['final_dwell_s']]
        radii = [r['stop_radius_m'] if d > 0 else r['pass_radius_m'] for d in dwell]
        radii[0] = r['stop_radius_m']
        return dict(points=np.array(points), dwell_s=dwell, radii_m=radii)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Use documented constructor configuration, not hidden reset overrides')
        episode_seed = int(self.np_random.integers(0, 2**31)) if self.training or seed is None else int(seed)
        self.episode_seed = episode_seed
        self.sensor_rng = seeded_rng(episode_seed, 220)
        initial_rng = seeded_rng(episode_seed, 330)
        kind = self.mission_override
        if kind is None:
            if self.training:
                probability = self.config['training_distribution']['route_probability']
                if not 0 <= probability <= 1:
                    raise ValueError('Route probability must be in [0,1]')
                kind = 'route' if seeded_rng(episode_seed, 440).random() < probability else 'hover'
            else: kind = 'hover'
        self.mission_kind = kind
        track = self._track(kind, episode_seed)
        self.mission = Mission(kind, self.waypoint_radius, **track)
        horizon = self.config['horizon_s'][kind] if self.seconds is None else self.seconds
        if track and self.seconds is None:
            # Long tracks: allow ~0.7 m/s average plus every hold, plus margin.
            length = float(np.linalg.norm(np.diff(self.mission.points, axis=0), axis=1).sum())
            horizon = min(300., max(20., length / .7 + float(np.sum(self.mission.dwell_s)) + 10.))
        self.max_steps = max(1, round(horizon / self.control_dt))
        if self.training:
            self.parameters = self._training_parameters(episode_seed)
        elif self.scenario in ('payload','joint'):
            self.parameters=scenario_spec('gust' if self.scenario=='joint' else 'nominal',episode_seed)
            self.parameters.update(name=self.scenario,mass_scale=1.2,
                                   payload_drop_s=4. if kind=='route' else 12.,payload_fraction=1/6)
        else:
            self.parameters=scenario_spec(self.scenario,episode_seed)
        self.payload_dropped = False
        self.payload_drop_event = None
        scale = self.parameters.get('mass_scale', 1.)
        self.mass = MASS * scale
        self.inertia = INERTIA * scale
        if self.physical_motors:
            cfg = DirectMotorConfig()
            self.inertia = np.asarray(cfg.body_inertia_diagonal_kgm2) * scale
            # Payload changes the rigid body, not rotor inertia or the actor's motor calibration.
            self.direct_model = DirectMotorModel(cfg)
            self.direct_model.reset(pwm=cfg.nominal_hover_pwm)
        pb.changeDynamics(self.body, -1, mass=self.mass, localInertiaDiagonal=self.inertia,
                          linearDamping=0, angularDamping=0, physicsClientId=self.cid)
        self.target = self.mission.target.copy()
        cfg = self.config['initial']
        position = self.target + initial_rng.uniform(-cfg['position_spread_m'], cfg['position_spread_m'], 3)
        velocity = initial_rng.uniform(-cfg['velocity_spread_mps'], cfg['velocity_spread_mps'], 3)
        rpy = initial_rng.uniform(-cfg['roll_pitch_spread_rad'], cfg['roll_pitch_spread_rad'], 3)
        rpy[2] = initial_rng.uniform(-np.pi, np.pi)
        self.yaw_reference = float(rpy[2])
        pid_config = dict(self.config.get('pid', {}))
        for key in ('velocity_xy_limit', 'acceleration_xy_limit'):
            if 'pid_' + key in self.parameters: pid_config[key] = self.parameters['pid_' + key]
        self.pid = CascadedPID(pid_config,self.parameters.get('pid_gain_scale',1.)) if self.architecture=='residual' else None
        if self.pid is not None:self.pid.reset(self.yaw_reference)
        if self.training:
            rpy[2] += seeded_rng(episode_seed, 460).uniform(-cfg['yaw_error_spread_rad'], cfg['yaw_error_spread_rad'])
        self.initial_state = dict(position=position.tolist(), velocity=velocity.tolist(),
                                  rpy=rpy.tolist(), body_rates=[0., 0., 0.])
        pb.resetBasePositionAndOrientation(self.body, position, pb.getQuaternionFromEuler(rpy), physicsClientId=self.cid)
        pb.resetBaseVelocity(self.body, velocity, [0, 0, 0], physicsClientId=self.cid)
        self.steps = 0
        self.previous_action = np.zeros(4)
        self.previous_motor_action = np.zeros(4)
        self.previous_target = None
        self.encoder.reset(self.yaw_reference)
        if callable(getattr(self.encoder,'set_task',None)):
            self.encoder.set_task(self.max_steps*self.control_dt,self.mission.kind)
        self.motor_forces = np.ones(4) * MASS * G / 4
        self.collective_force = MASS * G
        self.rate_integral = np.zeros(3)
        delay = self.parameters.get('delay_steps', 0)
        self.delay = deque([np.zeros(4) for _ in range(delay)])
        self.pwm_delay=deque([np.asarray(DirectMotorConfig().nominal_hover_pwm,dtype=float)
                              for _ in range(delay)])
        if callable(getattr(self.reward_tracker, 'reset', None)):
            self.reward_tracker.reset()
        self._ready, self._episode_done = True, False
        self.mission.update(0., position)
        observation = self.observation()
        info = self._info(self.state(), self.target, self.previous_target, 0, None, np.zeros(3),
                          False, 0., {}, np.zeros(4))
        info.update(initial_state=self.initial_state, parameters=self.parameters.copy(),
                    episode_seed=episode_seed, environment_version=ENVIRONMENT_VERSION,
                    config_sha256=self.config_sha256, observation_schema=self.observation_schema,
                    observation_size=self.encoder_config.size, command_schema=self.command_schema)
        return observation, info

    def _info(self, state, judged, previous, pre_index, event, force, failed,
              finite_distance, reward_terms, motor_action, completed_before=False):
        position, velocity, rotation, rates = state
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        yaw_error = float(np.arctan2(np.sin(self.yaw_reference - yaw), np.cos(self.yaw_reference - yaw)))
        return dict(position=position.tolist(), velocity=velocity.tolist(), rotation=rotation.tolist(),
            body_rates=rates.tolist(), target=np.asarray(judged).tolist(), target_before=np.asarray(judged).tolist(),
            next_target=self.target.tolist(), previous_target=None if previous is None else np.asarray(previous).tolist(),
            goal_index=int(self.mission.index), pre_goal_index=int(pre_index), judged_goal_index=int(pre_index),
            mission=self.mission.kind, mission_event=event, time=self.steps * self.control_dt, dt=self.control_dt,
            observation_schema=self.observation_schema, observation_size=self.encoder_config.size,
            command_schema=self.command_schema,
            route_completion_s=self.mission.completion_time, route_completed=self.mission.completion_time is not None,
            failed=bool(failed), yaw=yaw, yaw_error=yaw_error, yaw_reference=self.yaw_reference,
            force_enu_N=np.asarray(force).tolist(), motor_forces=self.motor_forces.tolist(),
            actual_motor_forces=self.motor_forces.tolist(), applied_motor_action=np.asarray(motor_action).tolist(),
            finite_distance=float(finite_distance), active_pre_leg=bool(self.mission.kind == 'route' and previous is not None
                and 1 <= pre_index <= self.mission.last_index and not completed_before),
            reward_terms=reward_terms, horizon_s=self.max_steps * self.control_dt,
            measured_state={name: value.tolist() for name, value in self.measured_state.items()},
            waypoint_radius_m=self.waypoint_radius, plant_model='detailed_rotors',
            mass_kg=self.mass,payload_dropped=self.payload_dropped,payload_drop_event=self.payload_drop_event,
            base_pwm=None if self.pid is None else self.pid.last_pwm.tolist(),
            baseline_kind=None if self.pid is None else self.pid.metadata()['kind'])

    def step(self, action):
        if not self._ready or self._episode_done:
            raise RuntimeError('Reset the episode before another action')
        desired = np.asarray(action, dtype=float)
        if desired.shape != (4,) or not np.isfinite(desired).all():
            raise ValueError('Four finite native actions required')
        desired = np.clip(desired, -1., 1.)
        applied = desired.copy()
        old_action = self.previous_action.copy()
        judged = self.target.copy()
        previous = None if self.previous_target is None else self.previous_target.copy()
        pre_index = int(self.mission.index)
        completed_before = self.mission.completion_time is not None
        error_before = float(np.linalg.norm(self.state()[0] - judged))
        force = force_N(self.steps * self.control_dt, self.parameters)
        # Both architectures share physical motor commands and their delays.
        # Only the residual baseline consumes the measured state, never true wind or mass.
        if self.physical_motors:
            if self.pid is None:
                pwm = self.direct_model.action_to_pwm(applied)
            else:
                base = self.pid.command(self.measured_state,judged,self.control_dt)
                pwm = np.rint(np.clip(base+applied*self.residual_limit_pwm,1100.,1900.))
            self.pwm_delay.append(np.asarray(pwm,dtype=float))
            pwm=self.pwm_delay.popleft()
            if self.pid is not None:self.pid.observe_applied_command(pwm,self.measured_state,self.control_dt)
            self.applied_pwm = np.asarray(pwm,dtype=float)
        t=self.steps*self.control_dt
        if not self.payload_dropped and t>=self.parameters.get('payload_drop_s',float('inf')):
            fraction=float(self.parameters['payload_fraction'])
            if not 0<fraction<1:raise ValueError('Payload drop fraction must lie in (0,1)')
            before=self.mass;self.mass*=1-fraction;self.inertia*=1-fraction
            pb.changeDynamics(self.body,-1,mass=self.mass,localInertiaDiagonal=self.inertia,physicsClientId=self.cid)
            self.payload_dropped=True
            self.payload_drop_event=dict(time_s=t,mass_before_kg=before,mass_after_kg=self.mass,
                                        model='instantaneous_centred_mass_release_no_impulse')
        for _ in range(self.decimation):
            position, velocity, rotation, rates = self.state()
            if self.physical_motors:
                efficiency = np.ones(4)
                if 'motor_index' in self.parameters:
                    efficiency[self.parameters['motor_index']]=self.parameters.get('motor_scale',1.)
                else:efficiency*=self.parameters.get('motor_scale',1.)
                result = self.direct_model.step_pwm(self.applied_pwm, self.dt, body_rates=rates,
                    body_velocity=rotation.T @ velocity, blade_efficiency=efficiency,
                    motor_torque_scale=np.ones(4)*self.parameters.get('motor_torque_scale',1.))
                self.motor_forces = result['forces_N'].copy()
                torque = result['body_torque_Nm']
            world_force = rotation[:, 2] * self.motor_forces.sum() + force
            if 'body_drag_N_per_mps' in self.plant_config:
                # Airframe/rotor drag of the physical-motor plant (Gazebo Iris LiftDrag:
                # ~1.52 N/(m/s) horizontal); absent key keeps the original drag-free plant.
                world_force -= np.asarray(self.plant_config['body_drag_N_per_mps']) * velocity
            pb.applyExternalForce(self.body, -1, world_force, position, pb.WORLD_FRAME, physicsClientId=self.cid)
            pb.applyExternalTorque(self.body, -1, rotation @ torque, pb.WORLD_FRAME, physicsClientId=self.cid)
            pb.stepSimulation(physicsClientId=self.cid)
        self.steps += 1
        self.previous_action = desired.copy()
        state = self.state()
        position, velocity, rotation, rates = state
        tracking_error = float(np.linalg.norm(position - judged))
        distance = finite_segment_distance(position, previous, judged)
        event = self.mission.update(self.steps * self.control_dt, position)
        self.target = self.mission.target.copy()
        if event is not None and not np.array_equal(self.target, judged):
            self.previous_target = judged.copy()
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        yaw_error = float(np.arctan2(np.sin(self.yaw_reference - yaw), np.cos(self.yaw_reference - yaw)))
        on_leg = self.mission.kind == 'route' and previous is not None and not completed_before
        failed = bool((distance if on_leg else tracking_error) > FAIL_DISTANCE_M or position[2] < .25 or position[2] > 10. or rotation[2, 2] < .2)
        horizon = self.steps >= self.max_steps
        completed = self.mission.completion_time is not None
        delta = desired - old_action
        reward_function = (getattr(self.reward_tracker, 'step', self.reward_tracker)
                           if self.reward_tracker is not None else (lambda **_: (0., {})))
        result = reward_function(mission=self.mission.kind, dt=self.control_dt,
            tracking_error=tracking_error, velocity_sq=float(velocity @ velocity), tilt_cos=float(rotation[2, 2]),
            yaw_error=yaw_error, rate_sq=float(rates @ rates), action_delta_sq=float(delta @ delta),
            distance_progress=error_before - tracking_error, arrival=event is not None,
            completion=bool(event is not None and completed), failed=failed, completed_before=completed_before,
            active_pre_leg=bool(self.mission.kind == 'route' and previous is not None
                                and 1 <= pre_index <= self.mission.last_index and not completed_before),
            finite_distance=distance, unfinished_horizon=bool(self.mission.kind == 'route' and horizon and not completed),
            position=position.copy(), velocity=velocity.copy(), rotation=rotation.copy(), body_rates=rates.copy(),
            target=judged.copy(), previous_target=None if previous is None else previous.copy(),
            action=desired.copy(), applied_action=applied.copy(), time_s=self.steps * self.control_dt,
            horizon_s=self.max_steps*self.control_dt,
            remaining_time_s=max(0.,(self.max_steps-self.steps)*self.control_dt))
        reward, terms = result if isinstance(result, tuple) else (result, {})
        reward = float(reward)
        if not math.isfinite(reward) or not isinstance(terms, dict):
            raise ValueError('Reward must return a finite number or (finite number, terms dict)')
        self._episode_done = bool(failed or horizon)
        # These are finite-duration tasks, including hover. Their remaining
        # duration and horizon are explicit observation features. No continuing
        # infinite-horizon bootstrap is used with gamma=1 and positive reward.
        observation = self.observation(state)
        motor_action = (applied if self.architecture == 'direct' else
                        2 * self.motor_forces / MAX_FORCE - 1)
        info = self._info(state, judged, previous, pre_index, event, force, failed, distance, terms,
                          motor_action, completed_before=completed_before)
        terminated=bool(failed or horizon)
        truncated=False
        info.update(error=tracking_error, altitude_error=float(position[2] - judged[2]), tilt_cos=float(rotation[2, 2]),
                    horizon_reached=horizon, finite_task_deadline=terminated and horizon,
                    applied_pwm=self.applied_pwm.tolist() if self.physical_motors else None,
                    **{'TimeLimit.truncated': truncated})
        return observation, reward, terminated, truncated, info

    def render(self):
        position = self.state()[0]
        view = pb.computeViewMatrixFromYawPitchRoll(position, 3.5, 45, -25, 0, 2)
        projection = pb.computeProjectionMatrixFOV(55, 4 / 3, .05, 50)
        _, _, rgba, _, _ = pb.getCameraImage(320, 240, view, projection,
            renderer=pb.ER_TINY_RENDERER, physicsClientId=self.cid)
        return np.asarray(rgba, dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

    def close(self):
        if pb.isConnected(self.cid): pb.disconnect(self.cid)
        self._ready = False

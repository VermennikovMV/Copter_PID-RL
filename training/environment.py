"""PyBullet quadrotor environment: Iris-class airframe with a physical rotor model.

Controllers
  direct    the action (4 values in [-1, 1], 0 = hover PWM) commands the motors (end-to-end);
  residual  the copy of the ArduPilot cascade (training/pid.py) commands the motors; corrections
            enter through its residual inputs (training.wrappers.AttitudeResidual).

The environment reward is 0: training.wrappers.TaskReward computes the task reward from `info`.
Domain randomisation (mass, payload drop, motors, delays, sensor noise, PID gains, gusts,
random tracks) is configured in configs/*.json; evaluation scenarios in experiments/scenarios.py.
"""
from collections import deque
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as pb

from controllers.base import COMMAND_SCHEMA_IDS, OBSERVATION_SCHEMA_ID, EncoderConfig, ObservationEncoder
from experiments.mission import Mission
from experiments.scenarios import force_N, scenario_spec, seeded_rng
from .physics import G, MASS, DirectMotorConfig, DirectMotorModel
from .pid import CascadedPID

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT = 0.02        # 50 Hz control
PHYSICS_DT = 0.005       # 200 Hz physics
FAIL_DISTANCE_M = 4.0    # loss of control: distance from the goal (hover) or from the active segment (route)
PWM_MIN, PWM_MAX = 1100., 1900.


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('schema') != 'quadrotor-training-config-v1':
        raise ValueError(f'{path}: unknown configuration schema')
    if cfg['control_dt_s'] != CONTROL_DT:
        raise ValueError(f'{path}: control_dt_s must be {CONTROL_DT}')
    return cfg


def segment_distance(position, a, b):
    """Distance from `position` to the segment a-b; 0 when there is no segment (a is None)."""
    if a is None:
        return 0.
    leg = np.asarray(b) - np.asarray(a)
    n2 = float(leg @ leg)
    if n2 <= 1e-15:
        return 0.
    f = np.clip(float((position - a) @ leg) / n2, 0., 1.)
    return float(np.linalg.norm(position - (a + f * leg)))


class QuadrotorEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 50}

    def __init__(self, controller='direct', training=False, scenario='nominal', mission=None, seconds=None,
                 waypoint_radius=0.4, config_path=None, render_mode=None):
        """controller: 'direct' | 'residual'; training: randomised episodes (configs "training_distribution",
        "randomization", random tracks) instead of the evaluation `scenario`; mission: 'hover' | 'route' |
        None (training: drawn per episode, evaluation: hover); seconds: episode length override."""
        super().__init__()
        if controller not in ('direct', 'residual'):
            raise ValueError('controller must be direct or residual')
        if mission not in (None, 'hover', 'route'):
            raise ValueError('mission must be hover, route or None')
        self.controller = controller
        self.training, self.scenario, self.mission_override = bool(training), scenario, mission
        self.seconds = None if seconds is None else float(seconds)
        self.waypoint_radius = float(waypoint_radius)
        self.render_mode = render_mode
        self.config_path = Path(config_path or ROOT / 'configs/compensator_eval.json')
        self.config = load_config(self.config_path)
        if (self.config['observation_schema'] != OBSERVATION_SCHEMA_ID
                or self.config.get('command_schemas', {}).get(controller) != COMMAND_SCHEMA_IDS[controller]):
            raise ValueError(f'{self.config_path}: observation or command schema differs from the code')
        self.drag = np.asarray(self.config['plant']['body_drag_N_per_mps'], float)   # N/(m/s), world x, y, z
        self.control_dt, self.dt = CONTROL_DT, PHYSICS_DT
        self.decimation = round(CONTROL_DT / PHYSICS_DT)
        self.motor_config = DirectMotorConfig()
        self.encoder = ObservationEncoder(EncoderConfig(**self.config.get('observation', {})))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.encoder.config.size,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)

        self.cid = pb.connect(pb.GUI if render_mode == 'human' else pb.DIRECT)
        pb.setGravity(0, 0, -G, physicsClientId=self.cid)
        pb.setTimeStep(self.dt, physicsClientId=self.cid)
        box = dict(halfExtents=[.13, .10, .035], physicsClientId=self.cid)
        self.body = pb.createMultiBody(MASS, pb.createCollisionShape(pb.GEOM_BOX, **box),
                                       pb.createVisualShape(pb.GEOM_BOX, rgbaColor=[.15, .35, .85, 1], **box),
                                       basePosition=[0, 0, 3], physicsClientId=self.cid)
        pb.createMultiBody(0, pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=self.cid),
                           physicsClientId=self.cid)
        self.measured_state = None
        self.pid = None

    # ------------------------------------------------------------------ state
    def state(self):
        """True state: position, velocity (world ENU), rotation body->world, body rates (FLU)."""
        position, quaternion = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        velocity, omega = pb.getBaseVelocity(self.body, physicsClientId=self.cid)
        rotation = np.asarray(pb.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return np.asarray(position), np.asarray(velocity), rotation, rotation.T @ np.asarray(omega)

    def observation(self, state=None):
        """Measured state (true state + sensor noise) -> network inputs. The measured state is also
        what the cascade copy and the wrappers see; nobody gets the true force or mass."""
        position, velocity, rotation, rates = self.state() if state is None else state
        p = self.parameters
        position = position + self.sensor_rng.normal(0., p.get('noise_position', 0.), 3)
        velocity = velocity + self.sensor_rng.normal(0., p.get('noise_velocity', 0.), 3)
        rates = rates + self.sensor_rng.normal(0., p.get('noise_rates', 0.), 3)
        tilt = self.sensor_rng.normal(0., p.get('noise_orientation', 0.), 3)
        rotation = rotation @ np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(tilt))).reshape(3, 3)
        self.measured_state = dict(position=position.copy(), velocity=velocity.copy(),
                                   rotation=rotation.copy(), body_rates=rates.copy())
        return np.asarray(self.encoder.observe(position, velocity, rotation, rates, self.target,
                                               self.previous_action, self.control_dt), dtype=np.float32)

    # ------------------------------------------------------------------ episode sampling
    def _training_episode(self, seed):
        """Random disturbances and vehicle parameters of a training episode."""
        rng = seeded_rng(seed, 450)
        dist, rnd = self.config['training_distribution'], self.config['randomization']
        horizon = self.max_steps * self.control_dt

        def gust(on_max):
            magnitude = float(rng.uniform(dist['force_min_N'], dist['force_max_N']))
            angle = float(rng.uniform(-np.pi, np.pi))
            on = float(rng.uniform(dist['on_min_s'], on_max))
            return magnitude, angle, on

        def event(on, off, magnitude, angle):
            return dict(on_s=on, off_s=off, force_enu_N=[float(magnitude * np.cos(angle)),
                                                         float(magnitude * np.sin(angle)), 0.])

        draw = rng.random()
        p_nominal, p_constant = dist['nominal_probability'], dist['constant_probability']
        regime = 'nominal' if draw < p_nominal else 'constant' if draw < p_nominal + p_constant else 'pulse'
        spec = dict(name='training', regime=regime, force_events=[])
        if regime != 'nominal':
            magnitude, angle, on = gust(dist['route_on_max_s'] if self.mission.kind == 'route' else dist['on_max_s'])
            off = horizon + 1. if regime == 'constant' else \
                on + float(rng.uniform(dist['pulse_duration_min_s'], dist['pulse_duration_max_s']))
            spec['force_events'].append(event(on, off, magnitude, angle))
            for _ in range(int(rng.integers(0, dist['extra_pulses_max'] + 1))):   # gust sequences
                magnitude = float(rng.uniform(dist['force_min_N'], dist['force_max_N']))
                angle = float(rng.uniform(-np.pi, np.pi))
                on = float(rng.uniform(dist['on_min_s'], max(dist['on_min_s'] + .1, horizon - 1.)))
                off = on + float(rng.uniform(dist['extra_pulse_min_s'], dist['extra_pulse_max_s']))
                spec['force_events'].append(event(on, off, magnitude, angle))

        def uniform(name):
            low, high = rnd[name]
            return float(rng.uniform(low, high))

        spec['mass_scale'] = uniform('mass_scale_range')
        spec['motor_scale'] = uniform('motor_scale_range')
        spec['motor_torque_scale'] = uniform('torque_scale_range')
        spec['pid_gain_scale'] = uniform('pid_scale_range')
        for name in ('position', 'velocity', 'orientation', 'rates'):
            spec['noise_' + name] = uniform(f'noise_{name}_range')
        low, high = rnd['delay_steps_range']
        spec['delay_steps'] = int(rng.integers(low, high + 1))
        if rng.random() < rnd['payload_probability']:
            # the payload is the part of the mass above nominal: it can be dropped in flight
            fraction = min(uniform('payload_fraction_range'), max(0., (spec['mass_scale'] - 1.) / spec['mass_scale']))
            if fraction > 1e-8:
                key = 'route_payload_drop_time_fraction_range' if self.mission.kind == 'route' \
                    else 'payload_drop_time_fraction_range'
                spec.update(payload_drop_s=horizon * uniform(key), payload_fraction=fraction)
        return spec

    def _evaluation_episode(self, seed):
        """Fixed evaluation scenario (experiments/scenarios.py); payload/joint carry 20 % extra mass
        dropped at 12 s (hover) or 4 s (route)."""
        if self.scenario not in ('payload', 'joint'):
            return scenario_spec(self.scenario, seed)
        spec = scenario_spec('gust' if self.scenario == 'joint' else 'nominal', seed)
        spec.update(name=self.scenario, mass_scale=1.2, payload_fraction=1 / 6,
                    payload_drop_s=4. if self.mission.kind == 'route' else 12.)
        return spec

    def _track(self, kind, seed):
        """Mission keyword arguments: random track in training, the configured fixed track in
        evaluation, {} for the default route of experiments/mission.py."""
        spec = self.config.get('track')
        if kind != 'route' or not spec:
            return {}
        if not self.training:
            fixed = spec['fixed']
            return dict(points=fixed['points'], dwell_s=fixed['dwell_s'], radii_m=fixed['radii_m'])
        r, rng = spec['random'], seeded_rng(seed, 470)
        n = int(rng.integers(r['legs'][0], r['legs'][1] + 1))
        points = [np.array([0., 0., float(rng.uniform(*r['altitude_m']))])]
        heading = float(rng.uniform(-np.pi, np.pi))
        for i in range(n):
            if i:
                heading += float(rng.uniform(-1, 1)) * np.radians(r['max_turn_deg'])
            length = float(rng.uniform(*r['leg_m']))
            nxt = points[-1] + np.array([np.cos(heading), np.sin(heading), 0.]) * length
            nxt[2] = float(rng.uniform(*r['altitude_m']))
            points.append(nxt)
        dwell = [.5] + [float(rng.uniform(*r['stop_dwell_s'])) if rng.random() < r['stop_probability'] else 0.
                        for _ in range(n - 1)] + [r['final_dwell_s']]
        radii = [r['stop_radius_m']] + [r['stop_radius_m'] if d > 0 else r['pass_radius_m'] for d in dwell[1:]]
        return dict(points=np.array(points), dwell_s=dwell, radii_m=radii)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed = int(self.np_random.integers(0, 2**31)) if self.training or seed is None else int(seed)
        self.episode_seed = seed
        self.sensor_rng = seeded_rng(seed, 220)
        initial_rng = seeded_rng(seed, 330)

        kind = self.mission_override
        if kind is None:
            route = self.training and seeded_rng(seed, 440).random() < self.config['training_distribution']['route_probability']
            kind = 'route' if route else 'hover'
        track = self._track(kind, seed)
        self.mission = Mission(kind, self.waypoint_radius, **track)
        horizon = self.seconds if self.seconds is not None else self.config['horizon_s'][kind]
        if track and self.seconds is None:   # long tracks: ~0.7 m/s average plus every stop, plus margin
            length = float(np.linalg.norm(np.diff(self.mission.points, axis=0), axis=1).sum())
            horizon = min(300., max(20., length / .7 + float(np.sum(self.mission.dwell_s)) + 10.))
        self.max_steps = max(1, round(horizon / self.control_dt))
        self.parameters = self._training_episode(seed) if self.training else self._evaluation_episode(seed)

        scale = self.parameters.get('mass_scale', 1.)
        self.mass = MASS * scale
        self.inertia = np.asarray(self.motor_config.body_inertia_diagonal_kgm2) * scale
        self.payload_dropped, self.payload_drop_event = False, None
        pb.changeDynamics(self.body, -1, mass=self.mass, localInertiaDiagonal=self.inertia,
                          linearDamping=0, angularDamping=0, physicsClientId=self.cid)
        self.motors = DirectMotorModel(self.motor_config)
        self.motors.reset(pwm=self.motor_config.nominal_hover_pwm)
        self.blade_efficiency = np.full(4, self.parameters.get('motor_scale', 1.))
        if 'motor_index' in self.parameters:   # single weak motor scenarios
            self.blade_efficiency = np.ones(4)
            self.blade_efficiency[self.parameters['motor_index']] = self.parameters['motor_scale']
        self.torque_scale = np.full(4, self.parameters.get('motor_torque_scale', 1.))

        self.target = self.mission.target.copy()
        init = self.config['initial']
        position = self.target + initial_rng.uniform(-init['position_spread_m'], init['position_spread_m'], 3)
        velocity = initial_rng.uniform(-init['velocity_spread_mps'], init['velocity_spread_mps'], 3)
        rpy = initial_rng.uniform(-init['roll_pitch_spread_rad'], init['roll_pitch_spread_rad'], 3)
        rpy[2] = initial_rng.uniform(-np.pi, np.pi)
        self.yaw_reference = float(rpy[2])
        if self.controller == 'residual':
            self.pid = CascadedPID(dict(self.config.get('pid', {})), self.parameters.get('pid_gain_scale', 1.))
            self.pid.reset(self.yaw_reference)
        if self.training:
            rpy[2] += seeded_rng(seed, 460).uniform(-init['yaw_error_spread_rad'], init['yaw_error_spread_rad'])
        pb.resetBasePositionAndOrientation(self.body, position, pb.getQuaternionFromEuler(rpy), physicsClientId=self.cid)
        pb.resetBaseVelocity(self.body, velocity, [0, 0, 0], physicsClientId=self.cid)

        self.steps = 0
        self.previous_action = np.zeros(4)
        self.previous_target = None
        self.pwm_delay = deque(np.asarray(self.motor_config.nominal_hover_pwm, float)
                               for _ in range(self.parameters.get('delay_steps', 0)))
        self.applied_pwm = np.asarray(self.motor_config.nominal_hover_pwm, float)
        self.encoder.reset(self.yaw_reference)
        self.mission.update(0., position)
        observation = self.observation()
        info = self._info(self.state(), self.target, None, 0, None, np.zeros(3), False, 0.)
        info.update(parameters=self.parameters.copy(), episode_seed=seed)
        return observation, info

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1., 1.)
        target = self.target.copy()                       # goal flown towards during this step
        previous = None if self.previous_target is None else self.previous_target.copy()
        goal_index = int(self.mission.index)
        completed_before = self.mission.completion_time is not None
        t = self.steps * self.control_dt
        force = force_N(t, self.parameters)

        if self.pid is None:
            pwm = self.motors.action_to_pwm(action)
        else:
            pwm = np.rint(np.clip(self.pid.command(self.measured_state, target, self.control_dt), PWM_MIN, PWM_MAX))
        self.pwm_delay.append(np.asarray(pwm, float))     # command delay (randomised in training)
        self.applied_pwm = np.asarray(self.pwm_delay.popleft(), float)
        if self.pid is not None:
            self.pid.observe_applied_command(self.applied_pwm, self.measured_state, self.control_dt)

        if not self.payload_dropped and t >= self.parameters.get('payload_drop_s', np.inf):
            before = self.mass
            self.mass *= 1 - self.parameters['payload_fraction']
            self.inertia *= 1 - self.parameters['payload_fraction']
            pb.changeDynamics(self.body, -1, mass=self.mass, localInertiaDiagonal=self.inertia, physicsClientId=self.cid)
            self.payload_dropped = True
            self.payload_drop_event = dict(time_s=t, mass_before_kg=before, mass_after_kg=self.mass)

        for _ in range(self.decimation):
            position, velocity, rotation, rates = self.state()
            rotor = self.motors.step_pwm(self.applied_pwm, self.dt, body_rates=rates, body_velocity=rotation.T @ velocity,
                                         blade_efficiency=self.blade_efficiency, motor_torque_scale=self.torque_scale)
            world_force = rotation[:, 2] * rotor['forces_N'].sum() + force
            world_force -= self.drag * velocity           # airframe drag (Gazebo Iris: ~1.52 N/(m/s))
            pb.applyExternalForce(self.body, -1, world_force, position, pb.WORLD_FRAME, physicsClientId=self.cid)
            pb.applyExternalTorque(self.body, -1, rotation @ rotor['body_torque_Nm'], pb.WORLD_FRAME,
                                   physicsClientId=self.cid)
            pb.stepSimulation(physicsClientId=self.cid)
        self.steps += 1
        self.previous_action = action.copy()

        state = self.state()
        position, rotation = state[0], state[2]
        on_leg = (self.mission.kind == 'route' and previous is not None and 1 <= goal_index <= self.mission.last_index
                  and not completed_before)
        distance = segment_distance(position, previous, target)
        event = self.mission.update(self.steps * self.control_dt, position)
        self.target = self.mission.target.copy()
        if event is not None and not np.array_equal(self.target, target):
            self.previous_target = target.copy()
        off_course = distance if on_leg else float(np.linalg.norm(position - target))
        failed = bool(off_course > FAIL_DISTANCE_M or not .25 <= position[2] <= 10. or rotation[2, 2] < .2)
        horizon = self.steps >= self.max_steps
        observation = self.observation(state)
        info = self._info(state, target, previous, goal_index, event, force, failed, distance, on_leg)
        info.update(horizon_reached=horizon, tilt_cos=float(rotation[2, 2]), applied_pwm=self.applied_pwm.tolist())
        # finite-duration task: the end of the episode is terminal, not a truncation
        return observation, 0., bool(failed or horizon), False, info

    def _info(self, state, target, previous, goal_index, event, force, failed, distance, on_leg=False):
        position, velocity, rotation, rates = state
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return dict(
            time=self.steps * self.control_dt, mission=self.mission.kind, mission_event=event,
            position=position.tolist(), velocity=velocity.tolist(), rotation=rotation.tolist(), body_rates=rates.tolist(),
            target=np.asarray(target).tolist(), goal_index=goal_index, next_target=self.target.tolist(),
            previous_target=None if previous is None else np.asarray(previous).tolist(),
            on_leg=bool(on_leg), finite_distance=float(distance),
            route_completed=self.mission.completion_time is not None, route_completion_s=self.mission.completion_time,
            yaw_error=float(np.arctan2(np.sin(self.yaw_reference - yaw), np.cos(self.yaw_reference - yaw))),
            failed=bool(failed), force_enu_N=np.asarray(force).tolist(), mass_kg=self.mass,
            payload_dropped=self.payload_dropped, payload_drop_event=self.payload_drop_event)

    def render(self):
        view = pb.computeViewMatrixFromYawPitchRoll(self.state()[0], 3.5, 45, -25, 0, 2)
        projection = pb.computeProjectionMatrixFOV(55, 4 / 3, .05, 50)
        _, _, rgba, _, _ = pb.getCameraImage(320, 240, view, projection, renderer=pb.ER_TINY_RENDERER,
                                             physicsClientId=self.cid)
        return np.asarray(rgba, dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

    def close(self):
        if pb.isConnected(self.cid):
            pb.disconnect(self.cid)

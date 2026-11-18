"""Gymnasium wrappers around training.environment.QuadrotorEnv.

TaskReward               - task reward computed from the environment `info`
AttitudeResidual         - bounded tilt/thrust corrections applied inside the ArduPilot cascade copy
GuidedShaping            - ArduPilot GUIDED target shaping (AC_PosControl::input_pos_NED_m)
PidTargetObservation     - adds the position-controller velocity/acceleration targets to the inputs
TrackObservation         - adds the track ahead (next points, stop flag, distances) to the inputs
RandomTrackStart         - training only: episodes start anywhere on the track, already moving
DisturbanceEstimator     - learned estimate of the disturbance acting on the vehicle
DisturbanceCompensation  - estimate -> physics compensation -> bounded residual input
"""
from dataclasses import dataclass
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

G = 9.81
ANGLE_MAX_RAD = 0.26   # 15 deg, firmware RESID_ANG_MAX
THRUST_MAX = 0.15      # firmware RESID_THR_MAX


def _append(env, extra):
    n = env.observation_space.shape[0] + extra
    return gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)


# --------------------------------------------------------------------------- reward
@dataclass
class RewardConfig:
    pos_lin: float = 2.0        # hover: per metre of position error
    pos_quad: float = 5.0       # hover: per metre^2
    z_extra: float = 2.0        # per metre of altitude error
    yaw: float = 1.0            # per radian of heading error
    vel: float = 0.1            # hover: per m/s
    rate: float = 0.02          # per (rad/s)^2 of body rate
    smooth: float = 0.02        # per sum((dPWM / 100 us)^2)
    seg_lin: float = 3.0        # route: per metre off the active segment
    seg_quad: float = 10.0      # route: per metre^2 off the active segment
    progress: float = 10.0      # route: per metre of progress towards the active waypoint
    crash: float = 100.0        # once, on loss of control
    action_rate: float = 0.1    # per |a_t - a_(t-1)|^2
    action_size: float = 0.05   # per |a_t|^2
    route_speed_mode: bool = False  # route: no per-step bonus until finished, time penalty instead
    route_time: float = 0.2     # (speed mode) per step until the route is finished
    route_quality: float = 0.2  # (speed mode) inaccuracy penalty: -k * (1 - exp(-cost)) per step
    unfinished_per_m: float = 5.0


class TaskReward(gym.Wrapper):
    """Per step: quality = exp(-sum of costs) in (0, 1]; on an active route leg also
    progress towards the waypoint. Crash: -crash once. A time limit is a truncation."""
    def __init__(self, env, config=None):
        super().__init__(env)
        self.cfg = config if isinstance(config, RewardConfig) else RewardConfig(**(config or {}))
        self.prev_pwm = self.prev_pos = self.action = self.prev_action = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev_pwm, self.prev_action = None, None
        self.prev_pos = np.asarray(info['position'], float)
        return obs, info

    def set_stage(self, stage):
        """Training curriculum (training/ppo.py): reward weights, random-start settings and the
        random-track geometry of this environment; takes effect from the next episode."""
        for k, v in stage.get('reward', {}).items():
            setattr(self.cfg, k, v)
        env = self.env
        while isinstance(env, gym.Wrapper):
            if isinstance(env, RandomTrackStart):
                env.p_random = stage.get('random_start', env.p_random)
                env.vmax = stage.get('start_vmax', env.vmax)
                env.max_off_m = stage.get('max_offtrack', env.max_off_m)
            env = env.env
        track = self.env.unwrapped.config.get('track', {}).get('random')
        if track is not None:
            track['leg_m'] = stage.get('legs_m', track['leg_m'])
            track['max_turn_deg'] = stage.get('max_turn_deg', track['max_turn_deg'])
        return True

    def terms(self, info):
        c = self.cfg
        p = np.asarray(info['position'], float)
        tgt = np.asarray(info['target'], float)
        v = np.asarray(info['velocity'], float)
        w = np.asarray(info['body_rates'], float)
        pwm = np.asarray(info['applied_pwm'], float) if info.get('applied_pwm') is not None else None
        e = float(np.linalg.norm(p - tgt))
        ez = abs(float(p[2] - tgt[2]))
        cost = dict(heading=c.yaw * abs(float(info['yaw_error'])), rate=c.rate * min(float(w @ w), 25.),
                    z=c.z_extra * min(ez, 2.))
        if pwm is not None and self.prev_pwm is not None:
            d = (pwm - self.prev_pwm) / 100.
            cost['smooth'] = c.smooth * float(d @ d)
        if self.action is not None:
            cost['action_size'] = c.action_size * float(self.action @ self.action)
            if self.prev_action is not None:
                da = self.action - self.prev_action
                cost['action_rate'] = c.action_rate * float(da @ da)
        t = {}
        if info['on_leg']:
            d = float(info['finite_distance'])
            cost['segment'] = c.seg_lin * d + c.seg_quad * d * d
            t['progress'] = c.progress * (float(np.linalg.norm(self.prev_pos - tgt)) - e)
        else:
            em = min(e, 2.)
            cost['position'] = c.pos_lin * em + c.pos_quad * em * em
            cost['velocity'] = c.vel * min(float(np.linalg.norm(v)), 3.)
        q = float(np.exp(-sum(cost.values())))
        if c.route_speed_mode and info['mission'] == 'route' and not info['route_completed']:
            t['quality'] = -c.route_quality * (1. - q)
            t['time'] = -c.route_time
        else:
            t['quality'] = q
        if info['failed']:
            t['crash'] = -c.crash
        if info['mission'] == 'route' and info['horizon_reached'] and not info['route_completed']:
            t['unfinished'] = -c.unfinished_per_m * e
        self.prev_pwm, self.prev_pos = pwm, p
        return t, cost

    def step(self, action):
        self.action = np.clip(np.asarray(action, float), -1, 1)
        obs, _, term, trunc, info = self.env.step(action)
        t, cost = self.terms(info)
        self.prev_action = self.action
        info['reward_terms'], info['reward_costs'] = t, cost
        if term and info['horizon_reached'] and not info['failed']:
            term, trunc = False, True            # time limit, not a failure
        done_t = info.get('route_completion_s')
        if done_t is not None and info['time'] - done_t > 3. and not term:
            trunc = True                         # 3 s of holding at the finish, then end
        return obs, float(sum(t.values())), term, trunc, info


# ------------------------------------------------------------------ cascade inputs
class AttitudeResidual(gym.Wrapper):
    """Action in [-1, 1]^4 -> tilt offset (roll, pitch, yaw; +-ANGLE_MAX_RAD) added to the attitude
    error before the angle P controller, and collective thrust fraction (+-THRUST_MAX).
    Same semantics as AC_Residual in the firmware (training.environment.QuadrotorEnv(controller='residual'))."""
    def __init__(self, env, delay_probability=0.05):
        super().__init__(env)
        self.delay_probability = delay_probability
        self.delay, self.pending = 0, np.zeros(4)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        base = self.env.unwrapped
        self.delay = int(np.random.default_rng(base.episode_seed + 5311).random() < self.delay_probability) \
            if base.training else 0
        self.pending = np.zeros(4)
        return obs, info

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        if self.delay:
            a, self.pending = self.pending, a
        pid = self.env.unwrapped.pid
        pid.angle_offset = a[:3] * ANGLE_MAX_RAD
        pid.thrust_offset = float(a[3] * THRUST_MAX)
        return self.env.step(a)


def shaping_params(stand_params_path):
    """GUIDED shaping limits read back from the SITL stand (configs/stand_params.json)."""
    w = json.loads(Path(stand_params_path).read_text(encoding='utf-8'))['wpnav_params']
    return dict(speed_ne_ms=w['wp_speed_ms'], accel_ne_mss=w['wp_accel_mss'], speed_up_ms=w['wp_speed_up_ms'],
                speed_down_ms=w['wp_speed_down_ms'], accel_d_mss=w['wp_accel_z_mss'],
                ang_vel_roll_max_degs=w['ang_vel_roll_max_degs'], ang_vel_pitch_max_degs=w['ang_vel_pitch_max_degs'],
                accel_roll_max_degss=w['accel_roll_max_degss'], accel_pitch_max_degss=w['accel_pitch_max_degss'])


class GuidedShaping(gym.Wrapper):
    """GUIDED as ArduPilot runs it: each point target is shaped (training/guided_shaper.py) into a
    moving target with velocity/acceleration feed-forward for the cascade copy. In training the
    speed (1-5 m/s) and acceleration (1.5-4 m/s^2) limits are randomised per episode."""
    def __init__(self, env, stand_params_path, randomize=False, overrides=None):
        super().__init__(env)
        self.base = shaping_params(stand_params_path)
        self.base.update(overrides or {})
        self.randomize = randomize

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        base = self.env.unwrapped
        if base.pid is None:
            return obs, info
        from training.guided_shaper import GuidedShaper, GuidedShapingParams
        p = dict(self.base)
        if self.randomize:
            rng = np.random.default_rng(base.episode_seed + 9173)
            p['speed_ne_ms'] = float(rng.uniform(1., 5.))
            p['accel_ne_mss'] = float(rng.uniform(1.5, 4.))
        shaper = GuidedShaper(GuidedShapingParams(**p))
        shaper.start(np.asarray(base.measured_state['position']), np.asarray(base.measured_state['velocity']))
        pid = base.pid
        plain = pid.command

        def command(measured, target, dt, velocity_ff=None, acceleration_ff=None):
            goal, vff, aff = shaper.update(target, dt)
            return plain(measured, goal, dt, velocity_ff=vff, acceleration_ff=aff)
        pid.command = command
        info['shaping'] = p
        return obs, info


# --------------------------------------------------------------------- observations
class _AppendObservation(gym.Wrapper):
    EXTRA = 0

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = _append(env, self.EXTRA)

    def features(self):
        raise NotImplementedError

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return np.r_[obs, self.features()].astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return np.r_[obs, self.features()].astype(np.float32), r, term, trunc, info


class PidTargetObservation(_AppendObservation):
    """Position-controller velocity target (/3 m/s) and acceleration target (/5 m/s^2) in the
    heading frame - the same values AC_PosControl exposes onboard."""
    EXTRA = 6

    def features(self):
        from controllers.base import heading_rotation
        base = self.env.unwrapped
        diag = getattr(base.pid, 'diagnostics', None) or {}
        if not diag or base.measured_state is None:
            return np.zeros(self.EXTRA, np.float32)
        _, h = heading_rotation(np.asarray(base.measured_state['rotation']))
        return np.r_[h @ np.asarray(diag['velocity_target'], float) / 3.,
                     h @ np.asarray(diag['acceleration_target'], float) / 5.].astype(np.float32)


def track_features(mission, position, rotation, target_distance=True):
    """Track ahead in the heading frame: the two points after the current target (/10 m),
    stop flag of the current target, distance along the track to the next stop (/20 m) and
    the distance to the current target (/10 m)."""
    from controllers.base import heading_rotation
    p = np.asarray(position, float)
    _, h = heading_rotation(np.asarray(rotation, float))
    pts, last = mission.points, mission.last_index
    i = min(mission.index, last)
    if mission.kind == 'hover':
        tgt = mission.target
        nxt, stop, dist = [tgt, tgt], 1., float(np.linalg.norm(p - tgt))
    else:
        nxt = [pts[min(i + 1, last)], pts[min(i + 2, last)]]
        stop = 1. if mission.dwell_s[i] > 0 else 0.
        dist, j = float(np.linalg.norm(p - pts[i])), i
        while mission.dwell_s[j] <= 0 and j < last:
            dist += float(np.linalg.norm(pts[j + 1] - pts[j])); j += 1
    f = np.r_[h @ (nxt[0] - p) / 10., h @ (nxt[1] - p) / 10., stop, min(dist, 60.) / 20.]
    if target_distance:
        f = np.r_[f, min(float(np.linalg.norm(p - mission.target)), 60.) / 10.]
    return f


class TrackObservation(_AppendObservation):
    def __init__(self, env, target_distance=True):
        self.target_distance = target_distance
        self.EXTRA = 9 if target_distance else 8
        super().__init__(env)

    def features(self):
        base = self.env.unwrapped
        ms = base.measured_state
        p = ms['position'] if ms else base.state()[0]
        rot = ms['rotation'] if ms else base.state()[2]
        return track_features(base.mission, p, rot, self.target_distance).astype(np.float32)


class RandomTrackStart(gym.Wrapper):
    """Training only, route episodes: with probability p_random start at a random point of a random
    leg, already moving along it (up to vmax; a third of them from rest); end after `waypoints`
    reached points or `seconds`; a deviation from the active segment above max_off_m is a crash."""
    def __init__(self, env, p_random=0.5, vmax=8.0, max_off_m=3.0, waypoints=3, seconds=25.):
        super().__init__(env)
        self.p_random, self.vmax, self.max_off_m = p_random, vmax, max_off_m
        self.waypoints, self.seconds = waypoints, seconds
        self.events = 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        b = self.env.unwrapped
        self.events = 0
        if not b.training or b.mission.kind != 'route':
            return obs, info
        import pybullet as pb
        rng = np.random.default_rng(b.episode_seed + 7717)
        m = b.mission
        b.max_steps = min(b.max_steps, int(round(self.seconds / b.control_dt)))
        if callable(getattr(b.encoder, 'set_task', None)):
            b.encoder.set_task(b.max_steps * b.control_dt, m.kind)
        if rng.random() >= self.p_random:
            return obs, info
        i = int(rng.integers(1, m.last_index + 1))
        a, c = m.points[i - 1], m.points[i]
        leg = c - a
        d = leg / max(np.linalg.norm(leg), 1e-6)
        pos = a + rng.uniform(0., .85) * leg + rng.normal(0, .25, 3)
        pos[2] = max(pos[2], 1.)
        vel = (0. if rng.random() < 1 / 3 else d * rng.uniform(0., self.vmax)) + rng.normal(0, .2, 3)
        tilt, az = np.radians(rng.uniform(0, 20)), rng.uniform(-np.pi, np.pi)
        rpy = [tilt * np.cos(az), tilt * np.sin(az), b.yaw_reference + rng.normal(0, .1)]
        pb.resetBasePositionAndOrientation(b.body, pos, pb.getQuaternionFromEuler(rpy), physicsClientId=b.cid)
        pb.resetBaseVelocity(b.body, vel, rng.normal(0, .3, 3), physicsClientId=b.cid)
        m.index, m.inside_since = i, None
        b.target, b.previous_target = m.target.copy(), a.copy()
        b.encoder.reset(b.yaw_reference)
        if callable(getattr(b.encoder, 'set_task', None)):
            b.encoder.set_task(b.max_steps * b.control_dt, m.kind)
        obs = b.observation()
        info = dict(info, position=pos.tolist(), random_start_leg=i)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        b = self.env.unwrapped
        if not b.training or b.mission.kind != 'route':
            return obs, r, term, trunc, info
        if info.get('mission_event'):
            self.events += 1
            if self.events >= self.waypoints and not term:
                trunc = True
        if info.get('on_leg') and info.get('finite_distance', 0.) > self.max_off_m and not term:
            info['failed'], term = True, True
        return obs, r, term, trunc, info


# ------------------------------------------------------- disturbance compensation
class ThrustHistory:
    """Collective thrust command of the cascade relative to hover thrust: raw and low-pass filtered
    (0.3 s, 1.5 s). Its average equals the effective mass. Onboard: throttle / RESID_THR_HOVR."""
    TAUS = (0.3, 1.5)

    def __init__(self, dt=0.02):
        self.alpha = [1. - np.exp(-dt / t) for t in self.TAUS]
        self.reset()

    def reset(self):
        self.now, self.ema, self.started = 1., [1.] * len(self.TAUS), False

    def push(self, pid):
        x = float((getattr(pid, 'diagnostics', None) or {}).get('thrust_ratio', 1.))
        self.now = x
        if not self.started:
            self.ema, self.started = [x] * len(self.TAUS), True
        else:
            self.ema = [e + a * (x - e) for e, a in zip(self.ema, self.alpha)]

    def features(self):
        return (np.array([self.now, *self.ema]) - 1.) * 5.


class DisturbanceEstimator:
    """Dense tanh MLP trained with supervision (training/estimator.py; keys W0, b0, W1, ..., y_mean, y_std).
    Input (83): 80 vehicle/cascade inputs + 3 thrust inputs. Output: disturbance acceleration in the
    heading frame (x, y), vertical imbalance (m/s^2), effective mass ratio."""
    def __init__(self, path):
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith('W'))
        self.layers = [(z[f'W{i}'], z[f'b{i}']) for i in range(n)]
        self.y_mean, self.y_std = z['y_mean'], z['y_std']

    def __call__(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x * self.y_std + self.y_mean


def compensation(estimate, kxy=1.0, kz=0.5):
    """Physics compensation in normalised residual units: tilt towards the disturbance
    (a / g, small angle) and a thrust fraction for the vertical imbalance."""
    c = np.zeros(4)
    c[:2] = kxy * np.array([estimate[1], -estimate[0]]) / G / ANGLE_MAX_RAD
    c[3] = -kz * estimate[2] / G / THRUST_MAX
    return np.clip(c, -1, 1)


class DisturbanceCompensation(gym.Wrapper):
    """On top of PidTargetObservation (80 inputs). Every step the estimator infers the disturbance,
    the physics compensation is applied through AttitudeResidual; an optional action is added to it.
    The observation is passed through unchanged."""
    def __init__(self, env, estimator_path, kxy=1.0, kz=0.5):
        super().__init__(env)
        self.est = DisturbanceEstimator(estimator_path)
        self.thrust = ThrustHistory()
        self.kxy, self.kz = kxy, kz
        self.comp, self.estimate = np.zeros(4), np.zeros(4)

    def _update(self, obs):
        self.thrust.push(self.env.unwrapped.pid)
        self.estimate = self.est(np.r_[np.asarray(obs, np.float64), self.thrust.features()])
        self.comp = compensation(self.estimate, self.kxy, self.kz)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.thrust.reset()
        self._update(obs)
        return obs, info

    def step(self, action):
        a = np.clip(self.comp + np.clip(np.asarray(action, float), -1, 1), -1, 1)
        obs, r, term, trunc, info = self.env.step(a)
        info['compensation'], info['estimate'] = self.comp.tolist(), self.estimate.tolist()
        self._update(obs)
        return obs, r, term, trunc, info

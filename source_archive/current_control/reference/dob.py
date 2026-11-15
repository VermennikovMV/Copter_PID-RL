"""Cascaded PID with a bounded, measured-only acceleration disturbance observer.

This is an analytical reference, not a learned controller or the thesis hybrid.
The predictor only consumes the public V2 observation. No plant truth enters it.
"""
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np


G = 9.80665
RATE_LIMIT = np.array([2.0, 2.0, 1.0])


@dataclass(frozen=True)
class DOBConfig:
    kp: tuple = (2.5, 2.5, 4.0)
    kd: tuple = (2.8, 2.8, 3.2)
    ki: tuple = (0.55, 0.55, 1.1)
    attitude_gain: float = 5.0
    integral_limit: float = 6.0
    horizontal_acceleration_limit: float = 5.0
    vertical_acceleration_limit: float = 5.0
    observer_tau_s: float = 0.25
    disturbance_limit: float = 5.0
    innovation_limit: float = 25.0
    nominal_motor_tau_s: float = 0.025
    nominal_command_delay_steps: int = 0
    nominal_drag_per_mass: float | tuple = 0.12 / 1.75
    maximum_specific_thrust: float = 2.0 * G
    enabled: bool = True


def _rotate_xy(vector, angle):
    result = np.asarray(vector, dtype=float).copy()
    c, s = np.cos(angle), np.sin(angle)
    result[:2] = np.array([[c, -s], [s, c]]) @ result[:2]
    return result


def _bound_xy_z(vector, limit_xy, limit_z):
    result = np.asarray(vector, dtype=float).copy()
    magnitude = np.linalg.norm(result[:2])
    if magnitude > limit_xy:
        result[:2] *= limit_xy / magnitude
    result[2] = np.clip(result[2], -limit_z, limit_z)
    return result


class DOBReference:
    """Shared-gain PID/DOB pair via ``config.enabled`` and ``predict(obs)``.

    Coordinates: ENU world, FLU body, then current heading canonicalization.
    Memory is transported exactly using sin/cos of the fixed yaw reference.
    The acceleration residual includes all unmodelled effects; it is not an
    oracle wind estimate, nor does the controller know the randomized mass.
    """
    architecture = "pid_dob_reference"

    def __init__(self, dt=0.02, config=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite controller dt required")
        self.dt = float(dt)
        self.config = config or DOBConfig()
        if self.config.observer_tau_s <= 0 or self.config.nominal_motor_tau_s < 0:
            raise ValueError("invalid observer or motor time constant")
        if self.config.nominal_command_delay_steps < 0 or int(self.config.nominal_command_delay_steps) != self.config.nominal_command_delay_steps:
            raise ValueError("nonnegative integer nominal delay steps required")
        drag = np.asarray(self.config.nominal_drag_per_mass)
        if drag.shape not in ((), (3,)) or not np.isfinite(drag).all() or np.any(drag < 0):
            raise ValueError("nominal drag must be a nonnegative scalar or xyz vector")
        if drag.shape == (3,) and not np.isclose(drag[0], drag[1]):
            raise ValueError("heading invariant observer requires equal nominal x/y drag")
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.disturbance = np.zeros(3)
        self.previous_velocity = None
        self.previous_body_z = None
        self.previous_yaw_error = None
        self.specific_thrust = G
        self.nominal_command_queue = deque([0.0] * self.config.nominal_command_delay_steps)
        self.diagnostics = {}

    def _transport_memory(self, yaw_error):
        if self.previous_yaw_error is None:
            return
        angle = yaw_error - self.previous_yaw_error
        self.integral = _rotate_xy(self.integral, angle)
        self.disturbance = _rotate_xy(self.disturbance, angle)
        self.previous_velocity = _rotate_xy(self.previous_velocity, angle)
        self.previous_body_z = _rotate_xy(self.previous_body_z, angle)

    def _observe_disturbance(self, velocity, body_z, previous_command):
        cfg, dt = self.config, self.dt
        if self.previous_velocity is None:
            return np.zeros(3), np.zeros(3)
        self.nominal_command_queue.append(float(np.clip(previous_command, -1, 1)))
        aligned_command = self.nominal_command_queue.popleft()
        commanded_specific_thrust = (aligned_command + 1) * 0.5 * cfg.maximum_specific_thrust
        if cfg.nominal_motor_tau_s > 0:
            decay = np.exp(-dt / cfg.nominal_motor_tau_s)
            average_thrust = commanded_specific_thrust + (self.specific_thrust - commanded_specific_thrust) * cfg.nominal_motor_tau_s / dt * (1 - decay)
            self.specific_thrust = commanded_specific_thrust + (self.specific_thrust - commanded_specific_thrust) * decay
        else:
            average_thrust = self.specific_thrust = commanded_specific_thrust
        # Trapezoidal measured attitude/velocity matches the preceding interval.
        model_acceleration = average_thrust * 0.5 * (body_z + self.previous_body_z)
        model_acceleration[2] -= G
        model_acceleration -= np.asarray(cfg.nominal_drag_per_mass) * 0.5 * (velocity + self.previous_velocity)
        measured_acceleration = (velocity - self.previous_velocity) / dt
        residual = np.clip(measured_acceleration - model_acceleration,
                           -cfg.innovation_limit, cfg.innovation_limit)
        alpha = -np.expm1(-dt / cfg.observer_tau_s)
        self.disturbance += alpha * (residual - self.disturbance)
        self.disturbance = _bound_xy_z(self.disturbance, cfg.disturbance_limit, cfg.disturbance_limit)
        return measured_acceleration, model_acceleration

    def predict(self, observation):
        obs = np.asarray(observation, dtype=float)
        if obs.ndim != 1 or obs.size not in (30, 74) or not np.all(np.isfinite(obs)):
            raise ValueError("finite V2 observation of size 30 or 74 required")
        cfg = self.config
        error, velocity = obs[:3] * 3.0, obs[3:6] * 3.0
        rotation = obs[6:15].reshape(3, 3)
        yaw_error = float(np.arctan2(obs[28], obs[29]))
        self._transport_memory(yaw_error)
        measured_acceleration, model_acceleration = self._observe_disturbance(velocity, rotation[:, 2], obs[18])
        compensation = self.disturbance if cfg.enabled else np.zeros(3)
        kp, kd, ki = map(np.asarray, (cfg.kp, cfg.kd, cfg.ki))
        candidate_integral = _bound_xy_z(self.integral + error * self.dt,
                                        cfg.integral_limit, cfg.integral_limit)
        raw_acceleration = kp * error - kd * velocity + ki * candidate_integral - compensation
        limited_acceleration = _bound_xy_z(raw_acceleration, cfg.horizontal_acceleration_limit,
                                          cfg.vertical_acceleration_limit)
        # Integrate only when feasible, or when error drives the request out of
        # saturation. This also bounds accumulated error during unreachable goals.
        accept = (np.abs(raw_acceleration - limited_acceleration) < 1e-9) | ((raw_acceleration - limited_acceleration) * error <= 0)
        self.integral = _bound_xy_z(np.where(accept, candidate_integral, self.integral),
                                   cfg.integral_limit, cfg.integral_limit)
        acceleration = kp * error - kd * velocity + ki * self.integral - compensation
        acceleration = _bound_xy_z(acceleration, cfg.horizontal_acceleration_limit,
                                   cfg.vertical_acceleration_limit)
        acceleration[2] += G
        desired_z = acceleration / np.linalg.norm(acceleration)
        desired_heading_x = np.array([np.cos(yaw_error), np.sin(yaw_error), 0.0])
        desired_y = np.cross(desired_z, desired_heading_x)
        desired_y /= np.linalg.norm(desired_y)
        desired_x = np.cross(desired_y, desired_z)
        desired_rotation = np.column_stack([desired_x, desired_y, desired_z])
        attitude_matrix = rotation.T @ desired_rotation - desired_rotation.T @ rotation
        attitude_error = 0.5 * np.array([attitude_matrix[2, 1], attitude_matrix[0, 2], attitude_matrix[1, 0]])
        rate_command = cfg.attitude_gain * attitude_error
        fraction = float(np.dot(acceleration, rotation[:, 2]) / cfg.maximum_specific_thrust)
        action = np.clip(np.r_[2 * fraction - 1, rate_command / RATE_LIMIT], -1, 1)
        self.previous_velocity = velocity.copy()
        self.previous_body_z = rotation[:, 2].copy()
        self.previous_yaw_error = yaw_error
        self.diagnostics = {
            "disturbance_heading_mps2": self.disturbance.tolist(),
            "integral_heading_ms": self.integral.tolist(),
            "measured_acceleration_heading_mps2": measured_acceleration.tolist(),
            "model_acceleration_heading_mps2": model_acceleration.tolist(),
            "observer_enabled": cfg.enabled,
        }
        return action

    def metadata(self):
        return {"architecture": self.architecture, "dt": self.dt,
                "config": asdict(self.config), "truth_access": False}

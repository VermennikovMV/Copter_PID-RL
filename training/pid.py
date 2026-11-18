"""Copy of the ArduPilot multicopter cascade for PyBullet.

  horizontal: position P -> velocity PID (+ target velocity / acceleration feed-forward) -> acceleration
  vertical:   position P -> velocity P -> acceleration PI
  acceleration -> thrust vector -> attitude command model (ATC_INPUT_TC, ATC_ACC_*_MAX) -> angle P
  (+ target angular velocity) -> body-rate PI -> mixer -> PWM

Defaults are the values of the SITL stand (configs/*.json "pid"); the copy is validated against
ArduPilot SITL on the same experiments (docs/results.md). Residual inputs, as AC_Residual in the
firmware: `angle_offset` (rad, added to the attitude error) and `thrust_offset` (collective fraction).
"""
from dataclasses import dataclass

import numpy as np

from .guided_shaper import shape_pos_vel_accel
from .physics import G, INERTIA, MASS, MAX_FORCE, DirectMotorModel


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def _expm(w):
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3) + np.sin(angle) * K + (1. - np.cos(angle)) * K @ K


def _rotvec(r):
    """Rotation vector of the rotation matrix r (log map)."""
    angle = float(np.arccos(np.clip((np.trace(r) - 1.) / 2., -1., 1.)))
    if angle < 1e-9:
        return .5 * _vee(r - r.T)
    return angle / (2. * np.sin(angle)) * _vee(r - r.T)


def limited_xy(vector, limit):
    """Limit the horizontal norm of a vector."""
    value = np.asarray(vector, dtype=float).copy()
    size = np.linalg.norm(value[:2])
    if size > limit:
        value[:2] *= limit / size
    return value


@dataclass(frozen=True)
class PIDConfig:
    position_xy_p: float = 1.5            # PSC_NE_POS_P
    velocity_xy_p: float = 4.             # PSC_NE_VEL_P/I/D
    velocity_xy_i: float = 2.
    velocity_xy_d: float = .5
    position_z_p: float = 1.              # PSC_D_POS_P
    velocity_z_p: float = 5.              # PSC_D_VEL_P
    velocity_z_i: float = 0.
    accel_z_p: float = 1.                 # PSC_D_ACC_P/I/IMAX, in m/s^2
    accel_z_i: float = 2.
    accel_z_imax: float = 15.7
    accel_z_filter_tau_s: float = .008
    attitude_p: float = 4.5               # ATC_ANG_*_P
    rate_p: float = 20.                   # rate PI (per unit inertia)
    rate_i: float = 10.
    yaw_momentum_gain: float = .35
    velocity_xy_limit: float = 3.         # position P output limit
    velocity_z_limit: float = 1.5
    acceleration_xy_limit: float = 5.66   # lean angle limit 30 deg
    acceleration_z_limit: float = 4.
    velocity_integral_limit: float = 2.
    velocity_z_integral_limit: float | None = None   # None: same as velocity_integral_limit
    derivative_tau_s: float = .032
    command_tau_s: float = .01
    attitude_input_tc_s: float = .1       # ATC_INPUT_TC
    attitude_accel_max_degss: tuple = (1100., 1100., 270.)   # ATC_ACC_R/P/Y_MAX
    attitude_loop_dt_s: float = .0025     # 400 Hz attitude loop


class CascadedPID:
    """Uses only the measured state, the target and its own PWM history (rotor speed model)."""

    def __init__(self, config=None, gain_scale=1.):
        self.config = PIDConfig(**(config or {}))
        self.gain_scale = float(gain_scale)   # randomised in training
        self.motor = DirectMotorModel()
        self.reset()

    def reset(self, yaw_reference=0.):
        self.yaw_reference = float(yaw_reference)
        self.velocity_integral = np.zeros(3)
        self.rate_integral = np.zeros(2)
        self.previous_velocity = None
        self.velocity_derivative = np.zeros(3)
        self.shaped_acceleration = np.zeros(3)
        self.shaped_rates = np.zeros(3)
        self.accel_z_measured = 0.
        self.accel_z_integral = 0.
        self.angle_offset = np.zeros(3)       # residual input: attitude offset, rad (roll, pitch, yaw)
        self.thrust_offset = 0.               # residual input: collective thrust fraction
        self.attitude_target = None
        self.ang_vel_target = np.zeros(3)
        self.ang_accel_target = np.zeros(3)
        self.motor.reset(pwm=self.motor.config.nominal_hover_pwm)
        self.last_pwm = np.asarray(self.motor.config.nominal_hover_pwm, dtype=float)
        self.diagnostics = {}

    def _gains(self, xy, z):
        return np.array([xy, xy, z])

    def command(self, measured, target, dt, velocity_ff=None, acceleration_ff=None):
        """Motor PWM for the target position. velocity_ff / acceleration_ff: target velocity and
        acceleration (world frame) from GUIDED shaping or the AUTO navigator."""
        cfg = self.config
        position, velocity, rotation, rates = (np.asarray(measured[k], dtype=float)
                                               for k in ('position', 'velocity', 'rotation', 'body_rates'))
        # position -> velocity target
        error = np.asarray(target, dtype=float) - position
        velocity_target = limited_xy(error * self._gains(cfg.position_xy_p, cfg.position_z_p), cfg.velocity_xy_limit)
        velocity_target[2] = np.clip(velocity_target[2], -cfg.velocity_z_limit, cfg.velocity_z_limit)
        if velocity_ff is not None:
            velocity_target = velocity_target + np.asarray(velocity_ff, dtype=float)

        # velocity PID -> acceleration (derivative on measurement, integrator with anti-windup)
        v_error = velocity_target - velocity
        derivative = np.zeros(3) if self.previous_velocity is None else -(velocity - self.previous_velocity) / dt
        self.velocity_derivative += (1 - np.exp(-dt / cfg.derivative_tau_s)) * (derivative - self.velocity_derivative)
        self.previous_velocity = velocity.copy()
        z_limit = cfg.velocity_integral_limit if cfg.velocity_z_integral_limit is None else cfg.velocity_z_integral_limit
        limit = np.array([cfg.velocity_integral_limit] * 2 + [z_limit])
        kp = self._gains(cfg.velocity_xy_p, cfg.velocity_z_p)
        ki = self._gains(cfg.velocity_xy_i, cfg.velocity_z_i)
        kd = self._gains(cfg.velocity_xy_d, 0.)
        candidate_integral = np.clip(self.velocity_integral + v_error * dt, -limit, limit)
        acceleration = self.gain_scale * (kp * v_error + ki * candidate_integral + kd * self.velocity_derivative)
        bounded = limited_xy(acceleration, cfg.acceleration_xy_limit)
        bounded[2] = np.clip(bounded[2], -cfg.acceleration_z_limit, cfg.acceleration_z_limit)
        accept = (np.abs(acceleration - bounded) < 1e-9) | ((acceleration - bounded) * v_error <= 0)
        self.velocity_integral = np.where(accept, candidate_integral, self.velocity_integral)
        acceleration = self.gain_scale * (kp * v_error + ki * self.velocity_integral + kd * self.velocity_derivative)
        if acceleration_ff is not None:
            acceleration = acceleration + np.asarray(acceleration_ff, dtype=float)
        acceleration = limited_xy(acceleration, cfg.acceleration_xy_limit)
        acceleration[2] = np.clip(acceleration[2], -cfg.acceleration_z_limit, cfg.acceleration_z_limit)
        self.shaped_acceleration += (1 - np.exp(-dt / cfg.command_tau_s)) * (acceleration - self.shaped_acceleration)

        # vertical acceleration PI
        specific_force = self.shaped_acceleration + np.array([0., 0., G])
        self.accel_z_measured += (1 - np.exp(-dt / cfg.accel_z_filter_tau_s)) * (-float(derivative[2]) - self.accel_z_measured)
        accel_error = float(self.shaped_acceleration[2]) - self.accel_z_measured
        self.accel_z_integral = float(np.clip(self.accel_z_integral + cfg.accel_z_i * accel_error * dt,
                                              -cfg.accel_z_imax, cfg.accel_z_imax))
        specific_force[2] += cfg.accel_z_p * accel_error + self.accel_z_integral

        # thrust vector + heading -> desired attitude
        desired_z = specific_force / np.linalg.norm(specific_force)
        heading = np.array([np.cos(self.yaw_reference), np.sin(self.yaw_reference), 0.])
        desired_y = np.cross(desired_z, heading)
        desired_y /= max(np.linalg.norm(desired_y), 1e-9)
        desired_rotation = np.column_stack([np.cross(desired_y, desired_z), desired_y, desired_z])

        # attitude command model (AC_AttitudeControl::input_thrust_vector_heading, 400 Hz substeps)
        if self.attitude_target is None:
            self.attitude_target = rotation.copy()
        accel_max = np.radians(np.asarray(cfg.attitude_accel_max_degss, dtype=float))
        n = max(1, int(round(dt / cfg.attitude_loop_dt_s)))
        h = dt / n
        for _ in range(n):
            self.attitude_target = self.attitude_target @ _expm(self.ang_vel_target * h)
            target_error = _rotvec(self.attitude_target.T @ desired_rotation)
            for i in range(3):
                self.ang_accel_target[i] = shape_pos_vel_accel(
                    float(target_error[i]), 0., 0., 0., float(self.ang_vel_target[i]), float(self.ang_accel_target[i]),
                    0., 0., -accel_max[i], accel_max[i], accel_max[i] / cfg.attitude_input_tc_s, h, True)
                self.ang_vel_target[i] += self.ang_accel_target[i] * h

        # angle P (+ residual offset) + target angular velocity -> body rates
        body_to_target = rotation.T @ self.attitude_target
        difference = body_to_target - body_to_target.T
        angle_error = .5 * np.array([difference[2, 1], difference[0, 2], difference[1, 0]]) + self.angle_offset
        self.shaped_rates = cfg.attitude_p * self.gain_scale * angle_error + body_to_target @ self.ang_vel_target

        # rate PI -> torques -> mixer
        rate_error = self.shaped_rates - rates
        self.rate_integral = np.clip(self.rate_integral + rate_error[:2] * dt, -.5, .5)
        torque = INERTIA[:2] * self.gain_scale * (cfg.rate_p * rate_error[:2] + cfg.rate_i * self.rate_integral)
        motor_cfg = self.motor.config
        speed = np.maximum(self.motor.speed_rad_s, 50.)
        yaw_row = np.asarray(motor_cfg.rotor_spin) / (2 * motor_cfg.thrust_coefficient * speed)
        predicted_force = motor_cfg.thrust_coefficient * speed**2
        yaw_request = yaw_row @ predicted_force - cfg.yaw_momentum_gain * INERTIA[2] / motor_cfg.rotor_inertia_kgm2 * rate_error[2]
        positions = self.motor.positions
        mixer = np.vstack([np.ones(4), positions[:, 1], -positions[:, 0], yaw_row])
        total_force = MASS * max(0., specific_force[2]) / max(.5, rotation[2, 2])   # tilt compensation
        total_force *= 1. + self.thrust_offset
        force = np.clip(np.linalg.solve(mixer, np.r_[total_force, torque, yaw_request]), 0, MAX_FORCE)
        self.last_pwm = np.clip(self.motor.nominal_pwm_from_forces(force), motor_cfg.pwm_min, motor_cfg.pwm_max)
        self.diagnostics = dict(thrust_ratio=float(total_force / (MASS * G)), velocity_target=velocity_target.tolist(),
                                acceleration_target=self.shaped_acceleration.tolist(),
                                desired_rates=self.shaped_rates.tolist(), base_pwm=self.last_pwm.tolist())
        return self.last_pwm.copy()

    def observe_applied_command(self, pwm, measured, dt):
        """Advance the rotor speed model with the PWM actually applied (after delays)."""
        rotation = np.asarray(measured['rotation'])
        self.motor.step_pwm(pwm, dt, body_rates=measured['body_rates'],
                            body_velocity=rotation.T @ np.asarray(measured['velocity']))

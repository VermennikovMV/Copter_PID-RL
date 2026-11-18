"""Port of ArduPilot's GUIDED position-target shaping for the PyBullet cascade copy.

In GUIDED, a position target is not fed straight into the position P controller:
AC_PosControl::input_pos_NED_m() moves an internal desired state (position, velocity,
acceleration) towards the target with a jerk- and acceleration-limited square-root
controller, and the position/velocity controllers then track that desired state with
velocity and acceleration feedforward.

Sources (ArduPilot master, 4.7.2):
  libraries/AC_AttitudeControl/AC_PosControl.cpp - input_pos_NED_m, NE/D_set_max_speed_accel_m
  libraries/AP_Math/control.cpp                  - update_pos_vel_accel(_xy), shape_pos_vel_accel(_xy),
                                                   shape_accel(_xy), sqrt_controller,
                                                   sqrt_controller_accel, limit_accel_corner_xy,
                                                   kinematic_limit
  ArduCopter/mode_guided.cpp                     - pva_control_start (limits from WP_* parameters)

Scope: no terrain offset, no motor-limit vectors (limit = 0), no overspeed gain (targets stay
within the speed limits). Internally NED like ArduPilot; the interface is ENU. ArduPilot runs
this at the main loop rate, so each control step is split into substeps of ``loop_dt``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .wpnav import GRAVITY_MSS, NED_FROM_ENU, kinematic_limit


def _safe_sqrt(x):
    return math.sqrt(x) if x > 0 else 0.


def sqrt_controller(error, p, second_ord_lim, dt):
    if second_ord_lim <= 0:
        rate = error * p
    elif p == 0:
        rate = math.copysign(_safe_sqrt(2. * second_ord_lim * abs(error)), error) if error else 0.
    else:
        linear_dist = second_ord_lim / (p * p)
        if error > linear_dist:
            rate = _safe_sqrt(2. * second_ord_lim * (error - linear_dist / 2.))
        elif error < -linear_dist:
            rate = -_safe_sqrt(2. * second_ord_lim * (-error - linear_dist / 2.))
        else:
            rate = error * p
    if dt > 0:
        return min(max(rate, -abs(error) / dt), abs(error) / dt)
    return rate


def sqrt_controller_vec(error, p, second_ord_lim, dt):
    length = float(np.linalg.norm(error))
    if length <= 0:
        return np.zeros_like(error)
    return error * (sqrt_controller(length, p, second_ord_lim, dt) / length)


def sqrt_controller_accel(error, rate_cmd, rate_state, p, second_ord_lim):
    if not rate_cmd * rate_state > 0:
        return 0.
    if second_ord_lim <= 0:
        return -p * rate_state
    if p <= 0:
        return 0. if rate_cmd == 0 else -(second_ord_lim / abs(rate_cmd)) * rate_state
    if abs(error) <= second_ord_lim / (p * p):
        return -p * rate_state
    return 0. if rate_cmd == 0 else -(second_ord_lim / abs(rate_cmd)) * rate_state


def _limit_length(v, limit):
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit > 0 or (limit == 0 and n > 0) else v


def limit_accel_corner_xy(vel, accel, accel_max):
    if accel_max <= 0:
        return accel
    if not np.any(vel):
        return _limit_length(accel, accel_max)
    accel = _limit_length(accel, 2. * accel_max)
    unit = vel / np.linalg.norm(vel)
    along = float(accel @ unit)
    accel_dir = unit * along
    cross = accel - accel_dir
    if along > 0:
        cross_mag = min(float(np.linalg.norm(cross)), accel_max)
        along_max = _safe_sqrt(accel_max ** 2 - cross_mag ** 2)
        return _limit_length(cross, accel_max) + _limit_length(accel_dir, along_max)
    along = max(along, -accel_max)
    cross_max = _safe_sqrt(accel_max ** 2 - along ** 2)
    return _limit_length(cross, cross_max) + unit * along


def shape_accel(accel_target, accel, jerk_max, dt):
    return accel + min(max(accel_target - accel, -jerk_max * dt), jerk_max * dt)


def shape_accel_xy(accel_target, accel, jerk_max, dt):
    return accel + _limit_length(accel_target - accel, jerk_max * dt)


def shape_pos_vel_accel_xy(pos_desired, vel_desired, accel_desired, pos, vel, accel,
                           vel_max, accel_max, jerk_max, dt, limit_total=False):
    k_v = jerk_max / accel_max
    vel_corr_cmd = np.zeros(2)
    pos_error = pos_desired - pos
    length = float(np.linalg.norm(pos_error))
    if length > 0:
        vel_corr_proj = float((vel - vel_desired) @ pos_error) / length
        cmd = sqrt_controller(length, k_v, accel_max, dt)
        cmd += sqrt_controller_accel(length, cmd, vel_corr_proj, k_v, accel_max) / k_v
        if vel_max > 0:
            cmd = min(max(cmd, -vel_max), vel_max)
        vel_corr_cmd = pos_error * (cmd / length)
    vel_target = vel_desired + vel_corr_cmd
    if limit_total and vel_max > 0:
        vel_target = _limit_length(vel_target, vel_max)
    accel_target = limit_accel_corner_xy(vel, (vel_target - vel) * k_v, accel_max) + accel_desired
    if limit_total:
        accel_target = _limit_length(accel_target, accel_max)
    return shape_accel_xy(accel_target, accel, jerk_max, dt)


def shape_pos_vel_accel(pos_desired, vel_desired, accel_desired, pos, vel, accel,
                        vel_min, vel_max, accel_min, accel_max, jerk_max, dt, limit_total=False):
    pos_error = pos_desired - pos
    accel_lim = -accel_min if pos_error > 0 else accel_max
    k_v = jerk_max / accel_lim
    vel_corr = vel - vel_desired
    cmd = sqrt_controller(pos_error, k_v, accel_lim, dt)
    cmd += sqrt_controller_accel(pos_error, cmd, vel_corr, k_v, accel_lim) / k_v
    if vel_min < 0:
        cmd = max(cmd, vel_min)
    if vel_max > 0:
        cmd = min(cmd, vel_max)
    vel_target = vel_desired + cmd
    if limit_total:
        if vel_min < 0:
            vel_target = max(vel_target, vel_min)
        if vel_max > 0:
            vel_target = min(vel_target, vel_max)
    accel_target = min(max((vel_target - vel) * k_v, accel_min), accel_max) + accel_desired
    if limit_total:
        accel_target = min(max(accel_target, accel_min), accel_max)
    return shape_accel(accel_target, accel, jerk_max, dt)


@dataclass(frozen=True)
class GuidedShapingParams:
    """Limits GUIDED takes from the WP_* / PSC_* / ATC_* parameters (record the stand values)."""
    speed_ne_ms: float = 3.0          # WP_SPD
    accel_ne_mss: float = 2.5         # WP_ACC
    speed_up_ms: float = 2.5          # WP_SPD_UP
    speed_down_ms: float = 1.5        # WP_SPD_DN
    accel_d_mss: float = 1.0          # WP_ACC_Z
    shaping_jerk_ne_msss: float = 5.0  # PSC_NE_JERK
    shaping_jerk_d_msss: float = 5.0   # PSC_D_JERK
    ang_vel_roll_max_degs: float = 0.  # ATC_RATE_R_MAX
    ang_vel_pitch_max_degs: float = 0.  # ATC_RATE_P_MAX
    accel_roll_max_degss: float = 1100.  # ATC_ACC_R_MAX
    accel_pitch_max_degss: float = 1100.  # ATC_ACC_P_MAX
    accel_d_filter_e_hz: float = 20.   # PSC_D_ACC_FLTE
    loop_dt: float = 0.0025            # main loop period (SCHED_LOOP_RATE 400 Hz)


class GuidedShaper:
    """AC_PosControl input_pos_NED_m: the desired state that GUIDED's controllers track."""

    def __init__(self, params: GuidedShapingParams | None = None):
        self.p = p = params or GuidedShapingParams()
        # NE_set_max_speed_accel_m (feedforward enabled, as ATC_RATE_FF_ENAB default)
        jerk = p.shaping_jerk_ne_msss
        rate = min(math.radians(p.ang_vel_roll_max_degs), math.radians(p.ang_vel_pitch_max_degs)) * GRAVITY_MSS
        snap = min(math.radians(p.accel_roll_max_degss), math.radians(p.accel_pitch_max_degss)) * GRAVITY_MSS
        if rate > 0:
            jerk = min(jerk, rate)
        if snap > 0:
            jerk = min(.5 * _safe_sqrt(p.accel_ne_mss * snap), jerk)
        self.jerk_ne = jerk
        # D_set_max_speed_accel_m
        jerk_d = p.shaping_jerk_d_msss
        if p.accel_d_filter_e_hz > 0:
            jerk_d = min(jerk_d, min(GRAVITY_MSS, p.accel_d_mss) * (2 * math.pi * p.accel_d_filter_e_hz) / 5.)
        self.jerk_d = jerk_d
        self.started = False

    def metadata(self):
        return dict(kind='ArduPilot_AC_PosControl_input_pos_port', parameters=asdict(self.p),
                    jerk_ne_msss=self.jerk_ne, jerk_d_msss=self.jerk_d)

    def start(self, position_enu, velocity_enu=(0., 0., 0.)):
        """NE/D_init_controller: the desired state starts at the measured state."""
        self.pos = NED_FROM_ENU @ np.asarray(position_enu, float)
        self.vel = NED_FROM_ENU @ np.asarray(velocity_enu, float)
        self.acc = np.zeros(3)
        self.started = True

    def _input_pos(self, target, dt):
        p = self.p
        # update_pos_vel_accel(_xy) with no limit vector
        self.pos = self.pos + self.vel * dt + self.acc * .5 * dt * dt
        self.vel = self.vel + self.acc * dt
        vel_max_ne = vel_max_d = 0.
        travel = target - self.pos
        n = float(np.linalg.norm(travel))
        if n > 0:
            unit = travel / n
            vel_max = kinematic_limit(float(np.hypot(*unit[:2])), float(unit[2]),
                                      p.speed_ne_ms, p.speed_up_ms, p.speed_down_ms)
            vel_max_ne = vel_max * float(np.hypot(*unit[:2]))
            vel_max_d = abs(vel_max * unit[2])
        self.acc[:2] = shape_pos_vel_accel_xy(target[:2], np.zeros(2), np.zeros(2), self.pos[:2], self.vel[:2],
                                              self.acc[:2], vel_max_ne, p.accel_ne_mss, self.jerk_ne, dt)
        self.acc[2] = shape_pos_vel_accel(float(target[2]), 0., 0., float(self.pos[2]), float(self.vel[2]),
                                          float(self.acc[2]), -vel_max_d, vel_max_d, -p.accel_d_mss,
                                          min(p.accel_d_mss, 7.5), self.jerk_d, dt)

    def update(self, target_enu, dt):
        """Advance by one control step; returns ENU (position, velocity_ff, acceleration_ff)."""
        if not self.started:
            raise RuntimeError('start() first')
        target = NED_FROM_ENU @ np.asarray(target_enu, float)
        n = max(1, int(round(dt / self.p.loop_dt)))
        for _ in range(n):
            self._input_pos(target, dt / n)
        return NED_FROM_ENU @ self.pos, NED_FROM_ENU @ self.vel, NED_FROM_ENU @ self.acc

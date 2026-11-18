"""Port of ArduPilot's AUTO-mode waypoint navigator for the PyBullet cascade copy.

Sources (ArduPilot master, AC_WPNav / AP_Math / ArduCopter):
  libraries/AP_Math/SCurve.cpp      - jerk-limited S-curve legs, corner blending
  libraries/AP_Math/control.cpp     - kinematic_limit
  libraries/AC_WPNav/AC_WPNav.cpp   - wp_and_spline_init_m, set_wp_destination_NED_m,
                                      set_wp_destination_next_NED_m,
                                      advance_wp_target_along_track,
                                      calc_scurve_jerk_and_snap
  ArduCopter/mode_auto.cpp          - do_nav_wp, set_next_wp, verify_nav_wp (delay p1)

Scope of the port: straight NAV_WAYPOINT legs (no splines, arcs or terrain), a constant
speed setting (so SCurve::set_speed_max and the WPNAV speed-change shaping are inactive:
_offset_vel stays at the desired speed and vel_dt_scalar == 1), one mission run from a
hover. Internally everything is NED like ArduPilot; the public interface is ENU.

Each control step ``WaypointNavigator.update`` returns (target_position, velocity_ff,
acceleration_ff) in ENU for ``CascadedPID.command``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

GRAVITY_MSS = 9.80665
NED_FROM_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])  # self-inverse

# segment indices (SCurve.cpp)
SEG_INIT, SEG_ACCEL_MAX, SEG_ACCEL_END = 0, 4, 7
SEG_SPEED_CHANGE_END, SEG_CONST, SEG_DECEL_START, SEG_DECEL_END = 14, 15, 15, 22
SEGMENTS_MAX = 23
CONSTANT_JERK, POSITIVE_JERK, NEGATIVE_JERK = 0, 1, 2


def _pos(x):
    return x > 1e-6   # is_positive (FLT_EPSILON-scale comparison in ArduPilot)


def _zero(x):
    return abs(x) < 1e-6


def _safe_sqrt(x):
    return math.sqrt(x) if x > 0 else 0.


def kinematic_limit(length_xy, length_z, max_xy, max_z_neg, max_z_pos):
    """control.cpp kinematic_limit(segment_length_xy, segment_length_z, ...). z is NED (down +)."""
    if _zero(max_xy) or _zero(max_z_pos) or _zero(max_z_neg):
        return 0.
    max_xy, max_z_pos, max_z_neg = abs(max_xy), abs(max_z_pos), abs(max_z_neg)
    length = _safe_sqrt(length_xy ** 2 + length_z ** 2)
    if not _pos(length):
        return 0.
    length_xy /= length
    length_z /= length
    if _zero(length_xy):
        return max_z_pos if _pos(length_z) else max_z_neg
    if _zero(length_z):
        return max_xy
    slope = length_z / length_xy
    if _pos(slope):
        if abs(slope) < max_z_pos / max_xy:
            return max_xy / length_xy
        return abs(max_z_pos / length_z)
    if abs(slope) < max_z_neg / max_xy:
        return max_xy / length_xy
    return abs(max_z_neg / length_z)


class SCurve:
    """SCurve.cpp, straight segments."""

    def __init__(self):
        self.init()

    def copy(self):
        other = SCurve.__new__(SCurve)
        other.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        other.seg = [list(s) for s in self.seg]
        return other

    def init(self):
        self.snap_max = self.jerk_max = self.accel_max = self.vel_max = 0.
        self.accel_z_max = 0.
        self.time = 0.
        # segment = [end_time, seg_type, jerk_ref, end_accel, end_vel, end_pos]
        self.seg = [[0., CONSTANT_JERK, 0., 0., 0., 0.] for _ in range(SEGMENTS_MAX)]
        self.num_segs = SEG_INIT
        self.num_segs = self._add_segment(self.num_segs, 0., CONSTANT_JERK, 0., 0., 0., 0.)
        self.seg_delta = np.zeros(3)
        self.seg_length = 0.

    # --- construction -------------------------------------------------------------
    def calculate_track(self, origin, destination, speed_xy, speed_up, speed_down,
                        accel_xy, accel_z, snap_maximum, jerk_maximum):
        self.init()
        speed_xy, speed_up, speed_down = abs(speed_xy), abs(speed_up), abs(speed_down)
        accel_xy, accel_z = abs(accel_xy), abs(accel_z)
        self.seg_delta = np.asarray(destination, float) - np.asarray(origin, float)
        if _zero(float(self.seg_delta @ self.seg_delta)):
            self.seg_delta = np.zeros(3)
            return
        length_ne = float(np.hypot(*self.seg_delta[:2]))
        self.seg_length = float(np.linalg.norm(self.seg_delta))
        self.snap_max, self.jerk_max = snap_maximum, jerk_maximum
        self.vel_max = kinematic_limit(length_ne, self.seg_delta[2], speed_xy, speed_up, speed_down)
        self.accel_max = kinematic_limit(length_ne, self.seg_delta[2], accel_xy, accel_z, accel_z)
        self.accel_z_max = accel_z
        if not (_pos(self.snap_max) and _pos(self.jerk_max) and _pos(self.accel_max) and _pos(self.vel_max)):
            raise ValueError('SCurve::calculate_track created zero length path')
        self._add_segments(self.seg_length)
        if not self.valid():
            raise ValueError('SCurve::calculate_track invalid path')

    def _add_segments(self, L):
        if _zero(L):
            return
        Jm, tj, t2, t4, t6 = self.calculate_path(self.snap_max, self.jerk_max, 0., self.accel_max, self.vel_max, L * .5)
        n = self.num_segs
        n = self._add_segments_jerk(n, tj, Jm, t2)
        n = self._add_segment_const_jerk(n, t4, 0.)
        n = self._add_segments_jerk(n, tj, -Jm, t6)
        self.seg[SEG_ACCEL_END][3] = 0.
        for _ in range(7):
            n = self._add_segment_const_jerk(n, 0., 0.)
        t15 = max(0., (L - 2. * self.seg[SEG_SPEED_CHANGE_END][5]) / self.seg[SEG_SPEED_CHANGE_END][4])
        n = self._add_segment_const_jerk(n, t15, 0.)
        n = self._add_segments_jerk(n, tj, -Jm, t6)
        n = self._add_segment_const_jerk(n, t4, 0.)
        n = self._add_segments_jerk(n, tj, Jm, t2)
        self.num_segs = n
        self.seg[SEG_DECEL_END][3] = 0.
        self.seg[SEG_DECEL_END][4] = 0.

    @staticmethod
    def calculate_path(Sm, Jm, V0, Am, Vm, L):
        """SCurve::calculate_path -> (Jm_out, tj_out, t2_out, t4_out, t6_out)."""
        if not (_pos(Sm) and _pos(Jm) and _pos(Am) and _pos(Vm) and _pos(L)):
            return 0., 0., 0., 0., 0.
        if V0 >= Vm:
            return 0., 0., 0., 0., 0.
        tj = Jm * math.pi / (2. * Sm)
        At = min(Am, (Vm - V0) / (2. * tj), (L - 4. * V0 * tj) / (4. * tj * tj))
        if not _pos(At):
            return 0., 0., 0., 0., 0.

        def t4_formula(Am, Jm):
            root = _safe_sqrt(Am ** 4 * .25 + Jm * Jm * V0 * V0 + Am * Am * Jm * Jm * tj * tj * .25
                              + Am * Jm * Jm * L * 2. - Am * Am * Jm * V0 + Am ** 3 * Jm * tj * .5
                              - Am * Jm * Jm * V0 * tj)
            a = (Am * Am * -1.5 + root - Jm * V0 - Am * Jm * tj * 1.5) / (Am * Jm)
            b = (Am * Am * -1.5 - root - Jm * V0 - Am * Jm * tj * 1.5) / (Am * Jm)
            return max(min(-(V0 - Vm + Am * tj + Am * Am / Jm) / Am, max(a, b)), 0.)

        if abs(At) < Jm * tj:
            if _zero(V0):
                tj = min(tj, ((L * math.pi) / (8. * Sm)) ** .25, ((Vm * math.pi) / (4. * Sm)) ** (1. / 3.),
                         _safe_sqrt((Am * math.pi) / (2. * Sm)))
                Jm = 2. * Sm * tj / math.pi
                Am = Jm * tj
            else:
                Am = At
                Jm = Am / tj
            if Vm <= V0 + 2. * Am * tj or L <= 4. * V0 * tj + 4. * Am * tj * tj:
                t2 = t4 = t6 = 0.
            else:
                t2, t4, t6 = 0., t4_formula(Am, Jm), 0.
        else:
            if (Vm < V0 + Am * tj + Am * Am / Jm or
                    L < 1. / (Jm * Jm) * (Am ** 3 + Am * Jm * (V0 * 2. + Am * tj * 2.)) + V0 * tj * 2. + Am * tj * tj):
                q = Jm * Jm * tj * tj / 9. - Jm * V0 * (2. / 3.)
                r = (-(Jm * Jm) * L * .5 + Jm ** 3 * tj ** 3 * (8. / 27.)
                     - Jm * tj * (Jm * Jm * tj * tj + Jm * V0 * 2.) / 3. + Jm * Jm * V0 * tj)
                s = _safe_sqrt(r * r - q ** 3)
                cube = (s - r) ** (1. / 3.) if s - r >= 0 else float('nan')  # powf of negative base -> NaN
                root_v = _safe_sqrt((V0 * -4. + Vm * 4. + Jm * tj * tj) / Jm)
                Am = min(Am, max(Jm * (tj + root_v) * -.5, Jm * (tj - root_v) * -.5),
                         Jm * tj * (-2. / 3.) + q / cube + cube)
                t2 = Am / Jm - tj
                t4 = 0.
                t6 = t2
            else:
                t2 = Am / Jm - tj
                t4 = t4_formula(Am, Jm)
                t6 = t2
        out = (Jm, tj, t2, t4, t6)
        if not all(math.isfinite(v) and v >= -1e-9 for v in out):
            return 0., tj, 0., 0., 0.
        return out

    def _add_segments_jerk(self, n, tj, Jm, Tcj):
        n = self._add_segment_incr_jerk(n, tj, Jm)
        n = self._add_segment_const_jerk(n, Tcj, Jm)
        return self._add_segment_decr_jerk(n, tj, Jm)

    def _add_segment_const_jerk(self, n, tj, J0):
        T0, _, _, A0, V0, P0 = self.seg[n - 1]
        if not _pos(tj):
            return self._add_segment(n, T0, CONSTANT_JERK, J0, A0, V0, P0)
        return self._add_segment(n, T0 + tj, CONSTANT_JERK, J0, A0 + J0 * tj,
                                 V0 + A0 * tj + .5 * J0 * tj * tj,
                                 P0 + V0 * tj + .5 * A0 * tj * tj + J0 * tj ** 3 / 6.)

    def _add_segment_incr_jerk(self, n, tj, Jm):
        T0, _, _, A0, V0, P0 = self.seg[n - 1]
        if not _pos(tj):
            return self._add_segment(n, T0, CONSTANT_JERK, 0., A0, V0, P0)
        Beta, Alpha = math.pi / tj, Jm * .5
        AT = Alpha * tj
        VT = Alpha * (tj * tj * .5 - 2. / Beta ** 2)
        PT = Alpha * ((-1. / Beta ** 2) * tj + tj ** 3 / 6.)
        return self._add_segment(n, T0 + tj, POSITIVE_JERK, Jm, A0 + AT, V0 + A0 * tj + VT,
                                 P0 + V0 * tj + .5 * A0 * tj * tj + PT)

    def _add_segment_decr_jerk(self, n, tj, Jm):
        T0, _, _, A0, V0, P0 = self.seg[n - 1]
        if not _pos(tj):
            return self._add_segment(n, T0, CONSTANT_JERK, 0., A0, V0, P0)
        Beta, Alpha = math.pi / tj, Jm * .5
        AT = Alpha * tj
        VT = Alpha * (tj * tj * .5 - 2. / Beta ** 2)
        PT = Alpha * ((-1. / Beta ** 2) * tj + tj ** 3 / 6.)
        A2T, V2T = Jm * tj, Jm * tj * tj
        P2T = Alpha * ((-1. / Beta ** 2) * 2. * tj + (4. / 3.) * tj ** 3)
        return self._add_segment(n, T0 + tj, NEGATIVE_JERK, Jm, (A0 - AT) + A2T,
                                 (V0 - VT) + (A0 - AT) * tj + V2T,
                                 (P0 - PT) + (V0 - VT) * tj + .5 * (A0 - AT) * tj * tj + P2T)

    def _add_segment(self, n, end_time, seg_type, jerk_ref, end_accel, end_vel, end_pos):
        self.seg[n] = [end_time, seg_type, jerk_ref, end_accel, end_vel, end_pos]
        return n + 1

    def valid(self):
        if self.num_segs != SEGMENTS_MAX:
            return False
        for i, (t, _, j, a, v, p) in enumerate(self.seg):
            if not all(map(math.isfinite, (t, j, a, v, p))) or v < -1e-6:
                return False
            if i and (t - self.seg[i - 1][0] < -1e-6 or p - self.seg[i - 1][5] < -1e-6):
                return False
        return _zero(self.seg[-1][3])

    # --- evaluation ---------------------------------------------------------------
    def _full(self):
        return self.num_segs == SEGMENTS_MAX

    def time_end(self):
        return self.seg[SEG_DECEL_END][0] if self._full() else 0.

    def get_time_remaining(self):
        return self.seg[SEG_DECEL_END][0] - self.time if self._full() else 0.

    def time_accel_end(self):
        return self.seg[SEG_ACCEL_END][0] if self._full() else 0.

    def time_decel_start(self):
        return self.seg[SEG_DECEL_START][0] if self._full() else 0.

    def finished(self):
        return self.time >= self.time_end()

    def advance_time(self, dt):
        self.time = min(self.time + dt, self.time_end())

    def get_jerk_accel_vel_pos_at_time(self, t):
        if not self._full():
            return 0., 0., 0., 0.
        n = self.num_segs
        pnt = n
        for i in range(n):
            if t < self.seg[n - 1 - i][0]:
                pnt = n - 1 - i
        if pnt == 0:
            jtype, Jm, tj = CONSTANT_JERK, 0., 0.
            T0, _, _, A0, V0, P0 = self.seg[0]
        elif pnt == n:
            jtype, Jm, tj = CONSTANT_JERK, 0., 0.
            T0, _, _, A0, V0, P0 = self.seg[n - 1]
        else:
            jtype, Jm = self.seg[pnt][1], self.seg[pnt][2]
            tj = self.seg[pnt][0] - self.seg[pnt - 1][0]
            T0, _, _, A0, V0, P0 = self.seg[pnt - 1]
        tau = t - T0
        if jtype == CONSTANT_JERK:
            J, A, V, P = (Jm, A0 + Jm * tau, V0 + A0 * tau + .5 * Jm * tau * tau,
                          P0 + V0 * tau + .5 * A0 * tau * tau + Jm * tau ** 3 / 6.)
        elif not _pos(tj):
            J, A, V, P = 0., A0, V0, P0
        elif jtype == POSITIVE_JERK:
            Alpha, Beta = Jm * .5, math.pi / tj
            J = Alpha * (1. - math.cos(Beta * tau))
            A = A0 + Alpha * tau - (Alpha / Beta) * math.sin(Beta * tau)
            V = (V0 + A0 * tau + Alpha * .5 * tau * tau + (Alpha / Beta ** 2) * math.cos(Beta * tau)
                 - Alpha / Beta ** 2)
            P = (P0 + V0 * tau + .5 * A0 * tau * tau + (-Alpha / Beta ** 2) * tau + Alpha * tau ** 3 / 6.
                 + (Alpha / Beta ** 3) * math.sin(Beta * tau))
        else:
            Alpha, Beta = Jm * .5, math.pi / tj
            AT = Alpha * tj
            VT = Alpha * (tj * tj * .5 - 2. / Beta ** 2)
            PT = Alpha * ((-1. / Beta ** 2) * tj + tj ** 3 / 6.)
            s = tau + tj
            J = Alpha * (1. - math.cos(Beta * s))
            A = (A0 - AT) + Alpha * s - (Alpha / Beta) * math.sin(Beta * s)
            V = ((V0 - VT) + (A0 - AT) * tau + .5 * Alpha * s * s + (Alpha / Beta ** 2) * math.cos(Beta * s)
                 - Alpha / Beta ** 2)
            P = ((P0 - PT) + (V0 - VT) * tau + .5 * (A0 - AT) * tau * tau + (-Alpha / Beta ** 2) * s
                 + (Alpha / 6.) * s ** 3 + (Alpha / Beta ** 3) * math.sin(Beta * s))
        return J, A, V, max(0., P)

    def _project(self, A, V, P, pos, vel, acc):
        if _zero(self.seg_length):
            return
        unit = self.seg_delta / self.seg_length
        pos += unit * P
        vel += unit * V
        acc += unit * A

    def move_from_pos_vel_accel(self, dt, pos, vel, acc):
        self.advance_time(dt)
        _, A, V, P = self.get_jerk_accel_vel_pos_at_time(self.time)
        self._project(A, V, P, pos, vel, acc)

    def move_to_pos_vel_accel(self, dt, pos, vel, acc):
        self.advance_time(dt)
        _, A, V, P = self.get_jerk_accel_vel_pos_at_time(self.time)
        self._project(A, V, P, pos, vel, acc)
        pos -= self.seg_delta

    def move_from_time_pos_vel_accel(self, t, pos, vel, acc):
        _, A, V, P = self.get_jerk_accel_vel_pos_at_time(t)
        self._project(A, V, P, pos, vel, acc)

    def advance_target_along_track(self, prev_leg, next_leg, wp_radius, accel_corner, fast_waypoint, dt,
                                   pos, vel, acc):
        prev_leg.move_to_pos_vel_accel(dt, pos, vel, acc)
        self.move_from_pos_vel_accel(dt, pos, vel, acc)
        s_finished = self.finished()
        time_to_destination = self.get_time_remaining()
        if (fast_waypoint and _zero(next_leg.time) and self.time >= self.time_decel_start()
                and self.get_time_remaining() <= next_leg.time_accel_end()):
            turn_pos, turn_vel, turn_acc = -self.seg_delta.copy(), np.zeros(3), np.zeros(3)
            self.move_from_time_pos_vel_accel(self.time + time_to_destination * .5, turn_pos, turn_vel, turn_acc)
            next_leg.move_from_time_pos_vel_accel(time_to_destination * .5, turn_pos, turn_vel, turn_acc)
            speed_min = min(self.vel_max, next_leg.vel_max)
            accel_z_lim = min(self.accel_z_max, next_leg.accel_z_max)
            if (np.linalg.norm(turn_pos) < wp_radius and np.linalg.norm(turn_vel) < speed_min
                    and np.hypot(*turn_acc[:2]) < accel_corner and abs(turn_acc[2]) < accel_z_lim):
                next_leg.move_from_pos_vel_accel(dt, pos, vel, acc)
        elif not _zero(next_leg.time):
            next_leg.move_from_pos_vel_accel(dt, pos, vel, acc)
            if next_leg.time >= self.get_time_remaining():
                s_finished = True
        return s_finished


@dataclass(frozen=True)
class WPNavParams:
    """ArduPilot parameters that shape AUTO navigation (record the stand's values)."""
    wp_speed_ms: float = 3.0          # WP_SPD
    wp_speed_up_ms: float = 2.5       # WP_SPD_UP
    wp_speed_down_ms: float = 1.5     # WP_SPD_DN
    wp_accel_mss: float = 2.5         # WP_ACC
    wp_accel_z_mss: float = 1.0       # WP_ACC_Z
    wp_accel_c_mss: float = 0.0       # WP_ACC_CNR (0 -> 2 * WP_ACC)
    wp_jerk_msss: float = 1.0         # WP_JERK
    wp_radius_m: float = 2.0          # WP_RADIUS_M
    pos_ne_p: float = 1.5             # PSC_NE_POS_P
    ang_vel_roll_max_degs: float = 0.  # ATC_RATE_R_MAX (0 = unlimited)
    ang_vel_pitch_max_degs: float = 0.  # ATC_RATE_P_MAX
    accel_roll_max_degss: float = 1100.  # ATC_ACC_R_MAX (deg/s/s)
    accel_pitch_max_degss: float = 1100.  # ATC_ACC_P_MAX
    input_tc: float = 0.15            # ATC_INPUT_TC

    @property
    def accel_xy(self):
        return self.wp_accel_mss if self.wp_accel_mss > 0 else 2.5

    @property
    def accel_corner(self):
        return self.wp_accel_c_mss if self.wp_accel_c_mss > 0 else 2. * self.accel_xy


class WaypointNavigator:
    """AC_WPNav (straight S-curve legs) driven by ModeAuto's NAV_WAYPOINT sequencing.

    ``waypoints_enu`` are the mission NAV_WAYPOINT positions after the start point;
    ``delays_s`` is param1 of each command (integer seconds in ArduPilot; 0 = pass-through,
    which lets the next leg be pre-loaded and the corner be blended)."""

    def __init__(self, waypoints_enu, delays_s, params: WPNavParams | None = None):
        self.p = params or WPNavParams()
        self.waypoints = [NED_FROM_ENU @ np.asarray(w, float) for w in waypoints_enu]
        self.delays = [float(d) for d in delays_s]
        if len(self.delays) != len(self.waypoints):
            raise ValueError('one delay per waypoint')
        # calc_scurve_jerk_and_snap
        p = self.p
        jerk = min(math.radians(p.ang_vel_roll_max_degs) * GRAVITY_MSS,
                   math.radians(p.ang_vel_pitch_max_degs) * GRAVITY_MSS)
        self.jerk_max = p.wp_jerk_msss if _zero(jerk) else min(jerk, p.wp_jerk_msss)
        snap = self.jerk_max * math.pi / (2. * max(p.input_tc, .1))
        snap_att = min(math.radians(p.accel_roll_max_degss), math.radians(p.accel_pitch_max_degss)) * GRAVITY_MSS
        if _pos(snap_att):
            snap = min(snap, snap_att)
        self.snap_max = .5 * snap
        self.started = False
        self.mission_complete = False

    def metadata(self):
        return dict(kind='ArduPilot_AC_WPNav_SCurve_port', parameters=asdict(self.p),
                    jerk_max_msss=self.jerk_max, snap_max_mssss=self.snap_max,
                    waypoints_enu=[(NED_FROM_ENU @ w).tolist() for w in self.waypoints], delays_s=self.delays)

    # --- AC_WPNav -----------------------------------------------------------------------
    def _calc_track(self, leg, origin, destination):
        p = self.p
        leg.calculate_track(origin, destination, p.wp_speed_ms, p.wp_speed_up_ms, p.wp_speed_down_ms,
                            p.accel_xy, p.wp_accel_z_mss, self.snap_max, self.jerk_max)

    def _wp_and_spline_init(self, stopping_point_ned):
        self.prev_leg, self.this_leg, self.next_leg = SCurve(), SCurve(), SCurve()
        self.track_dt_scalar = 1.
        self.reached_destination = True
        self.fast_waypoint = False
        self.origin = self.destination = np.asarray(stopping_point_ned, float).copy()
        self.pos_target = self.destination.copy()
        self.vel_desired = np.zeros(3)
        self.accel_desired = np.zeros(3)

    def _set_wp_destination(self, destination):
        self.prev_leg = SCurve()
        self.origin = self.destination.copy()
        self.prev_leg = self.this_leg   # preserve current leg profile for blending
        self.destination = np.asarray(destination, float).copy()
        if self.fast_waypoint and self.next_leg._full() and not self.next_leg.finished():
            self.this_leg = self.next_leg
        else:
            self.this_leg = SCurve()
            self._calc_track(self.this_leg, self.origin, self.destination)
        self.next_leg = SCurve()
        self.fast_waypoint = False
        self.reached_destination = False

    def _set_wp_destination_next(self, destination):
        self.next_leg = SCurve()
        self._calc_track(self.next_leg, self.destination, destination)
        self.fast_waypoint = True

    def _advance_wp_target_along_track(self, dt, position, velocity):
        track_dt_scalar = 1.
        speed2 = float(self.vel_desired @ self.vel_desired)
        if _pos(speed2):
            direction = self.vel_desired / math.sqrt(speed2)
            track_error = float((self.pos_target - position) @ direction)
            track_velocity = float(velocity @ direction)
            track_dt_scalar = min(max(.05 + (track_velocity - self.p.pos_ne_p * track_error) / math.sqrt(speed2), 0.), 1.)
        vel_dt_scalar = 1.   # constant WP_SPD: _offset_vel_ms == _wp_desired_speed_ne_ms
        tc = self.p.accel_xy / self.p.wp_jerk_msss if not _zero(self.p.wp_jerk_msss) else 1.
        self.track_dt_scalar += (track_dt_scalar - self.track_dt_scalar) * (dt / tc)
        pos, vel, acc = self.origin.copy(), np.zeros(3), np.zeros(3)
        s_finished = self.this_leg.advance_target_along_track(
            self.prev_leg, self.next_leg, self.p.wp_radius_m, self.p.accel_corner, self.fast_waypoint,
            self.track_dt_scalar * vel_dt_scalar * dt, pos, vel, acc)
        self.pos_target, self.vel_desired, self.accel_desired = pos, vel, acc
        if not self.reached_destination and s_finished:
            if self.fast_waypoint:
                self.reached_destination = True
            elif float(np.sum((position - self.destination) ** 2)) <= self.p.wp_radius_m ** 2:
                self.reached_destination = True

    # --- ModeAuto -----------------------------------------------------------------------
    def _do_nav_wp(self, index):
        self.index = index
        self._set_wp_destination(self.waypoints[index])
        self.loiter_time = None
        self.loiter_time_max = self.delays[index]
        if self.loiter_time_max > 0:
            return
        if index + 1 < len(self.waypoints):
            self._set_wp_destination_next(self.waypoints[index + 1])

    def _verify_nav_wp(self, t):
        if not self.reached_destination:
            return False
        if self.loiter_time is None:
            self.loiter_time = t
        return math.floor(t - self.loiter_time + 1e-9) >= self.loiter_time_max

    def start(self, t, position_enu):
        """Enter AUTO from a hover; the stopping point is the current position."""
        self._wp_and_spline_init(NED_FROM_ENU @ np.asarray(position_enu, float))
        self._do_nav_wp(0)
        self.started = True
        self.events = []

    def update(self, t, position_enu, velocity_enu, dt):
        """One control step: returns ENU (target_position, velocity_ff, acceleration_ff)."""
        if not self.started:
            raise RuntimeError('start() the mission first')
        position = NED_FROM_ENU @ np.asarray(position_enu, float)
        velocity = NED_FROM_ENU @ np.asarray(velocity_enu, float)
        if not self.mission_complete:
            if self._verify_nav_wp(t):
                self.events.append(dict(command=self.index, time_s=float(t)))
                if self.index + 1 < len(self.waypoints):
                    self._do_nav_wp(self.index + 1)
                else:
                    self.mission_complete = True
        self._advance_wp_target_along_track(dt, position, velocity)
        return (NED_FROM_ENU @ self.pos_target, NED_FROM_ENU @ self.vel_desired,
                NED_FROM_ENU @ self.accel_desired)

"""Stock ArduPilot AUTO mission on the evaluation track in Gazebo + SITL.

Each flight restarts Gazebo and SITL, takes off in GUIDED to the start point [0,0,3]
(relative to the spawn point on the ground), uploads the mission, switches to AUTO and
records a 50 Hz trace until the track evaluator reports completion.

Mission: NAV_WAYPOINT for every point after the start; param1 (delay, s) = 2 at the stop
point, 0 at pass-through points and at the finish.

  python gazebo/sitl.py --flights 3 --out runs/gazebo/auto_nominal [--wind 0 2 0]
"""
import os
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from simulation.interface import StateUnavailable  # noqa: E402
from simulation.launch import Session  # noqa: E402
from track_metrics import evaluate, load_track, write_trace  # noqa: E402

LOCATION_SCALING_FACTOR_INV = 89.83204953368922   # 1e7 deg per metre (ArduPilot Location)
# AC_WPNav parameter group prefix in this ArduPilot version is "WP_" (Parameters.cpp: GOBJECTPTR(wp_nav, "WP_", ...))
SET = dict(WP_SPD=3.0, WP_RADIUS_M=0.4, WP_YAW_BEHAVIOR=0)
READ = dict(wp_speed_ms='WP_SPD', wp_speed_up_ms='WP_SPD_UP', wp_speed_down_ms='WP_SPD_DN',
            wp_accel_mss='WP_ACC', wp_accel_z_mss='WP_ACC_Z', wp_accel_c_mss='WP_ACC_CNR',
            wp_jerk_msss='WP_JERK', wp_radius_m='WP_RADIUS_M', pos_ne_p='PSC_NE_POS_P',
            ang_vel_roll_max_degs='ATC_RATE_R_MAX', ang_vel_pitch_max_degs='ATC_RATE_P_MAX',
            accel_roll_max_degss='ATC_ACC_R_MAX', accel_pitch_max_degss='ATC_ACC_P_MAX',
            input_tc='ATC_INPUT_TC')
EXTRA = ('PSC_NE_VEL_P', 'PSC_NE_VEL_I', 'PSC_NE_VEL_D', 'PSC_D_POS_P', 'PSC_D_VEL_P', 'PSC_D_ACC_P',
         'PSC_D_ACC_I', 'PSC_D_ACC_IMAX', 'PSC_D_VEL_I', 'PSC_D_VEL_D', 'PSC_D_VEL_IMAX', 'PSC_NE_VEL_IMAX',
         'PSC_NE_VEL_FLTE', 'PSC_NE_VEL_FLTD', 'PSC_D_VEL_FLTE', 'PSC_D_ACC_FLTE', 'PSC_JERK_NE', 'PSC_JERK_D',
         'MOT_THST_HOVER', 'ATC_RAT_RLL_P', 'ATC_RAT_RLL_I', 'ATC_RAT_RLL_D', 'ATC_ANG_RLL_P', 'ATC_ANG_PIT_P', 'ATC_ANGLE_MAX', 'WP_YAW_BEHAVIOR', 'AHRS_EKF_TYPE')


def wait_state(v, timeout=2.):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return v.state(require_new=True)
        except StateUnavailable:
            if time.monotonic() > deadline:
                raise
            time.sleep(.002)


def set_mode(v, mode, timeout=10.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v.drain()
        hb = v.latest.get('HEARTBEAT')
        if hb is not None and hb.custom_mode == mode:
            return
        v.connection.mav.set_mode_send(v.system_id, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)
        time.sleep(.1)
    raise TimeoutError(f'mode {mode} not accepted')


def home_position(v, timeout=10.):
    requested = time.monotonic()
    while time.monotonic() - requested < timeout:
        v._command(mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 242.)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.:
            v.drain()
            if v.received.get('HOME_POSITION', 0) >= requested:
                return v.latest['HOME_POSITION']
            time.sleep(.01)
    raise TimeoutError('no HOME_POSITION')


def upload_mission(v, items, timeout=20.):
    """items: list of (lat_e7, lon_e7, rel_alt_m, delay_s); a home item is sent first."""
    conn = v.connection
    seq_items = [(0, 0, 0., 0.)] + list(items)
    started = time.monotonic()
    conn.mav.mission_count_send(v.system_id, 1, len(seq_items), 0)
    sent_after = {}
    deadline = started + timeout
    while time.monotonic() < deadline:
        v.drain()
        ack = v.latest.get('MISSION_ACK')
        if ack is not None and v.received.get('MISSION_ACK', 0) > started:
            if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f'mission rejected: {ack.type}')
            return
        for name in ('MISSION_REQUEST_INT', 'MISSION_REQUEST'):
            req = v.latest.get(name)
            stamp = v.received.get(name, 0)
            if req is None or stamp <= started or sent_after.get((name, req.seq)) == stamp:
                continue
            sent_after[(name, req.seq)] = stamp
            lat, lon, alt, delay = seq_items[req.seq]
            frame = (mavutil.mavlink.MAV_FRAME_GLOBAL if req.seq == 0
                     else mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT)
            conn.mav.mission_item_int_send(v.system_id, 1, req.seq, frame,
                                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1,
                                           float(delay), 0., 0., 0., int(lat), int(lon), float(alt), 0)
        time.sleep(.005)
    raise TimeoutError('mission upload timed out')


def read_params(v):
    values = {}
    for key, name in READ.items():
        values[key] = float(v.get_parameter(name))
    extra = {}
    for name in EXTRA:   # informational only; absent names are recorded as None
        try:
            extra[name] = float(v.get_parameter(name))
        except TimeoutError:
            extra[name] = None
    return values, extra


def fly(index, args, points, dwell, radii):
    out = args.out / f'flight_{index:02d}'
    prefix = ('/mnt/c/Windows/System32/wsl.exe', '-d', args.gazebo_distro, '--')
    with Session(out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',),
                 ardupilot=args.ardupilot) as session:
        v = session.clients['drone1']
        for name, value in SET.items():
            v.set_parameter(name, value)
        params, extra = read_params(v)
        initial = wait_state(v)
        origin = initial.position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        v.arm_and_takeoff(3.)
        home = home_position(v)
        home_enu = np.array([home.y, home.x, -home.z])
        start = origin + points[0]
        # settle at the start point in GUIDED
        deadline, stable = time.monotonic() + 60., None
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(start, math.pi / 2)
            if np.linalg.norm(s.position - start) <= .05 and np.linalg.norm(s.velocity) <= .1:
                stable = s.time_s if stable is None else stable
                if s.time_s - stable >= 1.:
                    break
            else:
                stable = None
        else:
            raise TimeoutError('start point not settled')
        items = []
        for i in range(1, len(points)):
            enu = origin + points[i] - home_enu
            lat = home.latitude + enu[1] * LOCATION_SCALING_FACTOR_INV
            scale = max(math.cos(math.radians((home.latitude + lat) * .5e-7)), .01)
            lon = home.longitude + enu[0] * LOCATION_SCALING_FACTOR_INV / scale
            delay = math.ceil(dwell[i]) if i < len(points) - 1 else 0
            items.append((round(lat), round(lon), float(enu[2]), float(delay)))
        if not args.guided:
            upload_mission(v, items)
        if args.wind is not None:
            session.world.set_wind(args.wind)
            settle_end = time.monotonic() + 5.
            while time.monotonic() < settle_end:
                wait_state(v)
                v.set_position(start, math.pi / 2)
        v.release_motors()
        s0 = wait_state(v)
        if not args.guided:
            set_mode(v, 3)   # AUTO
        t0 = s0.time_s
        rows, last_t, seq = [], None, 0
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            s = wait_state(v)
            if last_t is not None and s.time_s <= last_t:
                continue
            last_t = s.time_s
            if args.guided:
                # GUIDED point by point, as in the replica: the next point is sent once the
                # current one is accepted by the shared track rule
                if rows:
                    tt = np.array([r['t'] for r in rows] + [s.time_s - t0])
                    pp = np.array([[r['x'], r['y'], r['z']] for r in rows] + [list(s.position - origin)])
                    seq = min(evaluate(tt, pp, np.zeros_like(pp), points, dwell, radii)['reached_goals'],
                              len(points) - 1) if len(rows) % 5 == 0 else seq
                v.set_position(origin + points[seq], math.pi / 2)
            if not args.guided:
                cur = v.latest.get('MISSION_CURRENT')
                seq = int(cur.seq) if cur is not None else seq
            att = v.latest['ATTITUDE']
            p = s.position - origin
            rows.append(dict(t=round(s.time_s - t0, 4), x=p[0], y=p[1], z=p[2],
                             vx=s.velocity[0], vy=s.velocity[1], vz=s.velocity[2],
                             roll_deg=math.degrees(att.roll), pitch_deg=math.degrees(att.pitch),
                             yaw_deg=math.degrees(att.yaw), mission_seq=seq))
            if len(rows) % 50 == 0:
                t = np.array([r['t'] for r in rows])
                pos = np.array([[r['x'], r['y'], r['z']] for r in rows])
                m = evaluate(t, pos, np.zeros_like(pos), points, dwell, radii)
                if m['completed'] and t[-1] >= m['route_time_s'] + 1.:
                    break
        v.land()
    write_trace(out / 'trace.csv', rows)
    t = np.array([r['t'] for r in rows])
    pos = np.array([[r['x'], r['y'], r['z']] for r in rows])
    vel = np.array([[r['vx'], r['vy'], r['vz']] for r in rows])
    dt = np.diff(t)
    m = evaluate(t, pos, vel, points, dwell, radii)
    m.update(flight=index, wind_enu=args.wind, wpnav_params=params, other_params=extra,
             sample_interval_s=dict(median=float(np.median(dt)), max=float(dt.max())),
             mission_items=items, start_error_m=float(np.linalg.norm(pos[0] - points[0])))
    (out / 'summary.json').write_text(json.dumps(m, indent=1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flights', type=int, default=3)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--wind', type=float, nargs=3, help='Gazebo wind, ENU m/s')
    ap.add_argument('--timeout', type=float, default=90., help='wall seconds per AUTO flight')
    ap.add_argument('--guided', action='store_true',
                    help='fly the track in GUIDED, one point at a time (reference for the replica GUIDED)')
    ap.add_argument('--ardupilot', type=Path, default=Path.home() / 'ardupilot-ROSMeetUp')
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    points, dwell, radii = load_track()
    runs = []
    for i in range(args.first, args.first + args.flights):
        m = fly(i, args, points, dwell, radii)
        runs.append(m)
        print(f"flight {i}: time {m['route_time_s']} rms {m['seg_rms_m']:.3f} peak {m['seg_peak_m']:.3f} "
              f"vmax {m['peak_speed_xy_mps']:.2f} corner {m['corner_speed_xy_mps']} stops {m['stops']} "
              f"gates {m['gates']}", flush=True)
    (args.out / f'summary_{args.first}.json').write_text(json.dumps(runs, indent=1))


if __name__ == '__main__':
    main()

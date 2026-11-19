"""Firmware check of the onboard compensator across flight modes (Gazebo + SITL).

A constant side force (default 3 N, +x) acts on the hovering vehicle (RESID_ENABLE=1):
  0-15 s   GUIDED, hold the start point         compensator runs
  15-25 s  AUTO, one waypoint at the same point  compensator keeps its history (no restart)
  25-35 s  GUIDED velocity control (zero)        compensator is not called: RESID timeout (100 ms)
The applied correction is read as actual minus target attitude (ATTITUDE vs ATTITUDE_TARGET):
the attitude controller adds the correction to its error, so in steady state the vehicle leans by
the target plus the correction.

  python gazebo/check_modes.py --out runs/gazebo/check_modes
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disturbances import build_world, gazebo_prefix  # noqa: E402  (also puts the repository on sys.path)
from evaluate_ardupilot import STAND, arm_schedule  # noqa: E402
from simulation.launch import Session  # noqa: E402
from sitl import mission_items, set_mode, upload_mission, wait_state  # noqa: E402
from track_metrics import load_track  # noqa: E402

GUIDED, AUTO = 4, 3
VELOCITY_ONLY = 1 | 2 | 4 | 64 | 128 | 256 | 2048   # ignore position, acceleration and yaw rate


def euler_from_quaternion(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1., min(1., 2 * (w * y - z * x))))
    return roll, pitch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', type=float, default=3., help='side force, N (+x)')
    ap.add_argument('--ardupilot', type=Path, default=Path(os.environ.get('ARDUPILOT_PATH', Path.home() / 'ardupilot-ROSMeetUp')))
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    log_file = str((a.out / 'plugin.jsonl').resolve())
    prefix = gazebo_prefix(a.gazebo_distro)
    world = build_world(a.out, 'force', log_file, prefix)
    point = load_track()[0][0]
    rows = []
    with Session(a.out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',), ardupilot=a.ardupilot,
                 world=world) as session:
        v = session.clients['drone1']
        for name, value in STAND.items():
            v.set_parameter(name, value)
        v.set_parameter('RESID_ENABLE', 1.)
        v._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 83, 20000.)   # ATTITUDE_TARGET, 50 Hz
        origin = wait_state(v).position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        start = origin + point
        v.arm_and_takeoff(3.)
        deadline = time.monotonic() + 60.
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(start, math.pi / 2)
            if np.linalg.norm(s.position - start) < .03 and np.linalg.norm(s.velocity) < .05:
                break
        upload_mission(v, mission_items(v, origin, np.array([point, point]), np.array([0., 30.])))
        t_arm = arm_schedule(session, v, start, log_file, [f'force 0 40 {a.force} 0 0'])
        phase = 'guided'
        while True:
            s = wait_state(v)
            t = s.time_s - t_arm
            if phase == 'guided' and t >= 15.:
                set_mode(v, AUTO)
                phase = 'auto'
            elif phase == 'auto' and t >= 25.:
                set_mode(v, GUIDED)
                phase = 'velocity'
            if phase == 'guided':
                v.set_position(start, math.pi / 2)
            elif phase == 'velocity':
                v.connection.mav.set_position_target_local_ned_send(
                    0, v.system_id, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
            att, target = v.latest['ATTITUDE'], v.latest.get('ATTITUDE_TARGET')
            tr, tp = euler_from_quaternion(target.q) if target is not None else (math.nan, math.nan)
            rows.append(dict(t=round(t, 3), phase=phase, x=float(s.position[0] - start[0]),
                             y=float(s.position[1] - start[1]), z=float(s.position[2] - start[2]),
                             roll_deg=math.degrees(att.roll), pitch_deg=math.degrees(att.pitch),
                             roll_offset_deg=math.degrees(att.roll - tr), pitch_offset_deg=math.degrees(att.pitch - tp)))
            if t >= 35.:
                break
        v.land()
    (a.out / 'trace.json').write_text(json.dumps(rows))

    def window(t0, t1, key):
        x = np.array([r[key] for r in rows if t0 <= r['t'] < t1])
        return x
    dist = lambda t0, t1: np.hypot(window(t0, t1, 'x'), window(t0, t1, 'y'))   # noqa: E731
    print(f"GUIDED  10-15 s: deviation mean {100 * dist(10, 15).mean():.1f} cm, "
          f"roll correction {window(10, 15, 'roll_offset_deg').mean():+.1f} deg")
    print(f"AUTO    15-20 s: deviation mean {100 * dist(15, 20).mean():.1f} cm, max {100 * dist(15, 20).max():.1f} cm, "
          f"roll correction {window(15, 20, 'roll_offset_deg').mean():+.1f} deg")
    print(f"AUTO    20-25 s: deviation mean {100 * dist(20, 25).mean():.1f} cm")
    sw = [r for r in rows if 24.8 <= r['t'] <= 25.6]
    print('GUIDED velocity, around the switch at 25 s (t, roll correction deg):')
    print('   ' + '  '.join(f"{r['t']:.2f}:{r['roll_offset_deg']:+.1f}" for r in sw[::2]))
    print(f"GUIDED velocity 26-35 s: roll correction mean {window(26, 35, 'roll_offset_deg').mean():+.1f} deg")
    lean = np.degrees(np.arccos(np.cos(np.radians(window(0, 35, 'roll_deg'))) * np.cos(np.radians(window(0, 35, 'pitch_deg')))))
    print(f"largest lean {lean.max():.1f} deg (ATC_ANGLE_MAX limits the target plus the correction)")


if __name__ == '__main__':
    main()

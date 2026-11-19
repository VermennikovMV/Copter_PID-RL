"""How the compensator behaves with larger correction limits and forces beyond training (Gazebo + SITL).

Hover at the start point; a side force (+x) acts from 2 to 10 s, a downward force from 14 to 22 s.
For each force and limit: peak / RMS deviation and the applied tilt correction (actual minus target
attitude, 6-10 s) next to the ideal one atan(F / m g); thrust part: peak height error.

  python gazebo/limit_test.py --forces 3 5 7 --limits 0 15 25 30 --out runs/gazebo/limits
  (limit 0 = stock ArduPilot, RESID_ENABLE=0; otherwise RESID_ANG_MAX = limit deg, RESID_THR_MAX = limit %)
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
from sitl import wait_state  # noqa: E402
from track_metrics import load_track  # noqa: E402
from training.physics import MASS  # noqa: E402

G = 9.81


def euler(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1., min(1., 2 * (w * y - z * x)))))


def fly(force, limit, a, out):
    out.mkdir(parents=True, exist_ok=False)
    log_file = str((out / 'plugin.jsonl').resolve())
    prefix = gazebo_prefix(a.gazebo_distro)
    world = build_world(out, 'force', log_file, prefix)
    point = load_track()[0][0]
    rows = []
    with Session(out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',), ardupilot=a.ardupilot,
                 world=world) as session:
        v = session.clients['drone1']
        for name, value in STAND.items():
            v.set_parameter(name, value)
        v.set_parameter('RESID_ENABLE', 1. if limit else 0.)
        if limit:
            v.set_parameter('RESID_ANG_MAX', float(limit))
            v.set_parameter('RESID_THR_MAX', limit / 100.)
        v._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 83, 20000.)   # ATTITUDE_TARGET
        origin = wait_state(v).position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        start = origin + point
        v.arm_and_takeoff(3.)
        deadline = time.monotonic() + 60.
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(start, math.pi / 2)
            if np.linalg.norm(s.position - start) < .03 and np.linalg.norm(s.velocity) < .05:
                break
        t_arm = arm_schedule(session, v, start, log_file,
                            [f'force 2 10 {force} 0 0', f'force 14 22 0 0 {-force}'])
        while True:
            s = wait_state(v)
            t = s.time_s - t_arm
            v.set_position(start, math.pi / 2)
            att, target = v.latest['ATTITUDE'], v.latest.get('ATTITUDE_TARGET')
            tr, tp = euler(target.q) if target is not None else (math.nan, math.nan)
            d = s.position - start
            rows.append(dict(t=t, x=float(d[0]), y=float(d[1]), z=float(d[2]),
                             roll=math.degrees(att.roll), roll_off=math.degrees(att.roll - tr),
                             lean=math.degrees(math.acos(max(-1., min(1., math.cos(att.roll) * math.cos(att.pitch)))))))
            if t >= 26.:
                break
        v.land()
    (out / 'trace.json').write_text(json.dumps(rows))
    t = np.array([r['t'] for r in rows])
    h = np.hypot([r['x'] for r in rows], [r['y'] for r in rows])
    z = np.abs([r['z'] for r in rows])
    side, steady, down = (t >= 2) & (t < 10), (t >= 6) & (t < 10), (t >= 14) & (t < 22)
    corr = -np.mean([r['roll_off'] for r, m in zip(rows, steady) if m])   # force +x -> lean to -roll? sign-free below
    return dict(force=force, limit=limit, side_peak_cm=100 * h[side].max(), side_rms_cm=100 * math.sqrt(np.mean(h[side] ** 2)),
                tilt_correction_deg=abs(corr), ideal_deg=math.degrees(math.atan(force / (MASS * G))),
                lean_peak_deg=max(r['lean'] for r, m in zip(rows, side) if m),
                down_peak_cm=100 * z[down].max(), down_rms_cm=100 * math.sqrt(np.mean(z[down] ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forces', type=float, nargs='+', default=[3., 5., 7.])
    ap.add_argument('--limits', type=float, nargs='+', default=[0., 15., 25., 30.])
    ap.add_argument('--ardupilot', type=Path, default=Path(os.environ.get('ARDUPILOT_PATH', Path.home() / 'ardupilot-ROSMeetUp')))
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    res = []
    for f in a.forces:
        for lim in a.limits:
            name = a.out / f'F{f:g}_L{lim:g}'
            if (name / 'result.json').is_file():
                r = json.loads((name / 'result.json').read_text())
            else:
                r = fly(f, lim, a, name)
                (name / 'result.json').write_text(json.dumps(r))
            res.append(r)
            print(f"F={f:g} N limit={'stock' if not lim else f'{lim:g}'}: side peak {r['side_peak_cm']:.1f} cm, "
                  f"RMS {r['side_rms_cm']:.1f} cm, correction {r['tilt_correction_deg']:.1f} deg "
                  f"(ideal {r['ideal_deg']:.1f}), lean peak {r['lean_peak_deg']:.0f} deg | down peak "
                  f"{r['down_peak_cm']:.1f} cm, RMS {r['down_rms_cm']:.1f} cm", flush=True)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

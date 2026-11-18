"""Gazebo + SITL comparison: stock ArduPilot GUIDED (RESID_ENABLE=0) vs GUIDED with the onboard
disturbance compensator (RESID_ENABLE=1), same firmware build (ardupilot-ROSMeetUp).

Cases mirror the replica evaluation (same cases and metrics as the PyBullet eval):
  hover/nominal  hover/gust  hover/payload  hover/joint  route/nominal  route/gust  route/joint
  gust    = experiments/scenarios.py force_events('gust', seed), applied as a body force (plugin)
  payload = real 0.35 kg mass (0.2*m, inertia 0.2*I) on a DetachableJoint, released at 12 s (hover)
            or 4 s (route); joint = gust + payload
Time zero = the moment the disturbance schedule is armed in Gazebo, with the vehicle settled at the
start point [0,0,3]. The route is the evaluation track, GUIDED point by point (next point once the
shared acceptance rule reports the current one). ArduPilot shapes each position target itself.

Controllers (same firmware build, the network runs inside it):
  stock    RESID_ENABLE=0 (plain ArduPilot)
  network  RESID_ENABLE=1 (the onboard compensator's bounded correction is active in GUIDED)

  python gazebo/evaluate_ardupilot.py --controller stock --cases hover/gust --seeds 90000-90007 --out runs/gazebo/stock
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disturbances import build_world, gazebo_prefix  # noqa: E402  (also puts the repository on sys.path)
from experiments.scenarios import force_events  # noqa: E402
from simulation.launch import Session  # noqa: E402
from sitl import read_params, wait_state  # noqa: E402
from track_metrics import evaluate, load_track, segment_distance, write_trace  # noqa: E402

CASES = ('hover/nominal', 'hover/gust', 'hover/payload', 'hover/joint',
         'route/nominal', 'route/gust', 'route/joint')
HOVER_S = 30.
DROP_S = {'hover': 12., 'route': 4.}
STAND = dict(WP_SPD=3.0, WP_RADIUS_M=0.4, WP_YAW_BEHAVIOR=0, RESID_MODES=16,
             RESID_THR_HOVR=0.303)  # measured steady hover throttle of the stand (MOT_THST_HOVER stays 0.5)


def parse_seeds(text):
    if '-' in text:
        a, b = text.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(',')]


def schedule(case, seed):
    mission, scenario = case.split('/')
    forces = force_events('gust', seed) if scenario in ('gust', 'joint') else []
    lines = [f"force {e['on_s']} {e['off_s']} {e['force_enu_N'][0]} {e['force_enu_N'][1]} {e['force_enu_N'][2]}"
             for e in forces]
    drop = DROP_S[mission] if scenario in ('payload', 'joint') else None
    if drop is not None:
        lines.append(f'detach {drop}')
    return lines, drop, forces


def arm_schedule(session, v, goal, log_file, lines):
    """Publish the schedule until the plugin reports it armed (resend only after 3 s)."""
    text = '\\n'.join(lines)
    for _ in range(5):
        session.world.publish('/copter/disturbance', text)
        end = time.monotonic() + 3.
        while time.monotonic() < end:
            if 'schedule_armed' in Path(log_file).read_text():
                break
            wait_state(v)
            v.set_position(goal, math.pi / 2)
        events = [json.loads(x) for x in Path(log_file).read_text().splitlines() if x.strip()]
        armed = [e for e in events if e['event'] == 'schedule_armed']
        if armed:
            return armed[0]['sim_time']
    raise RuntimeError('disturbance schedule was not armed')


def fly(case, seed, controller, args, out):
    mission, scenario = case.split('/')
    points, dwell, radii = load_track()
    lines, drop, forces = schedule(case, seed)
    out.mkdir(parents=True, exist_ok=False)
    log_file = str((out / 'plugin.jsonl').resolve())
    prefix = gazebo_prefix(args.gazebo_distro)
    world = build_world(out, 'payload_drop' if drop is not None else 'force', log_file, prefix)
    rows = []
    with Session(out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',), ardupilot=args.ardupilot,
                 world=world) as session:
        v = session.clients['drone1']
        for name, value in STAND.items():
            v.set_parameter(name, value)
        v.set_parameter('RESID_ENABLE', 1. if controller == 'network' else 0.)
        params, extra = read_params(v)
        extra.update(RESID_ENABLE=v.get_parameter('RESID_ENABLE'), RESID_MODES=v.get_parameter('RESID_MODES'),
                     RESID_ANG_MAX=v.get_parameter('RESID_ANG_MAX'), RESID_THR_MAX=v.get_parameter('RESID_THR_MAX'))
        initial = wait_state(v)
        origin = initial.position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        start = origin + points[0]
        v.arm_and_takeoff(3.)
        deadline, stable = time.monotonic() + 60., None
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(start, math.pi / 2)
            if np.linalg.norm(s.position - start) <= .03 and np.linalg.norm(s.velocity) <= .05:
                stable = s.time_s if stable is None else stable
                if s.time_s - stable >= 2.:
                    break
            else:
                stable = None
        else:
            raise TimeoutError('start point not settled')
        t_arm = arm_schedule(session, v, start, log_file, lines)
        index, last, horizon = 0, None, (HOVER_S if mission == 'hover' else args.route_timeout)
        end = time.monotonic() + horizon * 4 + 30.
        while time.monotonic() < end:
            s = wait_state(v)
            if last is not None and s.time_s <= last:
                continue
            last = s.time_s
            t = s.time_s - t_arm
            p = s.position - origin
            if mission == 'route' and rows and len(rows) % 5 == 0:
                tt = np.array([r['t'] for r in rows if r['t'] >= 0] + [t])
                pp = np.array([[r['x'], r['y'], r['z']] for r in rows if r['t'] >= 0] + [list(p)])
                if len(tt) > 1:
                    index = min(evaluate(tt, pp, np.zeros_like(pp), points, dwell, radii)['reached_goals'],
                                len(points) - 1)
            target = origin + (points[index] if mission == 'route' else points[0])
            v.set_position(target, math.pi / 2)
            att = v.latest['ATTITUDE']
            rows.append(dict(t=round(t, 4), x=p[0], y=p[1], z=p[2], vx=s.velocity[0], vy=s.velocity[1],
                             vz=s.velocity[2], roll_deg=math.degrees(att.roll), pitch_deg=math.degrees(att.pitch),
                             target_index=index))
            if t >= horizon:
                break
            if mission == 'route' and len(rows) % 50 == 0:
                tt = np.array([r['t'] for r in rows if r['t'] >= 0])
                pp = np.array([[r['x'], r['y'], r['z']] for r in rows if r['t'] >= 0])
                m = evaluate(tt, pp, np.zeros_like(pp), points, dwell, radii)
                if m['completed'] and t >= m['route_time_s'] + 1.:
                    break
        v.land()
    write_trace(out / 'trace.csv', rows)
    result = metrics(mission, rows, points, dwell, radii, drop)
    result.update(case=case, seed=seed, controller=controller, forces=forces, drop_s=drop,
                  wpnav_params=params, other_params=extra, armed_gz_sim_time_s=t_arm)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    return result


def metrics(mission, rows, points, dwell, radii, drop):
    r = [x for x in rows if x['t'] >= 0]
    t = np.array([x['t'] for x in r])
    pos = np.array([[x['x'], x['y'], x['z']] for x in r])
    vel = np.array([[x['vx'], x['vy'], x['vz']] for x in r])
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians([x['roll_deg'] for x in r]))
                                        * np.cos(np.radians([x['pitch_deg'] for x in r])), -1, 1)))
    out = dict(tilt_peak_deg=float(tilt.max()), samples=len(r),
               max_sample_gap_s=float(np.diff(t).max()) if len(t) > 1 else None)
    if mission == 'hover':
        e = np.linalg.norm(pos - points[0], axis=1)
        m = t >= 2.   # same 2 s skip as the replica hover RMSE
        out.update(rmse=float(np.sqrt(np.mean(e[m] ** 2))), peak=float(e[m].max()), survived=bool(e.max() < 3.))
    else:
        m = evaluate(t, pos, vel, points, dwell, radii)
        out.update(route_time=m['route_time_s'], completed=m['completed'], seg_rms=m['seg_rms_m'],
                   seg_peak=m['seg_peak_m'], peak_speed_xy=m['peak_speed_xy_mps'], events=m['events'])
    if drop is not None:
        # payload_z_peak as in the replica eval: max |z - z0| within 3 s after the release, z0 = target
        # altitude at the release, counted only while the target altitude stays the same
        idx = np.array([x['target_index'] for x in r])
        z_target = points[idx if mission == 'route' else np.zeros_like(idx), 2]
        k0 = int(np.searchsorted(t, drop))
        z0 = z_target[min(k0, len(t) - 1)]
        k = (t >= drop) & (t <= drop + 3.) & (np.abs(z_target - z0) < 1e-9)
        out['payload_z_peak'] = float(np.abs(pos[k, 2] - z0).max()) if k.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controller', choices=('stock', 'network'), default='stock')
    ap.add_argument('--cases', default='all', help='comma list, e.g. hover/gust,route/joint')
    ap.add_argument('--seeds', default='90000-90007')
    ap.add_argument('--route-timeout', type=float, default=60., help='simulated seconds per route flight')
    ap.add_argument('--ardupilot', type=Path, default=Path(os.environ.get('ARDUPILOT_PATH', Path.home() / 'ardupilot-ROSMeetUp')))
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--wp-spd', type=float, default=None, help='override WP_SPD (m/s) for both controllers')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.wp_spd is not None:
        STAND['WP_SPD'] = args.wp_spd
    cases = CASES if args.cases == 'all' else tuple(args.cases.split(','))
    seeds = parse_seeds(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        # deterministic cases need no seed sweep: gust profiles depend on the seed, nominal/payload do not
        case_seeds = seeds if case.split('/')[1] in ('gust', 'joint') else seeds[:1]
        for seed in case_seeds:
            name = f"{case.replace('/', '_')}_{seed}"
            if (args.out / name / 'summary.json').is_file():
                results.append(json.loads((args.out / name / 'summary.json').read_text()))
                continue
            r = fly(case, seed, args.controller, args, args.out / name)
            results.append(r)
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                         if k in ('route_time', 'completed', 'seg_rms', 'seg_peak', 'rmse', 'peak', 'payload_z_peak',
                                  'tilt_peak_deg')}, flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()

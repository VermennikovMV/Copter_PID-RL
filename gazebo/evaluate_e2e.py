"""Gazebo + SITL evaluation of the end-to-end policy (network -> four motor PWM).

Same cases, disturbances (plugin), track and metrics as evaluate_ardupilot.py. ArduPilot takes off and
settles at the start point; then the motor relay holds the native servo output and the policy
drives the motors directly in synchronous 20 ms steps (20 x 1 ms physics per action).

  python gazebo/evaluate_e2e.py --policy models/e2e_policy.npz --cases all --seeds 90000-90007 --out runs/gazebo/e2e
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
from evaluate_ardupilot import CASES, HOVER_S, STAND, arm_schedule, metrics, parse_seeds, schedule  # noqa: E402
from simulation.launch import Session  # noqa: E402
from sitl import wait_state  # noqa: E402
from track_metrics import load_track, write_trace  # noqa: E402
from controllers.base import EncoderConfig, ObservationEncoder  # noqa: E402
from training.evaluate import Policy  # noqa: E402
from training.wrappers import track_features  # noqa: E402
from experiments.mission import Mission  # noqa: E402
from training.physics import DirectMotorModel  # noqa: E402

DT = .02


def fly(case, seed, actor, cfg, args, out):
    mission_kind, scenario = case.split('/')
    points, dwell, radii = load_track()
    lines, drop, forces = schedule(case, seed)
    out.mkdir(parents=True, exist_ok=False)
    log_file = str((out / 'plugin.jsonl').resolve())
    prefix = gazebo_prefix(args.gazebo_distro)
    world = build_world(out, 'payload_drop' if drop is not None else 'force', log_file, prefix)
    motors = DirectMotorModel()
    yaw_ref = math.pi / 2
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    horizon = HOVER_S if mission_kind == 'hover' else min(300., max(20., length / .7 + float(np.sum(dwell)) + 10.))
    rows = []
    with Session(out / 'simulation', gazebo_prefix=prefix, vehicle_ids=('drone1',), ardupilot=args.ardupilot,
                 world=world) as session:
        v = session.clients['drone1']
        for name, value in STAND.items():
            v.set_parameter(name, value)
        v.set_parameter('RESID_ENABLE', 0.)
        initial = wait_state(v)
        origin = initial.position - np.array([0., 0., session.vehicles[0].position_enu[2]])
        start = origin + points[0]
        v.arm_and_takeoff(3.)
        deadline, stable = time.monotonic() + 60., None
        while time.monotonic() < deadline:
            s = wait_state(v)
            v.set_position(start, yaw_ref)
            if np.linalg.norm(s.position - start) <= .03 and np.linalg.norm(s.velocity) <= .05:
                stable = s.time_s if stable is None else stable
                if s.time_s - stable >= 2.:
                    break
            else:
                stable = None
        else:
            raise TimeoutError('start point not settled')
        t_arm = arm_schedule(session, v, start, log_file, lines)
        mission = Mission(mission_kind, .4, points=points, dwell_s=dwell, radii_m=radii)
        encoder = ObservationEncoder(EncoderConfig(**cfg['observation']))
        encoder.reset(yaw_ref)
        encoder.set_task(horizon, mission_kind)
        v.pause_control()
        s = wait_state(v)
        prev = np.zeros(4)
        t0 = s.time_s
        mission.update(0., s.position - origin)
        while True:
            t = s.time_s - t_arm
            p = s.position - origin
            target = mission.target
            obs = encoder.observe(p, s.velocity, s.rotation, s.body_rates, target, prev, DT)
            obs = np.r_[obs, track_features(mission, p, s.rotation)]
            a = actor(obs.astype(np.float32).astype(np.float64))
            v.set_motor_pwm(motors.action_to_pwm(a))
            prev = a
            r = s.rotation
            rows.append(dict(t=round(t, 4), x=p[0], y=p[1], z=p[2], vx=s.velocity[0], vy=s.velocity[1],
                             vz=s.velocity[2], roll_deg=math.degrees(math.atan2(r[2, 1], r[2, 2])),
                             pitch_deg=math.degrees(-math.asin(max(-1., min(1., r[2, 0])))),
                             target_index=int(mission.index), action=a.tolist()))
            v.step_control(DT)
            s = wait_state(v)
            mission.update(s.time_s - t0, s.position - origin)
            done = (mission_kind == 'route' and mission.completion_time is not None
                    and s.time_s - t0 > mission.completion_time + 1.)
            pp = s.position - origin
            if mission_kind == "route" and mission.index > 0:
                a_, b_ = points[mission.index - 1], points[mission.index]
                leg = b_ - a_; q = float(np.clip((pp - a_) @ leg / (leg @ leg), 0, 1)); off = float(np.linalg.norm(pp - (a_ + q * leg)))
            else:
                off = float(np.linalg.norm(pp - target))
            crashed = off > 4. or pp[2] < .25 or s.rotation[2, 2] < .2
            if t >= (HOVER_S if mission_kind == 'hover' else args.route_timeout) or done or crashed:
                break
        v.resume_control()
    write_trace(out / 'trace.csv', [{k: x for k, x in r.items() if k != 'action'} for r in rows])
    result = metrics(mission_kind, rows, points, dwell, radii, drop)
    result.update(case=case, seed=seed, controller='e2e', forces=forces, drop_s=drop, crashed=bool(crashed),
                  armed_gz_sim_time_s=t_arm)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/e2e_eval.json')
    ap.add_argument('--cases', default='all')
    ap.add_argument('--seeds', default='90000-90007')
    ap.add_argument('--route-timeout', type=float, default=60.)
    ap.add_argument('--ardupilot', type=Path, default=Path(os.environ.get('ARDUPILOT_PATH', Path.home() / 'ardupilot-ROSMeetUp')))
    ap.add_argument('--gazebo-distro', default='Ubuntu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    actor = Policy(args.policy)
    cfg = json.loads(args.config.read_text())
    cases = CASES if args.cases == 'all' else tuple(args.cases.split(','))
    seeds = parse_seeds(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        case_seeds = seeds if case.split('/')[1] in ('gust', 'joint') else seeds[:1]
        for seed in case_seeds:
            name = f"{case.replace('/', '_')}_{seed}"
            if (args.out / name / 'summary.json').is_file():
                results.append(json.loads((args.out / name / 'summary.json').read_text()))
                continue
            r = fly(case, seed, actor, cfg, args, args.out / name)
            results.append(r)
            print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()
                         if k in ('route_time', 'completed', 'seg_rms', 'seg_peak', 'rmse', 'peak',
                                  'payload_z_peak', 'tilt_peak_deg', 'crashed')}, flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()

"""Matched PyBullet evaluation: ArduPilot cascade copy vs compensator vs end-to-end policy.

Every controller flies the same cases with the same seeds (disturbance schedules, initial states):
  hover/{nominal,gust,payload,joint}, route/{nominal,gust,joint}  (joint = gusts + payload drop)

  python -m training.evaluate --estimator models/estimator.npz --policy models/e2e_policy.npz \
      --seeds 8 --out results/pybullet.json
"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CASES = [('hover', 'nominal'), ('hover', 'gust'), ('hover', 'payload'), ('hover', 'joint'),
         ('route', 'nominal'), ('route', 'gust'), ('route', 'joint')]
METRICS = ['survived', 'rmse', 'peak', 'payload_z_peak', 'route_time', 'completed', 'seg_rms', 'seg_peak',
           'yaw_rms_deg', 'tilt_peak_deg']


class Policy:
    """Dense tanh actor exported with training.tools export-actor (keys w0, b0, w1, ...)."""
    def __init__(self, path):
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith('w'))
        self.layers = [(z[f'w{i}'].astype(np.float64), z[f'b{i}'].astype(np.float64)) for i in range(n)]

    def __call__(self, obs):
        x = np.asarray(obs, np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1., 1.)


def segment_distance(p, a, b):
    leg = b - a
    s = leg @ leg
    q = float(np.clip((p - a) @ leg / s, 0, 1)) if s else 0.
    return float(np.linalg.norm(p - (a + q * leg)))


def fly(job):
    method, args, mission, scenario, seed = job
    from training import stacks
    if method == 'e2e':
        env, policy = stacks.end_to_end(args['e2e_config'], mission=mission, scenario=scenario), Policy(args['policy'])
    elif method == 'compensator':
        env, policy = stacks.compensator(args['estimator'], args['config'], args['stand'], mission=mission,
                                         scenario=scenario, shaping_overrides=args.get('shaping')), None
    else:
        env, policy = stacks.ardupilot(args['config'], args['stand'], mission=mission, scenario=scenario,
                                       shaping_overrides=args.get('shaping')), None
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    errs, yaws, segs, tilts = [], [], [], []
    drop_t, z0, zpeak, failed = None, None, 0., False
    while True:
        obs, _, term, trunc, info = env.step(np.zeros(4) if policy is None else policy(obs))
        p, tgt = np.asarray(info['position']), np.asarray(info['target'])
        errs.append(float(np.linalg.norm(p - tgt)))
        yaws.append(abs(info['yaw_error']))
        tilts.append(math.degrees(math.acos(min(1., info['tilt_cos']))))
        if info['on_leg']:
            i = int(info['goal_index'])
            segs.append(segment_distance(p, base.mission.points[i - 1], base.mission.points[i]))
        if info.get('payload_drop_event') and drop_t is None:
            drop_t, z0 = info['payload_drop_event']['time_s'], float(tgt[2])
        if drop_t is not None and info['time'] <= drop_t + 3 and abs(tgt[2] - z0) < 1e-9:
            zpeak = max(zpeak, abs(p[2] - z0))
        failed |= bool(info['failed'])
        if term or trunc:
            break
    env.close()
    e = np.asarray(errs)[100:]   # hover: skip the first 2 s
    r = dict(method=method, mission=mission, scenario=scenario, seed=seed, survived=not failed,
             yaw_rms_deg=math.degrees(float(np.sqrt(np.mean(np.square(yaws))))), tilt_peak_deg=float(max(tilts)),
             payload_z_peak=zpeak if drop_t is not None else None)
    if mission == 'hover' and len(e):
        r.update(rmse=float(np.sqrt(np.mean(e ** 2))), peak=float(e.max()))
    if mission == 'route':
        r.update(route_time=info['route_completion_s'], completed=bool(info['route_completed']),
                 seg_rms=float(np.sqrt(np.mean(np.square(segs)))) if segs else None,
                 seg_peak=float(max(segs)) if segs else None)
    return r


def summarize(rows):
    table = {}
    for r in rows:
        table.setdefault(f"{r['mission']}/{r['scenario']}", {}).setdefault(r['method'], []).append(r)
    out = {}
    for case, methods in table.items():
        out[case] = {}
        for m, rs in methods.items():
            out[case][m] = {k: round(float(np.mean([float(x[k]) for x in rs if x.get(k) is not None])), 4)
                            for k in METRICS if any(x.get(k) is not None for x in rs)}
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--estimator', help='disturbance estimator .npz (adds the compensator)')
    ap.add_argument('--policy', help='end-to-end actor .npz (adds the end-to-end controller)')
    ap.add_argument('--no-ardupilot', action='store_true', help='skip the plain cascade baseline')
    ap.add_argument('--config', default=str(root / 'configs/compensator_eval.json'))
    ap.add_argument('--e2e-config', default=str(root / 'configs/e2e_eval.json'))
    ap.add_argument('--stand', default=str(root / 'configs/stand_params.json'))
    ap.add_argument('--wp-speed', type=float, help='override the GUIDED speed limit (m/s)')
    ap.add_argument('--cases', default='all', help='comma list, e.g. hover/gust,route/joint')
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--seed0', type=int, default=90000)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    args = dict(config=a.config, e2e_config=a.e2e_config, stand=a.stand, estimator=a.estimator, policy=a.policy,
                shaping=dict(speed_ne_ms=a.wp_speed) if a.wp_speed else None)
    methods = ([] if a.no_ardupilot else ['ardupilot']) + (['compensator'] if a.estimator else []) + \
        (['e2e'] if a.policy else [])
    cases = CASES if a.cases == 'all' else [tuple(c.split('/')) for c in a.cases.split(',')]
    jobs = [(m, args, mi, sc, a.seed0 + s) for m in methods for mi, sc in cases for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(fly, jobs, chunksize=1)
    table = summarize(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, table=table), indent=1))
    for case, ms in table.items():
        print(case)
        for m, d in ms.items():
            print(f'  {m:12s}', ' '.join(f'{k}={v}' for k, v in d.items()))


if __name__ == '__main__':
    main()

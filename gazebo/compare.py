"""Compare Gazebo result folders case by case on the common seeds; the first folder is the reference.

  python gazebo/compare.py runs/gazebo/stock runs/gazebo/network runs/gazebo/e2e
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ('rmse', 'peak', 'payload_z_peak', 'route_time', 'seg_rms', 'seg_peak', 'tilt_peak_deg')


def load(folder):
    runs = collections.defaultdict(dict)
    for f in Path(folder).glob('*/summary.json'):
        r = json.loads(f.read_text())
        runs[r['case']][r['seed']] = r
    return runs


def main(folders):
    ref = load(folders[0])
    for folder in folders[1:]:
        new = load(folder)
        print(f'== {Path(folder).name} vs {Path(folders[0]).name}')
        for case in sorted(new):
            seeds = sorted(set(new[case]) & set(ref[case]))
            if not seeds:
                continue
            cols = []
            for k in KEYS:
                a = [new[case][s][k] for s in seeds if new[case][s].get(k) is not None]
                b = [ref[case][s][k] for s in seeds if ref[case][s].get(k) is not None]
                if a and b:
                    cols.append(f'{k} {np.mean(b):.3f} -> {np.mean(a):.3f} ({(np.mean(a) / np.mean(b) - 1) * 100:+.0f}%)')
            ok = sum(1 for s in seeds if new[case][s].get('survived', new[case][s].get('completed', True)))
            print(f'{case:14s} {ok}/{len(seeds)} ok | ' + ' | '.join(cols))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])

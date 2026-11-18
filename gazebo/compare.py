"""Compare two Gazebo result folders case by case (same seeds): python gazebo/compare.py stock network [runs/gazebo]"""
import json, glob, sys, numpy as np, collections
from pathlib import Path
base = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / 'runs/gazebo'
def load(c):
    d = collections.defaultdict(list)
    for f in glob.glob(f'{base}/{c}/*/summary.json'):
        r = json.load(open(f)); d[r['case']].append(r)
    return d
ref, new = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('stock', 'network')
S, N = load(ref), load(new)
keys = ['rmse', 'peak', 'payload_z_peak', 'route_time', 'seg_rms', 'seg_peak', 'tilt_peak_deg']
for case in sorted(N):
    ss = [r for r in S[case] if any(r['seed'] == n['seed'] for n in N[case])]
    out = []
    for k in keys:
        a = [r[k] for r in N[case] if r.get(k) is not None]; b = [r[k] for r in ss if r.get(k) is not None]
        if a and b: out.append(f'{k} {np.mean(b):.3f}->{np.mean(a):.3f} ({(np.mean(a)/np.mean(b)-1)*100:+.0f}%)')
    ok = sum(1 for r in N[case] if r.get('survived', r.get('completed', True)) and not r.get('crashed', False))
    print(f'{case} {ok}/{len(N[case])} | ' + ' | '.join(out))

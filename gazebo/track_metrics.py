"""Track metrics shared by the Gazebo/SITL flights and the PyBullet replica.

The acceptance rule is the project's Mission rule (ordered points, per-point radius,
continuous dwell inside the radius), applied to a recorded 50 Hz trace of positions
relative to the track origin (ENU, metres).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'compensator_eval.json'   # track.fixed


def load_track(path=CONFIG):
    fixed = json.loads(Path(path).read_text(encoding='utf-8'))['track']['fixed']
    return (np.asarray(fixed['points'], float), np.asarray(fixed['dwell_s'], float),
            np.asarray(fixed['radii_m'], float))


def segment_distance(p, a, b):
    leg = b - a
    s = float(leg @ leg)
    q = float(np.clip((p - a) @ leg / s, 0., 1.)) if s else 0.
    return float(np.linalg.norm(p - (a + q * leg)))


def evaluate(t, pos, vel, points, dwell_s, radii_m, gap_limit_s=.05):
    """Return events and metrics. t starts at the moment the flight begins (s)."""
    t, pos, vel = np.asarray(t, float), np.asarray(pos, float), np.asarray(vel, float)
    index, inside_since, last, done = 0, None, None, None
    events, seg = [], []
    for k in range(len(t)):
        if last is not None and t[k] - last > gap_limit_s:
            inside_since = None
        last = t[k]
        if done is None and index >= 1:
            seg.append(segment_distance(pos[k], points[index - 1], points[index]))
        if done is not None:
            continue
        if np.linalg.norm(pos[k] - points[index]) > radii_m[index]:
            inside_since = None
            continue
        if inside_since is None:
            inside_since = t[k]
        if t[k] - inside_since + 1e-9 < dwell_s[index]:
            continue
        events.append(dict(goal_index=index, time_s=float(t[k]), dwell_started_s=float(inside_since)))
        inside_since = None
        if index == len(points) - 1:
            done = float(t[k])
        else:
            index += 1
    speed_xy = np.hypot(vel[:, 0], vel[:, 1])
    end = len(t) if done is None else int(np.searchsorted(t, done)) + 1
    gates = {}
    for i in range(1, len(points) - 1):
        if dwell_s[i] > 0:
            continue
        d = np.linalg.norm(pos[:end] - points[i], axis=1)
        j = int(np.argmin(d))
        gates[str(i)] = dict(closest_m=float(d[j]), speed_xy_mps=float(speed_xy[j]))
    stops = {str(e['goal_index']): dict(arrival_s=e['dwell_started_s'], release_s=e['time_s'])
             for e in events if 0 < e['goal_index'] < len(points) - 1 and dwell_s[e['goal_index']] > 0}
    seg = np.asarray(seg)
    return dict(completed=done is not None, route_time_s=done, events=events,
                reached_goals=len(events), stops=stops, gates=gates,
                seg_rms_m=float(np.sqrt(np.mean(seg ** 2))) if len(seg) else None,
                seg_peak_m=float(seg.max()) if len(seg) else None,
                peak_speed_xy_mps=float(speed_xy[:end].max()),
                corner_speed_xy_mps=float(np.mean([g['speed_xy_mps'] for g in gates.values()])) if gates else None)


def write_trace(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_trace(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    col = lambda *k: np.array([[float(r[x]) for x in k] for r in rows])
    return col('t')[:, 0], col('x', 'y', 'z'), col('vx', 'vy', 'vz')

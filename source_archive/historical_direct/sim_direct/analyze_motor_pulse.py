"""Causal motor-order check using physical output packets and Gazebo poses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args()
    folder=args.run
    actuator=records(folder/'actuators.jsonl')
    groups=[]
    previous=False
    for row in actuator:
        if row['override']:
            if not previous:groups.append([])
            groups[-1].append(row)
        previous=row['override']
    truth=records(folder/'groundtruth.jsonl')
    times=np.array([row['wall_time'] for row in truth])
    sim_times=np.array([row['sim_time'] for row in truth])
    rotations=Rotation.from_quat([row['quaternion_xyzw'] for row in truth])
    delta=rotations[:-1].inv()*rotations[1:]
    rates=delta.as_rotvec()/np.diff(sim_times)[:,None]
    midpoint=(times[1:]+times[:-1])/2
    expected=np.array([[-1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1]])
    results=[]
    for group in groups:
        start=group[0]['wall'];end=group[-1]['wall']
        change=np.array(group[0]['output_pwm'])-group[0]['input_pwm']
        indices=np.flatnonzero(change)
        if len(indices)!=(4 if args.balanced else 1):raise ValueError('Unexpected number of perturbed motors')
        rotor=0 if args.balanced else int(indices[0])
        before=(midpoint>=start-.08)&(midpoint<start)
        during=(midpoint>=start+.02)&(midpoint<=end+.02)
        if before.sum()<2 or during.sum()<2:raise ValueError('Insufficient independent pose samples')
        response=rates[during].mean(axis=0)-rates[before].mean(axis=0)
        observed=np.sign(response).astype(int)
        results.append({'rotor':rotor,'motor_name':'balanced_yaw' if args.balanced else ['FR','RL','FL','RR'][rotor],
            'pulse_pwm':int(change[rotor]),'override_packets':len(group),
            'wall_start':start,'wall_end':end,'mean_body_rate_change_rad_s':response.tolist(),
            'expected_roll_pitch_yaw_sign':expected[rotor].tolist(),
            'observed_sign':observed.tolist(),
            'normalized_yaw_sign':int(observed[2]*np.sign(change[0])) if args.balanced else None,
            'pass':bool(np.array_equal(observed,expected[rotor]*np.sign(change[rotor])))})
    result={'scope':'Development physical motor pulse; angular velocity independently derived from Gazebo world orientation differences, not MAVLink gyro.',
        'alignment':'Pose receive wall clock against proxy physical output send wall clock; short local run. Means bracket each actual override group.',
        'pass':len(results)==4 and (len({row['normalized_yaw_sign'] for row in results})==1 if args.balanced else all(row['pass'] for row in results)),
        'results':results,'source_hashes':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['actuators.jsonl','groundtruth.jsonl','telemetry.jsonl']}}
    if args.balanced:
        sign=results[0]['normalized_yaw_sign']
        result['inferred_yaw_torque_signs']=[sign,sign,-sign,-sign] if result['pass'] else None
    (folder/('balanced_yaw_validation.json' if args.balanced else 'motor_order_validation.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())

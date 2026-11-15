"""Validate immediate physical yaw response before native PID counteraction.

The complete FDM/actuator sequence is required. Its 1:1 correspondence provides
a simulation-time alignment; this check does not rely on cross-process clocks.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    folder=args.run
    packets=read(folder/'actuators.jsonl');physics=read(folder/'physics.jsonl');truth=read(folder/'groundtruth.jsonl')
    frames=np.array([r['frame'] for r in packets]);timestamps=np.array([r['timestamp'] for r in physics])
    if len(physics)!=len(packets)+1 or not np.array_equal(frames,np.arange(len(packets))):
        raise ValueError('Complete ordered command/FDM exchange required')
    if not np.allclose(np.diff(timestamps),.001,atol=1e-10):raise ValueError('Unexpected physical integration step')
    if not np.allclose(timestamps[:-1],frames*.001+timestamps[0],atol=1e-10):
        raise ValueError('Frame/simulation-time association is not exact')
    gyro=np.array([r['imu']['gyro'] for r in physics])*[1,-1,-1]
    times=np.array([r['sim_time'] for r in truth]);rot=Rotation.from_quat([r['quaternion_xyzw'] for r in truth])
    pose_rates=(rot[:-1].inv()*rot[1:]).as_rotvec()/np.diff(times)[:,None]
    midtimes=(times[1:]+times[:-1])/2
    starts=[i for i,r in enumerate(packets) if r['override'] and (i==0 or not packets[i-1]['override'])]
    results=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(starts),1,figsize=(9,8),sharex=True)
    for ax,index in zip(np.atleast_1d(axes),starts):
        start=timestamps[index]
        delta=np.array(packets[index]['output_pwm'])-packets[index]['input_pwm']
        if not (np.sign(delta)==np.sign(delta[0])*np.array([1,1,-1,-1])).all():
            raise ValueError('Not a balanced yaw pulse')
        pre=(timestamps>=start-.08)&(timestamps<start)
        early=(timestamps>=start+.003)&(timestamps<start+.023)
        ppre=(midtimes>=start-.08)&(midtimes<start)
        pearly=(midtimes>=start)&(midtimes<start+.035)
        response=float(gyro[early,2].mean()-gyro[pre,2].mean())
        pose_response=float(pose_rates[pearly,2].mean()-pose_rates[ppre,2].mean())
        expected=-np.sign(delta[0])
        passed=bool(np.sign(response)==expected and np.sign(pose_response)==expected and abs(response)>.1)
        results.append({'frame_start':int(frames[index]),'sim_time_start':start,
            'pwm_delta':delta.tolist(),'physical_gyro_delta_rad_s':response,
            'independent_pose_delta_rad_s':pose_response,'gyro_samples':int(early.sum()),
            'pose_samples':int(pearly.sum()),'expected_sign':int(expected),'pass':passed})
        window=(timestamps>=start-.1)&(timestamps<=start+.25)
        posewindow=(midtimes>=start-.1)&(midtimes<=start+.25)
        ax.plot(timestamps[window]-start,gyro[window,2]-gyro[pre,2].mean(),label='Gazebo FDM gyro (FLU)')
        ax.plot(midtimes[posewindow]-start,pose_rates[posewindow,2]-pose_rates[ppre,2].mean(),'o',ms=3,label='Independent pose derivative')
        ax.axvspan(0,.1,color='orange',alpha=.12,label='PWM pulse')
        ax.axvspan(.003,.023,color='green',alpha=.12,label='Initial response window')
        ax.set_ylabel('Yaw rate, rad/s');ax.set_title(f'PWM change {delta.tolist()}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Simulation seconds from actual actuator pulse')
    fig.tight_layout();fig.savefig(folder/'causal_yaw_response.png',dpi=160);fig.savefig(folder/'causal_yaw_response.pdf');plt.close(fig)
    result={'pass':len(results)==4 and all(r['pass'] for r in results),
        'scope':'Immediate yaw sign and rotor angular-momentum response; not a steady aerodynamic yaw coefficient fit.',
        'physical_motor_order':['FR','RL','FL','RR'],'body_reaction_sign':[-1,-1,1,1],
        'alignment':'Complete ordered actuator/FDM exchange, 1000Hz, contiguous unique frame numbers; feedback association has at most one 1ms integration-step convention uncertainty.',
        'important':'100ms average is confounded by native PID counteraction after about40ms. Earlier wall-clock-average diagnostic is inconclusive and preserved.',
        'rotor_inertia_kg_m2':.000167604,'command_speed_change_rad_s_for_20PWM':838*20/800,
        'ideal_angular_momentum_yaw_rate_jump_rad_s':4*.000167604*(838*20/800)/.02380042,
        'results':results,'source_hashes':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['actuators.jsonl','physics.jsonl','groundtruth.jsonl']}}
    (folder/'motor_yaw_causal_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())

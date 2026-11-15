"""Independent network delivery and three-way physical-state association.

An endpoint is identified by binary SERVO packets, never by a convenient clock
offset. Foreign orphan FDM remains counted and is excluded only by its address.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from packet_clock_ordered import ordered_clock


def inspect(run):
    summary=json.loads((run/'summary.json').read_text())
    with (run/'telemetry.csv').open(newline='') as file:
        rows=list(csv.DictReader(file))
    start=summary['measurement_boot_start_s']
    finish=start+float(rows[-1]['t'])
    capture=run/'network.pcap.gz'
    if not capture.exists(): capture=run/'network.pcap'
    clock=ordered_clock(capture,start,finish,summary.get('tcp_port',5760),summary.get('plugin_port',9002),include_fdm=True)
    physical=clock.pop('physical_samples')
    unique={float(t):p for t,p,v in physical}
    fdm_t=np.array(sorted(unique))
    fdm_ned=np.array([unique[t] for t in fdm_t])
    fdm_enu=fdm_ned[:,[1,0,2]]*np.array([1,1,-1])
    truth={}
    pose_source=run/'groundtruth.jsonl'
    observer_receipt=None
    if (run/'groundtruth_independent.jsonl').exists():
        observer_receipt=json.loads((run/'independent_pose_manifest.json').read_text())
        pose_source=run/'groundtruth_independent.jsonl'
        world_manifest=run/'gazebo_manifest.json'
        if not (observer_receipt.get('status')=='PASS' and observer_receipt.get('read_only') is True
                and observer_receipt['raw_pose_sha256']==hashlib.sha256(pose_source.read_bytes()).hexdigest()
                and observer_receipt['gazebo_manifest_sha256']==hashlib.sha256(world_manifest.read_bytes()).hexdigest()):
            raise ValueError('Dedicated independent pose observer receipt/seal is invalid')
    for line in pose_source.open():
        row=json.loads(line)
        if row.get('name')=='iris':truth[float(row['sim_time'])]=row['position']
    truth_t=np.array(sorted(truth));truth_xyz=np.array([truth[t] for t in truth_t])
    offset=clock['median_gazebo_minus_boot_s']
    sim_start,sim_end=start+offset,finish+offset
    selected=(truth_t>=sim_start)&(truth_t<=sim_end)
    sampled_t=truth_t[selected]
    interpolated_fdm=np.column_stack([np.interp(sampled_t,fdm_t,fdm_enu[:,i]) for i in range(3)])
    fdm_truth_error=np.linalg.norm(interpolated_fdm-truth_xyz[selected],axis=1)
    native_boot=np.array([float(r['boot_s']) for r in rows])
    native_position=np.array([[float(r[k]) for k in ('x_e','y_n','z_up')] for r in rows])
    origin=None
    for line in (run/'mavlink.jsonl').open():
        m=json.loads(line)['message']
        if m.get('mavpackettype')=='GPS_GLOBAL_ORIGIN':origin=m;break
    world=ET.parse(run/'gazebo_assets/world.sdf').getroot().find('world/spherical_coordinates')
    if abs(origin['latitude']*1e-7-float(world.findtext('latitude_deg')))>1e-7 or abs(origin['longitude']*1e-7-float(world.findtext('longitude_deg')))>1e-7:
        raise ValueError('Nonidentity geographic translation needs explicit conversion')
    origin_enu=np.array([0,0,origin['altitude']*.001-float(world.findtext('elevation'))])
    native_sim=native_boot+offset
    native_truth=np.column_stack([np.interp(native_sim,truth_t,truth_xyz[:,i]) for i in range(3)])-origin_enu
    native_error=np.linalg.norm(native_position-native_truth,axis=1)
    target=np.array([[float(r[k]) for k in ('target_e','target_n','target_up')] for r in rows])
    # The runner may record an endpoint20ms past the requested horizon.
    # Physical association uses it, but primary RMSE integrates only the horizon.
    weights=np.diff(np.r_[0,np.minimum([float(r['t']) for r in rows],summary['requested_duration_s'])])
    truth_rmse=float(np.sqrt(np.dot(weights,np.sum((native_truth-target)**2,axis=1))/weights.sum()))
    native_rmse=float(np.sqrt(np.dot(weights,np.sum((native_position-target)**2,axis=1))/weights.sum()))
    # Full measurement coverage remains mandatory. Nonmetric takeoff/landing
    # beyond the observer subscription window is a separate diagnostic.
    selected_fdm=fdm_t[(fdm_t>=sim_start)&(fdm_t<=sim_end)]
    active_range=len(selected_fdm)>100 and selected_fdm[0]>=truth_t[0]-.02 and selected_fdm[-1]<=truth_t[-1]+.02
    checks={
        'one_actual_SERVO_source_socket':len(clock['servo_sources'])==1,
        'no_active_FDM_simtime_backstep':clock['fdm_simtime_backsteps']==0,
        'measurement_active_FDM_within_recorded_world_time':bool(active_range),
        'active_clock_no_extreme_discontinuity':clock['outliers_over50ms']==0,
        'measurement_physically_covered':bool(len(sampled_t)>100 and fdm_t[0]<=sim_start and fdm_t[-1]>=sim_end-.02 and truth_t[0]<=sim_start and truth_t[-1]>=sim_end-.02),
        'FDM_matches_independent_world_pose':bool(len(fdm_truth_error)>100 and fdm_truth_error.max()<.01),
        'native_pose_matches_world_with_quantization_latency':bool(native_error.max()<.06),
    }
    return {'run':str(run),'status':'PASS' if all(checks.values()) else 'INVESTIGATE','checks':checks,
        'world_pose_source':pose_source.name,'dedicated_observer':observer_receipt,
        'nonmetric_takeoff_landing_pose_coverage':bool(fdm_t[0]>=truth_t[0]-.02 and fdm_t[-1]<=truth_t[-1]+.02),
        'clock':clock,'fdm_world_pose_samples':len(sampled_t),'fdm_world_error_m':{'rmse':float(np.sqrt(np.mean(fdm_truth_error**2))),'max':float(fdm_truth_error.max())},
        'native_world_error_m':{'rmse':float(np.sqrt(np.mean(native_error**2))),'max':float(native_error.max())},
        'native_rmse_m':native_rmse,'truth_rmse_m':truth_rmse,
        'measurement_sim_interval':[sim_start,sim_end],'captured_active_fdm_interval':[float(fdm_t[0]),float(fdm_t[-1])],
        'recorded_world_interval':[float(truth_t[0]),float(truth_t[-1])],
        'sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('summary.json',pose_source.name,'telemetry.csv')},
        'limitation':'10mm FDM/world and60mm native/world consistency guards detect gross wrong-world association; they are audit tolerances, not controller-performance criteria. Local clock association retains sensor/send latency.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--grid',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();runs=list(args.runs)
    if args.grid:runs+=sorted(p for p in args.grid.iterdir() if p.is_dir() and (p/'summary.json').exists())
    results=[]
    for run in runs:
        result=inspect(run);results.append(result)
        print(json.dumps({'run':run.name,'status':result['status'],'foreign_packets':sum(x['count'] for x in result['clock']['foreign_endpoint_fdm'].values()),
                          'fdm_truth_max':result['fdm_world_error_m']['max'],'native_truth_max':result['native_world_error_m']['max'],
                          'failed_checks':[name for name,value in result['checks'].items() if not value]}),flush=True)
    args.output.write_text(json.dumps({'status':'PASS' if all(r['status']=='PASS' for r in results) else 'INVESTIGATE','runs':results},indent=2)+'\n')

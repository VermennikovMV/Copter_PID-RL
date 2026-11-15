"""Read-only archived-trace diagnosis of the V4 seed84001 route peak.

No simulator, controller inference, training, or active source writes. New audit
receipt is the only output. Frozen source hashes are checked before/after.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/refinement/joint_v2'
GROUPS={'baseline':BASE/'nativeplant111_baseline/traces',
        'checkpoint_1048576':BASE/'nativeplant111/validation/checkpoint_1048576'}
OUT=Path(__file__).with_suffix('.json')
POINTS=np.array([[0,0,3],[2,0,3],[2,2,3.5],[0,2,3],[0,0,3.]],float)

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def col(rows,key):return np.array([float(r[key]) for r in rows])

def load(folder):
    with (folder/'trace.csv').open() as f:rows=list(csv.DictReader(f))
    summary=read(folder/'summary.json');geometry=read(folder/'geometry.json')
    t=col(rows,'time');p=np.array([[float(r[k]) for k in ('x','y','z')] for r in rows])
    v=np.array([[float(r[k]) for k in ('vx','vy','vz')] for r in rows])
    target=np.array([[float(r[k]) for k in ('target_x','target_y','target_z')] for r in rows])
    et=np.array([e['time_s'] for e in summary['mission_events']])
    active=np.minimum(np.searchsorted(et,t-1e-8,side='left'),4)
    assert np.allclose(target,POINTS[active])
    start=et[0];end=et[-1];transit=(t>start+1e-8)&(t<=end+1e-8)
    a=POINTS[np.maximum(active-1,0)];b=POINTS[active];delta=b-a
    denom=np.sum(delta*delta,axis=1);fraction=np.sum((p-a)*delta,axis=1)/np.where(denom>0,denom,1)
    nearest=a+np.clip(fraction,0,1)[:,None]*delta
    finite=np.linalg.norm(p-nearest,axis=1);line=np.linalg.norm(p-a-fraction[:,None]*delta,axis=1)
    delta_xy=delta[:,:2];denomxy=np.sum(delta_xy*delta_xy,axis=1)
    fracxy=np.sum((p[:,:2]-a[:,:2])*delta_xy,axis=1)/np.where(denomxy>0,denomxy,1)
    nearxy=a[:,:2]+np.clip(fracxy,0,1)[:,None]*delta_xy
    finite_xy=np.linalg.norm(p[:,:2]-nearxy,axis=1)
    force=np.array([[float(r[k]) for k in ('force_x','force_y','force_z')] for r in rows])
    action=np.array([[float(r['action'+str(i)]) for i in range(4)] for r in rows])
    with np.load(folder/'actor_io.npz',allow_pickle=False) as io:
        body_rates=io['body_rates'].copy();rotation=io['rotation'].copy();observation=io['observation'].copy();actor_time=io['time'].copy()
        actor_velocity=io['velocity'].copy();actor_position=io['position'].copy()
    assert len(rows)==2000 and np.allclose(t,np.arange(1,2001)*.02)
    assert observation.shape==(2000,74) and np.allclose(actor_time+.02,t)
    peak=int(np.where(transit)[0][np.argmax(finite[transit])])
    assert abs(finite[peak]-geometry['active_segment_peak_m'])<1e-10
    return locals()

def snapshot(d,i):
    i=int(i);j=int(d['active'][i]);a=POINTS[max(j-1,0)];b=POINTS[j];delta=b-a
    unit=delta/np.linalg.norm(delta) if j else np.zeros(3)
    uxy=delta[:2]/np.linalg.norm(delta[:2]) if j else np.zeros(2)
    accel=(d['v'][i]-d['actor_velocity'][i])/.02
    yaw=float(np.arctan2(d['rotation'][i,1,0],d['rotation'][i,0,0]));c,s=np.cos(yaw),np.sin(yaw)
    heading_to_world=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    observed_accel=heading_to_world@(d['observation'][i,25:28]*5.)
    xy_cross=d['p'][i,:2]-d['a'][i,:2]-d['fracxy'][i]*d['delta_xy'][i]
    return {'t_s':float(d['t'][i]),'active_goal':j,'position_enu_m':d['p'][i].tolist(),'velocity_enu_mps':d['v'][i].tolist(),
        'target_enu_m':b.tolist(),'finite_3D_m':float(d['finite'][i]),'finite_XY_m':float(d['finite_xy'][i]),
        'vertical_residual_to_finite_3D_projection_m':float(d['p'][i,2]-d['nearest'][i,2]),
        'vertical_goal_error_m':float(d['p'][i,2]-b[2]),'infinite_line_3D_m':float(d['line'][i]),
        'infinite_line_XY_m':float(np.linalg.norm(xy_cross)),
        'fraction_on_infinite_line':float(d['fraction'][i]),'incoming_along_overrun_3D_m':float(max(0,(d['p'][i]-b)@unit)),
        'incoming_along_overrun_XY_m':float(max(0,(d['p'][i,:2]-b[:2])@uxy)),
        'velocity_along_incoming_XY_mps':float(d['v'][i,:2]@uxy),'force_enu_N':d['force'][i].tolist(),
        'force_along_incoming_XY_N':float(d['force'][i,:2]@uxy),'yaw_rad':float(d['rows'][i]['yaw']),
        'yaw_reference_rad':d['summary']['yaw_reference'],
        'yaw_error_rad':float(d['rows'][i]['yaw_error']),'yaw_error_deg':float(np.rad2deg(float(d['rows'][i]['yaw_error']))),
        'tilt_deg':float(np.rad2deg(np.arccos(np.clip(float(d['rows'][i]['tilt_cos']),-1,1)))),
        'action_interval_begin_s':float(d['actor_time'][i]),'action':d['action'][i].tolist(),
        'actor_body_rates_interval_begin_flu_rad_s':d['body_rates'][i].tolist(),
        'actor_thrust_axis_interval_begin_enu':d['rotation'][i,:,2].tolist(),
        'actor_goal_error_heading_frame_m':(d['observation'][i,:3]*3.).tolist(),
        'actor_velocity_heading_frame_mps':(d['observation'][i,3:6]*3.).tolist(),
        'interval_mean_acceleration_enu_mps2':accel.tolist(),
        'interval_mean_acceleration_along_incoming_XY_mps2':float(accel[:2]@uxy),
        'known_external_force_acceleration_along_incoming_XY_mps2':float(d['force'][i,:2]@uxy/1.75),
        'deduced_rotor_acceleration_along_incoming_XY_mps2':float((accel[:2]-d['force'][i,:2]/1.75)@uxy),
        'actor_filtered_acceleration_enu_mps2':observed_accel.tolist(),
        'actor_filtered_acceleration_along_incoming_XY_mps2':float(observed_accel[:2]@uxy),
        'actor_integral_goal_error_enu_m_s':(heading_to_world@(d['observation'][i,22:25]*6.)).tolist(),
    }

def intervals(d,mask):
    starts=np.where(mask&np.r_[True,~mask[:-1]])[0];out=[]
    for start in starts:
        exits=np.where(~mask[start:])[0];last=start+int(exits[0])-1 if len(exits) else len(mask)-1
        out.append({'first_sample_s':float(d['t'][start]),'last_sample_s':float(d['t'][last]),
            'observed_span_s':float(d['t'][last]-d['t'][start])})
    return out

def wind_phase(d):
    changes=d['summary']['force_changes']
    ons=[e for e in changes if e['type']=='on_or_change']
    if not ons:return None
    on=ons[0]['time_s'];i=int(np.where(np.linalg.norm(d['force'],axis=1)>0)[0][0]);j=int(d['active'][i])
    onset=snapshot(d,i);leg=(d['active']==j);unit=(POINTS[j,:2]-POINTS[j-1,:2]);unit/=np.linalg.norm(unit)
    accel=(d['v']-d['actor_velocity'])/.02
    along_accel=accel[:,:2]@unit
    rotor_accel=(accel[:,:2]-d['force'][:,:2]/1.75)@unit
    relevant=leg&(d['t']>on)
    def sustained_negative(values):
        candidates=np.where(relevant&(values<0))[0]
        for candidate in candidates:
            if candidate+5<=len(values) and np.all(values[candidate:candidate+5]<0):return float(d['actor_time'][candidate])
        return None
    later=np.where(d['transit']&(d['t']>on))[0];peak=int(later[np.argmax(d['finite'][later])])
    goal3=np.where(d['active']==3)[0]
    return {'force_on_boundary_s':on,'first_affected_trace_sample':onset,
        'seconds_after_goal_activation':float(on-d['summary']['mission_events'][j-1]['time_s']) if j else None,
        'first_sustained_rotor_braking_interval_begin_s':sustained_negative(rotor_accel),
        'first_sustained_net_braking_interval_begin_s':sustained_negative(along_accel),
        'peak_finite_after_force':snapshot(d,peak),
        'goal3_max_horizontal_speed_mps':float(np.max(np.linalg.norm(d['v'][goal3,:2],axis=1))),
        'above_0p5m_finite_transit_intervals':intervals(d,d['transit']&(d['finite']>.5)),
        'active_leg_action_saturation_intervals_by_axis':{str(axis):intervals(d,leg&(np.abs(d['action'][:,axis])>=.999)) for axis in range(4)},
        'known_external_impulse_per_mass_on_incoming_axis_to_peak_mps':float(onset['force_along_incoming_XY_N']/1.75*(d['t'][peak]-on)),
        'impulse_interpretation':'Exact constant-force contribution integrated in time, not a counterfactual velocity or flight prediction; actor/attitude adapt concurrently.'}

def describe(d):
    peak=d['peak'];tpeak=d['t'][peak];events=d['summary']['mission_events'];j=int(d['active'][peak])
    previous=events[j-1] if j else None;following=events[j] if j<len(events) else None
    changes=d['summary']['force_changes'];earlier=[e for e in changes if e['time_s']<=tpeak]
    samples=[]
    for stamp in sorted(set([tpeak,*np.arange(max(.02,tpeak-1.2),tpeak+.61,.1),*(e['time_s'] for e in changes if abs(e['time_s']-tpeak)<2),*(e['time_s']+.02 for e in changes if abs(e['time_s']-tpeak)<2)])):
        samples.append(snapshot(d,np.argmin(abs(d['t']-stamp))))
    # Exact archived in-band runs show rejected arrival entries and dwell resets.
    leg=np.where(d['active']==j)[0];inside=np.linalg.norm(d['p'][leg]-POINTS[j],axis=1)<=.25
    starts=np.where(inside&np.r_[True,~inside[:-1]])[0];intervals=[]
    for k in starts:
        exits=np.where(~inside[k:])[0];last=k+int(exits[0])-1 if len(exits) else len(leg)-1
        intervals.append({'start_s':float(d['t'][leg[k]]),'last_inside_s':float(d['t'][leg[last]]),
            'continuous_observed_span_s':float(d['t'][leg[last]]-d['t'][leg[k]]),
            'exit_next_sample_s':float(d['t'][leg[last+1]]) if last+1<len(leg) else None,
            'entry_speed_mps':float(np.linalg.norm(d['v'][leg[k]]))})
    return {'run':rel(d['folder']),'scenario':d['summary']['scenario'],'seed':d['summary']['eval_seed'],
        'initial_state':d['summary']['initial_state'],'scenario_spec':d['summary']['scenario_spec'],
        'geometry':d['geometry'],'peak':snapshot(d,peak),'peak_previous_goal_confirmation':previous,
        'peak_next_goal_confirmation':following,'last_force_change_before_peak':earlier[-1] if earlier else None,
        'peak_active_goal_band_intervals':intervals,'peak_neighborhood':samples,'first_wind_phase':wind_phase(d),
        'source_hashes':{name:sha(d['folder']/name) for name in ('trace.csv','summary.json','geometry.json','actor_io.npz')}}

def main():
    frozen=[ROOT/'quadrl_joint'/n for n in ('__init__.py','native_rate_model_v3.py','env_v3.py','native_rate_fit_v3.json')]+[ROOT/'scripts/train_joint_nativeplant_v4.py',ROOT/'audit/joint_native_rate_physics_replay_v3.json']
    before={rel(p):sha(p) for p in frozen};all_data={};reports=[]
    for group,parent in GROUPS.items():
        for seed in range(84001,84005):
            for scenario in ('nominal','force_strong','gust'):
                d=load(parent/f'route_{scenario}_{seed}');all_data[group,scenario,seed]=d
                r=describe(d);r['group']=group;reports.append(r)
                print(group,scenario,seed,'peak',r['peak']['t_s'],r['peak']['active_goal'],r['peak']['finite_3D_m'],r['peak']['position_enu_m'],flush=True)
    after={rel(p):sha(p) for p in frozen};assert before==after
    comparisons=[]
    for group in GROUPS:
        strong=all_data[group,'force_strong',84001];gust=all_data[group,'gust',84001];nominal=all_data[group,'nominal',84001]
        prefix=min(e['off_s'] for e in strong['summary']['scenario_spec']['force_events'][:1]+gust['summary']['scenario_spec']['force_events'][:1])
        same=strong['t']<=prefix+1e-8
        changed=np.where(np.max(abs(strong['p']-nominal['p']),axis=1)>1e-9)[0]
        changed_action=np.where(np.max(abs(strong['action']-nominal['action']),axis=1)>1e-9)[0]
        comparisons.append({'group':group,'strong_gust_common_first_force_prefix_end_s':prefix,
            'strong_gust_position_max_difference_before_prefix_end_m':float(np.max(abs(strong['p'][same]-gust['p'][same]))),
            'strong_gust_action_max_difference_before_prefix_end':float(np.max(abs(strong['action'][same]-gust['action'][same]))),
            'wind_vs_nominal_first_position_difference_over_1e9_m_s':float(strong['t'][changed[0]]) if len(changed) else None,
            'wind_vs_nominal_position_max_difference_before_force_on_m':float(np.max(abs(strong['p'][strong['t']<=4.88]-nominal['p'][strong['t']<=4.88]))),
            'nominal_peak_m':float(nominal['finite'][nominal['peak']]),'wind_peak_m':float(strong['finite'][strong['peak']]),
            'nominal_and_wind_initial_state_identical':strong['summary']['initial_state']==nominal['summary']['initial_state'],
            'wind_vs_nominal_first_actor_action_difference_interval_begin_s':float(strong['actor_time'][changed_action[0]]) if len(changed_action) else None,
        })
    focus=next(r for r in reports if r['group']=='checkpoint_1048576' and r['scenario']=='force_strong' and r['seed']==84001)
    weather=[]
    for r in reports:
        if r['group']!='checkpoint_1048576' or r['scenario']!='force_strong':continue
        phase=r['first_wind_phase'];on=phase['first_affected_trace_sample']
        weather.append({'seed':r['seed'],'initial_yaw_deg':float(np.rad2deg(r['initial_state']['rpy'][2])),
            'force_on_s':phase['force_on_boundary_s'],'force_along_leg_XY_N':on['force_along_incoming_XY_N'],
            'speed_along_leg_at_first_force_sample_mps':on['velocity_along_incoming_XY_mps'],
            'net_accel_along_leg_at_first_force_sample_mps2':on['interval_mean_acceleration_along_incoming_XY_mps2'],
            'peak_after_force_m':phase['peak_finite_after_force']['finite_3D_m']})
    output={'schema':'joint-nativeplant-seed84001-peak-diagnosis-v4.0','scope':'Read-only completed archived traces; no simulator, policy inference, training or active source changes',
        'reports':reports,'same_seed_comparisons':comparisons,'frozen_hashes_verified_unchanged':after,
        'findings':{'peak':focus['peak'],'wind_phase':focus['first_wind_phase'],
            'dwell_intervals':focus['peak_active_goal_band_intervals'],'weather_alignment_across_four_seeds':weather,
            'baseline_vs_checkpoint_braking_phase':'Both begin sustained rotor braking5.28s and sustained net braking5.34s; max horizontal speed falls only3.5493→3.5329m/s; finite peak falls.78663→.76961m. The critical braking phase is essentially unchanged after1M.',
            'why_corner_guard_can_pass':'The .7696m deviation is before goal3 confirmation7.10s. Its post-confirmation outgoing corner window peaks only.07802m; the global finite-segment peak correctly retains the pre-confirmation incoming overshoot.',
            'reviewable_next_test':'A future separately authorized intervention should isolate wind assisting versus opposing the incoming leg at fixed initial state/yaw, and assess sphere-entry speed/braking timing. No such rollout is performed here.'},
        'interpretation':{
            'evidence':'Seed84001 receives a westward force .46s after activating westbound descending leg3. The same actor/initial-state nominal trajectory is exactly identical until force onset and remains below .25m peak. Wind strong/gust are identical until gust-off7.76, so the duplicate peak reflects a common first pulse. Peak is axial endpoint overshoot before successful arrival confirmation, with essentially zero vertical goal error.',
            'causal_hypothesis':'Along-track force accelerates an already fast westbound approach; the policy brakes but does not remove the extra incoming momentum before crossing the goal sphere. Dwell resets delay confirmation until recovery. This is supported by recorded force/velocity/action/acceleration timing, not by a new flight counterfactual.',
            'not_established':'Four seeds jointly vary weather direction/timing and initial yaw/position. Existing traces do not isolate their interaction independently; no claim that yaw is irrelevant. No matched comparison with Gazebo seed83301 is made, and the different weather case is not evidence of a rate-fit failure.',
            'next_iteration_basis':'Target diagnosis/evaluation of wind-assisted incoming-leg braking and too-fast sphere entry. Native shaper, collective mapping, reward and actor sources remain frozen during this audit.'},
        'analysis_script_sha256':sha(Path(__file__))}
    OUT.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8');print('Wrote',rel(OUT))

if __name__=='__main__':main()

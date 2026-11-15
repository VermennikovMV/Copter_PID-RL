"""Arithmetic interpretation of the completed10-new/4-reused diagnostic."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime,timezone
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadrl_route_transfer_v22.initial import rotation
RUN=ROOT/'artifacts/refinement/joint_v2/route_transfer_initial_v22_root01'
OUT=ROOT/'audit/route_transfer_initial_v22_interpretation_01.json'
P=np.array([[0,0,3],[2,0,3],[2,2,3.5],[0,2,3],[0,0,3]],float)
INPUTS={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p);INPUTS[p.relative_to(ROOT).as_posix()]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def local(p):return ROOT/str(p).split('/v2/',1)[-1]
def weights(t,start,end):return np.maximum(0.,np.minimum(t,end)-np.maximum(np.r_[0.,t[:-1]],start))


def analyze(record,intervention,is_new):
    trace=local(record['trace_path']);iofile=local(record['actor_io_path']);folder=trace.parent
    assert sha(bind(trace))==record['hashes']['trace.csv'] and sha(bind(iofile))==record['hashes']['actor_io.npz']
    with trace.open(newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==2000
    t=np.array([float(r['time']) for r in rows]);p=np.array([[float(r[k]) for k in ('x','y','z')] for r in rows]);v=np.array([[float(r[k]) for k in ('vx','vy','vz')] for r in rows]);force=np.array([[float(r[k]) for k in ('force_x','force_y','force_z')] for r in rows])
    events=record['mission_events'];et=np.array([e['time_s'] for e in events]);assert len(et)==5
    complete=record['metrics']['route_completion_time_s'];w=weights(t,et[0],complete)
    active=np.minimum(np.searchsorted(et,t-1e-8,side='left'),4);res=np.zeros_like(p);legs=[]
    for leg in range(1,5):
        mask=active==leg;d=P[leg]-P[leg-1];q=((p-P[leg-1])@d)/(d@d)
        error=p-P[leg-1]-np.clip(q,0,1)[:,None]*d;res[mask]=error[mask];lw=w*mask
        sse=np.sum(lw[:,None]*error**2,axis=0);idx=np.flatnonzero(lw>0)
        peak=idx[np.argmax(np.sum(error[idx]**2,axis=1))]
        before=q<0;after=q>1
        minimum_x=idx[np.argmin(p[idx,0])]
        legs.append(dict(leg=leg,start_s=float(et[leg-1]),end_s=float(et[leg]),duration_s=float(lw.sum()),SSE_XYZ_m2s=sse.tolist(),
            RMSE_XYZ_m=np.sqrt(sse/lw.sum()).tolist(),finite_RMSE_m=float(np.sqrt(sse.sum()/lw.sum())),
            finite_peak_m=float(np.linalg.norm(error[peak])),peak_time_s=float(t[peak]),peak_position_m=p[peak].tolist(),peak_velocity_mps=v[peak].tolist(),
            X_SSE_projection_before_inside_after_m2s=[float(np.sum(lw*category*error[:,0]**2)) for category in (before,(~before)&(~after),after)],
            incoming_longitudinal_after_endpoint_peak_m=float(np.max(np.maximum((p[idx]-P[leg])@(d/np.linalg.norm(d)),0.))),
            minimum_X_m=float(p[minimum_x,0]),minimum_X_time_s=float(t[minimum_x]),forced_seconds=float(np.sum(lw*(np.linalg.norm(force,axis=1)>1e-8)))))
    sse=np.sum(w[:,None]*res**2,axis=0);rmse=float(np.sqrt(sse.sum()/w.sum()))
    assert abs(rmse-record['geometry']['segment_rmse_m'])<1e-12
    peak=float(np.max(np.linalg.norm(res[w>0],axis=1)))
    assert abs(peak-record['geometry']['active_segment_peak_m'])<1e-12
    with np.load(iofile,allow_pickle=False) as io:
        first={name:io[name][0].tolist() for name in ('position','velocity','rotation','body_rates','previous_action','effective_observation')}
        assert np.array_equal(io['observation'],io['effective_observation'])
        assert np.array_equal(io['previous_action'][0],np.zeros(4))
        if is_new:
            proof=record['initial_intervention_proof'];initial=proof['injected_initial']
            errors={name:float(np.max(abs(io[name][0]-np.asarray(value)))) for name,value in (('position',initial['position']),('velocity',initial['velocity']),('rotation',rotation(initial['rpy'])),('body_rates',initial['body_rates']))}
            assert max(errors.values())<1e-11
            assert proof['hidden_physics_steps']==0 and proof['steps']==0 and proof['base74_history_rows']==1
            assert proof['base74_time_s']==0. and proof['raw_env_80_observations_after_reset']==1
            assert proof['sensor_rng_restored_before_one_effective_observation'] and proof['plant_rate_queues_zero_before_effective_observation']
            original=record['frozen_original_summary'];assert sha(bind(local(original['path'])))==original['sha256']
            assert local(record['intervention_summary_path']).exists();bind(local(record['intervention_summary_path']))
        else:errors=None
    onsets=[]
    for event in record['force_changes']:
        i=int(np.argmin(abs(t-(event['time_s']+.02))))
        if event['type']!='off':
            j=int(active[i]);onsets.append(dict(applied_control_interval_start_s=event['time_s'],PRE_leg=j,seconds_after_leg_transition=float(event['time_s']-et[max(j-1,0)]),measured_row_end_s=float(t[i]),position_m=p[i].tolist(),velocity_mps=v[i].tolist(),force_N=event['force_enu_N']))
    return dict(intervention=intervention,mode=record['conditioning_mode'],scenario=record['scenario'],new_actual_episode=is_new,steps=2000,
        completion_s=float(complete),finite_RMSE_m=rmse,finite_peak_m=peak,absolute_peak_guard_pass=peak<=.5,mean_corner_peak_m=record['geometry']['mean_corner_post_peak_m'],
        axis_SSE_m2s=sse.tolist(),axis_RMSE_m=np.sqrt(sse/w.sum()).tolist(),legs=legs,force_onsets=onsets,force_changes=record['force_changes'],first_IO_state=first,first_state_max_errors=errors,
        trace_sha256=record['hashes']['trace.csv'],IO_sha256=record['hashes']['actor_io.npz'])


def main():
    assert not OUT.exists()
    bind(Path(__file__));bind(ROOT/'quadrl_route_transfer_v22/initial.py')
    complete=read(RUN/'complete.json');plan=read(RUN/'prospective_plan.json')
    assert complete['new_episodes']==10 and complete['coverage_episodes']==14
    assert complete['actual_counts']==dict(control_steps=20000,physical_substeps=160000,NN_forwards=20000)
    assert complete['optimizer']==complete['backward']==0 and not complete['policy_promotion']
    rows=[]
    for original in complete['reused_cases']:
        record=read(ROOT/original['run']/'summary.json');assert sha(ROOT/original['run']/'summary.json')==original['summary_sha256']
        rows.append(analyze(record,'native_original',False))
    for record in complete['cases']:rows.append(analyze(record,record['intervention'],True))
    common=plan['source_measured_Gazebo_initial']['state'];pairs=[]
    for intervention,scenarios in [('native_original',('nominal','force_strong')),('yaw_only',('nominal','force_strong')),('full_initial',('nominal','force_strong')),('full_initial_physical_pulse',('force_strong',))]:
        for scenario in scenarios:
            raw=next(x for x in rows if x['intervention']==intervention and x['scenario']==scenario and x['mode']=='raw')
            zero=next(x for x in rows if x['intervention']==intervention and x['scenario']==scenario and x['mode']=='zero_route_integral')
            assert all(np.array_equal(raw['first_IO_state'][key],zero['first_IO_state'][key]) for key in raw['first_IO_state'])
            if intervention.startswith('full_initial'):
                for row in (raw,zero):
                    assert np.max(abs(np.asarray(row['first_IO_state']['position'])-common['position']))<1e-12
                    assert np.max(abs(np.asarray(row['first_IO_state']['rotation'])-common['rotation']))<1e-11
            pairs.append(dict(intervention=intervention,scenario=scenario,raw_finite_RMSE_m=raw['finite_RMSE_m'],zero_finite_RMSE_m=zero['finite_RMSE_m'],zero_vs_raw_change_percent=100*(zero['finite_RMSE_m']/raw['finite_RMSE_m']-1),
                raw_peak_m=raw['finite_peak_m'],zero_peak_m=zero['finite_peak_m'],zero_absolute_peak_guard_pass=zero['absolute_peak_guard_pass'],
                raw_corner_mean_m=raw['mean_corner_peak_m'],zero_corner_mean_m=zero['mean_corner_peak_m'],raw_completion_s=raw['completion_s'],zero_completion_s=zero['completion_s'],
                zero_minus_raw_axis_SSE_m2s=(np.asarray(zero['axis_SSE_m2s'])-raw['axis_SSE_m2s']).tolist(),
                leg_differences=[dict(leg=a['leg'],zero_minus_raw_SSE_XYZ_m2s=(np.asarray(b['SSE_XYZ_m2s'])-a['SSE_XYZ_m2s']).tolist(),raw_after_endpoint_X_SSE_m2s=a['X_SSE_projection_before_inside_after_m2s'][2],zero_after_endpoint_X_SSE_m2s=b['X_SSE_projection_before_inside_after_m2s'][2]) for a,b in zip(raw['legs'],zero['legs'])]))
    prior=read(ROOT/'audit/route_memory_transfer_gap_v22_01.json')
    gz=[{key:x[key] for key in ('name','finite_RMSE_m','axis_RMSE_m','complete_s')} for x in prior['cases'] if x['backend']=='Gazebo' and x['mode']!='PID']
    failure_rows=[dict(intervention=x['intervention'],mode=x['mode'],scenario=x['scenario'],peak_m=x['finite_peak_m']) for x in rows if not x['absolute_peak_guard_pass']]
    assert len(failure_rows)==2
    proposal=dict(status='PROPOSED_NOT_AUTHORIZED_OR_EXECUTED',sole_changed_factor='Route initial-heading training distribution, not reward weights or plant refit.',
        reason='Yaw-only intervention flips zero-versus-raw tail ranking while preserving p/v/roll/pitch/bodyrates; no dynamics or force timing changed. Full state and pulse retain failure; peak geometry newlyfails despite lower corner mean.',
        control='Frozen original training route initial yaw sampler.',
        treatment='Equal-priority route-heading strata: retain original sampler for half route episodes; set only initial yaw/yaw_reference to the declared measuredGazebo heading for other half. Initial p/v/r/p, task sampler, wind curriculum, source weights/prior, architecture, PPO settings, reward and interaction budget identical; hover unchanged.',
        allocation='50/50 is prospective equal-stratum coverage allocation, not chosen by validation search. Freeze RNG/hash-based assignment and log counts before run; do not silently change sampler RNG draws.',
        common_source='ROOT selects and freezes ONE common prior/initializer only after current v21p1 outcome; no assumption that pendingcandidate passed.',
        main_budget='Same bounded131072critic+262144PPO per arm/three candidates, not automaticextension. Allmetrics mapped and officialpreflight/rootreviewbeforelearning.',
        selection='Keep original24native gates and add all5 matched-start diagnosticconditions per candidate with fixed prospective guards. Entire active-leg finiteerror and absolutepeak remain required; low post-transition cornermean cannot hide pre-transition overshoot. No finalreserved seeds.',
        attribution='Tests route-heading coverage for this integrated controller. Does not separately isolate network yawdependence versus native roll/pitch model asymmetry; no universal heading-invariance or Gazeboacceptance claim.')
    assert all(sha(ROOT/name)==value for name,value in INPUTS.items())
    report=dict(schema='joint-transfer-initial-v22-interpretation-v1.0',status='PASS_ARITHMETIC_INTERPRETATION_WITH_VALID_FAILURES',created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=INPUTS,all_inputs_unchanged=True,interpreted_counts=dict(new_episodes=10,reused_episodes=4,trace_rows=28000,archived_IO_files=14,new_NN_forwards=0,new_physics=0,new_backward=0,new_optimizer=0),
        rows=rows,pair_table=pairs,prior_Gazebo_anchors=gz,new_valid_absolute_peak_failures=failure_rows,
        conclusion='Yaw-only is sufficient to reproduce the adverse tail ranking sign in native. Remaining full-state/pulse changes affect magnitude but do not restore the old favorable ranking. Old randomized84001native was not an adequate sole transfer test.',
        next_PPO_proposal=proposal,limitations=['Yaw intervention changes body-relative route/force and integrated network+anisotropic plant response; cannot assign cause solely to policy or plant.',
        'Full initial matches measuredp/v/R/bodyrates, not hiddenArduPilotcontrollerstates; pulsequantizes to4.90/19.86 in originalnative50Hz lookup.',
        'One fixed developmentseed, dependent raw/zero cases and4reusedbaselines; not independentalgorithm ranking or finalconfirmation.','A conservativefull-initialnative peak failure is valid even if post-transitioncorner means improve.'])
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],receipt_sha256=sha(OUT),pair_table=pairs,valid_peak_failures=failure_rows)))


if __name__=='__main__':main()

"""DIAGNOSTIC: autonomous recorded-command replay through real PyBullet bodies.

Creates only the new namespace candidate fit and this audit's JSON receipt.
No Gazebo, training, policy update, or frozen-source mutation is performed.
"""
from __future__ import annotations
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit.joint_nonlinear_rate_diagnosis_v2 import read, lines, sha, rel, load_run

DIAG=ROOT/'audit/joint_nonlinear_rate_diagnosis_v2.json'
FIT=ROOT/'quadrl_joint/native_rate_fit_v3.json'
OUT=Path(__file__).with_suffix('.json')


def write_candidate_fit():
    receipt=read(DIAG); model=receipt['effective_rate_model_comparison']
    rp=model['fitted_parameters']['native_shaper_plus_refit_remainder'][:2]
    old=read(ROOT/'quadrl/plant_config.json')
    fit={'schema':'native-rate-fit-v3.0','status':'PROSPECTIVE_DEVELOPMENT_PHYSICS_ONLY',
        'model_version':'native-rate-rp-refit-v3.0','diagnosis_receipt':rel(DIAG),'diagnosis_receipt_sha256':sha(DIAG),
        'diagnosis_script_sha256':receipt['analysis_script_sha256'],
        'native_shaper':{'dt_s':.0025,'RP_input_tc_s':[.025,.025],
            'RP_accel_limit_rad_s2':[float(np.deg2rad(1100.))]*2,
            'source_evidence':receipt['source_evidence']},
        'RP_remainder':{'gain':[r['gain'] for r in rp],'tau_s':[r['tau_s'] for r in rp],'delay_s':[r['delay_s'] for r in rp],
            'integration':'physical feedback coefficient (1-exp(-dt/tau))/dt with fixed nominal inertia; no RP integral'},
        'yaw_retained':{'gain':old['rate_gain'][2],'delay_s':old['rate_delay_s'][2],
            'kp':old['rate_kp'][2],'ki':old['rate_ki'][2],'integral_limit':old['rate_integral_limit'],
            'queue_rule':'round(delay_s/dt), matching frozen QuadEnv at the selected physical dt'},
        'collective_retained_tau_s':old['collective_tau_s'],'frozen_config_sha256':sha(ROOT/'quadrl/plant_config.json'),
        'parameter_bounds':{'gain':[.85,1.15],'tau_s':[.001,.15],'delay_s':[0.,.10]},
        'fit_scope':model['objective'],'training_runs':model['training_runs'],
        'training_and_held_out_cells':model['comparisons'],
        'source_and_input_hashes':{r['run']:r['hashes'] for r in receipt['runs']},
        'forward_model_inputs':model['forward_model_inputs'],
        'validation_limits':receipt['findings']['validation_limits'],
        'physical_embedding_audit':rel(OUT),
        'claims':'Nominal physical-response candidate. No randomized-fault validation, policy improvement, Gazebo acceptance, or production readiness claimed.'}
    if FIT.exists():
        if read(FIT)!=fit:raise RuntimeError('Candidate fit already exists with different contents; inspect before revising it')
    else:FIT.write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return fit


def set_initial(env, row):
    import pybullet as pb
    from scipy.spatial.transform import Rotation
    # Recorded attitude is NED/FRD; environment is ENU/FLU.
    ned=Rotation.from_euler('xyz',[float(row[k]) for k in ('roll_ned','pitch_ned','yaw_ned')]).as_matrix()
    rot=np.array([[0,1,0],[1,0,0],[0,0,-1.]])@ned@np.diag([1.,-1.,-1.])
    quat=Rotation.from_matrix(rot).as_quat()
    pos=np.array([float(row[k]) for k in ('x_e','y_n','z_up')])
    vel=np.array([float(row[k]) for k in ('vx_e','vy_n','vz_up')])
    rates=np.array([float(row[k]) for k in ('p_frd','q_frd','r_frd')])*[1,-1,-1]
    pb.resetBasePositionAndOrientation(env.body,pos,quat,physicsClientId=env.cid)
    pb.resetBaseVelocity(env.body,vel,rot@rates,physicsClientId=env.cid)
    env.yaw_reference=float(np.arctan2(rot[1,0],rot[0,0]));env.encoder.reset(env.yaw_reference)
    return {'position':pos.tolist(),'velocity':vel.tolist(),'rotation_enu_flu':rot.tolist(),'body_rates_flu':rates.tolist(),
        'initial_collective':'Both models start with nominal mass*g and equal rotor forces; unknown archived PWM state excluded by 1s warmup.'}


def stationary(env):
    import pybullet as pb
    env.reset(seed=0)
    pb.resetBasePositionAndOrientation(env.body,[0,0,3],[0,0,0,1],physicsClientId=env.cid)
    pb.resetBaseVelocity(env.body,[0,0,0],[0,0,0],physicsClientId=env.cid)
    env.yaw_reference=0.;env.encoder.reset(0.)


def unit_checks(fit):
    import ast
    import copy
    import inspect
    import textwrap
    import pybullet as pb
    from quadrl.env import QuadEnv
    from quadrl.policy import MASS,INERTIA,MAX_FORCE,NumpyPolicy
    from quadrl.protocol import Mission
    from quadrl_joint.env_v3 import QuadEnvJointV3
    from quadrl_joint.native_rate_model_v3 import NativeRateModelV3,FractionalDelay
    checks={};m=NativeRateModelV3()
    # Compare the executable AST after the integration loop. Ignore only the
    # added diagnostic version metadata; all judge/reward/return logic matches.
    def judge_tail(cls):
        function=ast.parse(textwrap.dedent(inspect.getsource(cls.step))).body[0]
        tail=[];started=False
        for node in function.body:
            if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Attribute) and node.target.attr=='steps':started=True
            if started:tail.append(copy.deepcopy(node))
        for node in tail:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict):
                pairs=[(k,v) for k,v in zip(node.value.keys,node.value.values) if not isinstance(k,ast.Constant) or k.value!='joint_rate_model_version']
                node.value.keys=[k for k,v in pairs];node.value.values=[v for k,v in pairs]
        return ast.dump(ast.Module(body=tail,type_ignores=[]),include_attributes=False)
    assert judge_tail(QuadEnv)==judge_tail(QuadEnvJointV3)
    assert QuadEnvJointV3.observation is QuadEnv.observation
    checks['unchanged_judge_reward_termination_observation_AST']={'status':'PASS','method':'Exact AST equality after steps increment, ignoring only new version metadata; observation method inherited by identity.'}
    rng=np.random.default_rng(731);old_a=m.target_accel.copy();old_v=m.target_rate.copy()
    peak_a=np.zeros(2);peak_j=np.zeros(2)
    for k in range(2000):
        command=rng.uniform([-2,-2,-1],[2,2,1]) if k%31==0 else command
        m.step(command);jerk=abs(m.target_accel-old_a)/m.dt
        assert np.all(abs(m.target_accel)<=m.accel_limit+1e-12)
        assert np.all(jerk<=m.jerk_limit+1e-9)
        assert np.allclose((m.target_rate-old_v)/m.dt,m.target_accel,rtol=0,atol=1e-12)
        peak_a=np.maximum(peak_a,abs(m.target_accel));peak_j=np.maximum(peak_j,jerk)
        old_a=m.target_accel.copy();old_v=m.target_rate.copy()
    checks['native_acceleration_jerk_caps']={'status':'PASS','peak_a':peak_a.tolist(),'peak_j':peak_j.tolist()}
    d=FractionalDelay(.00625,.0025);impulse=[d.step(1. if k==0 else 0.) for k in range(8)]
    assert np.allclose(impulse,[0,0,.5,.5,0,0,0,0])
    a=NativeRateModelV3();b=NativeRateModelV3();prefix=[]
    for k in range(100):
        ca=np.array([.2,-.3,.1]);cb=ca.copy() if k<60 else -ca
        oa=a.step(ca);ob=b.step(cb)
        if k<60:assert np.array_equal(oa,ob)
    checks['constant_fractional_delay_causality']={'status':'PASS','impulse':impulse}
    # Compare whole physical yaw trajectories to original equations at the same
    # .0025 integration, preventing an RP refit from silently changing yaw.
    old=QuadEnv(mission='hover');new=QuadEnvJointV3(mission='hover')
    try:
        old.dt=.0025;old.decimation=8;old.control_dt=.02
        pb.setTimeStep(old.dt,physicsClientId=old.cid)
        stationary(old);stationary(new)
        errors=[]
        for k in range(150):
            command=[0,0,0,.2*np.sin(k*.07)]
            old.step(command);new.step(command)
            errors.append(float(np.max(abs(old.state()[3]-new.state()[3]))))
            assert np.allclose(old.motor_forces,new.motor_forces,rtol=0,atol=1e-11)
        assert max(errors)<1e-11
        checks['original_yaw_physics_at_matching_dt']={'status':'PASS','peak_rate_difference_rad_s':max(errors),'delay_queue_steps':new.rate_model.yaw_queue_length}
        assert new.observation().shape==(74,)
        assert new.mass==MASS and np.array_equal(new.inertia,INERTIA)
        checks['actor74_mass_inertia_and_forces']={'status':'PASS','observation_shape':[74],
            'mass_kg':new.mass,'inertia':new.inertia.tolist(),'rotor_limit_N':MAX_FORCE}
    finally:old.close();new.close()
    # Stationary hover/route keep the complete original horizon and judge.
    for mission,expected in (('hover',1500),('route',2000)):
        env=QuadEnvJointV3(mission=mission)
        try:
            stationary(env);assert isinstance(env.mission,Mission);assert env.max_steps==expected
            trunc=[];events=[]
            for k in range(expected):
                obs,reward,terminated,truncated,info=env.step(np.zeros(4))
                assert obs.shape==(74,) and np.isfinite(obs).all() and np.isfinite(reward)
                assert not terminated and np.min(info['actual_motor_forces'])>=0 and np.max(info['actual_motor_forces'])<=MAX_FORCE
                if truncated:trunc.append(k+1)
                if info['mission_event']:events.append(info['mission_event'])
            assert trunc==[expected]
            checks[mission+'_full_original_horizon']={'status':'PASS','steps':expected,'duration_s':info['time'],'events':events}
        finally:env.close()
    policy_folder=next(r for r in read(DIAG)['runs'] if isinstance(r['kind'],dict) and r['kind']['method']=='body_collective')
    policy=NumpyPolicy(ROOT/policy_folder['run']/'source_snapshot/policy.npz')
    action=policy.predict(np.zeros(74,dtype=np.float32))
    assert action.shape==(4,) and np.isfinite(action).all()
    checks['archived_actor_loads_unchanged']={'status':'PASS','input_size':74,'output_size':4}
    return checks


def replay(folder, record=None):
    import pybullet as pb
    from quadrl.env import QuadEnv
    from quadrl.policy import MASS,INERTIA,MAX_FORCE
    from quadrl_joint.env_v3 import QuadEnvJointV3
    data=load_run(folder,record);commands=list(lines(folder/'commands.jsonl'))
    cb=np.array([r['boot_s'] for r in commands]);actions=np.array([r['action'] for r in commands])
    scenario=record['job']['scenario'] if record else 'nominal'
    mission=record['job']['mission'] if record else 'hover'
    seed=record['job']['environment_seed'] if record else 0
    start=cb[0];end=data['offset']+data['end'];steps=int(np.floor((end-start)/.02+1e-7))
    initial_row=min(data['telemetry'],key=lambda row:abs(float(row['boot_s'])-start))
    trajectories={};initials={}
    for label,cls in (('frozen_physical',QuadEnv),('native_v3_physical',QuadEnvJointV3)):
        env=cls(scenario=scenario,mission=mission)
        try:
            env.reset(seed=seed);initials[label]=set_initial(env,initial_row)
            assert env.mass==MASS and np.array_equal(env.inertia,INERTIA)
            tt=[];rates=[];positions=[];first_contact=None;first_flag=None;rotor_clip_steps=0
            for k in range(steps):
                now=start+k*.02
                index=int(np.clip(np.searchsorted(cb,now+1e-8,side='right')-1,0,len(cb)-1))
                _,_,terminated,_,info=env.step(actions[index])
                tt.append(start+(k+1)*.02);rates.append(env.state()[3]);positions.append(info['position'])
                if terminated and first_flag is None:first_flag=info['time']
                if pb.getContactPoints(bodyA=env.body,physicsClientId=env.cid) and first_contact is None:first_contact=start+(k+1)*.02
                assert np.all(np.isfinite(rates[-1]))
                forces=np.array(info['actual_motor_forces']);assert np.min(forces)>=0 and np.max(forces)<=MAX_FORCE+1e-10
                rotor_clip_steps+=int(np.any(forces<1e-10)|np.any(forces>MAX_FORCE-1e-10))
            trajectories[label]={'t':np.array(tt),'rates':np.array(rates),'first_contact_boot_s':first_contact,
                'first_termination_flag_elapsed_s':first_flag,'rotor_clip_control_steps':rotor_clip_steps,'last_position':positions[-1]}
        finally:env.close()
    assert initials['frozen_physical']==initials['native_v3_physical']
    common_end=min([end,*[r['first_contact_boot_s'] for r in trajectories.values() if r['first_contact_boot_s'] is not None]])
    result={'run':rel(folder),'split':'fit' if rel(folder) in read(FIT)['training_runs'] else 'held_out',
        'initial_state':initials['native_v3_physical'],'warmup_excluded_s':1.,'common_score_end_boot_s':common_end,
        'requested_end_boot_s':end,'full_scored_span_without_ground_contact':bool(common_end>=end-1e-6),
        'recorded_thrust_semantics':record['job']['method'] if record else 'legacy_system_id_small_signal',
        'input_rule':'Exact recorded normalized actions held causally at 50Hz; no recorded target, attitude, gyro or pose is fed after initialization.',
        'models':{},'source_hashes':data['report']['hashes']}
    for label,row in trajectories.items():
        m=(row['t']>=start+1.)&(row['t']<common_end-1e-7)
        assert m.sum()>50,(folder,'Insufficient contact-free scored replay')
        truth=np.column_stack([np.interp(row['t'][m],data['rate'][:,0],data['rate'][:,4+a]) for a in range(3)])*[1,-1,-1]
        error=row['rates'][m]-truth
        result['models'][label]={'MSE_RPY_rad_s_squared':np.mean(error**2,axis=0).tolist(),
            'RMSE_RPY_rad_s':np.sqrt(np.mean(error**2,axis=0)).tolist(),'samples':int(m.sum()),
            **{k:v for k,v in row.items() if k not in ('t','rates')}}
    print(json.dumps({'run':result['run'],'full_span':result['full_scored_span_without_ground_contact'],
        'old_RMSE':result['models']['frozen_physical']['RMSE_RPY_rad_s'],'v3_RMSE':result['models']['native_v3_physical']['RMSE_RPY_rad_s']},ensure_ascii=False),flush=True)
    return result


def main():
    frozen_paths=[ROOT/'quadrl'/name for name in ('env.py','policy.py','protocol.py','reward.py','plant_config.json')]
    frozen_before={rel(p):sha(p) for p in frozen_paths}
    fit=write_candidate_fit();checks=unit_checks(fit);receipt=read(DIAG);results=[]
    complete=read(ROOT/'artifacts/refinement/joint_v2/gazebo_thrust_ablation_recovery/complete.json')
    for r in complete['results']:
        if r['status']=='PASS' and r['job']['mission']=='route':results.append(replay(Path(r['run']),r))
    for name in ('system_id_multisine_1001','system_id_validation_1002'):
        results.append(replay(ROOT/'artifacts/gazebo'/name))
    held=[r for r in results if r['split']=='held_out' and '/runs/' in r['run']]
    means={name:np.mean([r['models'][name]['MSE_RPY_rad_s_squared'] for r in held],axis=0).tolist() for name in ('frozen_physical','native_v3_physical')}
    benefit=1-float(np.mean(means['native_v3_physical'][:2])/np.mean(means['frozen_physical'][:2]))
    matching=[r for r in held if r['recorded_thrust_semantics']=='body_collective']
    primary_means={name:np.mean([r['models'][name]['MSE_RPY_rad_s_squared'] for r in matching],axis=0).tolist() for name in ('frozen_physical','native_v3_physical')}
    primary_benefit=1-float(np.mean(primary_means['native_v3_physical'][:2])/np.mean(primary_means['frozen_physical'][:2]))
    small=next(r for r in results if r['run'].endswith('system_id_validation_1002'))
    small_better=np.mean(small['models']['native_v3_physical']['MSE_RPY_rad_s_squared'][:2])<np.mean(small['models']['frozen_physical']['MSE_RPY_rad_s_squared'][:2])
    status='PASS' if primary_benefit>.5 and all(r['full_scored_span_without_ground_contact'] for r in matching) and small['full_scored_span_without_ground_contact'] and small_better else 'REVIEW_REQUIRED'
    frozen_after={rel(p):sha(p) for p in frozen_paths};assert frozen_before==frozen_after
    checks['frozen_files_preserved']={'status':'PASS','hashes':frozen_after}
    out={'schema':'joint-native-rate-physical-replay-v3.0','status':status,'scope':'DIAGNOSTIC prospective physical subsystem verification, not Gazebo or new policy acceptance',
        'checks':checks,'replays':results,'held_out_route_mean_MSE_RPY':means,'held_out_RP_mean_MSE_reduction_fraction':benefit,
        'primary_matching_body_collective_interface':{'held_out_wind_routes':len(matching),'mean_MSE_RPY':primary_means,'RP_MSE_reduction_fraction':primary_benefit,
            'small_signal_validation1002_held_out':small,'all_full_spans_without_contact':bool(all(r['full_scored_span_without_ground_contact'] for r in matching) and small['full_scored_span_without_ground_contact'])},
        'supplemental_legacy_interface':'Legacy-preboost normalized action0 commands target native preboost, whereas this namespace targets body collective. Their common contact-free prefixes are retained as supplemental rate diagnostics; old physical branch ground contacts invalidate a full-span legacy comparison. They do not determine the matching-interface PASS verdict.',
        'limits':'Open-loop command replay can diverge spatially and trigger mission termination flags; no new position or policy performance inferred. Ground contact truncates both models at their common earliest contact. No fault/randomization validation claimed.',
        'artifact_hashes':{rel(p):sha(p) for p in (Path(__file__),FIT,ROOT/'quadrl_joint/native_rate_model_v3.py',ROOT/'quadrl_joint/env_v3.py',ROOT/'quadrl/env.py',ROOT/'quadrl/plant_config.json',DIAG)}}
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8');print('Wrote',rel(OUT),status,flush=True)


if __name__=='__main__':main()

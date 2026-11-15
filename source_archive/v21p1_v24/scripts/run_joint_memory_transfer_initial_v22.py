"""Prepare or ROOT-authorized run of10 new initial-state native diagnostics."""
from pathlib import Path
import argparse
import ast
import builtins
import copy
import csv
import hashlib
import json
import os
import sys
import symtable
import tempfile
from datetime import datetime,timezone

os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from quadrl_route_transfer_v22.initial import rotation,euler,target_initial

DOC=ROOT/'docs/joint_route_transfer_v22'
DEFAULT_PLAN=DOC/'physical_plan.root02.DRAFT.json'
STATIC=ROOT/'audit/joint_route_transfer_initial_v22_static02.json'
SOURCE_CASE='artifacts/refinement/joint_v2/gazebo_route80_development9_v5_metric_cost16_root01/runs/route80_v5_route_force_strong_env84001'
OLD_BASE='artifacts/refinement/joint_v2/route_memory_baseline_v20_root01'
PROTOTYPE='artifacts/refinement/joint_v2/route_memory_prototype_v20/root02'
BUDGET=dict(new_evaluation_episodes=10,reused_archived_episodes=4,coverage_episodes=14,
    maximum_new_control_steps=20000,reused_archived_control_steps=8000,maximum_new_physical_substeps=160000,
    maximum_new_NN_state_forwards=20000,additional_fixtures=0,training_steps=0,backward=0,optimizer=0)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
    if Path(p).exists():raise FileExistsError(p)
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()


def write_intervention_record(destination,record):
    """Frozen evaluator already wrote summary.json; preserve its exact bytes."""
    original=destination/'summary.json'
    original_sha256=sha(original)
    augmented=copy.deepcopy(record)
    augmented['frozen_original_summary']=dict(path=str(original),sha256=original_sha256,bytes_preserved=True)
    augmented['intervention_summary_path']=str(destination/'intervention_summary.json')
    write(destination/'intervention_summary.json',augmented)
    assert sha(original)==original_sha256
    return augmented


def static_undefined_names(paths):
    """Check every referenced global in nested scopes without importing modules."""
    findings={}
    for path in paths:
        module=symtable.symtable(path.read_text(),str(path),'exec')
        defined={symbol.get_name() for symbol in module.get_symbols() if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
        allowed=defined|set(dir(builtins))|{'__file__','__name__','__package__','__spec__','__builtins__','__doc__'}
        missing=set()
        def visit(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in allowed:missing.add(symbol.get_name())
            for child in scope.get_children():visit(child)
        visit(module)
        findings[path.relative_to(ROOT).as_posix()]=sorted(missing)
    assert not any(findings.values()),findings
    return findings


def static_output_lifecycle():
    """Exercise metadata file ownership only, with0environment/NN/physics."""
    frozen_text=(ROOT/'quadrl_route_memory_v20/eval.py').read_text()
    tree=ast.parse(frozen_text)
    original_writes=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
        and node.func.attr=='write_text' and isinstance(node.func.value,ast.BinOp)
        and isinstance(node.func.value.right,ast.Constant) and node.func.value.right.value=='summary.json']
    assert len(original_writes)==1,'Frozen run_episode owns summary.json'
    with tempfile.TemporaryDirectory(prefix='v22_metadata_lifecycle_') as temporary:
        folder=Path(temporary)
        (folder/'summary.json').write_text('{"fixture":"already_written_by_frozen_evaluator"}\n',encoding='utf-8')
        before=(folder/'summary.json').read_bytes()
        augmented=write_intervention_record(folder,dict(initial_intervention_proof='metadata_only_fixture'))
        assert (folder/'summary.json').read_bytes()==before
        assert read(folder/'intervention_summary.json')==augmented
        try:write_intervention_record(folder,{})
        except FileExistsError:pass
        else:raise AssertionError('Sidecar overwrite must be rejected')
        assert (folder/'summary.json').read_bytes()==before
    return dict(frozen_evaluator_summary_writer_AST_confirmed=True,sidecar_written_successfully=True,
        original_summary_bytes_preserved=True,sidecar_overwrite_rejected=True,temporary_metadata_fixture_only=True,
        environment_constructions=0,NN_forwards=0,physics_steps=0)


def matrix():
    result=[]
    for intervention,scenarios in [('yaw_only',('nominal','force_strong')),('full_initial',('nominal','force_strong')),('full_initial_physical_pulse',('force_strong',))]:
        for scenario in scenarios:
            for mode in ('raw','zero_route_integral'):
                result.append(dict(name=f'{intervention}_{mode}_{scenario}_84001',intervention=intervention,mode=mode,scenario=scenario,mission='route',seed=84001,seconds=40))
    return result


def source_hashes():
    old=read(ROOT/'docs/joint_route_memory_v20/preparation_freeze.json')['source_sha256']
    assert all(sha(ROOT/n)==v for n,v in old.items()),'Old frozen source changed'
    own=[Path(__file__),*sorted((ROOT/'quadrl_route_transfer_v22').glob('*.py'))]
    return {**old,**{p.relative_to(ROOT).as_posix():sha(p) for p in own}}


def prepare():
    assert not DEFAULT_PLAN.exists() and not STATIC.exists()
    sources=source_hashes(); folder=ROOT/SOURCE_CASE
    old_draft=DOC/'physical_plan.DRAFT.json';old_static=ROOT/'audit/joint_route_transfer_initial_v22_static01.json'
    assert sha(old_draft)=='4911b0ad27f79d925375e14c054f0a321f87164e13356c9cb6a5b3d992892e36'
    assert sha(old_static)=='2e9aa7a530c1780e7c270ac47bd6c89e54b6a9bd586fe77582abe9148e4a4774'
    old_sources=read(old_draft)['source_sha256'];archived={}
    for name in ('scripts/run_joint_memory_transfer_initial_v22.py','quadrl_route_transfer_v22/env.py','quadrl_route_transfer_v22/initial.py'):
        archived_path=DOC/'history/static01_sources'/name
        assert sha(archived_path)==old_sources[name]
        archived[archived_path.relative_to(ROOT).as_posix()]=sha(archived_path)
    own=[Path(__file__),*sorted((ROOT/'quadrl_route_transfer_v22').glob('*.py'))]
    global_names=static_undefined_names(own)
    lifecycle=static_output_lifecycle()
    whole=read(ROOT/'audit/joint_route80_development9_v5_root01.json')
    oldcase=next(x for x in whole['cases'] if str(x['run']).endswith(SOURCE_CASE));assert oldcase['status']=='PASS'
    with (folder/'telemetry.csv').open(newline='') as handle:first=next(csv.DictReader(handle))
    with (folder/'commands.jsonl').open() as handle:command=json.loads(next(handle))
    assert command['t']==float(first['t']) and command['policy_observation_size']==80
    R=np.array([[0,1,0],[1,0,0],[0,0,-1]])@rotation([float(first[k]) for k in ('roll_ned','pitch_ned','yaw_ned')])@np.diag([1,-1,-1])
    initial=dict(position=[float(first[k]) for k in ('x_e','y_n','z_up')],velocity=[float(first[k]) for k in ('vx_e','vy_n','vz_up')],
        rotation=R.tolist(),rpy=euler(R).tolist(),body_rates=(np.array([float(first[k]) for k in ('p_frd','q_frd','r_frd')])*[1,-1,-1]).tolist())
    assert np.max(abs(rotation(initial['rpy'])-R))<1e-12
    pulse=copy.deepcopy(oldcase['force']['physical_intervals'][0]);assert len(oldcase['force']['physical_intervals'])==1
    bindings={n:sha(ROOT/n) for n in ('audit/joint_route80_development9_v5_root01.json','audit/joint_route_memory_baseline_v20_root01.json',SOURCE_CASE+'/summary.json',SOURCE_CASE+'/telemetry.csv',SOURCE_CASE+'/commands.jsonl')}
    for name in (SOURCE_CASE+'/telemetry.csv',SOURCE_CASE+'/commands.jsonl'):
        bound=[v for n,v in whole['input_sha256'].items() if str(n).endswith(name)];assert bound==[bindings[name]]
    policies={mode:dict(path=f'{PROTOTYPE}/{mode}/policy.npz',sha256=sha(ROOT/f'{PROTOTYPE}/{mode}/policy.npz')) for mode in ('raw','zero_route_integral')}
    original=ROOT/'artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz'
    with np.load(original,allow_pickle=False) as z:weights={n:z[n].copy() for n in ('w0','b0','w1','b1','w2','b2')}
    for policy in policies.values():
        with np.load(ROOT/policy['path'],allow_pickle=False) as z:assert all(np.array_equal(z[n],v) for n,v in weights.items())
    reused=[]
    for mode in policies:
        for scenario in ('nominal','force_strong'):
            path=ROOT/OLD_BASE/mode/f'route_{scenario}_84001';summary=read(path/'summary.json')
            assert summary['steps']==2000
            for name in ('trace.csv','actor_io.npz'):assert sha(path/name)==summary['hashes'][name]
            reused.append(dict(mode=mode,scenario=scenario,seed=84001,run=path.relative_to(ROOT).as_posix(),summary_sha256=sha(path/'summary.json'),hashes=summary['hashes'],steps=2000))
            for name in ('summary.json','trace.csv','actor_io.npz'):bindings[(path/name).relative_to(ROOT).as_posix()]=sha(path/name)
    native=read(ROOT/OLD_BASE/'raw/route_nominal_84001/summary.json')['initial_state']
    mapped=target_initial(native,initial,'yaw_only')
    assert mapped['position']==native['position'] and mapped['velocity']==native['velocity'] and mapped['rpy'][:2]==native['rpy'][:2]
    assert mapped['rpy'][2]==initial['rpy'][2]
    assert np.max(abs(euler(rotation(mapped['rpy']))-mapped['rpy']))<1e-12
    assert np.max(abs(rotation(initial['rpy']).T@(rotation(initial['rpy'])@initial['body_rates'])-initial['body_rates']))<1e-12
    for name in sources:
        if name.endswith('.py'):compile((ROOT/name).read_text(),name,'exec')
    supersedes=dict(old_draft=old_draft.relative_to(ROOT).as_posix(),old_draft_sha256=sha(old_draft),old_static=old_static.relative_to(ROOT).as_posix(),old_static_sha256=sha(old_static),
        old_source_snapshots=archived,static01_scientific_status='NOGO_BEFORE_ANY_PHYSICS_OR_ROOT_GO',
        reason='Frozen v20 run_episode already writes summary.json; original wrapper would reject its second exclusive write after first episode. ROOT02 writes intervention_summary.json only.')
    plan=dict(schema='joint-route-transfer-initial-plan-v22.0',revision='ROOT02_METADATA_ONLY',status='DRAFT_NOT_AUTHORIZED_FOR_PHYSICS',created_utc=utc(),source_sha256=sources,input_sha256=bindings,supersedes=supersedes,
        output='artifacts/refinement/joint_v2/route_transfer_initial_v22_root01',budget=BUDGET,cases=matrix(),
        source_measured_Gazebo_initial=dict(run=SOURCE_CASE,first_command_time_s=command['t'],first_received_prefix_index=command['raw_message_index'],state=initial,
            scope='First actor-visible AHRS/telemetry sample; injected as native rigid-body initial state, not Gazebo truth or hidden AP state.'),
        source_physical_Gazebo_pulse=dict(run=SOURCE_CASE,from_proof='audit/joint_route80_development9_v5_root01.json /cases matching run /force/physical_intervals/0',event=pulse,
            native_realization='Existing force_N evaluated once per50Hz control interval; applied onset ceil(source_on/.02)*.02, no control-time interpolation or model change.'),
        initial_interventions=dict(yaw_only='Keep originalnative84001 p/v/r/p/bodyrates; replace only yaw and yaw reference with selectedcommonGz yaw.',full_initial='Replace p/v/R/bodyrates and yawreference with ONE fixed selectedGz sample; reset both raw encoders/history and native queues before first effectiveobserve.',full_initial_physical_pulse='Same fullinitial plus selectedactualphysicalpulse asENV intervention, never actor input.'),
        policies=policies,source_policy_sha256=sha(original),six_weight_arrays_exact=True,reused_archived_cases=reused,
        no_default_baseline_rerun=True,quality_limits='Exact original metric modules and applicability contract frozen; this limiteddiagnostic cannot accept/promote anypolicy.',
        retries=0,reserved_seeds_opened=[],controller_features='Only frozen measured80 and ownhistory; initial state/force schedule are ENV settings, never appendedfeatures.',
        root_physical_authorization=None)
    write(DEFAULT_PLAN,plan)
    receipt=dict(status='PASS_STATIC_PREPARATION_NO_ENVIRONMENT_OR_NN',created_utc=utc(),draft_plan_sha256=sha(DEFAULT_PLAN),source_sha256=sources,input_sha256=bindings,
        source_sample_and_pulse_frozen_before_any_physics=True,original_source_bindings_verified=True,yaw_only_mapping_and_rotation_roundtrip=True,
        world_angular_velocity_to_body_roundtrip=True,six_actor_weight_arrays_unchanged=True,reused_cases=4,new_cases_declared=10,
        compiled_source_files=True,global_undefined_name_check=global_names,output_lifecycle_metadata_fixture=lifecycle,supersedes=supersedes,
        intervention_summary_is_separate=True,environment_constructions=0,physics_steps=0,NN_forwards=0,backward=0,optimizer=0,
        reset_runtime_not_yet_physically_tested=True,actual_execution_authorized=False)
    write(STATIC,receipt);print(json.dumps(dict(status=receipt['status'],plan=str(DEFAULT_PLAN.relative_to(ROOT)),plan_sha256=sha(DEFAULT_PLAN),static_receipt=str(STATIC.relative_to(ROOT)),static_sha256=sha(STATIC),budget=BUDGET)))


def validate(path):
    plan=read(path)
    assert plan['status']=='READY_FOR_ROOT_AUTHORIZED_NATIVE_DIAGNOSTIC'
    assert plan['schema']=='joint-route-transfer-initial-plan-v22.0' and plan['budget']==BUDGET and plan['cases']==matrix()
    assert plan['source_sha256']==source_hashes()
    assert all(sha(ROOT/n)==v for n,v in plan['input_sha256'].items())
    for policy in plan['policies'].values():assert sha(ROOT/policy['path'])==policy['sha256']
    binding=plan['root_physical_authorization']; authpath=ROOT/binding['path'];assert sha(authpath)==binding['sha256']
    auth=read(authpath)
    value=auth
    for part in binding['pointer'].strip('/').split('/'):value=value[part]
    assert value is True and binding['equals'] is True
    assert auth['source_sha256']==plan['source_sha256'] and auth['budget']==BUDGET
    assert not (ROOT/plan['output']).exists()
    return plan


def run(path):
    # Validate BEFORE importing environment modules or constructing a policy.
    plan=validate(path)
    from quadrl_route_transfer_v22.env import TransferNativeEnv
    from quadrl_route_memory_v20.policy import NumpyPolicyRouteMemory
    from quadrl_route_memory_v20.eval import run_episode
    from scripts import train_joint_route_basis_v15 as frozen
    output=ROOT/plan['output'];output.mkdir(parents=True,exist_ok=False);write(output/'prospective_plan.json',plan)
    counts=dict(control_steps=0,physical_substeps=0,NN_forwards=0); records=[]
    class CountedController:
        def __init__(self,p):self.inner=p
        def __getattr__(self,n):return getattr(self.inner,n)
        def predict(self,*a,**kw):
            if counts['NN_forwards']>=20000:raise RuntimeError('NN budget exhausted')
            counts['NN_forwards']+=1;return self.inner.predict(*a,**kw)
    def profile(frame,event,arg):
        if event=='c_call' and getattr(arg,'__name__','')=='stepSimulation':
            if counts['physical_substeps']>=160000:raise RuntimeError('C physical substep budget exhausted')
            counts['physical_substeps']+=1
    prior=sys.getprofile();sys.setprofile(profile)
    try:
        for case in plan['cases']:
            controller=CountedController(NumpyPolicyRouteMemory(ROOT/plan['policies'][case['mode']]['path'],expected_mode=case['mode']))
            env=TransferNativeEnv(intervention=case['intervention'],measured_initial=plan['source_measured_Gazebo_initial']['state'],
                physical_pulse=plan['source_physical_Gazebo_pulse']['event'],budget=counts,basis_kind='velocity_residual',
                architecture='hybrid',training=False,scenario=case['scenario'],mission='route',seconds=40.)
            destination=output/case['name']
            try:
                record=run_episode(env,controller,'hybrid',True,case['scenario'],'route',84001,destination)
                assert len(env.transfer_reset_receipts)==1
                record['initial_intervention_proof']=env.transfer_reset_receipts[0]
            finally:env.close()
            with (destination/'trace.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
            record['geometry']=frozen.geometry_metrics([float(r['time']) for r in rows],[[float(r[k]) for k in ('x','y','z')] for r in rows],[[float(r[k]) for k in ('target_x','target_y','target_z')] for r in rows],record['mission_events'],record['metrics']['route_completion_time_s'])
            record.update(intervention=case['intervention'],new_actual_evaluation=True,training_steps=0,policy_promotion=False)
            assert record['steps']<=2000
            if record['metrics']['survived']:assert record['steps']==2000
            # Frozen evaluator owns summary.json; save additions in a NEW sidecar.
            record=write_intervention_record(destination,record);records.append(record)
            write(output/(case['name']+'_progress.json'),dict(completed_cases=len(records),counts=counts.copy(),case=case))
        assert counts['physical_substeps']==counts['control_steps']*8 and counts['NN_forwards']==counts['control_steps']
        assert source_hashes()==plan['source_sha256'] and all(sha(ROOT/n)==v for n,v in plan['input_sha256'].items())
        report=dict(status='COMPLETE_DIAGNOSTIC_NOT_MODEL_ACCEPTANCE',completed_utc=utc(),plan_sha256=sha(path),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],all_inputs_unchanged=True,
            new_episodes=len(records),reused_cases=plan['reused_archived_cases'],coverage_episodes=len(records)+4,actual_counts=counts,budget=BUDGET,cases=records,
            valid_failures_preserved=True,retries=0,backward=0,optimizer=0,policy_promotion=False,Gazebo_executed=False,
            limits='Measured p/v/R/rates match only; hidden AP controllerstate is not matched. Same frozen50Hz force lookup quantizes sourcepulse. Not final independent confirmation.')
        write(output/'complete.json',report);print(json.dumps(dict(status=report['status'],counts=counts,receipt=str(output/'complete.json'))))
    except Exception as exc:
        write(output/'error.json',dict(status='ERROR_RETAINED_NO_AUTOMATIC_RETRY',utc=utc(),error=repr(exc),counts=counts));raise
    finally:sys.setprofile(prior)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-static',action='store_true');p.add_argument('--physical-plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.prepare_static:
        assert not args.run;prepare();return 0
    path=args.physical_plan if args.physical_plan.is_absolute() else ROOT/args.physical_plan
    if args.run:run(path)
    else:validate(path);print(json.dumps(dict(status='ROOT_BOUND_PLAN_VALID_NO_EXECUTION',budget=BUDGET)))
    return 0


if __name__=='__main__':raise SystemExit(main())

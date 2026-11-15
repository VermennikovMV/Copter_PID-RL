#!/usr/bin/env python3
"""Bound manual Gazebo recording replay; preview/verify never starts a flight."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'scripts', ROOT/'audit'):
    sys.path.insert(0, str(directory))
REGISTRY = ROOT/'development_models_memory20_v1.json'
SNAPSHOT = ROOT/'artifacts/refinement/joint_v2/presentation_memory20_v1/source_snapshot'
CAMPAIGN = ROOT/'artifacts/refinement/joint_v2/gazebo_memory_development9_v20_root01'
POLICY = 'artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/zero_route_integral/policy.npz'
ACTOR_SHA = 'f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'
RUNNER = 'sim_joint/run_experiment_v5_memory.py'
RUNNER_SHA = '04f31f69f03c6f0e411ea7c57fd73e9b3513eb8bdd324566e3c37aa288e9d047'
AUDITOR = 'audit/verify_joint_hybrid_v5_memory.py'
PID = 'artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'
PID_SHA = '086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd'
PLAN_SHA = '49228743db8afc64de5e9553e278d3efb129631de074d45d2a8ee4e6a002669a'
SCENARIOS = ('nominal', 'force_mild', 'force_strong', 'gust')
OWNED_CHILD = """import json,os,pathlib,sys
record=pathlib.Path(sys.argv[1]);command=sys.argv[2:]
if os.getsid(0)!=os.getpid():os.setsid()
ticks=int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
record.write_text(json.dumps(dict(pid=os.getpid(),start_ticks=ticks,argv=command,process_group=os.getpgrp())))
os.execv(command[0],command)
"""
STOP_OWNED_CHILD = """import json,os,pathlib,signal,sys,time
record=json.loads(pathlib.Path(sys.argv[1]).read_text());pid=record['pid'];run=pathlib.Path(sys.argv[2])
proc=pathlib.Path('/proc')/str(pid)
if not proc.exists():print('owned helper already exited');raise SystemExit(0)
stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
argv=proc.joinpath('cmdline').read_bytes().split(b'\\0');argv=[v.decode() for v in argv if v]
if int(stat[19])!=record['start_ticks'] or argv!=record['argv'] or os.getpgid(pid)!=pid:raise RuntimeError('owned helper identity changed; no signal sent')
state=run/'gui_client_state.json'
if sys.argv[3]=='gui' and state.exists():
 value=json.loads(state.read_text());gui_pid=value.get('gui_pid');gui_proc=pathlib.Path('/proc')/str(gui_pid)
 if value.get('helper_pid')==pid and type(gui_pid) is int and gui_proc.exists():
  gui_stat=gui_proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
  if int(gui_stat[1])==pid and os.getpgid(gui_pid)==gui_pid:os.killpg(gui_pid,signal.SIGTERM)
os.killpg(pid,signal.SIGTERM);time.sleep(.3)
if proc.exists() and int(proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19])==record['start_ticks']:
 os.killpg(pid,signal.SIGKILL)
print('exact owned helper group stopped')
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local(value):
    value = str(value).replace('\\', '/')
    if value.startswith('/mnt/'):
        value = value.rsplit('/v2/', 1)[1]
    path = (ROOT/value).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Bound path escaped project: '+str(path))
    return path


def build_registry():
    files = {}
    def bind(value, expected=None):
        path = local(value)
        name = path.relative_to(ROOT).as_posix()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Frozen input changed: '+name)
        files[name] = actual
        return dict(path=name, sha256=actual)
    bind(CAMPAIGN/'plan.json', PLAN_SHA)
    plan, complete = read(CAMPAIGN/'plan.json'), read(CAMPAIGN/'complete.json')
    if (complete['status'] != 'COMPLETE' or complete['policy_sha256'] != ACTOR_SHA
        or len(complete['results']) != 9 or complete['basis_kind'] != 'velocity_residual'
        or complete['new_actual_flights'] != 9 or complete['total_observation_coverage'] != 20):
        raise ValueError('Completed exact9NEW+11SHARED Route80 matrix required')
    proof_name = 'audit/joint_memory_development9_v20_root01.json'
    proof = read(local(proof_name))
    if proof['status'] != 'PASS' or proof['policy_sha256'] != ACTOR_SHA:
        raise ValueError('Completed independent Route80 integrity audit required')
    proofs = {proof_name:bind(proof_name)}
    bind(CAMPAIGN/'complete.json')
    bind(complete['quality_receipt'], complete['quality_receipt_sha256'])
    for key in ('source_sha256','original_frozen_runtime_sha256','evidence_sha256'):
        for name, digest in plan[key].items(): bind(name, digest)
    models = []
    for item in complete['results']:
        job = item['job']
        if (job['controller'] != 'hybrid' or job['basis_kind'] != 'velocity_residual'
            or job['policy_sha256'] != ACTOR_SHA or item['status'] != 'PASS'
            or any(item[k+'_audit'] != 'PASS' for k in ('actor','physical','force','metric'))):
            raise ValueError('All9 actual Route80 integration cases must have integrity PASS')
        bind(local(item['run'])/'summary.json', item['summary_sha256'])
        receipts = {kind:bind(item[kind+'_audit_receipt'],item[kind+'_audit_sha256']) for kind in ('actor','physical','force','metric')}
        models.append(dict(scenario=job['scenario'],mission=job['mission'],environment_seed=job['environment_seed'],instance=job['instance'],duration_s=job['duration_s'],run=local(item['run']).relative_to(ROOT).as_posix(),audits=receipts))
    for name in ('show_joint_memory20.cmd','show_joint_memory20.ps1','START_JOINT_MEMORY20.txt',
                 'scripts/joint_memory20_presentation_v1.py','scripts/joint_gazebo_gui_v2.py',
                 'scripts/gazebo_gui_client.py','scripts/gazebo_recording_gui.config',
                 'scripts/independent_pose_recorder.py',AUDITOR,'quadrl_route_basis_v14/policy.py'):
        bind(name)
    table = 'reports/joint_memory20_gazebo_v7/presentation.png'
    table_binding = bind(table) if local(table).exists() else None
    quality=read(local(complete['quality_receipt']))
    primary_cells={}
    for check in quality['checks']:
        if check['case'][0]==83301:
            key=tuple(check['case']);primary_cells[key]=primary_cells.get(key,True) and check['pass_']
    actor, pid = bind(POLICY,ACTOR_SHA), bind(PID,PID_SHA)
    bind(RUNNER,RUNNER_SHA)
    for name, original in (('policy.npz',POLICY),('frozen_native_pid.json',PID),
        ('run_experiment_v5_memory.py',RUNNER),('thrust_adapter.py','sim_joint/thrust_adapter.py'),
        ('policy_base74.py','quadrl/policy.py'),('policy_path77.py','quadrl_path_v7/policy.py'),
        ('policy_route80.py','quadrl_route_basis_v14/policy.py'),('policy_memory20.py','quadrl_route_memory_v20/policy.py')):
        path=SNAPSHOT/name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/original,path)
        bind(path,sha(ROOT/original))
    return dict(schema='joint-memory20-presentation-v1.0',status='DEVELOPMENT_MEASURED_EXPERIMENTAL',
        model_role='DIAGNOSTIC_DEVELOPMENT_MEMORY_CONDITIONING_NOT_PROMOTED',observation_contract='measured-memory-route-integral-v5.0',conditioning_mode='zero_route_integral',additional_PPO_steps=0,final_applicability_confirmed=False,
        training_models=1,training_seed=111,primary_environment_seeds=[83301],stress_environment_seeds=[84001],
        reserved_environment_seeds=[93001,93002,93003],basis_kind='velocity_residual',actor_observations=80,
        default=dict(controller='rl',scenario='force_strong',mission='route',environment_seed=83301,instance=0),
        horizon_s=dict(hover=30.,route=40.),policy=actor,pid_profile=pid,
        policy_snapshot=(SNAPSHOT/'policy.npz').relative_to(ROOT).as_posix(),pid_snapshot=(SNAPSHOT/'frozen_native_pid.json').relative_to(ROOT).as_posix(),
        hover_thrust=plan['hover_thrust'],thrust_semantics='body_collective',adapter_attitude_max_age_s=.1,
        runtime=dict(runner=RUNNER,actor_distro='Ubuntu-24.04',actor_user='misha',python='[external-home]/vkr_sim/venv-cu128/bin/python',gui_distro='Ubuntu',gui_user='misha3'),
        original_external_runtime=plan['original_external_runtime'],evidence_cases=models,evidence=proofs,
        quality=dict(primary=quality['mandatory_eight_condition_operational_gate'],tailwind=quality['additional_tailwind84001_operational_gate'],six_wind_cell_macro=quality['original_thesis_wind_macro']['status'],primary_cells_pass=sum(primary_cells.values()),primary_cells_total=8,quality_receipt=complete['quality_receipt']),
        coverage=dict(new_actual_flights=9,shared_PID_flights=9,shared_retention_flights=2,total_observation_coverage=20),
        audience_note='Development hybrid memory20: exact trained source68 weights, zero additional PPO; only observed-leg integral inputs22:25 are conditioned. Quality: primary '+quality['mandatory_eight_condition_operational_gate']+', tailwind '+quality['additional_tailwind84001_operational_gate']+', six-cell wind macro '+quality['original_thesis_wind_macro']['status']+'. One viewed development observation per condition; unpromoted and no final confirmation.',
        table=table,table_binding=table_binding,automatic_video_recording=False,files_sha256=dict(sorted(files.items())))


def verify_registry(external=False):
    registry = read(REGISTRY)
    for name, expected in registry['files_sha256'].items():
        if sha(local(name)) != expected:
            raise ValueError('Demonstration source/evidence changed: '+name)
    if (registry['schema'] != 'joint-memory20-presentation-v1.0' or registry['policy']['sha256'] != ACTOR_SHA
            or registry['pid_profile']['sha256'] != PID_SHA or registry['files_sha256'][RUNNER] != RUNNER_SHA
            or registry['basis_kind'] != 'velocity_residual' or registry['actor_observations'] != 80
            or registry['final_applicability_confirmed'] is not False
            or registry['conditioning_mode']!='zero_route_integral'
            or registry['observation_contract']!='measured-memory-route-integral-v5.0'):
        raise ValueError('Expected exact experimental memory20 Route80 registry')
    if external:
        from build_joint_development_registry_v2 import verify_external
        verify_external(dict(external_runtime=registry['original_external_runtime']))
    return registry


def preview(args, registry):
    from quadrl.protocol import scenario_spec
    out = args.output.resolve()
    base = (ROOT/'artifacts/user_runs').resolve()
    if out == base or not out.is_relative_to(base) or out.exists():
        raise ValueError('Fresh artifacts/user_runs output required')
    if args.seed < 0 or args.seed in registry['reserved_environment_seeds']:
        raise ValueError('Nonnegative development seed required; reserved final seeds rejected')
    matched=[case for case in registry['evidence_cases'] if
        case['environment_seed']==args.seed and case['scenario']==args.scenario and case['mission']==args.mission]
    instance=args.instance if args.instance is not None else (matched[0]['instance'] if len(matched)==1 else 0)
    run = out/'runs/flight'
    controller = 'hybrid' if args.controller == 'rl' else 'pid'
    duration = registry['horizon_s'][args.mission]
    command = [sys.executable,'-B',str(ROOT/RUNNER),'--controller',controller,
        '--scenario',args.scenario,'--mission',args.mission,'--seed',str(args.seed),'--instance',str(instance),
        '--split','demo','--duration',str(duration),'--pid-profile',str(ROOT/registry['pid_snapshot']),
        '--hover-thrust',str(registry['hover_thrust']),'--thrust-semantics','body_collective',
        '--adapter-attitude-max-age-s','0.1','--basis-kind','velocity_residual','--output',str(run)]
    if args.controller == 'rl':
        command += ['--policy',str(ROOT/registry['policy_snapshot']),'--train-seed','111']
    def owned_helper(kind, script, extra):
        return ['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','misha3','--','/usr/bin/python3','-B',
            '-c',OWNED_CHILD,str(out/(kind+'_owned_linux_pid.json')),'/usr/bin/python3','-B',
            str(ROOT/script),'--run',str(run),*extra]
    observer = owned_helper('observer','scripts/independent_pose_recorder.py',['--flush-gate',str(out/'observer_flush_gate')])
    gui = owned_helper('gui','scripts/joint_gazebo_gui_v2.py',[]) if not args.no_gui else None
    job = dict(name='flight',controller=controller,scenario=args.scenario,mission=args.mission,
        environment_seed=args.seed,instance=instance,duration_s=duration,
        basis_kind='velocity_residual',conditioning_mode='zero_route_integral',training_seed=111 if args.controller == 'rl' else None,policy_sha256=ACTOR_SHA if args.controller == 'rl' else None)
    return dict(schema='joint-memory20-recording-case-v1', status='DRAFT_FOR_ROOT_GUI_REVIEW',
        registry_sha256=sha(REGISTRY), source_sha256=registry['files_sha256'], output=str(out),
        job=job, controller_command=command, observer_command=observer, gui_command=gui,
        scenario_spec=scenario_spec(args.scenario,args.seed), automatic_video_recording=False,
        new_actual_flights=0, planned_new_actual_flights=1,
        scope='Manual demonstration integration only; not a new primary comparison or independent replication',
        limits=registry['audience_note'])


def stop_owned(process, record, run, kind, seconds=15):
    if process is None or process.poll() is not None:
        return
    cleanup=subprocess.run(['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','misha3','--',
        '/usr/bin/python3','-B','-c',STOP_OWNED_CHILD,str(record),str(run),kind],
        capture_output=True,text=True,timeout=10)
    if cleanup.returncode:
        raise RuntimeError('Exact owned-helper cleanup failed: '+cleanup.stderr)
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.terminate();process.wait(timeout=5)


def execute(args, registry, plan):
    from joint_gazebo_screen_v2 import process_preflight
    import run_joint_path77_development11_v3 as development
    from run_joint_path77_smoke_v3 import scheduled_force_proof, physical_audit, audit_metrics, from_gazebo_run
    from verify_joint_hybrid_v5_memory import audit as actor_audit
    if sys.version_info < (3,11):
        raise RuntimeError('Scientific controller Python>=3.11 required')
    verify_registry(external=True)
    out=Path(plan['output']);run=out/'runs/flight'
    lock=ROOT/'artifacts/refinement/joint_v2/.gazebo_native_queue.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=process_preflight()
        out.mkdir();(out/'runs').mkdir();(out/'audits').mkdir()
        shutil.copy2(REGISTRY,out/'development_models_memory20_v1.json')
        write(out/'plan.json',plan)
        write(out/'execution_start.json',dict(utc=utc(),pid=os.getpid(),queue=queue,registry_sha256=sha(REGISTRY),
            advisory_lock_scope='Cooperating new launchers; all-three-port/process preflight also passed'))
        observer=gui=controller=None;code=None;error=None;logs=[];result={}
        try:
            for key,command in (('observer',plan['observer_command']),('gui',plan['gui_command']),('controller',plan['controller_command'])):
                if command is None:continue
                stream=(out/(key+'.log')).open('x');logs.append(stream)
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=(key=='controller'))
                if key=='observer':observer=process
                elif key=='gui':gui=process
                else:controller=process
            try:
                code=controller.wait(timeout=360)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid,signal.SIGINT)
                try:controller.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(controller.pid,signal.SIGKILL);controller.wait()
                raise RuntimeError('Bounded controller timeout; raw attempt retained')
        except BaseException as exc:
            error=repr(exc)
        finally:
            if controller is not None and controller.poll() is None:
                os.killpg(controller.pid,signal.SIGINT)
                try:controller.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(controller.pid,signal.SIGKILL);controller.wait()
            (out/'observer_flush_gate').write_text('flush\n')
            if observer is not None:
                try:observer.wait(timeout=105)
                except subprocess.TimeoutExpired:
                    try:stop_owned(observer,out/'observer_owned_linux_pid.json',run,'observer')
                    except Exception as cleanup_error:error=error or repr(cleanup_error)
                    error=error or 'Independent observer timeout; attempt retained'
                if observer.returncode != 0:error=error or 'Independent observer failed; see observer.log'
            if gui is not None:
                if run.exists():(run/'STOP_GUI_CLIENT').write_text('stop\n')
                try:gui.wait(timeout=105)
                except subprocess.TimeoutExpired:
                    try:stop_owned(gui,out/'gui_owned_linux_pid.json',run,'gui')
                    except Exception as cleanup_error:error=error or repr(cleanup_error)
                    error=error or 'GUI helper timeout; attempt retained'
                if gui.returncode != 0:error=error or 'GUI/target-marker helper failed; see gui.log'
            for stream in logs:stream.close()
        try:
            if error:raise RuntimeError(error)
            summary=read(run/'summary.json')
            job=plan['job']
            if summary['status'] not in ('PASS','FAIL') or code != (0 if summary['status']=='PASS' else 1):
                raise RuntimeError('Controller infrastructure error retained')
            if (summary['joint_interface']['runner_sha256'] != RUNNER_SHA or
                sha(run/'source_snapshot/pid_profile.json') != PID_SHA or
                summary['scenario_spec'] != plan['scenario_spec']):
                raise RuntimeError('Runtime/PID/scenario identity mismatch')
            result=dict(status=summary['status'],metrics={key:summary.get(key) for key in development.METRICS},audits={})
            reports=dict(force=scheduled_force_proof(run,summary,plan['scenario_spec']),physical=physical_audit(run))
            if args.controller=='rl' and summary['status']=='PASS':
                if sha(run/'source_snapshot/policy.npz') != ACTOR_SHA:raise RuntimeError('Wrong measured actor')
                reports['actor']=actor_audit(run, expected_kind='velocity_residual')
            reports['metric']=audit_metrics(run,summary,reports['physical'])
            for kind,report in reports.items():
                write(out/'audits'/(kind+'.json'),report)
                result['audits'][kind]=report['status']
                if report['status']!='PASS':raise RuntimeError('Independent '+kind+' audit failed')
            if args.mission=='route':
                geometry=from_gazebo_run(run);write(out/'audits/geometry.json',geometry);result['geometry']=geometry['geometry']
            if not args.no_gui:
                states={name:read(run/name) for name in ('gui_client_state.json','joint_current_target_state.json')}
                if any(value['status']!='PASS' for value in states.values()):raise RuntimeError('GUI/marker verification failed')
                if args.mission=='route' and states['joint_current_target_state.json']['target_updates']!=5:
                    raise RuntimeError('Five actual route targets were not verified')
                result['gui']=states
            verify_registry()
            result['queue_release']=process_preflight()
        except BaseException as exc:
            error=repr(exc)
        receipt=dict(schema='joint-memory20-recording-complete-v1',status='ERROR' if error else 'COMPLETE',
            utc=utc(),error=error,result=result,controller_returncode=code,
            new_actual_flights=int((run/'summary.json').exists()),automatic_video_recording=False,
            registry_sha256=sha(REGISTRY),plan_sha256=sha(out/'plan.json'),
            primary_comparison_or_final_replication=False,policy_promotion=False)
        write(out/'complete.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
        if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('build-registry','verify','plan-only','run'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--external',action='store_true')
    parser.add_argument('--controller',choices=('rl','pid'),default='rl')
    parser.add_argument('--scenario',choices=SCENARIOS,default='force_strong')
    parser.add_argument('--mission',choices=('hover','route'),default='route')
    parser.add_argument('--seed',type=int,default=83301)
    parser.add_argument('--instance',type=int,choices=(0,1,2),default=None,help='Defaults to the measured case lane, otherwise0')
    parser.add_argument('--no-gui',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-output',type=Path)
    parser.add_argument('--expected-plan-sha256')
    args=parser.parse_args()
    if args.build_registry:
        registry=build_registry();write(REGISTRY,registry)
        print(json.dumps(dict(status='REGISTRY_FROZEN',sha256=sha(REGISTRY),bound_files=len(registry['files_sha256']),new_actual_flights=0)));return
    registry=verify_registry(args.external)
    if args.verify:
        print(json.dumps(dict(status='PASS',bound_files=len(registry['files_sha256']),new_actual_flights=0)));return
    if args.output is None:parser.error('--output fresh artifacts/user_runs path required')
    plan=preview(args,registry)
    if args.plan_only:
        if args.plan_output:
            write(args.plan_output,plan);print(json.dumps(dict(status=plan['status'],path=str(args.plan_output),sha256=sha(args.plan_output),new_actual_flights=0)))
        else:print(json.dumps(plan,indent=2,ensure_ascii=False))
        return
    if args.expected_plan_sha256:
        if not args.plan_output or sha(args.plan_output)!=args.expected_plan_sha256 or read(args.plan_output)!=plan:
            parser.error('Root-reviewed preview differs; no simulation launched')
    execute(args,registry,plan)


if __name__=='__main__':main()

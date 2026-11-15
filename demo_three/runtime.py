#!/usr/bin/env python3
"""Three real aircraft, one Gazebo world; presentation demo, not a research matrix."""
import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo_three.assets import data_root,resolve_control_assets
DATA_ROOT=data_root(ROOT)
if DATA_ROOT!=ROOT:sys.path.insert(1,str(DATA_ROOT))
from demo_three.sitl import VEHICLES,check_ports,launch_sitl,sha,stop_owned,write_json

PID='artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'
BUNDLE='artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json'
DIRECT='artifacts/training/main_direct_seed111_v2/best_policy.npz'
CALIBRATION='artifacts/gazebo_direct/calibration_nominal_1104/calibration.json'
PINS={PID:'086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd',
    BUNDLE:'c3fc1c9c799f21200c98fc5995dba9908b1b13a517eb78bd60759e91eb8818a4',
    DIRECT:'5d0d19e02c30c62ed3a0efdcf003d8b6b635ae1eea9b1d32b4737836c46ff900',
    CALIBRATION:'2e20df83b9dadf5231c2f93e8070d89c03ffbdbe17c14e277935d7c87743a649'}
def message(text):print(text,flush=True)
def read_json(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,json.JSONDecodeError):return None
def load_hybrid(public_root=None):
    if public_root:
        repository=Path(public_root).resolve()
        package=repository/'v2' if (repository/'v2/quadrl_phase_expert_inference_v1/policy.py').is_file() else repository
        source=package/'quadrl_phase_expert_inference_v1/policy.py'
        spec=importlib.util.spec_from_file_location('demo_three_public_phase_inference',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bundle=(repository if package!=repository else repository.parent)/'models/phase29_public_inference.json'
        policy=module.NumpyPhaseExpertInference(bundle)
        return policy,dict(kind='distinct_public_inference_contract',contract=policy.contract,bundle_path=str(bundle),
            bundle_sha256=sha(bundle),numeric_controller='original_phase29',original_ZIP_verified_locally=False,
            expert_paths=[str(module.ROOT/module.F0_PATH),str(module.ROOT/module.ROUTE_PATH)],
            source68_path=str(module.ROOT/module.SOURCE_PATH),inference_source_path=str(source))
    from quadrl_phase_expert_v29.policy import NumpyPhaseExpertPolicy
    bundle=DATA_ROOT/BUNDLE
    from quadrl_phase_expert_v29.policy import F0_PATH,ROUTE_PATH
    return NumpyPhaseExpertPolicy(bundle),dict(kind='original_strict_phase29',bundle_path=str(bundle),bundle_sha256=sha(bundle),original_ZIP_verified_locally=True,
        expert_paths=[str(DATA_ROOT/F0_PATH),str(DATA_ROOT/ROUTE_PATH)],source68_path=str(DATA_ROOT/'artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz'))
def scene_command(out,sequence,force,vehicles,sim_t0):
    targets={}
    for v in vehicles:
        target=v.mission.target.tolist() if v.mission else [0.,0.,3.]
        offset=v.spec['offset_enu']
        targets[v.spec['model']]=[target[i]+offset[i] for i in range(3)]
    write_json(out/'SCENE_COMMAND.json',dict(seq=sequence,measurement_sim_start_s=sim_t0,force_enu_N=list(force),targets_world_enu=targets))
def wait_scene(out,process,timeout=60):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if process.poll() is not None:raise RuntimeError('Shared Gazebo worker exited before readiness')
        state=read_json(out/'SCENE_STATE.json')
        if state and len(state.get('models',{}))==3:return state
        time.sleep(.05)
    raise TimeoutError('Shared scene state not ready')
def fair_start_state(vehicles,sim_time):
    import numpy as np
    states={}
    for v in vehicles:
        position,velocity,_,_=v.task_state()
        error=float(np.linalg.norm(position-np.array([0.,0.,3.])))
        speed=float(np.linalg.norm(velocity))
        states[v.role]=dict(position=position.tolist(),velocity=velocity.tolist(),goal0_error_m=error,speed_mps=speed,
            within_position_tolerance=error<=.03,within_speed_tolerance=speed<=.05,mav_boot_s=v.rec.boot_s,receive_cutoff=v.rec.n)
    return dict(sim_time=sim_time,states=states,passed=all(v['within_position_tolerance'] and v['within_speed_tolerance'] for v in states.values()),
        position_tolerance_m=.03,speed_tolerance_mps=.05,required_stable_s=1.)
def measurement_finished(elapsed,slot,expected_slots,duration):
    return elapsed>=duration or slot>=expected_slots
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scenario',choices=['nominal','force_strong','gust'],default='force_strong')
    parser.add_argument('--mission',choices=['hover','route'],default='route');parser.add_argument('--seed',type=int,choices=[83301,84001],default=83301)
    parser.add_argument('--duration',type=float);parser.add_argument('--gui',action='store_true');parser.add_argument('--record',action='store_true')
    parser.add_argument('--start-file',type=Path,help='After READY, hold all3 until this file exists')
    parser.add_argument('--finish-hold-s',type=float,default=3.)
    parser.add_argument('--public-root',type=Path);parser.add_argument('--ardupilot',type=Path,default=Path.home()/'vkr_sim/ardupilot')
    parser.add_argument('--gazebo-distro',default='Ubuntu');parser.add_argument('--gazebo-user',default=None)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();duration=args.duration or (40. if args.mission=='route' else 30.)
    if args.public_root is None and DATA_ROOT!=ROOT and (ROOT/'models/phase29_public_inference.json').is_file():args.public_root=ROOT
    if not 0<duration<=60:parser.error('Presentation measurement must be in (0,60] seconds')
    plan=dict(schema='shared-world-three-demo-v1.0',world='demo_three',partition='demo_three',vehicles=VEHICLES,
        scenario=args.scenario,seed=args.seed,mission=args.mission,duration_s=duration,physical_world_count=1,
        public_root=str(args.public_root) if args.public_root else None,task_scope='visual demo only; no research-matrix or new-training claim',
        common_clock='Gazebo sim_time after all-three takeoff/settle and optional capture gate',new_training_steps=0,
        direct_interface='unchanged motor actuator proxy; not stock MAVLink deployment',original_model_lineage=PINS)
    if args.plan_only:print(json.dumps(plan,indent=2));return
    if out.exists():parser.error('Use a fresh output folder; preserve actual attempts')
    import fcntl
    import numpy as np
    from quadrl.policy import load_policy
    from quadrl.protocol import force_N,scenario_spec
    from sim_direct.proxy import Proxy
    from demo_three.adapters import VehicleController
    controls=resolve_control_assets(ROOT)
    if not args.public_root and sha(DATA_ROOT/BUNDLE)!=PINS[BUNDLE]:raise ValueError('Original strict phase bundle differs')
    plan['control_asset_identity']=controls['identity']
    out.mkdir(parents=True);write_json(out/'plan.json',plan)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for file in (Path(__file__),ROOT/'demo_three/assets.py',ROOT/'demo_three/adapters.py',ROOT/'demo_three/sitl.py',ROOT/'demo_three/telemetry.py',ROOT/'demo_three/scene.py',DATA_ROOT/'sim_direct/proxy.py',DATA_ROOT/'sim_joint/thrust_adapter.py',DATA_ROOT/'quadrl/protocol.py'):
        target=snapshot/(file.parent.name+'_'+file.name);shutil.copy2(file,target)
    queue_path=DATA_ROOT/'artifacts/refinement/joint_v2/.gazebo_native_queue.lock';queue_path.parent.mkdir(parents=True,exist_ok=True)
    lock=queue_path.open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=[];logs=[];vehicles=[];proxy=None;worker=None;capture=None
    start_wall=time.monotonic();exit_status='ERROR'
    try:
        check_ports()
        profile=read_json(controls['pid_path'])['parameters'];calibration=read_json(controls['calibration_path'])
        hybrid,hybrid_meta=load_hybrid(args.public_root)
        direct=load_policy(controls['direct_path'])
        if direct.architecture!='direct':raise ValueError('Direct111 model architecture differs')
        assets=snapshot/'models';assets.mkdir()
        actor_inputs={'direct111.npz':controls['direct_path'],'calibration.json':controls['calibration_path'],'pid.json':controls['pid_path'],
            'phase_bundle.json':Path(hybrid_meta['bundle_path']),'F0.npz':Path(hybrid_meta['expert_paths'][0]),
            'route_final.npz':Path(hybrid_meta['expert_paths'][1]),'source68.npz':Path(hybrid_meta['source68_path'])}
        for name,path in actor_inputs.items():shutil.copy2(path,assets/name)
        write_json(snapshot/'model_inputs.json',{name:dict(original_path=str(path),sha256=sha(path),snapshot_sha256=sha(assets/name)) for name,path in actor_inputs.items()})
        proxy_folder=out/'direct';proxy_folder.mkdir()
        proxy=Proxy(proxy_folder,listen_port=9032,physics_port=9022);proxy.thread.start()
        capture_command=['/mnt/c/Windows/System32/wsl.exe','-d',args.gazebo_distro,'-u','root','--','/usr/bin/python3',str(ROOT/'demo_three/sitl.py'),'--capture','--output',str(out)]
        log=(out/'capture_worker.log').open('x');logs.append(log)
        capture=subprocess.Popen(capture_command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append(capture)
        scene_args=['/mnt/c/Windows/System32/wsl.exe','-d',args.gazebo_distro]
        if args.gazebo_user:scene_args+=['-u',args.gazebo_user]
        scene_args+=['--','/usr/bin/python3',str(ROOT/'demo_three/scene.py'),
            '--serve','--output',str(out),'--world-name','demo_three','--partition','demo_three','--seed',str(args.seed)]
        if args.gui:scene_args.append('--gui')
        # Actual GUI recording is owned by demo_three.capture/presentation;
        # scene --serve only starts the real server and optional GUI.
        message('Starting one Gazebo world with PID, Hybrid phase29 and Direct111.')
        log=(out/'scene_worker.log').open('x');logs.append(log)
        worker=subprocess.Popen(scene_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append(worker)
        # Start all3 JSON partners promptly: the one physics world may otherwise
        # wait for an unconnected lockstep plugin before publishing pose state.
        for spec in VEHICLES:
            process,log=launch_sitl(spec,out,args.ardupilot,profile);processes.append(process);logs.append(log)
            vehicles.append(VehicleController(spec,out,policy=hybrid if spec['role']=='hybrid' else direct if spec['role']=='direct' else None,
                proxy=proxy if spec['role']=='direct' else None,calibration=calibration))
        write_json(out/'manifest.json',dict(plan=plan,controller_pid=os.getpid(),one_scene_worker_pid=worker.pid,
            scene_command=scene_args,hybrid=hybrid_meta,SITL_pids=[p.pid for p in processes[2:]],source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'demo_three/adapters.py',ROOT/'demo_three/sitl.py')}))
        message('Three SITL processes started; normal arming and takeoff.')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(v.connect_and_takeoff,profile):v for v in vehicles}
            for future in concurrent.futures.as_completed(futures):
                v=futures[future]
                try:future.result()
                except BaseException:
                    (out/'STOP_DEMO').write_text('Owned startup failure cancels all startup waits.\n');raise
                message(v.role.upper()+': takeoff complete.')
        state=wait_scene(out,worker);settle_start=float(state['sim_time']);last_print=-1.;force_seq=0
        stable_since=None;gate_deadline=time.monotonic()+240.;gate_log=(out/'start_gate.jsonl').open('x',buffering=1)
        while True:
            if (out/'STOP_DEMO').exists():raise InterruptedError('Owned demo stop during common start gate')
            if time.monotonic()>gate_deadline:raise TimeoutError('All-three start-state gate not reached; no measurement begun')
            if worker.poll() is not None:raise RuntimeError('Owned shared world exited')
            state=read_json(out/'SCENE_STATE.json')
            if not state:time.sleep(.01);continue
            for v in vehicles:v.drain();v.hold()
            scene_command(out,force_seq,[0.,0.,0.],vehicles,None)
            now_sim=float(state['sim_time']);gate=fair_start_state(vehicles,now_sim)
            if gate['passed'] and now_sim-settle_start>=5.:
                if stable_since is None:stable_since=now_sim
            else:stable_since=None
            gate['stable_since_sim_time']=stable_since;gate_log.write(json.dumps(gate)+'\n')
            if stable_since is not None and now_sim-stable_since>=1.:break
            time.sleep(.01)
        ready=dict(status='READY_THREE_REAL_VEHICLES_HOVERING',sim_time=float(state['sim_time']),world='demo_three',
            start_file=str(args.start_file) if args.start_file else None,models=[v.spec['model'] for v in vehicles],
            coordinate_origins={v.role:v.origin.tolist() for v in vehicles},scene_state=state,qualified_start_gate=gate)
        write_json(out/'READY.json',ready)
        message('READY: all three aircraft airborne in one world. Waiting for capture start.' if args.start_file else 'READY: all three aircraft airborne in one world.')
        stable_since=None;post_start_deadline=None
        while True:
            if (out/'STOP_DEMO').exists():raise RuntimeError('Owned demo stop requested before measurement')
            if worker.poll() is not None:raise RuntimeError('Shared world exited while waiting for capture')
            for v in vehicles:v.drain();v.hold()
            state=read_json(out/'SCENE_STATE.json')
            if not state:time.sleep(.01);continue
            now_sim=float(state['sim_time']);gate=fair_start_state(vehicles,now_sim)
            requested=args.start_file is None or args.start_file.exists()
            if requested:
                if post_start_deadline is None:post_start_deadline=time.monotonic()+240.
                if time.monotonic()>post_start_deadline:raise TimeoutError('Post-capture common start-state gate not reached')
                if gate['passed']:
                    if stable_since is None:stable_since=now_sim
                else:stable_since=None
                gate['stable_since_sim_time']=stable_since;gate_log.write(json.dumps(gate)+'\n')
                if stable_since is not None and now_sim-stable_since>=1.:break
            scene_command(out,force_seq,[0.,0.,0.],vehicles,None);time.sleep(.02)
        gate_log.close()
        state=read_json(out/'SCENE_STATE.json');sim_t0=float(state['sim_time'])+.1
        for v in vehicles:v.drain();v.begin(args.mission,sim_t0)
        start_receipt=dict(world_sim_time=sim_t0,world='demo_three',wall_time=time.time(),
            qualified_start_gate=gate,common_force_schedule=scenario_spec(args.scenario,args.seed),
            mission_epoch_boot_ms=None,mission_epoch_boot_s=None,
            mission_epoch_source='first_common_actual_sensor_pair_before_any_controlled_forward')
        write_json(out/'MEASUREMENT_START.json',start_receipt)
        message('Measurement started: one Gazebo clock, identical physical wind and translated identical routes.')
        next_control=sim_t0;last_force=None;last_print=-1.;deadline=time.monotonic()+max(180,duration*12)
        slot=0;missed_slots=0;expected_slots=round(duration*50);mission_epoch_bound=False
        schedule=(out/'control_schedule.jsonl').open('x',buffering=1);logs.append(schedule)
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Shared physical world measurement timeout')
            if worker.poll() is not None:raise RuntimeError('Shared world exited during measurement')
            if proxy.error:raise RuntimeError('Direct physical proxy error: '+proxy.error)
            if (out/'STOP_DEMO').exists():raise RuntimeError('Owned demo stop requested')
            state=read_json(out/'SCENE_STATE.json')
            if not state:time.sleep(.002);continue
            sim_time=float(state['sim_time']);elapsed=sim_time-sim_t0
            for v in vehicles:v.drain()
            if measurement_finished(elapsed,slot,expected_slots,duration):break
            if sim_time+1e-8<next_control:time.sleep(.001);continue
            skipped=int(max(0.,sim_time-next_control+1e-8)//.02)
            if skipped:
                schedule.write(json.dumps(dict(event='missed_control_slots',first_slot=slot,count=skipped,observed_sim_time=sim_time,scheduled_sim_time=next_control))+'\n')
                missed_slots+=skipped;slot+=skipped;next_control+=skipped*.02
            if slot>=expected_slots:break
            if not all(v.has_fresh_input_pair() for v in vehicles):
                time.sleep(.001);continue
            if not mission_epoch_bound:
                pairs={v.role:list(v.input_pair()) for v in vehicles}
                epochs={max(pair) for pair in pairs.values()}
                if len(epochs)!=1:
                    raise RuntimeError('First common control group lacks one identical actual-sensor epoch across all3 aircraft')
                epoch=epochs.pop()
                for v in vehicles:v.set_mission_epoch(epoch)
                start_receipt.update(mission_epoch_boot_ms=epoch,mission_epoch_boot_s=epoch*.001,
                    mission_epoch_observer_world_sim_time=sim_time,mission_epoch_observer_measurement_t=elapsed,
                    mission_epoch_raw_evidence={v.role:dict(pair_boot_ms=pairs[v.role],receive_cutoff=v.rec.n,
                        position_record_index=v.rec.message_indices['LOCAL_POSITION_NED'],
                        attitude_record_index=v.rec.message_indices['ATTITUDE']) for v in vehicles})
                write_json(out/'MEASUREMENT_START.json',start_receipt);mission_epoch_bound=True
            schedule.write(json.dumps(dict(event='shared_fresh_control_slot',slot=slot,scheduled_sim_time=next_control,
                observed_sim_time=sim_time,lateness_s=sim_time-next_control,input_pairs_boot_ms={v.role:list(v.input_pair()) for v in vehicles},
                input_receive_prefixes={v.role:dict(receive_cutoff=v.rec.n,position_record_index=v.rec.message_indices['LOCAL_POSITION_NED'],
                    attitude_record_index=v.rec.message_indices['ATTITUDE'],
                    pair_skew_s=abs(v.input_pair()[0]-v.input_pair()[1])*.001,
                    position_receive_monotonic_s=v.rec.received['LOCAL_POSITION_NED'],attitude_receive_monotonic_s=v.rec.received['ATTITUDE']) for v in vehicles}))+'\n')
            force=force_N(max(0.,elapsed),args.scenario,args.seed)
            if last_force is None or not np.array_equal(force,last_force):force_seq+=1;last_force=force.copy()
            status=[]
            for v in vehicles:
                try:
                    item=v.step(max(0.,elapsed),sim_time,slot_index=slot,scheduled_sim_time=next_control)
                    if item:status.append(v.role.upper()+' g'+str(v.mission.index)+' error '+format(float(np.linalg.norm(item['task_position']-item['target_before_update'])),'.2f')+'m')
                except Exception as exc:
                    v.fail_and_land(exc);status.append(v.role.upper()+' FAILED (actual native LAND)')
            scene_command(out,force_seq,force.tolist(),vehicles,sim_t0)
            slot+=1;next_control+=.02
            if int(max(0.,elapsed))!=last_print:
                message('Gazebo t='+format(max(0.,elapsed),'.1f')+'s | '+' | '.join(status));last_print=int(max(0.,elapsed))
            if all(v.failed for v in vehicles):break
        force_seq+=1;scene_command(out,force_seq,[0.,0.,0.],vehicles,sim_t0)
        expected_slots=round(duration*50)
        coverage={v.role:dict(samples=v.samples,expected_slots=expected_slots,coverage=v.samples/expected_slots,
            passed=v.samples>=.95*expected_slots and v.samples<=1.05*expected_slots) for v in vehicles}
        cadence_pass=all(row['passed'] for row in coverage.values())
        summary=dict(status='COMPLETE' if not any(v.failed for v in vehicles) else 'COMPLETE_WITH_CONTROLLER_FAILURES',
            one_physical_world=True,world='demo_three',measurement_sim_start_s=sim_t0,measurement_sim_end_s=sim_time,
            requested_duration_s=duration,roles={v.role:dict(samples=v.samples,failure=v.failed,route_completed=v.mission.completion_time is not None if args.mission=='route' else None,
                route_completion_time_s=v.mission.completion_time,mission_events=v.mission.events) for v in vehicles},
            shared_force_vector_and_sequence_requested=True,actual_force_tick_identity='VERIFY_PER_MODEL_FORCE_PHYSICS_LOGS',
            control_cadence_gate=dict(status='PASS' if cadence_pass else 'FAIL',target_hz=50.,minimum_expected_slot_fraction=.95,
                missed_slots=missed_slots,roles=coverage,common_fresh_position_and_attitude_barrier=True,no_duplicate_observation_forwards=True),
            scene_worker_pid=worker.pid,new_training_steps=0,scientific_matrix_updated=False,wall_seconds=time.monotonic()-start_wall)
        if not cadence_pass:summary['status']='COMPLETE_WITH_CONTROL_CADENCE_FAILURE'
        write_json(out/'complete.json',summary);exit_status=summary['status']
        for v in vehicles:v.land()
        if not cadence_pass:raise RuntimeError('Actual shared50Hz coverage below prescribed95%; raw attempt retained, no full-demo acceptance')
        message('Finished: '+exit_status+'. Real common-world traces and commands saved.')
        finish=time.monotonic()+max(0.,args.finish_hold_s)
        while time.monotonic()<finish:
            for v in vehicles:v.drain()
            time.sleep(.02)
    except BaseException as exc:
        write_json(out/'error.json',dict(status='ERROR',error=repr(exc),traceback=traceback.format_exc(),preserve_actual_attempt=True))
        message('Stopped: '+str(exc));raise
    finally:
        for v in vehicles:
            try:v.close()
            except Exception:pass
        if proxy:proxy.close()
        (out/'STOP_SHARED_WORLD').write_text('Owned three-aircraft scene cleanup.\n')
        (out/'STOP_CAPTURE').write_text('Owned passive capture cleanup.\n')
        for process in reversed(processes):
            try:
                if process in (worker,capture):process.wait(timeout=15)
                else:stop_owned(process,5)
            except subprocess.TimeoutExpired:stop_owned(process,3)
        for log in logs:log.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
        write_json(out/'cleanup.json',dict(status=exit_status,owned_process_returncodes={str(p.pid):p.poll() for p in processes},no_foreign_process_killed=True))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Simulation-only actuator replacement, preserving genuine Gazebo JSON sensors.

Native ArduPilot flies takeoff/settle/land. During DIRECT measurement the proxy
replaces motor PWM with four learned rotor commands and logs both inputs/outputs.
This is deliberately not a stock MAVLink deployment interface.
"""
import argparse,csv,datetime,hashlib,json,math,os,signal,socket,struct,subprocess,sys,threading,time,traceback,shutil
from pathlib import Path
import numpy as np
from pymavlink import mavutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim.run_experiment import Telemetry,send_position,command
from quadrl.policy import load_policy
from quadrl.protocol import Mission,force_N,scenario_spec,SCENARIOS
from sim_direct.proxy import Proxy, MOTOR_ORDER
from contract import verify_contract


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--controller',choices=['passthrough','pulse','direct'],default='passthrough')
    ap.add_argument('--scenario',choices=['nominal','force_mild','force_strong','gust','mass_20','mass_40'],default='nominal')
    ap.add_argument('--policy',type=Path)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--duration',type=float,default=None)
    ap.add_argument('--mission',choices=['hover','route'],default='hover')
    ap.add_argument('--split',choices=['development','validation','heldout','demo'],default='development')
    ap.add_argument('--pulse-pwm',type=float,default=10.)
    ap.add_argument('--pulse-pattern',choices=['single','yaw'],default='single')
    ap.add_argument('--pid-profile',type=Path)
    ap.add_argument('--train-seed',type=int)
    ap.add_argument('--instance',type=int,choices=[0,1,2],default=0)
    ap.add_argument('--freeze-file',type=Path)
    ap.add_argument('--seed',type=int,default=301)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--record',action='store_true')
    ap.add_argument('--gui',action='store_true')
    a=ap.parse_args()
    tcp_port=5790+10*a.instance;plugin_port=19002+20*a.instance
    proxy_port=plugin_port+10;world_name=f'vkr_direct_v2_{a.instance}'
    ap_root=Path('[external-home]/vkr_sim/ardupilot')
    gazebo_upstream=Path('[external-home]/ardupilot_gazebo')
    mission=Mission(a.mission)
    if a.duration is None:a.duration=mission.horizon_s
    if a.duration<=0:ap.error('Duration must be positive')
    if a.controller=='pulse' and (a.scenario!='nominal' or a.mission!='hover' or a.duration<15):ap.error('Pulse identification requires nominal hover >=15s')
    if abs(a.pulse_pwm)>20:ap.error('Development pulse is limited to 20 PWM units')
    if a.controller=='direct' and (a.policy is None or a.calibration is None):ap.error('Direct requires frozen policy and fixed nominal calibration')
    if a.controller!='direct' and (a.policy is not None or a.calibration is not None):ap.error('Native/pulse cannot receive policy/calibration')
    if a.split=='heldout' and (a.freeze_file is None or not a.freeze_file.is_file()):ap.error('An existing --freeze-file is required before heldout')
    pid_profile=json.loads(a.pid_profile.read_text()) if a.pid_profile else {'label':'native_default','parameters':{}}
    if any(not k.startswith('PSC_NE_') for k in pid_profile['parameters']):ap.error('Only native outer-loop PSC_NE parameters may change')
    contract_receipt=None
    if a.split=='heldout':
        contract_receipt=verify_contract(a.freeze_file,backend='gazebo',controller='pid' if a.controller=='passthrough' else a.controller,
            env_seed=a.seed,mission=a.mission,scenario=a.scenario,duration_s=a.duration,policy=a.policy,
            train_seed=a.train_seed,pid_profile=a.pid_profile,calibration=a.calibration,
            external_paths={'ardupilot_binary':ap_root/'build/sitl/bin/arducopter',
                'default_copter':ap_root/'Tools/autotest/default_params/copter.parm',
                'default_gazebo':ap_root/'Tools/autotest/default_params/gazebo-iris.parm',
                'gazebo_plugin':gazebo_upstream/'build/libArduPilotPlugin.so',
                'iris_base_sdf':gazebo_upstream/'models/iris_with_standoffs/model.sdf',
                'iris_control_sdf':gazebo_upstream/'models/iris_with_ardupilot/model.sdf'})
    # Exclusive probes detect active helpers across the WSL distributions.
    for probe_port in (plugin_port,proxy_port):
        probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:probe.bind(('127.0.0.1',probe_port))
        except OSError as exc:raise RuntimeError(f'UDP{probe_port} already occupied; no experiment started') from exc
        finally:probe.close()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    snap=out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),ROOT/'sim_direct/proxy.py',ROOT/'sim/gazebo_worker.py',ROOT/'sim/run_experiment.py',ROOT/'quadrl/policy.py',ROOT/'quadrl/protocol.py']:
        shutil.copy2(p,snap/(p.parent.name+'_'+p.name))
    shutil.copy2(ROOT/'contract.py',snap/'contract.py')
    for name in ['network_capture.py','metrics.py','finalize_metrics.py','pcap_clock.py']:
        shutil.copy2(ROOT/'sim'/name,snap/name)
    archived_force=snap/'force_plugin';(archived_force/'build').mkdir(parents=True)
    for name in ['ForceSystem.cc','CMakeLists.txt','build/libVkrForceSystem.so']:
        shutil.copy2(ROOT/'sim/force_plugin'/name,archived_force/name)
    if a.freeze_file:shutil.copy2(a.freeze_file,snap/'frozen_contract.json')
    if a.pid_profile:shutil.copy2(a.pid_profile,snap/'pid_profile.json')
    sys.path.insert(0,str(snap))
    from finalize_metrics import finalize
    policy=None;cal=None
    if a.policy:
        shutil.copy2(a.policy,snap/'policy.npz');policy=load_policy(snap/'policy.npz')
        if policy.architecture!='direct':raise ValueError('Policy is not motor-direct')
    if a.calibration:
        shutil.copy2(a.calibration,snap/'calibration.json');cal=json.loads(a.calibration.read_text())
        hp=np.asarray(cal.get('hover_pwm_gazebo_order'),dtype=float)
        if not cal.get('nominal_only') or hp.shape!=(4,) or not np.isfinite(hp).all() or np.any(hp<=1100) or np.any(hp>=1900):
            raise ValueError('Calibration must be fixed nominal with four finite PWM strictly inside 1100..1900')
    logfiles=[];worker=None;capture=None;sitl=None;proxy=None;rec=None;conn=None;rows=[]
    summary={'controller':'pid' if a.controller=='passthrough' else a.controller,'scenario':a.scenario,'seed':a.seed,'mission':a.mission,'split':a.split,'backend':'gazebo','interface':'direct_actuator_proxy','requested_duration_s':a.duration,'status':'ERROR','utc_started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    summary.update(env_seed=a.seed,train_seed=a.train_seed,pid_profile=pid_profile,plugin_port=plugin_port,tcp_port=tcp_port,contract_verification=contract_receipt)
    summary['policy_sha256']=sha(snap/'policy.npz') if policy is not None else None
    ap_root=Path('[external-home]/vkr_sim/ardupilot')
    started=time.time()
    try:
        check=socket.socket();check.settimeout(.2)
        if check.connect_ex(('127.0.0.1',tcp_port))==0:raise RuntimeError(f'Isolated TCP{tcp_port} occupied')
        check.close()
        proxy=Proxy(out,listen_port=proxy_port,physics_port=plugin_port);proxy.thread.start()
        params=out/'experiment.parm'
        params.write_text('FRAME_CLASS 1\nFRAME_TYPE 1\nSIM_SPEEDUP 1\nFS_THR_ENABLE 0\nFS_GCS_ENABLE 0\nFENCE_ENABLE 0\nLOG_DISARMED 1\nLOG_BITMASK 65535\nMOT_PWM_MIN 1100\nMOT_PWM_MAX 1900\nMOT_THST_HOVER 0.5\nMOT_HOVER_LEARN 0\n')
        with params.open('a') as profile_stream:
            for name,value in pid_profile['parameters'].items():profile_stream.write(f'{name} {float(value)}\n')
        spec=scenario_spec(a.scenario,a.seed)
        summary['scenario_spec']=spec
        (out/'scenario.json').write_text(json.dumps(spec,indent=2))
        worker_cmd=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','--','/usr/bin/python3',str(snap/'sim_gazebo_worker.py'),'--output',str(out),'--scenario',a.scenario,'--seed',str(a.seed),'--plugin-port',str(plugin_port),'--partition',world_name,'--world-name',world_name,'--mass-scale',str(spec.get('mass_scale',1.)),'--upstream',str(gazebo_upstream)]
        if a.record:worker_cmd.append('--record')
        if a.gui:worker_cmd.append('--gui')
        wl=open(out/'worker.log','w');logfiles.append(wl);worker=subprocess.Popen(worker_cmd,stdout=wl,stderr=subprocess.STDOUT)
        capture_cmd=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','root','--','/usr/bin/python3',str(snap/'network_capture.py'),'--output',str(out),'--udp-port',str(plugin_port),'--tcp-port',str(tcp_port)]
        cl=open(out/'network_worker.log','w');logfiles.append(cl);capture=subprocess.Popen(capture_cmd,stdout=cl,stderr=subprocess.STDOUT)
        time.sleep(2)
        sd=out/'sitl';sd.mkdir()
        defaults=','.join(str(p) for p in [ap_root/'Tools/autotest/default_params/copter.parm',ap_root/'Tools/autotest/default_params/gazebo-iris.parm',params])
        sitl_cmd=[str(ap_root/'build/sitl/bin/arducopter'),'--model','JSON','--speedup','1','--defaults',defaults,'--home','-35.363262,149.165237,584,0',f'-I{a.instance+3}','--sim-address=127.0.0.1','--sim-port-out',str(proxy_port),'--sim-port-in',str(proxy_port+1)]
        sl=open(out/'sitl.log','w');logfiles.append(sl);sitl=subprocess.Popen(sitl_cmd,cwd=sd,stdout=sl,stderr=subprocess.STDOUT,start_new_session=True)
        manifest={'argv':sys.argv,'sitl_command':sitl_cmd,'worker_command':worker_cmd,'capture_command':capture_cmd,'sitl_pid':sitl.pid,'worker_pid':worker.pid,
                  'proxy_pid':os.getpid(),'source_commit':subprocess.check_output(['git','-C',str(ap_root),'rev-parse','HEAD'],text=True).strip(),
                  'binary_sha256':sha(ap_root/'build/sitl/bin/arducopter'),'policy_sha256':sha(snap/'policy.npz') if policy else None,
                  'original_policy_path':str(a.policy) if a.policy else None,'calibration':cal,
                  'motor_order':'Gazebo [FR,RL,FL,RR] <- V2 policy indices [0,1,2,3]',
                  'actuator_conversion':'PWM_i=1100+(nominal_hover_PWM_i-1100)*sqrt(action_i+1); clamp 1100..1900',
                  'split':a.split,'scenario_spec':spec,'pulse_pwm':a.pulse_pwm if a.controller=='pulse' else None,'pulse_pattern':a.pulse_pattern,
                  'scope':'sim-only actuator proxy. Native PID for takeoff/land, bypassed during direct measurement. Official Gazebo motor joint velocity loops remain actuator physics.'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        deadline=time.monotonic()+45
        while conn is None:
            try:conn=mavutil.mavlink_connection(f'tcp:127.0.0.1:{tcp_port}',source_system=254,source_component=190)
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(.25)
        if conn.wait_heartbeat(timeout=45) is None:raise TimeoutError('No isolated SITL heartbeat')
        rec=Telemetry(conn,out);rec.event('connected')
        rec.wait(lambda:(out/'force_physics.jsonl').exists(),10,'ForceSystem did not initialize')
        rec.wait(lambda:(out/'network_capture_manifest.json').exists(),10,'Passive packet capture did not initialize')
        if capture.poll() is not None:raise RuntimeError('Packet capture exited before arming')
        readback={name:rec.param(name) for name in ['AHRS_EKF_TYPE','EK3_ENABLE','MOT_PWM_MIN','MOT_PWM_MAX']}
        for name,value in pid_profile['parameters'].items():
            readback[name]=rec.param(name)
            if abs(readback[name]-value)>1e-4:raise ValueError(f'PID parameter readback mismatch {name}')
        (out/'parameter_readback.json').write_text(json.dumps(readback,indent=2))
        for msgid,hz in [(30,50),(32,50),(36,50)]:command(conn,mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,msgid,1e6/hz)
        rec.wait(lambda:'LOCAL_POSITION_NED'in rec.latest and 'ATTITUDE'in rec.latest,75,'No local pose')
        deadline=time.monotonic()+90
        while True:
            conn.mav.set_mode_send(conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,4)
            command(conn,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1)
            try:rec.wait(lambda:bool(rec.latest.get('HEARTBEAT')) and rec.latest['HEARTBEAT'].base_mode&128,2,'Not armed yet');break
            except TimeoutError:
                if time.monotonic()>deadline:raise
        rec.event('armed')
        command(conn,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,3)
        rec.wait(lambda:rec.state()[0][2]>2.7,60,'Takeoff failed')
        settle=rec.boot_s
        while rec.boot_s-settle<5:
            rec.drain();send_position(conn,np.array([0.,0.,3.]),rec.boot_s);time.sleep(.02)
        t0=rec.boot_s;last=-1;previous=np.zeros(4,dtype=np.float32);failure=None;cal_samples=[]
        summary['measurement_boot_start_s']=t0
        initial_R=rec.state()[2];yaw_reference=float(np.arctan2(initial_R[1,0],initial_R[0,0]))
        if policy:policy.reset(yaw_reference)
        active_target=mission.target
        force_sequence=0;previous_force=None
        summary['yaw_reference_rad']=yaw_reference
        rec.event('measurement_start',position_enu=rec.state()[0].tolist())
        deadline=time.monotonic()+max(120,a.duration*4)
        while rec.boot_s-t0<a.duration:
            rec.drain()
            if proxy.error:raise RuntimeError(proxy.error)
            if time.monotonic()>deadline:raise TimeoutError('Sim measurement deadline')
            if time.monotonic()-rec.last_recv>3:raise TimeoutError('Stale MAVLink')
            t=rec.boot_s-t0
            if t-last<.019:time.sleep(.001);continue
            elapsed=.02 if last<0 else t-last
            last=t;pos,vel,R,rates=rec.state();target=active_target.copy();hb=rec.latest['HEARTBEAT']
            reached=mission.update(t,pos)
            if reached:rec.event('goal_reached',mission_event=reached)
            next_target=mission.target
            force=force_N(t,a.scenario,a.seed)
            if previous_force is None or not np.array_equal(force,previous_force):
                force_sequence+=1
                tmp=out/'FORCE_COMMAND.tmp';tmp.write_text(json.dumps({'seq':force_sequence,'t':t,'boot_s':rec.boot_s,'force_enu_N':force.tolist()}));tmp.replace(out/'FORCE_COMMAND.json')
                previous_force=force.copy()
            tmp=out/'TARGET.tmp';tmp.write_text(json.dumps({'target_enu':next_target.tolist(),'boot_s':rec.boot_s}));tmp.replace(out/'TARGET.json')
            send_position(conn,next_target,rec.boot_s,yaw_reference) # native output discarded in direct measurement
            action=None;obs=None;pulse_rotor=None
            if policy:
                if max(time.monotonic()-rec.received[k] for k in ['LOCAL_POSITION_NED','ATTITUDE'])>.25:raise TimeoutError('Stale control state')
                obs=policy.encoder.observe(pos,vel,R,rates,next_target,previous,elapsed);action=np.clip(policy.predict(obs),-1,1)
                if not np.isfinite(action).all():raise ValueError('Nonfinite direct action')
                proxy.set_action(action,cal['hover_pwm_gazebo_order'])
                previous=np.asarray(action,dtype=np.float32)
            elif a.controller=='pulse':
                for rotor in range(4):
                    if 5+2*rotor<=t<5.10+2*rotor:pulse_rotor=rotor
                if pulse_rotor is not None and proxy.latest_pwm is not None:
                    pwm=np.array(proxy.latest_pwm,dtype=float)
                    if a.pulse_pattern=='single':pwm[pulse_rotor]+=a.pulse_pwm
                    else:pwm+=np.array([1,1,-1,-1])*a.pulse_pwm*(1 if pulse_rotor%2==0 else -1)
                    pwm=np.rint(np.clip(pwm,1100,1900)).astype(int).tolist()
                    proxy.pwm=tuple(pwm);proxy.updated=time.monotonic();proxy.active=True
                else:proxy.active=False
            elif proxy.latest_pwm is not None and t>5:cal_samples.append(proxy.latest_pwm.copy())
            err=float(np.linalg.norm(pos-target));att=rec.latest['ATTITUDE']
            row={'t':t,'boot_s':rec.boot_s,'x_e':pos[0],'y_n':pos[1],'z_up':pos[2],'target_e':target[0],'target_n':target[1],'target_up':target[2],'error_m':err,
                 'roll':att.roll,'pitch':att.pitch,'yaw':att.yaw,'mode':hb.custom_mode,'armed':bool(hb.base_mode&128),
                 'velocity':vel.tolist(),'rotation':R.reshape(-1).tolist(),'body_rates':rates.tolist(),
                 'position':pos.tolist(),'target':target.tolist(),'action_target':next_target.tolist(),
                 'observation':obs.tolist() if obs is not None else None,
                 'action':action.tolist() if action is not None else None,'direct_pwm':proxy.pwm if proxy.active else None,
                 'pulse_rotor':pulse_rotor,'force_command_N':force.tolist(),
                 'yaw_error_rad':float(np.arctan2(np.sin(yaw_reference-np.arctan2(R[1,0],R[0,0])),np.cos(yaw_reference-np.arctan2(R[1,0],R[0,0]))))}
            rows.append(row)
            active_target=next_target.copy()
            if err>4 or pos[2]<.25 or pos[2]>10 or R[2,2]<.2 or not hb.base_mode&128 or hb.custom_mode!=4:
                failure='flight_envelope_or_mode';rec.event('flight_failed',state=row);break
        proxy.active=False;rec.event('measurement_end',failure=failure)
        force_sequence+=1
        tmp=out/'FORCE_COMMAND.tmp';tmp.write_text(json.dumps({'seq':force_sequence,'t':rec.boot_s-t0,'boot_s':rec.boot_s,'force_enu_N':[0,0,0]}));tmp.replace(out/'FORCE_COMMAND.json')
        summary.update(route_completed=mission.completion_time is not None if a.mission=='route' else None,route_completion_time_s=mission.completion_time,completion_time_s=mission.completion_time,mission_events=mission.events)
        if a.controller=='passthrough' and cal_samples:
            (out/'calibration.json').write_text(json.dumps({'hover_pwm_gazebo_order':np.mean(cal_samples,axis=0).tolist(),'n_samples':len(cal_samples),
                'source_run':str(out),'window':'measurement time >5 seconds','reference_fraction':.5,'nominal_only':a.scenario=='nominal'},indent=2))
        errors=np.array([r['error_m'] for r in rows]);summary.update(status='FAIL' if failure else 'PASS',failure=failure,duration_s=rows[-1]['t'],samples=len(rows),rmse_position_m=float(np.sqrt(np.mean(errors**2))),max_error_m=float(errors.max()),measurement_boot_start_s=t0)
        conn.mav.set_mode_send(conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
        try:rec.wait(lambda:bool(rec.latest.get('HEARTBEAT')) and not rec.latest['HEARTBEAT'].base_mode&128,60,'Landing timeout');summary['landed_disarmed']=True
        except TimeoutError:summary['landed_disarmed']=False
    except Exception:
        error=traceback.format_exc()
        controller_failure=('Stale direct action' in error or 'Nonfinite direct action' in error) and 'measurement_boot_start_s' in summary
        summary.update(status='FAIL' if controller_failure else 'ERROR',error=error)
        if controller_failure:summary['failure']='controller_action_invalid_or_stale'
        print(summary['error'],flush=True)
    finally:
        if proxy:proxy.active=False
        (out/'STOP_GAZEBO').write_text('stop')
        if sitl:
            try:os.killpg(sitl.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:sitl.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(sitl.pid,signal.SIGKILL);sitl.wait()
        if worker:
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=5)
        if capture:
            try:capture.wait(timeout=30)
            except subprocess.TimeoutExpired:capture.terminate();capture.wait(timeout=5)
        if proxy:proxy.close();summary['proxy']={'packets':proxy.count,'overridden':proxy.overridden,'sensor_packets':proxy.sensors,'error':proxy.error}
        if rec:rec.close()
        if conn:conn.close()
        for f in logfiles:f.close()
        if rows:
            with (out/'telemetry.jsonl').open('w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            with (out/'telemetry.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary.update(wall_duration_s=time.time()-started,utc_finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if a.controller=='direct' and summary.get('proxy',{}).get('overridden',0)==0:
            summary.update(status='ERROR',error='No observed actuator override')
        for required in ['groundtruth.jsonl','physics.jsonl']:
            if not (out/required).is_file() or (out/required).stat().st_size==0:
                summary.update(status='ERROR',error=f'No observed {required}')
        if proxy and proxy.error:summary.update(status='ERROR',error=proxy.error)
        finished=out/'gazebo_finished.json'
        if finished.is_file():summary['gazebo_worker']=json.loads(finished.read_text())
        if a.record and summary.get('gazebo_worker',{}).get('video_frames',0)==0:
            summary.update(status='ERROR',error='Requested camera recording contains zero frames')
        if not (out/'network.pcap.gz').exists():
            summary.update(status='ERROR',error='Required passive packet capture is missing')
        if rows:
            try:
                metrics=finalize(out,summary);summary.update(metrics)
                (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
            except Exception:
                summary.update(status='ERROR',error='Metric verification failed: '+traceback.format_exc())
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())

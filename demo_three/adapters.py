"""Unmodified controllers behind measured-coordinate and public actuator adapters."""
import json
import math
from pathlib import Path
import time
import numpy as np
from pymavlink import mavutil
from demo_three.mission import DemoMission
from demo_three.telemetry import Telemetry,command,send_position
from sim_joint.thrust_adapter import body_collective_to_preboost

HOVER_THRUST=.302651215314865
class VehicleController:
    def __init__(self,spec,output,policy=None,proxy=None,calibration=None):
        self.spec=spec;self.role=spec['role'];self.folder=Path(output)/self.role
        self.folder.mkdir(exist_ok=True);self.policy=policy;self.proxy=proxy;self.calibration=calibration
        self.conn=None;self.rec=None;self.origin=None;self.yaw=None;self.mission=None;self.last_boot=None;self.failed=None
        self.previous=np.zeros(4,np.float32);self.samples=0;self.commands=(self.folder/'commands.jsonl').open('x',buffering=1)
        self.last_input_pair=None
    def check_stop(self):
        if (self.folder.parent/'STOP_DEMO').exists():raise InterruptedError('Owned STOP_DEMO requested')
    def connect_and_takeoff(self,pid_parameters,timeout=600,simulation_timeout=90):
        end=time.monotonic()+timeout
        while self.conn is None:
            self.check_stop()
            try:self.conn=mavutil.mavlink_connection('tcp:127.0.0.1:'+str(self.spec['tcp_port']),source_system=250+self.spec['instance'],source_component=190)
            except OSError:
                if time.monotonic()>end:raise
                time.sleep(.2)
        heartbeat=None;heartbeat_deadline=min(end,time.monotonic()+45)
        while heartbeat is None and time.monotonic()<heartbeat_deadline:
            self.check_stop();heartbeat=self.conn.wait_heartbeat(timeout=.5)
        if heartbeat is None:raise TimeoutError(self.role+' no SITL heartbeat')
        identity=dict(heartbeat_source_system=heartbeat.get_srcSystem(),heartbeat_source_component=heartbeat.get_srcComponent(),
            connection_target_system=self.conn.target_system,expected_system=self.spec['instance']+1,parameter_name='MAV_SYSID')
        (self.folder/'connection_identity.json').write_text(json.dumps(identity,indent=2))
        if identity['heartbeat_source_system']!=identity['expected_system'] or self.conn.target_system!=identity['expected_system']:
            raise RuntimeError(self.role+' actual heartbeat source/target system differs from its assigned unique ID')
        self.rec=Telemetry(self.conn,self.folder,stop_file=self.folder.parent/'STOP_DEMO')
        for msgid,hz in ((30,50),(32,50),(83,50),(36,50)):
            command(self.conn,mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,msgid,1e6/hz)
        readback={k:self.rec.param(k) for k in (*pid_parameters,'AHRS_EKF_TYPE','EK3_ENABLE','GUID_OPTIONS','ATC_ANGLE_BOOST','MAV_SYSID')}
        if readback['MAV_SYSID']!=self.spec['instance']+1:raise RuntimeError('Unique MAV_SYSID readback differs')
        if any(abs(readback[k]-float(v))>1e-5 for k,v in pid_parameters.items()):raise RuntimeError(self.role+' frozen PID readback differs')
        if self.role=='hybrid' and abs(readback['ATC_ANGLE_BOOST']-1)>1e-6:raise RuntimeError('Hybrid requires existing native angle boost')
        (self.folder/'parameter_readback.json').write_text(json.dumps(readback,indent=2))
        self.rec.wait(lambda:'LOCAL_POSITION_NED' in self.rec.latest and 'ATTITUDE' in self.rec.latest,60,'No public measured state')
        ground=self.rec.state()[0]
        # Fixed preflight translation from measured ground pose and KNOWN model
        # spawn height. No ongoing Gazebo truth or disturbance enters any actor.
        self.origin=ground-np.array([0.,0.,.195])
        self.yaw=float(np.arctan2(self.rec.state()[2][1,0],self.rec.state()[2][0,0]))
        self.rec.event('task_coordinate_origin',origin_mav_enu=self.origin.tolist(),measured_ground_mav_enu=ground.tolist(),known_spawn_height_m=.195)
        sim_start=None;last_status=0.
        while time.monotonic()<end:
            self.check_stop()
            try:
                scene=json.loads((self.folder.parent/'SCENE_STATE.json').read_text())
                sim_now=float(scene['sim_time'])
                if sim_start is None:sim_start=sim_now
                if sim_now-sim_start>simulation_timeout:raise TimeoutError(self.role+' normal prearm exceeded actual simulation-time budget')
            except (OSError,json.JSONDecodeError,KeyError):sim_now=None
            if time.monotonic()-last_status>10:
                status=self.rec.latest.get('STATUSTEXT')
                print(self.role.upper()+': waiting normal arming; Gazebo '+str(None if sim_now is None else round(sim_now,2))+'s; '+str(getattr(status,'text','startup calibration')),flush=True)
                last_status=time.monotonic()
            self.conn.mav.set_mode_send(self.conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,4)
            command(self.conn,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1)
            try:self.rec.wait(lambda:bool(self.rec.latest.get('HEARTBEAT')) and self.rec.latest['HEARTBEAT'].base_mode&128,2,'Normal arm not yet accepted');break
            except TimeoutError:pass
        else:raise TimeoutError(self.role+' refused normal arming')
        command(self.conn,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,3.)
        self.rec.wait(lambda:self.task_state()[0][2]>2.7,max(1.,min(300.,end-time.monotonic())),self.role+' takeoff failed')
        self.rec.event('takeoff_completed')
    def task_state(self):
        pos,vel,R,rates=self.rec.state()
        return pos-self.origin,vel,R,rates
    def drain(self):self.rec.drain()
    def hold(self):
        send_position(self.conn,np.array([0.,0.,3.])+self.origin,self.rec.boot_s,self.yaw)
    def begin(self,mission,sim_t0,waypoint_radius_m=.25):
        self.mission=DemoMission(mission,waypoint_radius_m);self.previous.fill(0);self.last_boot=None;self.last_input_pair=None
        self.mission_epoch_boot_ms=None
        if self.policy:self.policy.reset(self.yaw)
        self.rec.event('common_measurement_start',world_sim_time=sim_t0,receive_cutoff=self.rec.n,yaw_reference=self.yaw)
    def set_mission_epoch(self,epoch_boot_ms):
        if self.mission_epoch_boot_ms is not None or self.samples!=0:
            raise RuntimeError('Common measured mission epoch may be bound once before any controlled sample')
        self.mission_epoch_boot_ms=int(epoch_boot_ms)
    def input_pair(self):
        return (int(self.rec.latest['LOCAL_POSITION_NED'].time_boot_ms),int(self.rec.latest['ATTITUDE'].time_boot_ms))
    def has_fresh_input_pair(self):
        if self.failed:return True
        pair=self.input_pair()
        return self.last_input_pair is None or all(current>previous for current,previous in zip(pair,self.last_input_pair))
    def step(self,elapsed,world_sim_time,slot_index=None,scheduled_sim_time=None):
        if self.failed:return None
        # Runtime already captured all three latest received-state prefixes.
        # Never re-drain one vehicle while another uses the shared frame group.
        now=time.monotonic()
        if max(now-self.rec.received[k] for k in ('LOCAL_POSITION_NED','ATTITUDE'))>.25:raise TimeoutError(self.role+' stale public state')
        pos,vel,R,rates=self.task_state();target=self.mission.target
        pair=self.input_pair()
        sample_boot_s=max(pair)*.001
        if self.mission_epoch_boot_ms is None:raise RuntimeError('Common actual-sensor mission epoch is not bound')
        mission_time_s=(max(pair)-self.mission_epoch_boot_ms)*.001
        dt=.02 if self.last_boot is None else sample_boot_s-self.last_boot
        if dt<=0:raise RuntimeError('Fresh shared frame group has non-increasing measured-state time')
        self.last_boot=sample_boot_s;action=None;observation=None;mapping=None;diagnostic=None;frd=None
        if self.last_input_pair is not None and not all(current>prior for current,prior in zip(pair,self.last_input_pair)):
            raise RuntimeError('A scheduled observation must have fresh position AND attitude timestamps')
        self.last_input_pair=pair
        if self.role=='pid':
            send_position(self.conn,target+self.origin,sample_boot_s,self.yaw)
        elif self.role=='hybrid':
            prediction=self.policy.predict_enu(pos,vel,R,rates,target,dt=dt)
            action=np.clip(np.asarray(prediction['action'],float),-1,1);observation=prediction['observation']
            age=now-self.rec.received['ATTITUDE']
            mapping=body_collective_to_preboost(float(np.clip((action[0]+1)*HOVER_THRUST,0,1)),float(R[2,2]),float(age),.1)
            frd=action[1:]*np.array([2.,-2.,-1.])
            self.conn.mav.set_attitude_target_send(int(sample_boot_s*1000)&0xffffffff,self.conn.target_system,self.conn.target_component,128,[0.,0.,0.,0.],*frd,mapping.preboost_thrust)
            if hasattr(self.policy,'diagnostics'):diagnostic=self.policy.diagnostics()
        else:
            observation=self.policy.encoder.observe(pos,vel,R,rates,target,self.previous,dt)
            action=np.clip(np.asarray(self.policy.predict(observation),float),-1,1)
            if not np.isfinite(action).all():raise ValueError('Nonfinite unchanged Direct111 action')
            pwm=self.proxy.set_action(action,self.calibration['hover_pwm_gazebo_order'])
            diagnostic=dict(motor_pwm=pwm,interface='simulation-only native motor packet replacement',policy_unchanged=True)
            send_position(self.conn,target+self.origin,sample_boot_s,self.yaw)
        previous=self.previous.copy()
        if action is not None:self.previous=action.astype(np.float32)
        event=self.mission.update(mission_time_s,pos)
        item=dict(world_sim_time=world_sim_time,measurement_t=elapsed,mav_boot_s=self.rec.boot_s,control_dt_s=dt,role=self.role,
            waypoint_radius_m=self.mission.waypoint_radius_m,arrival_dwell_s=self.mission.acceptance['arrival_dwell_s'],
            observation_boot_s=sample_boot_s,dt_source='max_actual_position_attitude_boot_ms',
            mission_epoch_boot_ms=self.mission_epoch_boot_ms,mission_epoch_boot_s=self.mission_epoch_boot_ms*.001,
            mission_time_s=mission_time_s,mission_clock_source='actual_sensor_pair_relative_to_one_common_raw_epoch',
            shared_control_slot=slot_index,scheduled_control_sim_time=scheduled_sim_time,
            measured_input_pair_boot_ms=list(pair),
            receive_cutoff=self.rec.n,position_record_index=self.rec.message_indices['LOCAL_POSITION_NED'],attitude_record_index=self.rec.message_indices['ATTITUDE'],
            task_position=pos,velocity=vel,rotation=R,body_rates=rates,target_before_update=target,target_next=self.mission.target,
            actual_previous_action=previous,action=action,observation=observation,adapter=mapping.to_dict() if mapping else None,
            raw_mav_position=pos+self.origin,task_origin_mav_enu=self.origin,
            world_position_from_fixed_translation=pos+np.asarray(self.spec['offset_enu']),
            target_world_enu=target+np.asarray(self.spec['offset_enu']),target_mav_enu=target+self.origin,
            mavlink_type_mask=128 if self.role=='hybrid' else 2552,native_rates_frd=frd,
            guid_thrust=mapping.preboost_thrust if mapping else None,
            actual_send_arguments=dict(message='SET_ATTITUDE_TARGET' if self.role=='hybrid' else 'SET_POSITION_TARGET_LOCAL_NED',
                time_boot_ms=int(sample_boot_s*1000)&0xffffffff,target_system=self.conn.target_system,target_component=self.conn.target_component,
                type_mask=128 if self.role=='hybrid' else 2552,
                position_ned=(np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])@(target+self.origin)).tolist() if self.role!='hybrid' else None,
                yaw_ned=math.pi/2-self.yaw if self.role!='hybrid' else None,quaternion=[0.,0.,0.,0.] if self.role=='hybrid' else None,
                rates_frd=frd,thrust=mapping.preboost_thrust if mapping else None),
            diagnostics=diagnostic,goal_event=event,mode=self.rec.latest['HEARTBEAT'].custom_mode,armed=bool(self.rec.latest['HEARTBEAT'].base_mode&128))
        def plain(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [plain(x) for x in v]
            return v
        self.commands.write(json.dumps(plain(item),allow_nan=False)+'\n');self.samples+=1
        hb=self.rec.latest['HEARTBEAT']
        if np.linalg.norm(pos-target)>4 or pos[2]<.25 or pos[2]>10 or R[2,2]<.2 or not hb.base_mode&128 or hb.custom_mode!=4:
            raise RuntimeError(self.role+' observed flight envelope/mode failure')
        return item
    def fail_and_land(self,reason):
        self.failed=str(reason)
        if self.proxy:self.proxy.active=False
        if self.rec:self.rec.event('demo_controller_failed',reason=self.failed,fallback='native LAND; measurement failure remains')
        if self.conn:self.conn.mav.set_mode_send(self.conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
    def land(self):
        if self.proxy:self.proxy.active=False
        if self.rec:self.rec.event('measurement_finished_native_land')
        if self.conn:self.conn.mav.set_mode_send(self.conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
    def close(self):
        if self.proxy:self.proxy.active=False
        if self.conn:
            try:self.conn.mav.set_mode_send(self.conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
            except Exception:pass
        self.commands.close()
        if self.rec:self.rec.close()
        if self.conn:self.conn.close()

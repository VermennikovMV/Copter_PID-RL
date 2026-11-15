#!/usr/bin/env python3
"""Shared phase-expert v29 runtime, unchanged v2.1 receive/body-collective interface.

Dormant preparation: a future root-frozen case must explicitly bind its basis kind.

Run under Ubuntu-24.04 with Python >=3.11 containing numpy/pymavlink.
The Gazebo worker uses Ubuntu's installed Harmonic Python bindings.
Ubuntu's worker Python3.10 cannot parse the unchanged frozen base74 policy.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_joint.thrust_adapter import INTERFACE_VERSION, SEMANTICS, ThrustAdapterError, body_collective_to_preboost
RECEIVE_EVIDENCE_VERSION = 'joint-receive-index-v2.1'
OBSERVATION_CONTRACT = 'measured-shared-phase-expert-v29.0'
OBSERVATION_SIZE = 80
BASIS_KINDS = ('velocity_residual',)
CONDITIONING_MODES = ('zero_route_integral',)

F0_PATH='artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/zero_route_integral/policy.npz'
F0_SHA='f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'
ROUTE_EXPERT_PATH='artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.npz'
ROUTE_EXPERT_SHA='ce86bb56d7e0f5c1775266d4a45cd3417e979303d27cc662a58e8070bf34e345'
ROUTE_ZIP_PATH='artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.zip'
ROUTE_ZIP_SHA='62eaf3fb179f915a91cb15686720c06c5dd8cc80595d1308279ee1895ce13c11'
TRAINING_PROOF_PATH='audit/review_joint_prior_ablation_v28_pilot_01.json'
TRAINING_PROOF_SHA='8b9936e55e286587ae8ee66d9c487bbc08471a1d0abef9fcad53b8d858e1e2a4'
EXPERT_SOURCE_SHA256=(F0_SHA,ROUTE_EXPERT_SHA)
PHASE_POLICY_SHA='0827ff86abd77b2945b1f5a5f845ccae32a5695396e5843a1eed143832a91cf7'
PHASE_INIT_SHA='fac9e760f1dea3aa7158695037607840233d24a98b3a127175bcaecff6858460'

def expected_bundle():
    conditioning=dict(mode='zero_route_integral',integral_input_columns=[22,23,24],
        raw_old_contract='measured-memory-route-basis-v4.0',raw_encoder_calls_per_observation=1,
        gating='own observed-target history valid_leg',reset_timing='observed target change before current dt integration',
        world_integral_rule='original first-sample skip; error*dt; XYnorm and separate Z clamp',
        other77_inputs_bit_exact=True,hidden_or_future_inputs=False)
    return dict(schema='joint-causal-two-expert-bundle-v29.0',observation_contract=OBSERVATION_CONTRACT,
        architecture='hybrid',conditioning=conditioning,shared_encoder=True,
        selector=dict(predicate='CURRENT shared encoder valid_leg AFTER observe',false_expert=0,true_expert=1,
            latched=False,reset_on_switch=False,uses_mission_kind_or_future_waypoint=False),
        experts=[dict(index=0,role='F0_without_observed_leg',policy=dict(path=F0_PATH,sha256=F0_SHA)),
            dict(index=1,role='prior0_FINAL_with_observed_leg',policy=dict(path=ROUTE_EXPERT_PATH,sha256=ROUTE_EXPERT_SHA),
                model_zip=dict(path=ROUTE_ZIP_PATH,sha256=ROUTE_ZIP_SHA),
                training_proof=dict(path=TRAINING_PROOF_PATH,sha256=TRAINING_PROOF_SHA,pointer='/candidate_claims/prior0/final'))],
        new_optimizer_steps=0,new_training_steps=0,
        development_choice='Manual causal composition after known v28 quality; no new PPO training or inherited cell acceptance')

ENU_FROM_NED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FRD_FROM_FLU = np.diag([1., -1., -1.])

class ControllerFailure(RuntimeError):
    """Controller output failure belongs to the evaluated controller, not infra."""


def identification_signal(t,profile='fit'):
    """Four sequential 20-second balanced multisines after 5-second baseline.

    Returns the physical FLU angular-rate offset rad/s, or normalized raw a0
    offset for thrust. It is recorded separately and added to a holding PD.
    """
    index=int((t-5)//20)
    if t<5 or index>=4: return 'settle',0.
    local=t-5-index*20
    if index<3:
        frequencies=[.5,1.1,2.3,4.7] if profile=='fit' else [.7,1.7,3.7,6.1]
        value=sum(amp*math.sin(2*math.pi*freq*local) for amp,freq in zip([.15,.12,.08,.06],frequencies))
    else:
        frequencies=[.35,.8,1.6,3.] if profile=='fit' else [.55,1.15,2.15,4.15]
        value=sum(amp*math.sin(2*math.pi*freq*local) for amp,freq in zip([.04,.025,.015,.01],frequencies))
    return ['roll','pitch','yaw','thrust'][index],value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_phase_bundle(path,expected_kind,expected_mode,expected_bundle_sha256):
    if expected_kind!='velocity_residual' or expected_mode!='zero_route_integral':
        raise ValueError('Explicit fixed phase-expert basis/zeroI mode required')
    if (not isinstance(expected_bundle_sha256,str) or len(expected_bundle_sha256)!=64
            or any(c not in '0123456789abcdef' for c in expected_bundle_sha256)
            or digest(path)!=expected_bundle_sha256):
        raise ValueError('Externally bound exact expert bundle SHA required')
    if json.loads(Path(path).read_text(encoding='utf-8-sig'))!=expected_bundle():raise ValueError('Exact two-expert causal bundle required')
    if digest(ROOT/'quadrl_phase_expert_v29/policy.py')!=PHASE_POLICY_SHA:raise ValueError('Frozen phase-policy source changed')
    from quadrl_phase_expert_v29.policy import NumpyPhaseExpertPolicy
    policy=NumpyPhaseExpertPolicy(path)
    if policy.contract!=OBSERVATION_CONTRACT or policy.encoder.config.size!=80 or len(policy.experts)!=2:
        raise ValueError('Strict shared80/two-stateless-expert topology required')
    return policy

def phase_diagnostics(policy):
    def plain(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    return plain(policy.diagnostics())

def freeze_memory_package(source_dir):
    """Exact original package layout keeps memory policy ROOT/provenance honest."""
    package = source_dir/'memory_package'
    for name in ('quadrl','quadrl_path_v7','quadrl_route_basis_v14','quadrl_route_memory_v20','quadrl_route_memory_train_v21','quadrl_phase_expert_v29'):
        for filename in ('__init__.py','policy.py'):
            relative = Path(name)/filename
            target = package/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative,target)
            if digest(ROOT/relative) != digest(target):
                raise RuntimeError('Memory package changed during snapshot')
    relative = Path('artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz')
    target = package/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/relative,target)
    if digest(target) != '68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5':
        raise RuntimeError('Exact original source68 archive required')


    relative=Path('artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.zip')
    target=package/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/relative,target)
    if digest(target)!='0df1b6cd2811f5196395e8b707a3eee0387c8b0050375479a912511df779bf40':raise RuntimeError('Initialization ZIP changed while archiving')


    for relative,expected in ((F0_PATH,F0_SHA),(ROUTE_EXPERT_PATH,ROUTE_EXPERT_SHA),
            (ROUTE_ZIP_PATH,ROUTE_ZIP_SHA),(TRAINING_PROOF_PATH,TRAINING_PROOF_SHA)):
        target=package/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
        if digest(target)!=expected:raise RuntimeError('Frozen phase expert source/proof changed')


def load_archived_route_policy(source_dir, policy_path, expected_kind, expected_mode, expected_bundle_sha256):
    """Import exact archived packages, including the real original source NPZ."""
    if expected_kind!='velocity_residual' or expected_mode not in CONDITIONING_MODES:
        raise ValueError('Wrong explicit trained-memory basis/mode')
    package = source_dir/'memory_package'
    names = ('quadrl','quadrl_path_v7','quadrl_route_basis_v14','quadrl_route_memory_v20','quadrl_route_memory_train_v21','quadrl_phase_expert_v29')
    aliases = [name for name in sys.modules if any(name == p or name.startswith(p+'.') for p in names)]
    previous = {name:sys.modules.pop(name) for name in aliases}
    sys.path.insert(0,str(package))
    try:
        import importlib
        module = importlib.import_module('quadrl_phase_expert_v29.policy')
        if Path(module.__file__).resolve() != (package/'quadrl_phase_expert_v29/policy.py').resolve():
            raise ValueError('Archived memory package was not imported')
        if digest(package/'artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.zip')!='0df1b6cd2811f5196395e8b707a3eee0387c8b0050375479a912511df779bf40':raise ValueError('Archived initialization ZIP mismatch')
        if digest(policy_path)!=expected_bundle_sha256:raise ValueError('Archived bundle SHA differs')
        policy = module.NumpyPhaseExpertPolicy(policy_path)
    finally:
        sys.path.remove(str(package))
        for name in list(sys.modules):
            if any(name == p or name.startswith(p+'.') for p in names):sys.modules.pop(name)
        sys.modules.update(previous)
    if policy.contract != OBSERVATION_CONTRACT or policy.encoder.config.size != OBSERVATION_SIZE:
        raise ValueError('Archived memory80 contract mismatch')
    return policy


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp,cp*sr,cp*cr]])


class Telemetry:
    def __init__(self, conn, output):
        self.conn = conn
        self.raw = open(output/'mavlink.jsonl', 'w', buffering=1)
        self.events = open(output/'events.jsonl', 'w', buffering=1)
        self.latest = {}
        self.boot_s = 0.
        self.n = 0
        self.last_recv = time.monotonic()
        self.last_hb = 0.
        self.received = {}
        self.message_indices = {}

    def event(self, event, **fields):
        item = {'event':event, 'wall_time':time.time(), 'boot_s':self.boot_s, **fields}
        self.events.write(json.dumps(item,default=str)+'\n')
        print(json.dumps(item,default=str),flush=True)

    def drain(self):
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None: break
            if msg.get_type() == 'BAD_DATA': continue
            now = time.monotonic()
            self.last_recv = now
            self.n += 1
            self.latest[msg.get_type()] = msg
            self.received[msg.get_type()] = now
            self.message_indices[msg.get_type()] = self.n
            if hasattr(msg,'time_boot_ms'): self.boot_s=max(self.boot_s,msg.time_boot_ms/1000.)
            self.raw.write(json.dumps({'wall_time':time.time(),'message':msg.to_dict(),'record_index':self.n,'receive_monotonic_s':now},default=str)+'\n')
        if time.monotonic()-self.last_hb > 1:
            self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
            self.last_hb=time.monotonic()

    def wait(self, predicate, timeout, description):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.drain()
            if predicate(): return
            time.sleep(.01)
        raise TimeoutError(description)

    def param(self, name, value=None):
        for _ in range(4):
            self.latest.pop('PARAM_VALUE',None)
            if value is None:
                self.conn.mav.param_request_read_send(self.conn.target_system,self.conn.target_component,name.encode(),-1)
            else:
                self.conn.mav.param_set_send(self.conn.target_system,self.conn.target_component,name.encode(),float(value),mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            end=time.monotonic()+1.5
            while time.monotonic()<end:
                self.drain()
                m=self.latest.get('PARAM_VALUE')
                if m and m.param_id.strip('\x00')==name and (value is None or abs(m.param_value-value)<1e-4):
                    return float(m.param_value)
                time.sleep(.005)
        raise RuntimeError('Missing parameter readback '+name)

    def state(self):
        p=self.latest.get('LOCAL_POSITION_NED'); att=self.latest.get('ATTITUDE')
        if p is None or att is None: raise RuntimeError('Missing position or attitude')
        pos=ENU_FROM_NED@np.array([p.x,p.y,p.z])
        vel=ENU_FROM_NED@np.array([p.vx,p.vy,p.vz])
        R=ENU_FROM_NED@rotation(att.roll,att.pitch,att.yaw)@FRD_FROM_FLU
        omega=np.array([att.rollspeed,-att.pitchspeed,-att.yawspeed])
        return pos,vel,R,omega

    def close(self):
        self.raw.close(); self.events.close()


def command(conn, cmd, *values):
    values=list(values)+[0.]*(7-len(values))
    conn.mav.command_long_send(conn.target_system,conn.target_component,cmd,0,*values)


def send_position(conn, target_enu, boot_s, yaw_enu=None):
    x,y,z=ENU_FROM_NED@target_enu
    mask=3576 if yaw_enu is None else 2552
    yaw=0 if yaw_enu is None else math.pi/2-yaw_enu
    conn.mav.set_position_target_local_ned_send(int(boot_s*1000)&0xffffffff,conn.target_system,conn.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mask,x,y,z,0,0,0,0,0,0,yaw,0)


def pd_action(pos,vel,R,omega,target,previous):
    # Explicit analytic diagnostic controller; it is never reported as RL.
    acceleration=np.clip(1.4*(target-pos)-1.8*vel,[-3,-3,-4],[3,3,4])+np.array([0,0,9.80665])
    desired_z=acceleration/np.linalg.norm(acceleration)
    desired_x=np.array([0,1,0.]); desired_y=np.cross(desired_z,desired_x); desired_y/=np.linalg.norm(desired_y)
    desired_x=np.cross(desired_y,desired_z)
    Rd=np.column_stack([desired_x,desired_y,desired_z])
    error=.5*(np.cross(R[:,0],Rd[:,0])+np.cross(R[:,1],Rd[:,1])+np.cross(R[:,2],Rd[:,2]))
    rate=4*(R.T@error)
    thrust=np.dot(acceleration,R[:,2])/(2*9.80665)
    return np.clip(np.r_[2*thrust-1,rate/np.array([2,2,1])],-1,1)


def main():
    p=argparse.ArgumentParser()
    from quadrl.protocol import Mission,scenario_spec,force_N
    p.add_argument('--controller',choices=['pid','hybrid','rate_pd'],default='pid')
    p.add_argument('--scenario',choices=['nominal','force_mild','force_strong','gust','mass_20','mass_40'],default='nominal')
    p.add_argument('--seed','--env-seed',dest='seed',type=int,default=82001)
    p.add_argument('--train-seed',type=int)
    p.add_argument('--duration',type=float,help='Defaults30s hover,40s route; shortened runs are development only')
    p.add_argument('--split',choices=['development','validation','heldout','demo'],default='development')
    p.add_argument('--freeze-file',type=Path)
    p.add_argument('--pid-profile',type=Path,help='Frozen JSON label/parameters override for native outer-loop gains')
    p.add_argument('--instance',type=int,choices=[0,1,2],default=0,help='Isolated lanes:0=5760/9002,1=5770/9012,2=5780/9022; lane2 conflicts with direct SITL and requires coordination')
    p.add_argument('--mission',choices=['hover','route'],default='hover')
    p.add_argument('--expert-bundle',type=Path,help='Explicit strict two-expert JSON; this is not an NPZ actor')
    p.add_argument('--bundle-sha256',help='Exact externally bound bundle SHA before launch')
    p.add_argument('--conditioning-mode',choices=CONDITIONING_MODES,required=True,help='Fixed shared zero_route_integral preprocessing')
    p.add_argument('--basis-kind',choices=BASIS_KINDS,required=True,help='Explicit future case binding; must exactly match80 policy metadata')
    p.add_argument('--hover-thrust',type=float,default=.302651215314865,help='Fixed nominal calibration from smoke_pid_101 ATTITUDE_TARGET, boot 40..50s')
    p.add_argument('--thrust-semantics',choices=SEMANTICS,default='legacy_preboost',help='Explicit NN collective interface; native PID is unchanged')
    p.add_argument('--adapter-attitude-max-age-s',type=float,default=.1,help='Public measured attitude freshness limit for body_collective; at most 0.1s')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ardupilot',type=Path,default=Path('[external-home]/vkr_sim/ardupilot'))
    p.add_argument('--gazebo-upstream',type=Path,default=Path('[external-home]/ardupilot_gazebo'))
    p.add_argument('--gui',action='store_true')
    p.add_argument('--record',action='store_true')
    a=p.parse_args()
    if not math.isfinite(a.adapter_attitude_max_age_s) or not 0<a.adapter_attitude_max_age_s<=.1:p.error('Adapter attitude age limit must be in (0,0.1] seconds')
    # A campaign owner can yield the simulation lane between flights without
    # interrupting an active vehicle or replacing any recorded attempt.
    while (a.output.parent/'PAUSE').exists():time.sleep(.5)
    mission=Mission(a.mission);spec=scenario_spec(a.scenario,a.seed)
    if a.split=='validation' and a.seed not in range(82001,82009):p.error('Validation environment seeds are82001..82008 in the prospective protocol')
    tcp_port=5760+10*a.instance;plugin_port=9002+10*a.instance
    world_name='vkr_v2' if a.instance==0 else f'vkr_v2_lane{a.instance}'
    pid_profile=json.loads(a.pid_profile.read_text()) if a.pid_profile else {'label':'native_default','parameters':{}}
    if any(not name.startswith('PSC_NE_') for name in pid_profile['parameters']):p.error('PID profile may change only PSC_NE outer-loop parameters')
    if a.duration is None:a.duration=mission.horizon_s
    if a.split=='heldout' and (a.freeze_file is None or not a.freeze_file.is_file()):p.error('Heldout runs require an existing frozen protocol/model manifest --freeze-file')
    if a.split=='heldout':
        # An original frozen contract does not freeze this new interface.
        joint_freeze=json.loads(a.freeze_file.read_text())
        required_joint_sources=['sim_joint/run_experiment_v8_phase_expert.py','sim_joint/thrust_adapter.py','quadrl/policy.py','quadrl_path_v7/policy.py','quadrl_route_basis_v14/policy.py','quadrl_route_memory_v20/policy.py','quadrl_route_memory_train_v21/policy.py','quadrl_phase_expert_v29/policy.py','quadrl_phase_expert_v29/__init__.py',F0_PATH,ROUTE_EXPERT_PATH,ROUTE_ZIP_PATH,TRAINING_PROOF_PATH,'artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.zip','artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz']
        if joint_freeze.get('joint_interface_version')!=INTERFACE_VERSION or joint_freeze.get('joint_receive_evidence_version')!=RECEIVE_EVIDENCE_VERSION or joint_freeze.get('joint_thrust_semantics')!=a.thrust_semantics or joint_freeze.get('joint_adapter_attitude_max_age_s')!=a.adapter_attitude_max_age_s or joint_freeze.get('joint_observation_contract')!=OBSERVATION_CONTRACT or joint_freeze.get('joint_observation_size')!=OBSERVATION_SIZE or joint_freeze.get('joint_bundle_sha256')!=a.bundle_sha256 or joint_freeze.get('joint_conditioning_mode')!=a.conditioning_mode or joint_freeze.get('joint_basis_kind')!=a.basis_kind or any(joint_freeze.get('joint_runtime_sha256',{}).get(name)!=digest(ROOT/name) for name in required_joint_sources):p.error('Heldout joint Route80 v4 requires frozen interface, receive evidence version, observation contract/size, semantics, attitude age limit and exact new source hashes')
    if a.split=='heldout' and a.duration!=mission.horizon_s:p.error('Heldout duration must equal the common mission horizon')
    if a.controller=='hybrid' and (a.expert_bundle is None or not a.expert_bundle.is_file()): p.error('--expert-bundle must point to a strict two-expert JSON')
    policy=None
    if a.controller=='hybrid':
        policy=validate_phase_bundle(a.expert_bundle,a.basis_kind,a.conditioning_mode,a.bundle_sha256)
        if policy.architecture!='hybrid':p.error('Expected hybrid phase-expert policy')
        # Strict phase predict requires an actual current shared observation.
        # Topology/schema are validated above; no synthetic observe/NN probe.
    contract_receipt=None
    if a.split=='heldout':
        from contract import verify_contract
        try:
            contract_receipt=verify_contract(a.freeze_file,backend='gazebo',controller=a.controller,env_seed=a.seed,mission=a.mission,scenario=a.scenario,duration_s=a.duration,policy=a.expert_bundle,train_seed=a.train_seed,pid_profile=a.pid_profile,hover_thrust=a.hover_thrust,external_paths={'ardupilot_binary':a.ardupilot/'build/sitl/bin/arducopter','default_copter':a.ardupilot/'Tools/autotest/default_params/copter.parm','default_gazebo':a.ardupilot/'Tools/autotest/default_params/gazebo-iris.parm','gazebo_plugin':a.gazebo_upstream/'build/libArduPilotPlugin.so','iris_base_sdf':a.gazebo_upstream/'models/iris_with_standoffs/model.sdf','iris_control_sdf':a.gazebo_upstream/'models/iris_with_ardupilot/model.sdf'})
        except (ValueError,KeyError,OSError) as exc:p.error(str(exc))
    output=a.output.resolve()
    # Gazebo transport partitions do not isolate the JSON UDP socket. A leftover
    # worker can shareSO_REUSEADDR and emit cachedFDM even when TCP is free.
    udp_probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:udp_probe.bind(('127.0.0.1',plugin_port))
    except OSError as exc:p.error(f'JSON UDP{plugin_port} already bound; inspect existing simulator process before starting: {exc}')
    finally:udp_probe.close()
    if output.exists(): p.error('Output already exists; use a fresh path to preserve evidence')
    output.mkdir(parents=True)
    original_policy_path=str(a.expert_bundle.resolve()) if a.expert_bundle is not None else None
    source_dir=output/'source_snapshot'; source_dir.mkdir()
    for source in [Path(__file__),ROOT/'sim_joint/thrust_adapter.py',ROOT/'sim/gazebo_worker.py',ROOT/'sim/network_capture.py',ROOT/'quadrl/policy.py',ROOT/'quadrl/protocol.py',ROOT/'sim/metrics.py',ROOT/'sim/finalize_metrics.py',ROOT/'sim/pcap_clock.py',ROOT/'contract.py']:
        if source.exists(): shutil.copy2(source,source_dir/source.name)
    shutil.copy2(ROOT/'quadrl_path_v7/policy.py',source_dir/'path_policy.py')
    shutil.copy2(ROOT/'quadrl_route_basis_v14/policy.py',source_dir/'route_policy.py')
    native_source_dir=source_dir/'native_angle_boost';native_source_dir.mkdir()
    for relative in ['ArduCopter/mode_guided.cpp','libraries/AC_AttitudeControl/AC_AttitudeControl.cpp','libraries/AC_AttitudeControl/AC_AttitudeControl_Multi.cpp']:
        shutil.copy2(a.ardupilot/relative,native_source_dir/Path(relative).name)
    archived_force=source_dir/'force_plugin';(archived_force/'build').mkdir(parents=True)
    for name in ['ForceSystem.cc','CMakeLists.txt','build/libVkrForceSystem.so']:
        shutil.copy2(ROOT/'sim/force_plugin'/name,archived_force/name)
    freeze_memory_package(source_dir)
    if a.expert_bundle is not None:
        # Freeze bytes before launch; later training cannot mutate this experiment.
        shutil.copy2(a.expert_bundle,source_dir/'expert_bundle.json')
        a.expert_bundle=source_dir/'expert_bundle.json'
        if contract_receipt is not None and digest(a.expert_bundle)!=contract_receipt['policy_sha256']:raise RuntimeError('Policy changed while freezing run snapshot')
        policy=load_archived_route_policy(source_dir,a.expert_bundle,a.basis_kind,a.conditioning_mode,a.bundle_sha256)
    if a.freeze_file is not None:shutil.copy2(a.freeze_file,source_dir/'frozen_protocol.json')
    if a.pid_profile is not None:shutil.copy2(a.pid_profile,source_dir/'pid_profile.json')
    sys.path.insert(0,str(source_dir))
    from finalize_metrics import finalize
    start_wall=time.time()
    sitl=None; gazebo=None; capture=None; rec=None; conn=None; files=[]
    summary={'status':'ERROR','controller':a.controller,'scenario':a.scenario,'seed':a.seed,'mission':a.mission,'requested_duration_s':a.duration,'utc_started':dt.datetime.now(dt.timezone.utc).isoformat(),'output':str(output)}
    summary.update(split=a.split,env_seed=a.seed,train_seed=a.train_seed,scenario_spec=spec)
    summary['model_family']='gazebo_iris_ardupilot'
    summary.update(tcp_port=tcp_port,plugin_port=plugin_port,pid_profile=pid_profile)
    summary['contract_receipt']=contract_receipt
    summary['joint_interface']={'version':INTERFACE_VERSION,'receive_evidence_version':RECEIVE_EVIDENCE_VERSION,'thrust_semantics':a.thrust_semantics,'effective_semantics':'native_position_target' if a.controller=='pid' else a.thrust_semantics,'runner_sha256':digest(source_dir/Path(__file__).name),'adapter_sha256':digest(source_dir/'thrust_adapter.py'),'adapter_attitude_max_age_s':a.adapter_attitude_max_age_s,'original_runner_sha256':digest(ROOT/'sim/run_experiment.py'),'frozen_runtime_scope':'Original contract checks are inherited dependencies only; this is the separate joint shared phase-expert v29 runner with v2.1 receive evidence, not execution of the original frozen runner.'}
    summary['joint_observation']={'contract':OBSERVATION_CONTRACT,'size':OBSERVATION_SIZE,'base_size':74,'path_size':77,'basis_kind':a.basis_kind,'route_basis_config':{'basis_kind':a.basis_kind,'position_gain_s_inv':1.5,'reference_speed_cap_mps':2.,'velocity_residual_scale_mps':1.},'base_policy_sha256':digest(source_dir/'policy.py'),'path_policy_sha256':digest(source_dir/'path_policy.py'),'route_policy_sha256':digest(source_dir/'route_policy.py'),'memory_policy_sha256':digest(source_dir/'memory_package/quadrl_route_memory_v20/policy.py'),'source68_policy_sha256':digest(source_dir/'memory_package/artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz'),'train_policy_sha256':digest(source_dir/'memory_package/quadrl_route_memory_train_v21/policy.py'),'initial_policy_zip_sha256':digest(source_dir/'memory_package/artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.zip'),'conditioning_mode':a.conditioning_mode,'expert_bundle_sha256':digest(a.expert_bundle) if a.expert_bundle else None,'expert_source_paths':[F0_PATH,ROUTE_EXPERT_PATH],'expert_source_sha256':list(EXPERT_SOURCE_SHA256),'route_model_zip_sha256':ROUTE_ZIP_SHA,'training_proof_sha256':TRAINING_PROOF_SHA,'phase_policy_sha256':digest(source_dir/'memory_package/quadrl_phase_expert_v29/policy.py'),'shared_encoder':True,'selector':'CURRENT shared encoder valid_leg AFTER observe','actor_input_scope':'Manual two-expert composition after known v28 quality; no new SGD. One measured shared RAW80/zeroI80 history and actual previous mixed action; current valid_leg AFTER observe selects F0false/prior0FINALtrue. No latch/reset/mission oracle; no inherited acceptance.'}
    try:
        sock=socket.socket(); sock.settimeout(.2)
        if sock.connect_ex(('127.0.0.1',tcp_port))==0: raise RuntimeError(f'TCP{tcp_port} occupied: another SITL may be running')
        sock.close()
        capture_cmd=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','root','--','/usr/bin/python3',str(source_dir/'network_capture.py'),'--output',str(output),'--udp-port',str(plugin_port),'--tcp-port',str(tcp_port)]
        capture_log=open(output/'network_worker.log','w'); files.append(capture_log)
        capture=subprocess.Popen(capture_cmd,stdout=capture_log,stderr=subprocess.STDOUT)
        params=output/'experiment.parm'
        params.write_text('FRAME_CLASS 1\nFRAME_TYPE 1\nSIM_SPEEDUP 1\nFS_THR_ENABLE 0\nFS_GCS_ENABLE 0\nFENCE_ENABLE 0\nLOG_DISARMED 1\nLOG_BITMASK 65535\nGUID_OPTIONS 8\nMOT_PWM_MIN 1100\nMOT_PWM_MAX 1900\nMOT_THST_HOVER 0.5\nMOT_HOVER_LEARN 0\n')
        with params.open('a') as profile_file:
            for key,value in pid_profile['parameters'].items():profile_file.write(f'{key} {float(value)}\n')
        worker_cmd=['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','--','/usr/bin/python3',str(source_dir/'gazebo_worker.py'),'--output',str(output),'--scenario',a.scenario,'--seed',str(a.seed),'--mass-scale',str(spec.get('mass_scale',1.))]
        worker_cmd+=['--plugin-port',str(plugin_port),'--partition',world_name,'--world-name',world_name]
        worker_cmd+=['--upstream',str(a.gazebo_upstream)]
        if a.gui: worker_cmd.append('--gui')
        if a.record: worker_cmd.append('--record')
        worker_log=open(output/'gazebo_worker.log','w',buffering=1); files.append(worker_log)
        gazebo=subprocess.Popen(worker_cmd,stdout=worker_log,stderr=subprocess.STDOUT)
        time.sleep(2)
        sitl_dir=output/'sitl'; sitl_dir.mkdir()
        defaults=str(a.ardupilot/'Tools/autotest/default_params/copter.parm')+','+str(a.ardupilot/'Tools/autotest/default_params/gazebo-iris.parm')+','+str(params)
        sitl_cmd=[str(a.ardupilot/'build/sitl/bin/arducopter'),'--model','JSON','--speedup','1','--defaults',defaults,'--home','-35.363262,149.165237,584,0','--sim-address=127.0.0.1',f'-I{a.instance}','--sim-port-out',str(plugin_port),'--sim-port-in',str(plugin_port+1)]
        sitl_log=open(output/'sitl.log','w',buffering=1); files.append(sitl_log)
        sitl=subprocess.Popen(sitl_cmd,cwd=sitl_dir,stdout=sitl_log,stderr=subprocess.STDOUT,start_new_session=True)
        manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'sitl_command':sitl_cmd,'gazebo_worker_command':worker_cmd,'sitl_pid':sitl.pid,'worker_windows_pid':gazebo.pid,'controller_pid':os.getpid(),'source_commit':subprocess.check_output(['git','-C',str(a.ardupilot),'rev-parse','HEAD'],text=True).strip(),'source_status':subprocess.check_output(['git','-C',str(a.ardupilot),'status','--porcelain'],text=True),'binary_sha256':digest(a.ardupilot/'build/sitl/bin/arducopter'),'script_sha256':digest(__file__),'worker_script_sha256':digest(ROOT/'sim/gazebo_worker.py'),'params_sha256':digest(params),'policy_sha256':digest(a.expert_bundle) if a.expert_bundle else None,'python':sys.version,'thrust_mapping':{'fast_hover_fraction':.5,'fixed_nominal_guid_hover':a.hover_thrust,'formula':'GUID thrust = fast_fraction * fixed_nominal_guid_hover / 0.5'},'axes':'MAV NED/FRD -> policy ENU/FLU'}
        manifest['seed_scope']='Gazebo RNG (--seed); ArduPilot 4.7.2 has no SIM_RANDOM_SEED parameter. Repeated fresh processes, not guaranteed independent autopilot noise seeds.'
        manifest['metric_scope']='Integrated-system comparison: native PID internal AHRS10 and hybrid external50Hz quantized MAVLink. Common external state metrics; independent Gazebo truth/PCAP retained. Wind loads are Newton force proxies, not wind speed.'
        manifest['scenario_spec']=spec
        manifest['split']=a.split
        manifest['train_seed']=a.train_seed
        manifest['pid_profile']=pid_profile
        manifest['model_family']='gazebo_iris_ardupilot'
        manifest['contract_receipt']=contract_receipt
        manifest['joint_interface']=summary['joint_interface']
        manifest['joint_observation']=summary['joint_observation']
        manifest['thrust_mapping'].update(semantics=a.thrust_semantics,effective_semantics=summary['joint_interface']['effective_semantics'],body_collective_inverse='preboost = clip(desired_body * clip(measured_Rzz,0.1,1) / clip(10*measured_Rzz,0,1),0,1)',native_source='AC_AttitudeControl_Multi::get_throttle_boosted; thrust angle comes from current body in AC_AttitudeControl::thrust_vector_rotation_angles')
        manifest['control_authority']={'hybrid_action_bounds':[-1,1],'hybrid_body_rate_limits_flu_rad_s':[2,2,1],'hybrid_guid_thrust_range':[0,2*a.hover_thrust],'native_guid_thrust_range':[0,1],'scope':'Integrated-controller limits differ; do not infer equal extreme thrust authority. Nominal calibration and bounds fixed across scenarios.'}
        manifest['script_sha256']=digest(source_dir/Path(__file__).name)
        manifest['worker_script_sha256']=digest(source_dir/'gazebo_worker.py')
        manifest['source_snapshot_sha256']={str(path.relative_to(source_dir)):digest(path) for path in source_dir.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        summary['joint_interface']['source_snapshot_sha256']=manifest['source_snapshot_sha256']
        manifest['failure_envelope']={'position_error_m':4.,'altitude_min_m':.25,'altitude_max_m':10.,'body_z_world_z_min':.2,'state_receive_age_max_s':.25,'required_mode':4,'required_armed':True}
        manifest['original_policy_path']=original_policy_path
        if original_policy_path:
            original_dir=Path(original_policy_path).parent
            for name in ['config.json','manifest.json','training_config.json','summary.json']:
                source=original_dir/name
                if source.is_file():
                    shutil.copy2(source,source_dir/('training_'+name))
            summary['original_policy_path']=original_policy_path
            summary['policy_sha256']=manifest['policy_sha256']
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        end=time.monotonic()+40
        while True:
            try:
                conn=mavutil.mavlink_connection(f'tcp:127.0.0.1:{tcp_port}',source_system=255,source_component=190)
                break
            except OSError:
                if time.monotonic()>end: raise
                time.sleep(.25)
        hb=conn.wait_heartbeat(timeout=45)
        if hb is None: raise TimeoutError('No SITL heartbeat')
        rec=Telemetry(conn,output)
        rec.event('connected',system=conn.target_system,component=conn.target_component)
        rec.wait(lambda:(output/'force_physics.jsonl').exists(),10,'ForceSystem plugin did not initialize; see Gazebo log')
        readback={}
        for key,value in [('GUID_OPTIONS',8),('FS_THR_ENABLE',0),('FS_GCS_ENABLE',0),('FENCE_ENABLE',0),('MOT_HOVER_LEARN',0)]: readback[key]=rec.param(key,value)
        for key in ['MOT_THST_HOVER','MOT_THST_EXPO','MOT_PWM_MIN','MOT_PWM_MAX','AHRS_EKF_TYPE','EK3_ENABLE']:
            try: readback[key]=rec.param(key)
            except RuntimeError: readback[key]=None
        if a.controller!='pid' and a.thrust_semantics=='body_collective':
            readback['ATC_ANGLE_BOOST']=rec.param('ATC_ANGLE_BOOST')
            if not math.isfinite(readback['ATC_ANGLE_BOOST']) or abs(readback['ATC_ANGLE_BOOST']-1.)>1e-6:raise RuntimeError('Body collective interface requires read-only ATC_ANGLE_BOOST=1 readback')
        for key,value in pid_profile['parameters'].items():
            readback[key]=rec.param(key)
            if abs(readback[key]-float(value))>1e-5:raise RuntimeError('PID profile readback mismatch '+key)
        (output/'parameter_readback.json').write_text(json.dumps(readback,indent=2))
        for msgid,hz in [(30,50),(32,50),(83,50),(36,50),(74,10),(193,5)]:
            command(conn,mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,msgid,1e6/hz)
        command(conn,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,148)
        rec.wait(lambda:'LOCAL_POSITION_NED' in rec.latest and 'ATTITUDE' in rec.latest,75,'No AHRS local position')
        # Wait for actual prearm acceptance instead of setting ARMING_CHECK=0.
        deadline=time.monotonic()+90
        armed=False
        while time.monotonic()<deadline:
            conn.mav.set_mode_send(conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,4)
            command(conn,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1)
            interval=time.monotonic()+2
            while time.monotonic()<interval:
                rec.drain(); hb=rec.latest.get('HEARTBEAT')
                if hb and hb.base_mode&mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    armed=True; break
                time.sleep(.01)
            if armed: break
        if not armed: raise RuntimeError('SITL refused normal arming')
        rec.event('armed')
        command(conn,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,3.)
        rec.wait(lambda:rec.state()[0][2]>2.7,60,'Takeoff did not reach 2.7 m')
        settle_start=rec.boot_s
        while rec.boot_s-settle_start<5:
            rec.drain(); send_position(conn,np.array([0,0,3.]),rec.boot_s); time.sleep(.02)
        measurement_start_evidence={'receive_evidence_version':RECEIVE_EVIDENCE_VERSION,'raw_message_index':rec.n,'attitude_record_index':rec.message_indices['ATTITUDE'],'position_record_index':rec.message_indices['LOCAL_POSITION_NED'],'attitude_time_boot_ms':rec.latest['ATTITUDE'].time_boot_ms,'position_time_boot_ms':rec.latest['LOCAL_POSITION_NED'].time_boot_ms,'attitude_receive_monotonic_s':rec.received['ATTITUDE'],'position_receive_monotonic_s':rec.received['LOCAL_POSITION_NED'],'boot_s':rec.boot_s,'start_monotonic_s':time.monotonic()}
        rec.event('measurement_start',position_enu=rec.state()[0].tolist(),measurement_start_evidence=measurement_start_evidence)
        t0=rec.boot_s
        summary['measurement_boot_start_s']=t0
        yaw_reference=math.atan2(rec.state()[2][1,0],rec.state()[2][0,0])
        if policy is not None:policy.reset(yaw_reference)
        summary['yaw_reference_enu_rad']=yaw_reference
        summary['measurement_start_evidence']=measurement_start_evidence
        columns=['t','boot_s','x_e','y_n','z_up','vx_e','vy_n','vz_up','roll_ned','pitch_ned','yaw_ned','p_frd','q_frd','r_frd','target_e','target_n','target_up','error_m','action_thrust','action_p','action_q','action_r','guid_thrust','mode','armed','pos_boot_s','att_boot_s','pos_receive_age_s','att_receive_age_s','yaw_error_rad','goal_index','force_e_N','force_n_N','force_up_N']
        original_columns=columns.copy()
        columns+=['raw_message_index','position_record_index','attitude_record_index','desired_body_thrust','preboost_thrust','measured_cos','boost_denominator','inverted_factor','inverse_factor','adapter_clipped']
        csv_file=open(output/'telemetry.csv','w',newline='',buffering=1); files.append(csv_file)
        writer=csv.DictWriter(csv_file,fieldnames=columns); writer.writeheader()
        cmd_file=open(output/'commands.jsonl','w',buffering=1); files.append(cmd_file)
        previous=np.zeros(4,dtype=np.float32); last_control=-1e9; rows=[]; failure=None; displayed_target=None;last_force=None;force_seq=0
        timeout_wall=time.monotonic()+max(120,a.duration*3)
        while rec.boot_s-t0<a.duration:
            rec.drain()
            if time.monotonic()-rec.last_recv>3: raise TimeoutError('MAVLink stream stale')
            if time.monotonic()>timeout_wall: raise TimeoutError('Simulation time stopped')
            t=rec.boot_s-t0
            if t-last_control<.019:
                time.sleep(.001); continue
            control_dt=t-last_control if last_control>=0 else .02
            last_control=t
            target=mission.target;goal_index=mission.index
            if (a.gui or a.record) and tuple(target)!=displayed_target:
                temporary=output/'TARGET.json.tmp'
                temporary.write_text(json.dumps({'target_enu':target.tolist(),'boot_s':rec.boot_s}))
                temporary.replace(output/'TARGET.json'); displayed_target=tuple(target)
            desired_force=force_N(t,spec,a.seed)
            if last_force is None or tuple(desired_force)!=last_force:
                temporary=output/'FORCE_COMMAND.json.tmp'
                temporary.write_text(json.dumps({'seq':force_seq,'t':t,'boot_s':rec.boot_s,'force_enu_N':desired_force.tolist()}));temporary.replace(output/'FORCE_COMMAND.json')
                rec.event('force_requested',seq=force_seq,t=t,force_enu_N=desired_force.tolist());force_seq+=1;last_force=tuple(desired_force)
            pos,vel,R,omega=rec.state()
            att=rec.latest['ATTITUDE']; hb=rec.latest.get('HEARTBEAT')
            pos_age=time.monotonic()-rec.received['LOCAL_POSITION_NED']; att_age=time.monotonic()-rec.received['ATTITUDE']
            if max(pos_age,att_age)>.25: raise TimeoutError('Control state stale: position or attitude >250ms')
            adapter=None;desired_body_thrust=None;prediction=None
            if a.controller=='pid':
                send_position(conn,target,rec.boot_s,yaw_reference)
                action=np.full(4,np.nan); thrust=None
            else:
                if policy is not None:
                    try:
                        prediction=policy.predict_enu(pos,vel,R,omega,target,dt=control_dt)
                        action=np.asarray(prediction['action'],dtype=np.float64).reshape(4)
                    except Exception as exc:raise ControllerFailure('Policy inference failed: '+str(exc)) from exc
                else: action=pd_action(pos,vel,R,omega,target,previous)
                if not np.all(np.isfinite(action)): raise ControllerFailure('Nonfinite policy action')
                action=np.clip(action,-1,1)
                thrust=float(np.clip((action[0]+1)*.5*a.hover_thrust/.5,0,1))
                if a.thrust_semantics=='body_collective':
                    desired_body_thrust=thrust
                    try:adapter=body_collective_to_preboost(desired_body_thrust,float(R[2,2]),att_age,a.adapter_attitude_max_age_s)
                    except ThrustAdapterError as exc:raise ControllerFailure('Thrust adapter failed closed: '+str(exc)) from exc
                    thrust=adapter.preboost_thrust
                rates=action[1:]*[2,2,1]*[1,-1,-1]
                conn.mav.set_attitude_target_send(int(rec.boot_s*1000)&0xffffffff,conn.target_system,conn.target_component,128,[0.,0.,0.,0.],*rates,thrust)
                previous=action.astype(np.float32)
            error=float(np.linalg.norm(pos-target))
            yaw=math.atan2(R[1,0],R[0,0]);yaw_error=math.atan2(math.sin(yaw_reference-yaw),math.cos(yaw_reference-yaw))
            row=dict(zip(original_columns,[t,rec.boot_s,*pos,*vel,att.roll,att.pitch,att.yaw,att.rollspeed,att.pitchspeed,att.yawspeed,*target,error,*action,thrust,hb.custom_mode if hb else None,bool(hb and hb.base_mode&128),rec.latest['LOCAL_POSITION_NED'].time_boot_ms/1000,att.time_boot_ms/1000,pos_age,att_age,yaw_error,goal_index,*desired_force]))
            row.update(raw_message_index=rec.n,position_record_index=rec.message_indices['LOCAL_POSITION_NED'],attitude_record_index=rec.message_indices['ATTITUDE'],desired_body_thrust=desired_body_thrust,preboost_thrust=thrust,measured_cos=float(R[2,2]),boost_denominator=adapter.denominator if adapter else None,inverted_factor=adapter.inverted_factor if adapter else None,inverse_factor=adapter.inverse_factor if adapter else None,adapter_clipped=adapter.clipped if adapter else None)
            writer.writerow(row); rows.append(row)
            mission_event=mission.update(t,pos)
            if mission_event:rec.event('mission_transition',**mission_event)
            target_feedback=rec.latest.get('ATTITUDE_TARGET')
            cmd_file.write(json.dumps({'t':t,'boot_s':rec.boot_s,'controller':a.controller,'target_enu':target.tolist(),'target_next_enu':mission.target.tolist(),'goal_index':goal_index,'policy_observation_contract':OBSERVATION_CONTRACT if prediction is not None else None,'policy_observation_size':OBSERVATION_SIZE if prediction is not None else None,'policy_basis_kind':a.basis_kind if prediction is not None else None,'policy_raw_base77_observe_calls':policy.encoder.raw_encoder.base77_observe_calls if prediction is not None else None,'policy_raw_encoder_calls':policy.encoder.observation_calls if prediction is not None else None,'policy_conditioning_mode':policy.mode if prediction is not None else None,'policy_memory_valid_leg':bool(policy.encoder.path_context.valid_leg) if prediction is not None else None,'policy_memory_target_changed':bool(policy.encoder.target_changed) if prediction is not None else None,'policy_raw_observation':policy.encoder.raw_observation.tolist() if prediction is not None else None,'policy_expert_diagnostics':phase_diagnostics(policy) if prediction is not None else None,'policy_observation':prediction['observation'].tolist() if prediction is not None else None,'yaw_reference_enu_rad':yaw_reference,'control_dt_s':control_dt,'action':action.tolist() if a.controller!='pid' else None,'guid_thrust':thrust,'type_mask':128 if a.controller!='pid' else 2552,'joint_interface_version':INTERFACE_VERSION,'thrust_semantics':a.thrust_semantics,'raw_message_index':rec.n,'position_record_index':rec.message_indices['LOCAL_POSITION_NED'],'attitude_record_index':rec.message_indices['ATTITUDE'],'attitude_receive_monotonic_s':rec.received['ATTITUDE'],'state_sample_monotonic_s':rec.received['ATTITUDE']+att_age,'measured_cos':float(R[2,2]),'desired_body_thrust':desired_body_thrust,'preboost_thrust':thrust,'thrust_adapter':adapter.to_dict() if adapter else None,'attitude_target_diagnostic':{'record_index':rec.message_indices['ATTITUDE_TARGET'],'time_boot_ms':target_feedback.time_boot_ms,'q':list(target_feedback.q)} if target_feedback is not None else None})+'\n')
            if error>4 or pos[2]<.25 or pos[2]>10 or R[2,2]<.2 or not row['armed'] or row['mode']!=4:
                failure='flight_envelope_or_mode'; rec.event('flight_failed',row=row); break
        rec.event('measurement_end',reason=failure or 'duration_completed')
        temporary=output/'FORCE_COMMAND.json.tmp';temporary.write_text(json.dumps({'seq':force_seq,'t':rec.boot_s-t0,'boot_s':rec.boot_s,'force_enu_N':[0,0,0]}));temporary.replace(output/'FORCE_COMMAND.json')
        summary['mission_events']=mission.events
        summary['completion_time_s']=mission.completion_time
        errors=np.array([r['error_m'] for r in rows])
        summary.update(status='FAIL' if failure else 'PASS',failure=failure,duration_s=rows[-1]['t'] if rows else 0,samples=len(rows),rmse_position_m=float(np.sqrt(np.mean(errors**2))),p95_position_m=float(np.quantile(errors,.95)),max_position_error_m=float(errors.max()),min_altitude_m=float(min(r['z_up'] for r in rows)),mean_altitude_m=float(np.mean([r['z_up'] for r in rows])),control_hz=len(rows)/(rows[-1]['t'] if rows else 1),raw_message_count=rec.n,measurement_boot_start_s=t0)
        # All controllers return to the same native LAND after the measured segment.
        conn.mav.set_mode_send(conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
        try:
            rec.wait(lambda: bool(rec.latest.get('HEARTBEAT')) and not rec.latest['HEARTBEAT'].base_mode&128,60,'Landing did not disarm')
            summary['landed_disarmed']=True
        except TimeoutError: summary['landed_disarmed']=False
        rec.event('finished',status=summary['status'],landed_disarmed=summary['landed_disarmed'])
    except Exception as exc:
        summary.update(status='FAIL' if isinstance(exc,ControllerFailure) else 'ERROR',error=str(exc),traceback=traceback.format_exc())
        if rec is not None and 'measurement_boot_start_s' in summary:
            summary.update(duration_s=max(0.,rec.boot_s-summary['measurement_boot_start_s']),mission_events=mission.events,completion_time_s=mission.completion_time)
        if not isinstance(exc,ControllerFailure):summary.update(valid_controller_comparison=False,infrastructure_error=str(exc))
        if isinstance(exc,ControllerFailure):
            summary.update(failure='controller_output_failure',survived=False,penalized_rmse_position_m=4.,heading_full_flight_eligible=False)
            stop_force={'seq':force_seq,'t':rec.boot_s-summary['measurement_boot_start_s'],'boot_s':rec.boot_s,'force_enu_N':[0,0,0]}
            temporary=output/'FORCE_COMMAND.json.tmp';temporary.write_text(json.dumps(stop_force));temporary.replace(output/'FORCE_COMMAND.json')
            rec.event('force_requested',**stop_force)
            summary.update(duration_s=max(0.,rec.boot_s-summary['measurement_boot_start_s']),mission_events=mission.events,completion_time_s=mission.completion_time)
            # The observation at a failed inference is still a real measurement.
            # Preserve it even if the first action failed, so the common horizon
            # penalty does not depend on whether an earlier command existed.
            failure_t=rec.boot_s-summary['measurement_boot_start_s']
            if not rows or failure_t>rows[-1]['t']:
                pos,vel,R,omega=rec.state();att=rec.latest['ATTITUDE'];hb=rec.latest.get('HEARTBEAT');target=mission.target
                yaw=math.atan2(R[1,0],R[0,0]);yaw_error=math.atan2(math.sin(yaw_reference-yaw),math.cos(yaw_reference-yaw))
                failure_row=dict(zip(columns,[failure_t,rec.boot_s,*pos,*vel,att.roll,att.pitch,att.yaw,att.rollspeed,att.pitchspeed,att.yawspeed,*target,float(np.linalg.norm(pos-target)),*np.full(4,np.nan),None,hb.custom_mode if hb else None,bool(hb and hb.base_mode&128),rec.latest['LOCAL_POSITION_NED'].time_boot_ms/1000,att.time_boot_ms/1000,time.monotonic()-rec.received['LOCAL_POSITION_NED'],time.monotonic()-rec.received['ATTITUDE'],yaw_error,mission.index,*force_N(failure_t,spec,a.seed)]))
                writer.writerow(failure_row);rows.append(failure_row)
                cmd_file.write(json.dumps({'t':failure_t,'boot_s':rec.boot_s,'controller':a.controller,'action':None,'controller_failure':str(exc),'command_sent':False})+'\n')
                rec.event('controller_output_failure',reason=str(exc),t=failure_t)
            if conn is not None:
                conn.mav.set_mode_send(conn.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,9)
                try:
                    rec.wait(lambda: bool(rec.latest.get('HEARTBEAT')) and not rec.latest['HEARTBEAT'].base_mode&128,60,'Controller failure landing did not disarm')
                    summary['landed_disarmed']=True
                except Exception:summary['landed_disarmed']=False
        print(summary['traceback'],file=sys.stderr,flush=True)
    finally:
        (output/'STOP_GAZEBO').write_text('stop\n')
        if sitl is not None:
            try: os.killpg(sitl.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: sitl.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(sitl.pid,signal.SIGKILL); sitl.wait()
        if gazebo is not None:
            try: gazebo.wait(timeout=15)
            except subprocess.TimeoutExpired: gazebo.terminate(); gazebo.wait(timeout=5)
        if capture is not None:
            try: capture.wait(timeout=15)
            except subprocess.TimeoutExpired: capture.terminate(); capture.wait(timeout=5)
        if rec is not None: rec.close()
        if conn is not None: conn.close()
        for f in files: f.close()
        summary['wall_duration_s']=time.time()-start_wall
        summary['utc_finished']=dt.datetime.now(dt.timezone.utc).isoformat()
        truth_path=output/'groundtruth.jsonl'
        summary['gazebo_groundtruth_bytes']=truth_path.stat().st_size if truth_path.exists() else 0
        finished_path=output/'gazebo_finished.json'
        if finished_path.exists(): summary['gazebo_worker']=json.loads(finished_path.read_text())
        if summary['status']=='PASS' and summary['gazebo_groundtruth_bytes']==0:
            summary['status']='ERROR'; summary['error']='No independently observed Gazebo ground truth'
        if a.record and summary.get('gazebo_worker',{}).get('video_frames',0)==0:
            summary['status']='ERROR'; summary['error']='Requested Gazebo video has no observed camera frames'
        expected_force=spec.get('force_events') and summary.get('duration_s',0)>spec['force_events'][0]['on_s']+.1
        physical_path=output/'force_physics.jsonl'
        actual_force=[json.loads(line) for line in physical_path.read_text().splitlines()] if physical_path.exists() else []
        if expected_force and not any(e.get('link_valid') and any(e['force_enu_N']) for e in actual_force):
            summary['status']='ERROR';summary['error']='No authoritative physical force application despite a scheduled disturbance';summary['valid_controller_comparison']=False
        event_path=output/'events.jsonl'
        force_requests=[event for event in (json.loads(line) for line in event_path.read_text().splitlines()) if event.get('event')=='force_requested'] if event_path.exists() else []
        physical_by_seq={event['seq']:event for event in actual_force}
        missing_force=[];previous_requested_force=np.zeros(3);required_transitions=0
        for request in force_requests:
            actual=physical_by_seq.get(request['seq'])
            required=not np.allclose(previous_requested_force,request['force_enu_N'],atol=1e-10,rtol=0)
            required_transitions+=int(required);previous_requested_force=np.array(request['force_enu_N'])
            if (required and actual is None) or (actual is not None and (not actual.get('link_valid') or not np.allclose(actual['force_enu_N'],request['force_enu_N'],atol=1e-10,rtol=0))):missing_force.append(request['seq'])
        if missing_force:
            summary['status']='ERROR';summary['error']='Missing or mismatched physical force transitions '+str(missing_force);summary['valid_controller_comparison']=False
        summary['force_transition_verification']={'requested_sequences':len(force_requests),'required_force_changes':required_transitions,'missing_or_mismatched_sequences':missing_force,'status':'PASS' if not missing_force else 'FAIL','redundant_zero_scope':'Initial/repeated zero command need not survive a latest-value coalescing transport; every actual force change, including OFF, is required'}
        csv_path=output/'telemetry.csv'
        if csv_path.exists():
            recorded_rows=list(csv.DictReader(csv_path.open()))
            if recorded_rows:
                try:
                    metrics=finalize(output,summary)
                    if a.controller!='pid':
                        actions=np.array([[float(row[key]) for key in ['action_thrust','action_p','action_q','action_r']] for row in recorded_rows])
                        valid_actions=actions[np.all(np.isfinite(actions),axis=1)]
                        if len(valid_actions):
                            metrics['action_saturation_fraction_by_channel']=np.mean(np.abs(valid_actions)>=.999,axis=0).tolist()
                            metrics['max_abs_action_by_channel']=np.max(np.abs(valid_actions),axis=0).tolist()
                    summary.update(metrics)
                    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
                except Exception as exc:
                    summary['postprocess_error']=str(exc);summary['status']='ERROR';summary['valid_controller_comparison']=False
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__': sys.exit(main())

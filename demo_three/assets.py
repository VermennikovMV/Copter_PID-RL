"""Exact controller numerics with distinct, explicit public metadata projections."""
import hashlib
import json
from pathlib import Path

PID_PATH='artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.json'
PID_PUBLIC_PATH='artifacts/gazebo/native_pid_tuning_v1/frozen_native_pid.public.json'
PID_SHA='086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd'
PID_NUMBERS={'PSC_NE_POS_P':1.5,'PSC_NE_VEL_P':4.,'PSC_NE_VEL_I':2.,'PSC_NE_VEL_D':.5}
CAL_PATH='artifacts/gazebo_direct/calibration_nominal_1104/calibration.json'
CAL_PUBLIC_PATH='artifacts/gazebo_direct/calibration_nominal_1104/calibration.public.json'
CAL_SHA='2e20df83b9dadf5231c2f93e8070d89c03ffbdbe17c14e277935d7c87743a649'
CAL_NUMBERS=dict(hover_pwm_gazebo_order=[1533.216,1533.215,1533.214,1533.217],n_samples=1000,
    window='measurement time >5 seconds',reference_fraction=.5,nominal_only=True)
DIRECT_PATH='artifacts/training/main_direct_seed111_v2/best_policy.npz'
DIRECT_SHA='5d0d19e02c30c62ed3a0efdcf003d8b6b635ae1eea9b1d32b4737836c46ff900'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def data_root(code_root):
    code_root=Path(code_root).resolve()
    return code_root/'v2' if (code_root/'v2/quadrl/policy.py').is_file() else code_root
def public_layout(path):
    path=Path(path).resolve()
    if (path/'v2/quadrl_phase_expert_inference_v1/policy.py').is_file():return path,path/'v2'
    if (path/'quadrl_phase_expert_inference_v1/policy.py').is_file():return path.parent,path
    raise FileNotFoundError('Explicit public inference package is absent')
def resolve_control_assets(code_root):
    root=data_root(code_root);pid=root/PID_PATH;cal=root/CAL_PATH
    identity={}
    if pid.exists():
        if digest(pid)!=PID_SHA:raise ValueError('Existing original PID bytes differ; no projection fallback')
        pid_contract='original_exact_PID_document'
    else:
        pid=root/PID_PUBLIC_PATH;value=read(pid)
        if value.get('_public_projection',{}).get('original_sha256')!=PID_SHA:
            raise ValueError('Explicit public PID projection lacks original lineage')
        pid_contract='public_PID_projection_exact_parameters_original_document_not_present'
    pid_value=read(pid)
    if pid_value.get('parameters')!=PID_NUMBERS:raise ValueError('Frozen numerical PID gains differ')
    if cal.exists():
        if digest(cal)!=CAL_SHA:raise ValueError('Existing original calibration bytes differ; no projection fallback')
        cal_contract='original_exact_calibration_document'
    else:
        cal=root/CAL_PUBLIC_PATH;value=read(cal);meta=value.get('_meta',{})
        if meta.get('original_sha256')!=CAL_SHA or meta.get('changed_fields')!=['source_run']:
            raise ValueError('Explicit public calibration projection has unexpected changes')
        if set(value)-{'_meta'}!=set(CAL_NUMBERS)|{'source_run'}:
            raise ValueError('Public calibration schema differs beyond declared source_run metadata')
        cal_contract='public_calibration_projection_exact_numerics_original_document_not_present'
    cal_value=read(cal)
    if any(cal_value.get(key)!=expected for key,expected in CAL_NUMBERS.items()):
        raise ValueError('Fixed original nominal motor calibration numerics differ')
    direct=root/DIRECT_PATH
    if digest(direct)!=DIRECT_SHA:raise ValueError('Exact unchanged Direct111 NPZ required')
    identity=dict(data_root=str(root),
        PID=dict(path=str(pid),sha256=digest(pid),original_document_sha256=PID_SHA,contract=pid_contract,parameters_exact=True),
        calibration=dict(path=str(cal),sha256=digest(cal),original_document_sha256=CAL_SHA,contract=cal_contract,numerics_exact=True),
        Direct111=dict(path=str(direct),sha256=DIRECT_SHA,contract='original_exact_NPZ'))
    return dict(root=root,pid_path=pid,calibration_path=cal,direct_path=direct,identity=identity)

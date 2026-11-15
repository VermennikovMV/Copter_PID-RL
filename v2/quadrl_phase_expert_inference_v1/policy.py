"""Public inference-only binding; exact NPZ/proof, unchanged v29 numeric methods.

The private original FINAL ZIP is lineage only. This new contract does not claim
to validate that absent ZIP or replace the strict research/Gazebo wire contract.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from quadrl_phase_expert_v29.policy import (NumpyPhaseExpertPolicy,StatelessMean80,
    bundle_payload as original_bundle_payload,F0_PATH,F0_SHA,ROUTE_PATH,ROUTE_SHA,
    PROOF_PATH,PROOF_SHA,ZIP_SHA)
from quadrl_route_memory_v20.policy import ObservationEncoderRouteMemory,raw_config,conditioning_config,SOURCE_PATH,SOURCE_SHA

ROOT=Path(__file__).resolve().parents[1]
CONTRACT_VERSION='public-shared-phase-expert-inference-v1.0'
SCHEMA='public-phase-expert-inference-bundle-v1.0'
ORIGINAL_BUNDLE_PATH='artifacts/refinement/joint_v2/phase_expert_preparation_v29_root01/bundle.json'
ORIGINAL_BUNDLE_SHA='c3fc1c9c799f21200c98fc5995dba9908b1b13a517eb78bd60759e91eb8818a4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def public_bundle_payload():
    original=original_bundle_payload()
    return dict(schema=SCHEMA,observation_contract=CONTRACT_VERSION,architecture='hybrid',
        numeric_controller='v29',shared_encoder=True,conditioning=original['conditioning'],selector=original['selector'],
        exact_assets=dict(F0=dict(path=F0_PATH,sha256=F0_SHA),route=dict(path=ROUTE_PATH,sha256=ROUTE_SHA),
            source68=dict(path=SOURCE_PATH.as_posix(),sha256=SOURCE_SHA),
            training_proof=dict(path=PROOF_PATH,sha256=PROOF_SHA,pointer='/candidate_claims/prior0/final'),
            original_bundle=dict(path=ORIGINAL_BUNDLE_PATH,sha256=ORIGINAL_BUNDLE_SHA)),
        original_model_ZIP=dict(sha256=ZIP_SHA,lineage_only=True,present_locally=False,verified_locally=False),
        scope='CPU inference only; distinct public export contract, not a validated Gazebo wire or full training contract',
        new_optimizer_steps=0,new_training_steps=0)

def validate_public_sources(path):
    if read(path)!=public_bundle_payload():raise ValueError('Exact distinct public inference bundle required')
    for asset in public_bundle_payload()['exact_assets'].values():
        if sha(ROOT/asset['path'])!=asset['sha256']:raise ValueError('Actual public input SHA mismatch: '+asset['path'])
    if read(ROOT/ORIGINAL_BUNDLE_PATH)!=original_bundle_payload():raise ValueError('Original controller payload changed')
    claim=read(ROOT/PROOF_PATH)['candidate_claims']['prior0']['final']
    if not (claim['policy_npz_sha256']==ROUTE_SHA and claim['model_zip_sha256']==ZIP_SHA and
            claim['actual_checkpoint_PPO_updates']==2048 and claim['all6_actor_arrays_changed'] and claim['log_std_exact_source68']):
        raise ValueError('Original actual route-training claim required; ZIP digest is lineage, not a local ZIP verification')
    arrays=[]
    for name in (F0_PATH,ROUTE_PATH):
        with np.load(ROOT/name,allow_pickle=False) as archive:arrays.append({k:archive[k].copy()for k in archive.files})
    for value in arrays:
        if str(value['architecture'])!='hybrid' or int(value['layers'])!=3:raise ValueError('Exact hybrid topology required')
        if json.loads(str(value['conditioning_config']))!=conditioning_config('zero_route_integral'):raise ValueError('Conditioning changed')
        if [value[f'w{i}'].shape for i in range(3)]!=[(128,80),(128,128),(4,128)]:raise ValueError('Wrong actor shapes')
        if [value[f'b{i}'].shape for i in range(3)]!=[(128,),(128,),(4,)]:raise ValueError('Wrong bias shapes')
        if not all(value[f'{prefix}{i}'].dtype==np.float32 and np.isfinite(value[f'{prefix}{i}']).all()for prefix in ('w','b')for i in range(3)):
            raise ValueError('Finite float32 arrays required')
    for key in ('encoder_config','architecture','layers'):
        if not np.array_equal(arrays[0][key],arrays[1][key]):raise ValueError('Encoder/topology mismatch')
    if raw_config(arrays[0])!=raw_config(arrays[1]):raise ValueError('Expert preprocessing differs')
    return arrays

class NumpyPhaseExpertInference(NumpyPhaseExpertPolicy):
    """Only construction/provenance differ; reset/predict/ENU/math are inherited."""
    def __init__(self,path):
        arrays=validate_public_sources(Path(path))
        self.experts=[StatelessMean80(value)for value in arrays]
        self.encoder=ObservationEncoderRouteMemory(raw_config(arrays[0]),'zero_route_integral')
        self.architecture='hybrid';self.mode='zero_route_integral';self.contract=CONTRACT_VERSION
        self.bundle_sha256=sha(Path(path));self.original_controller_bundle_sha256=ORIGINAL_BUNDLE_SHA
        self.original_ZIP_verified_locally=False
        self.reset()

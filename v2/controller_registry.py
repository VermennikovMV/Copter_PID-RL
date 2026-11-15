"""Runnable NumPy controllers with explicit backend/version scope."""
from pathlib import Path
import hashlib
from quadrl.policy import NumpyPolicy
from sim_direct.controller import DirectController111
from quadrl_phase_expert_inference_v1.policy import NumpyPhaseExpertInference

ROOT=Path(__file__).resolve().parents[1]
LEGACY_HYBRID='artifacts/training/main_hybrid_seed111_v1/best_policy.npz'
LEGACY_HYBRID_SHA='a80e039ee35a718a2659de51f6b649237f9c8bc9c39eeae9bd3917484794d3a5'

def load_controller(name):
    if name=='direct':return DirectController111()
    if name=='hybrid':return NumpyPhaseExpertInference(ROOT/'models/phase29_public_inference.json')
    if name=='historical-hybrid':
        path=ROOT/'v2'/LEGACY_HYBRID
        if hashlib.sha256(path.read_bytes()).hexdigest()!=LEGACY_HYBRID_SHA:raise ValueError('Historical Hybrid111 SHA differs')
        result=NumpyPolicy(path)
        if result.architecture!='hybrid':raise ValueError('Historical hybrid architecture required')
        return result
    if name=='pid-fast':
        from quadrl.baseline import CascadedPID
        return CascadedPID()
    raise ValueError('Controller must be hybrid, direct, historical-hybrid or pid-fast')

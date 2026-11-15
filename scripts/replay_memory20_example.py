"""Replay 64 archived encoded inputs through exact F0; no simulator or training."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v2'))
from quadrl_route_memory_v20.policy import NumpyPolicyRouteMemory

def main():
    model=ROOT/'models/route_memory20_zero_integral.npz';example=ROOT/'examples/memory20_encoded_obs64.npz'
    assert hashlib.sha256(model.read_bytes()).hexdigest()=='f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'
    assert hashlib.sha256(example.read_bytes()).hexdigest()=='feceb462ec6eef8efbb976700618694e88b94bf329f7f8c42169f612990cbc00'
    policy=NumpyPolicyRouteMemory(model,expected_mode='zero_route_integral')
    with np.load(example,allow_pickle=False) as io:
        raw=io['raw_observation'];effective=io['effective_observation'];valid=io['valid_leg']
        assert raw.shape==effective.shape==(64,80) and valid.shape==(64,)
        expected=raw.copy();expected[valid,22:25]=0.
        assert np.array_equal(expected,effective)
        actual=np.stack([policy.predict(row) for row in effective])
        error=float(np.max(np.abs(actual-io['action'])))
    assert error<=1e-6 and 'torch' not in sys.modules and 'pybullet' not in sys.modules
    print(json.dumps(dict(status='PASS_ARCHIVED_FORWARD_REPLAY_NOT_SIMULATION',archived_rows=64,
        action_max_error=error,effective_conditioning_exact=True,Torch_or_PyBullet_imported=False,
        new_environment_steps=0,optimizer_steps=0,independent_quality_replication=False),indent=2))

if __name__=='__main__':main()

"""Read-only exact diagnostic80 inspection/inference; never starts a simulator."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"v2"))
from quadrl_route_basis_v14.policy import NumpyPolicyRoute80,CONTRACT_VERSION

POLICY=ROOT/"models/diagnostic80_metric_cost16.npz"
SHA="68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations",type=Path,help="Optional .npy array with final dimension80; these are inputs, not simulated flight states")
    parser.add_argument("--output",type=Path,help="Fresh actions .npy file under runs/")
    args=parser.parse_args()
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest()==SHA
    policy=NumpyPolicyRoute80(POLICY,expected_kind="velocity_residual")
    result=dict(status="DIAGNOSTIC_ONLY_NOT_PROMOTED",policy_sha256=SHA,contract=CONTRACT_VERSION,
        observation_size=80,basis_kind="velocity_residual",native_operational_gate="FAIL",
        main_policy_77_unchanged=True,simulation_or_training_started=False)
    if args.observations:
        assert args.output is not None,"Provide an explicit fresh --output under runs/"
        observations=np.load(args.observations,allow_pickle=False)
        assert observations.ndim in (1,2) and observations.shape[-1]==80 and np.isfinite(observations).all()
        output=(args.output if args.output.is_absolute() else ROOT/args.output).resolve()
        assert output.suffix==".npy" and output.is_relative_to((ROOT/"runs").resolve()) and not output.exists()
        actions=policy.predict(observations)
        assert actions.shape==observations.shape[:-1]+(4,) and np.isfinite(actions).all()
        output.parent.mkdir(parents=True,exist_ok=True);np.save(output,actions)
        result.update(actions_shape=list(actions.shape),output=output.relative_to(ROOT).as_posix())
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()

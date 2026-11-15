"""Read exact v20 conditioned policy on CPU; no simulation, learning or server."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v2'))
from quadrl_route_memory_v20.policy import NumpyPolicyRouteMemory,CONTRACT_VERSION,SOURCE_PATH,SOURCE_SHA

POLICY=ROOT/'models/route_memory20_zero_integral.npz'
POLICY_SHA='f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,help='Optional effective observations .npy; final dimension80')
    parser.add_argument('--output',type=Path,help='Fresh actions .npy under runs/')
    args=parser.parse_args()
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest()==POLICY_SHA
    assert hashlib.sha256((ROOT/'v2'/SOURCE_PATH).read_bytes()).hexdigest()==SOURCE_SHA
    policy=NumpyPolicyRouteMemory(POLICY,expected_mode='zero_route_integral')
    result=dict(status='V20_DEVELOPMENT_PRIMARY8_OF8_TAIL_FAIL_NOT_PROMOTED',policy_sha256=POLICY_SHA,
        source68_sha256=SOURCE_SHA,contract=CONTRACT_VERSION,observation_size=80,
        conditioning_mode='zero_route_integral',new_v20_SGD_updates=0,
        source68_original_relative_path_preserved=True,simulation_or_training_started=False)
    if args.observations:
        assert args.output is not None,'Provide an explicit fresh --output under runs/'
        observations=np.load(args.observations,allow_pickle=False)
        assert observations.ndim in (1,2) and observations.shape[-1]==80 and np.isfinite(observations).all()
        output=(args.output if args.output.is_absolute() else ROOT/args.output).resolve()
        assert output.suffix=='.npy' and output.is_relative_to((ROOT/'runs').resolve()) and not output.exists()
        actions=policy.predict(observations)
        assert actions.shape==observations.shape[:-1]+(4,) and np.isfinite(actions).all()
        output.parent.mkdir(parents=True,exist_ok=True);np.save(output,actions)
        result.update(actions_shape=list(actions.shape),output=output.relative_to(ROOT).as_posix())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Export and verify three unchanged-weight source68 conditioning prototypes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
from scripts import train_joint_route_basis_v15 as frozen
from quadrl_route_memory_v20.policy import (SOURCE_PATH,SOURCE_SHA,CONTRACT_VERSION,MODES,
    NumpyPolicyRouteMemory,export_conditioned_policy,source_values,sha)
from quadrl_route_basis_v14.policy import NumpyPolicyRoute80
write_json,read,utc=frozen.write_json,frozen.read,frozen.utc


def source_hashes():
    paths=[Path(__file__),ROOT/"scripts/run_joint_route_memory_baseline_v20.py"]
    paths+=sorted((ROOT/"quadrl_route_memory_v20").glob("*.py"))
    return {**frozen.source_hashes(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths},SOURCE_PATH.as_posix():SOURCE_SHA}


def assert_frozen(sources):
    path=ROOT/"docs/joint_route_memory_v20/preparation_freeze.json"
    assert read(path)["source_sha256"]==sources,"Changed source after zero-update preparation freeze"
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name


def archived_sample():
    directory=(ROOT/SOURCE_PATH).parent
    selection=read(directory/"selection.json")
    validation=read(directory/(selection["checkpoint"]+"_validation.json"))
    assert validation["policy_sha256"]==SOURCE_SHA
    record=next(r for r in validation["episodes"] if (r["scenario"],r["mission"],r["eval_seed"])==("nominal","route",84001))
    path=Path(record["actor_io_path"]);path=path if path.is_absolute() else ROOT/path
    assert sha(path)==record["hashes"]["actor_io.npz"]
    return record,path


def prototype(output,receipt):
    output.mkdir(parents=True,exist_ok=False);assert not receipt.exists()
    sources=source_hashes();source_values()
    original=NumpyPolicyRoute80(ROOT/SOURCE_PATH,expected_kind="velocity_residual")
    x=np.random.default_rng(174920).normal(size=(1000,80)).astype(np.float32)
    records={};policies={}
    source_record,archive_path=archived_sample()
    with np.load(archive_path,allow_pickle=False) as io:
        assert len(io["observation"])>=1000
        for mode in MODES:
            folder=output/mode;folder.mkdir();path=folder/"policy.npz"
            export_conditioned_policy(path,mode);policy=NumpyPolicyRouteMemory(path,expected_mode=mode);policies[mode]=policy
            error=float(np.max(np.abs(original.predict(x)-policy.predict(x))))
            assert error==0.,"Same numerical inputs must produce bit-exact unchanged-weight predictions"
            policy.reset(source_record["yaw_reference"])
            changed_rows=0
            for index in range(1000):
                effective=policy.encoder.observe(*(io[name][index] for name in ("position","velocity","rotation","body_rates","target","previous_action","dt")))
                raw=policy.encoder.raw_observation
                assert np.array_equal(raw,io["observation"][index]),"Original80 measured replay differs"
                changed_rows+=int(not np.array_equal(raw,effective))
                if mode=="raw":assert np.array_equal(policy.predict(effective),original.predict(raw))
                if mode=="zero_route_integral" and policy.encoder.path_context.valid_leg:assert np.array_equal(effective[22:25],np.zeros(3,np.float32))
            records[mode]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),same_numeric_input_forward_parity1000_max_abs=error,
                original80_archive_replay_samples=1000,effective_changed_rows_on_original_trajectory=changed_rows,
                no_claim_of_BC_route_closed_loop_parity=True,raw_encoder_calls=policy.encoder.observation_calls)
    # Isolated hover samples contain no observed target change. All histories,
    # world-integral clamps and full effective observations must remain exact.
    for policy in policies.values():policy.reset(.2)
    for index in range(1000):
        t=index*.02;c,s=np.cos(.2),np.sin(.2);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        sample=(np.array([.5*np.sin(t),.3*np.cos(t),1.+.2*np.sin(t)]),np.zeros(3),rotation,np.zeros(3),np.array([0.,0.,1.]),np.zeros(4),.02)
        effective=[p.encoder.observe(*sample) for p in policies.values()]
        assert all(np.array_equal(effective[0],v) for v in effective[1:])
    # Reset occurs BEFORE integrating the current changed-target interval.
    policy=policies["reset_on_target"];policy.reset(0.)
    sample=(np.zeros(3),np.zeros(3),np.eye(3),np.zeros(3),np.array([1.,0.,1.]),np.zeros(4),.02)
    policy.encoder.observe(*sample);policy.encoder.observe(*sample)
    changed=(*sample[:4],np.array([1.,1.,1.]),*sample[5:]);policy.encoder.observe(*changed)
    np.testing.assert_array_equal(policy.encoder.integral_world,np.array([.02,.02,.02]))
    assert policy.encoder.path_context.valid_leg and policy.encoder.target_changed
    # A changed target can be too close to form a valid observed leg. Own C
    # memory resets, but applying conditioned integral remains gated off.
    policy.reset(0.);policy.encoder.observe(*sample);policy.encoder.observe(*sample)
    degenerate=(*sample[:4],sample[4]+np.array([1e-9,0.,0.]),*sample[5:])
    effective=policy.encoder.observe(*degenerate)
    assert not policy.encoder.path_context.valid_leg and policy.encoder.target_changed
    assert np.array_equal(effective,policy.encoder.raw_observation)
    assert source_hashes()==sources
    result=dict(schema="joint-route-memory-prototype-v20.0",status="PASS_STATIC_ZERO_ENV_UPDATES",utc=utc(),
        source_sha256=sources,source_policy=dict(path=SOURCE_PATH.as_posix(),sha256=SOURCE_SHA),
        contract=CONTRACT_VERSION,policies=records,unchanged_all_actor_weights_and_original_metadata=True,
        source_archive=dict(path=archive_path.relative_to(ROOT).as_posix(),sha256=sha(archive_path)),
        isolated_hover_effective_and_action_numeric_parity_samples=1000,
        target_change_reset_before_current_integration=True,first_sample_integral_skip_original=True,
        degenerate_observed_target_change_conditioning_gate_off_raw_exact=True,
        physical_steps=0,environment_constructions=0,backward_calls=0,optimizer_steps=0,
        native_closed_loop_baseline_executed=False,Gazebo_acceptance=False,
        interpretation="Changed memory conditioning only; no new learned weights or BC-route closed-loop equivalence claim")
    write_json(output/"prototype_manifest.json",result);write_json(receipt,result)
    print(json.dumps(dict(status=result["status"],receipt=str(receipt),sha256=sha(receipt))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,default=ROOT/"audit/joint_route_memory_static_v20_root01.json")
    args=parser.parse_args()
    for name in ("output","receipt"):
        path=getattr(args,name);setattr(args,name,(path if path.is_absolute() else ROOT/path).resolve())
    prototype(args.output,args.receipt)


if __name__=="__main__":main()

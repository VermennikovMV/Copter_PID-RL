"""Conditioned PPO versus each own zero-update baseline; A/B is diagnostic."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import joint_research_preflight_v1 as gate
from scripts import run_joint_critic_calibration_pair_v18 as previous
from scripts import prototype_joint_route_memory_v20 as v20
from quadrl_route_memory_train_v21.policy import MODES,INITIAL_ZIP,INITIAL_ZIP_SHA,SOURCE_SHA
ARMS=MODES
LABELS,BUDGET=previous.LABELS,previous.BUDGET
DRIVER="scripts/train_joint_route_memory_v21p1.py"
BASELINES={mode:f"artifacts/refinement/joint_v2/route_memory_baseline_v20_root01/{mode}/evaluation.json" for mode in ARMS}
read,option=previous.read,previous.option


def memory_replay(trace,mode):
    from quadrl_route_memory_v20.policy import ObservationEncoderRouteMemory,raw_config,source_values
    config=raw_config(source_values());shape=trace["observations"].shape
    encoders=[ObservationEncoderRouteMemory(config,mode) for unused in range(shape[1])]
    for step in range(shape[0]):
        for lane,encoder in enumerate(encoders):
            if step==0 or trace["episode_starts"][step,lane]:encoder.reset(float(trace["sample_yaw_reference"][step,lane]))
            effective=encoder.observe(*(trace["sample_"+name][step,lane] for name in ("position","velocity","rotation","body_rates","target","previous_action","dt")))
            assert np.array_equal(effective,trace["observations"][step,lane]),"Actual PPO effective memory replay differs"
            assert np.array_equal(encoder.raw_observation,trace["sample_raw_observation"][step,lane])
    return dict(status="PASS_ACTUAL_EFFECTIVE_AND_RAW80_PURE_MEMORY_REPLAY",rows=shape[0]*shape[1],
        changed_columns_only=[22,23,24],extra_forward_physics_or_updates=0)


def semantic_evidence(output):
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    from scripts.bc_trust_region_probe_v1 import actor_name
    initial=load_from_zip_file(ROOT/INITIAL_ZIP,device="cpu")[1]["policy"]
    results={}
    for mode in ARMS:
        folder=Path(output)/mode
        warm=load_from_zip_file(folder/"critic_warmed_model.zip",device="cpu")[1]
        assert len(warm["policy.optimizer"]["state"])==6
        assert all(torch.equal(value,warm["policy"][name]) for name,value in initial.items() if actor_name(name))
        reset=read(folder/"optimizer_reset.json");assert reset["initializer_sha256"]==INITIAL_ZIP_SHA and reset["after_clear"]==0
        transition=read(folder/"lr_transition.json")
        assert transition["Adam_states"]==6 and transition["optimizer_values_steps_and_other_groups_digest_before"]==transition["optimizer_values_steps_and_other_groups_digest_after"]
        with np.load(folder/"actual_first_ppo_rollout.npz",allow_pickle=False) as trace:
            assert np.array_equal(trace["observations"],trace["sample_effective_observation"])
            gae=previous.gae_readback(trace);memory=memory_replay(trace,mode)
        results[mode]=dict(warm_actor7_exact_source68=True,optimizer_reset=reset,LR_transition=transition,
            first_PPO_GAE=gae,first_PPO_memory=memory)
    return dict(status="PASS_EACH_MODE_CONDITIONED_DATA_AND_GAE_NOT_CROSS_MODE_TRAJECTORY_PARITY",arms=results,
        initial_weights_identical=True,warm_physical_critic_reward_GAE_cross_mode_equality_required=False,
        reason="Different effective inputs change actions and on-policy physical data even while actor weights are frozen")


def pair_comparison(output):
    output=Path(output)
    records={mode:{label:read(output/mode/(label+"_validation.json")) for label in LABELS} for mode in ARMS}
    labels={mode:read(output/mode/"selection.json")["checkpoint"] for mode in ARMS}
    selected={mode:records[mode][labels[mode]] for mode in ARMS}
    own={mode:selected[mode]["own_baseline_admission"] for mode in ARMS}
    diagnostic=previous.compare_records(selected[ARMS[0]],selected[ARMS[1]])
    diagnostic["status"]="TRAINED_RAW_VS_CONDITIONED_INFORMATIONAL_NOT_PPO_GAIN_GATE"
    return dict(status="CONDITIONED_MEMORY_TRUE_PPO_GAIN_VS_OWN_BASELINE",selected_labels=labels,
        own_zero_update_baseline_admission=own,
        all_three_candidate_admissions={mode:{label:record["own_baseline_admission"] for label,record in records[mode].items()} for mode in ARMS},
        selected_raw_vs_conditioned_informational=diagnostic,
        all_three_phase_raw_vs_conditioned_informational={label:previous.compare_records(records[ARMS[0]][label],records[ARMS[1]][label]) for label in LABELS},
        actual_initialization_warm_and_first_PPO_evidence=semantic_evidence(output),
        conditioned_selected_true_PPO_gain_eligible_for_root_review_only=own[ARMS[1]]["passed"],
        reused_zero_update_baseline_episodes=48,new_baseline_episodes=0,
        source_actor68_unchanged_conditioning_gain_is_not_PPO_gain=True,
        zero_update_conditioned_baseline_remains_independent_candidate=True,
        automatic_extension_or_Gazebo_authorized=False)


def pair_plan(raw_plan,conditioned_plan):
    rows=[]
    for mode,supplied in zip(ARMS,(raw_plan,conditioned_plan)):
        path=gate.local_path(ROOT,supplied);plan=read(path);gate.validate(plan)
        assert plan["stage"]=="bounded_pilot" and plan["budget"]==BUDGET and plan["sources"]["driver"]["path"]==DRIVER
        argv=plan["argv"]
        assert option(argv,"--variant")==mode and option(argv,"--research-plan")==supplied
        assert option(argv,"--initial")==INITIAL_ZIP.as_posix() and plan["sources"]["initial_policy"]==dict(path=INITIAL_ZIP.as_posix(),sha256=INITIAL_ZIP_SHA)
        assert option(argv,"--baseline")==BASELINES[mode]
        assert option(argv,"--retention-reference")==previous.RETENTION and gate.sha(ROOT/previous.RETENTION)==previous.RETENTION_SHA
        assert option(argv,"--seed")=="9611" and option(argv,"--warmup-seed")=="9511"
        output=gate.local_path(ROOT,option(argv,"--output"));assert output.name==mode
        filtered=[];index=0
        while index<len(argv):
            if argv[index] in ("--variant","--baseline","--output","--research-plan"):index+=2;continue
            filtered.append(argv[index]);index+=1
        rows.append(dict(arm=mode,output=output,common_argv=filtered,
            command=[sys.executable,"-B",str(ROOT/"scripts/joint_research_preflight_v1.py"),"--plan",supplied,"--run"]))
    assert rows[0]["common_argv"]==rows[1]["common_argv"] and rows[0]["output"].parent==rows[1]["output"].parent
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-plan");parser.add_argument("--conditioned-plan");parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if not args.raw_plan and not args.conditioned_plan and not args.run:
        print(json.dumps(dict(status="PLAN_ONLY_NO_EXECUTION",arms=ARMS,budget=BUDGET,reused_baseline_episodes=48)));return 0
    if not args.raw_plan or not args.conditioned_plan:parser.error("Both bound plans required")
    try:rows=pair_plan(args.raw_plan,args.conditioned_plan)
    except (gate.PlanError,OSError,ValueError,TypeError,AssertionError) as exc:
        print(json.dumps(dict(status="BLOCKED",reason=str(exc))));return 2
    print(json.dumps(dict(status="PAIR_PLAN_PASS_NOT_ACCEPTANCE",will_execute=args.run)),flush=True)
    if not args.run:return 0
    if os.name!="posix":raise RuntimeError("Use WSL/Linux official preflight")
    output=rows[0]["output"].parent;output.mkdir(parents=True,exist_ok=False);jobs=[];streams=[]
    try:
        for row in rows:
            stream=(output/(row["arm"]+".stdout.log")).open("x",encoding="utf-8");streams.append(stream)
            jobs.append((row,subprocess.Popen(row["command"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)))
        results=[]
        for row,process in jobs:
            code=process.wait();path=row["output"]/"complete.json"
            results.append(dict(variant=row["arm"],returncode=code,complete=read(path) if path.exists() else None))
        status="complete" if all(r["returncode"]==0 and r["complete"] and r["complete"]["status"]=="complete" for r in results) else "error"
        receipt=dict(status=status,arms=results,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if status=="complete":
            try:
                result=pair_comparison(output)
                (output/"pair_comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
                receipt.update(comparison_sha256=gate.sha(output/"pair_comparison.json"),conditioned_selected_true_PPO_gain_eligible_for_root_review_only=result["conditioned_selected_true_PPO_gain_eligible_for_root_review_only"])
            except Exception as exc:status=receipt["status"]="error";receipt["pair_validation_error"]=repr(exc)
        (output/"paired_complete.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
        return 0 if status=="complete" else 1
    finally:
        for row,process in jobs:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for row,process in jobs:
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        for stream in streams:stream.close()


if __name__=="__main__":raise SystemExit(main())

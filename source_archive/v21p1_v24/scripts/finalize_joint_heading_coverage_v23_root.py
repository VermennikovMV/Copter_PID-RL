"""Read-only scientific inputs, fresh ROOT ledger after direct official preflight.

The frozen orchestrator's unused run_main passes summaries without episodes to
compare_records. This separate metadata writer uses the full selected records;
it never starts a learner, environment, policy forward or simulation.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/joint_route_heading_plans_v23/zero_route_integral.bound.json"
PLAN_SHA = "3b6aa7d35126adfbd8c2e8e0f50653915182055d49798037bbd002c1298191d7"
CONTROL = ROOT / "artifacts/refinement/joint_v2/route_memory_train_v21p1_111/zero_route_integral"
B = ROOT / "artifacts/refinement/joint_v2/route_heading_pilot_v23_111/zero_route_integral"
AUX = "audit/joint_heading_coverage_control_aux_v23_root01.json"
LABELS = ("checkpoint_131072", "checkpoint_262144", "final")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    args = parser.parse_args()
    execution_path = (ROOT / args.execution_receipt).resolve()
    assert execution_path.is_relative_to(ROOT)
    execution = read(execution_path)
    assert execution["actual_tool_exit_code"] == 0
    assert execution["plan_sha256"] == sha(ROOT / PLAN) == PLAN_SHA
    assert execution["entrypoint"] == "scripts/joint_research_preflight_v1.py --run DIRECT"
    assert execution["training_orchestrator_run_main_executed"] is False
    complete = read(B / "complete.json")
    assert complete["status"] == "complete"
    assert complete["critic_transitions"] == 131072 and complete["ppo_transitions"] == 262144
    assert complete["optimizer"]["actual_optimizer_steps"] == dict(critic=1024, ppo=2048)
    for name, digest in complete["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    aux = read(ROOT / AUX)
    assert aux["status"] == "COMPLETE_REUSED_A_PROFILES_NO_NEW_TRAINING"
    a_selection, b_selection = read(CONTROL / "selection.json"), read(B / "selection.json")
    a_path = CONTROL / (a_selection["checkpoint"] + "_validation.json")
    b_path = B / (b_selection["checkpoint"] + "_validation.json")
    a, b = read(a_path), read(b_path)
    assert len(a["episodes"]) == len(b["episodes"]) == 24
    assert complete["selected_policy_sha256"] == sha(B / "best_policy.npz") == b["policy_sha256"]
    count, controls, inputs = 15, aux["actual_counts"]["control_steps"], {}
    for label in LABELS:
        path = B / (label + "_validation.json")
        record = read(path)
        assert len(record["episodes"]) == 24
        extra = record["additional_matched_profiles_admission"]
        assert len(extra["episodes"]) == 5 and extra["not_averaged_into_original24"] is True
        count += 29
        controls += sum(e["steps"] for e in record["episodes"] + extra["episodes"])
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    assert count == 102 and controls <= 186000
    # This established helper is arithmetic on the full 24-case JSON records.
    # No memory_replay / NPZ loop, learner or evaluator is invoked.
    from scripts.run_joint_route_memory_pair_v21p1 import previous
    comparison = previous.compare_records(a, b)
    ledger = dict(
        status="ONE_NEW_B_COMPLETE_REUSED_A_NOT_INDEPENDENT_REPLICATION",
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        B_complete_sha256=sha(B / "complete.json"), A_control_aux_receipt_sha256=sha(ROOT / AUX),
        A_original_selected_admission=a_selection["own_baseline_admission"],
        B_selected_combined_admission=b_selection["own_baseline_admission"],
        B_eligible_for_ROOT_review_only=b_selection["own_baseline_admission"]["passed"],
        trained_A_vs_B_informational=comparison,
        new_evaluation_episodes=count, new_evaluation_control_steps_actual=controls,
        new_evaluation_control_steps_planned=186000, reused_coverage_episodes=101,
        reused_control_steps=178000, coverage_episodes=203, coverage_control_steps_planned=364000,
        A_training_newly_run=False, B_training_newly_run=True,
        no_automatic_extension_or_Gazebo=True,
        root_postprocessing=dict(
            source=Path(__file__).relative_to(ROOT).as_posix(), sha256=sha(Path(__file__)),
            execution_receipt=execution_path.relative_to(ROOT).as_posix(),
            execution_receipt_sha256=sha(execution_path), direct_official_preflight=True,
            unused_frozen_orchestrator_bug="Summary selection.json omits episodes; full *_validation.json required",
            frozen_training_or_evaluation_sources_modified=False,
            selected_A_full_validation_sha256=sha(a_path), selected_B_full_validation_sha256=sha(b_path),
            candidate_validation_sha256=inputs, new_NN_forwards=0,
            new_environment_constructions=0, new_physics_or_optimizer_steps=0))
    output = B.parent / "controlled_complete.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=ledger["status"], output=str(output), sha256=sha(output),
                          B_eligible=ledger["B_eligible_for_ROOT_review_only"])))


if __name__ == "__main__":
    main()

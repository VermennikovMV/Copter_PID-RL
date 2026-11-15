"""Matched finite-objective pair via two official preflights; plan-only default."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import joint_research_preflight_v1 as preflight
from scripts import run_joint_route_basis_pair_v15 as v15pair

ARMS = ("w8_control", "metric_cost")
LABELS, BUDGET = v15pair.LABELS, v15pair.BUDGET
DRIVER = "scripts/train_joint_metric_cost_v16.py"
BASELINE = "artifacts/refinement/joint_v2/route_basis_v15_baseline111/velocity_residual/baseline.json"
INITIAL, INITIAL_SHA = v15pair.INITIAL, v15pair.INITIAL_SHA
RETENTION, RETENTION_SHA = v15pair.RETENTION, v15pair.RETENTION_SHA
read, option = v15pair.read, v15pair.option


def warmup_physical_equivalence(output):
    """Reward/critic may differ, while the still-frozen warm actor must not."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    from scripts.bc_trust_region_probe_v1 import actor_name
    folders = [Path(output) / arm for arm in ARMS]
    states = [load_from_zip_file(folder / "critic_warmed_model.zip", device="cpu")[1] for folder in folders]
    for name, left in states[0]["policy"].items():
        if actor_name(name): assert torch.equal(left, states[1]["policy"][name]), name
    assert all(len(state["policy.optimizer"]["state"]) == 6 for state in states)
    critic_differences = [name for name in states[0]["policy"] if not actor_name(name)
        and not torch.equal(states[0]["policy"][name], states[1]["policy"][name])]
    with np.load(folders[0] / "warm_trace.npz", allow_pickle=False) as a, np.load(folders[1] / "warm_trace.npz", allow_pickle=False) as b:
        assert a.files == b.files and str(a["basis_kind"]) == str(b["basis_kind"]) == "velocity_residual"
        for name in a.files:
            if name == "reward": continue
            assert np.array_equal(a[name], b[name]), "Warm physical/actor/objective-input trace differs: " + name
        rows, reward_identical = len(a["time"]), np.array_equal(a["reward"], b["reward"])
        if not reward_identical: assert critic_differences, "Different warm objectives did not change any critic parameter"
    return dict(status="PASS_WARM_ACTOR_PHYSICS_EXACT_CRITIC_REWARD_MAY_DIFFER", physical_rows_per_arm=rows,
        actor_all_tensors_exact=True, physical_actions_common80_inputs_exact=True,
        actual_warm_rewards_identical=bool(reward_identical), changed_critic_tensors_between_arms=critic_differences,
        critic_Adam_equality_required=False, Adam_parameter_states_per_arm=6,
        snapshots=[dict(path=(folder / "critic_warmed_model.zip").relative_to(ROOT).as_posix(), sha256=preflight.sha(folder / "critic_warmed_model.zip")) for folder in folders])


def pair_comparison(output):
    output = Path(output)
    records = {arm: {label: read(output / arm / (label + "_validation.json")) for label in LABELS} for arm in ARMS}
    selected = {arm: read(output / arm / "selection.json")["checkpoint"] for arm in ARMS}
    alabel, blabel = selected[ARMS[0]], selected[ARMS[1]]
    against_selected = v15pair.compare_records(records[ARMS[0]][alabel], records[ARMS[1]][blabel])
    against_phase = v15pair.compare_records(records[ARMS[0]][blabel], records[ARMS[1]][blabel])
    source = v15pair.source77_nonregression(records[ARMS[1]][blabel])
    return dict(status="MATCHED_FINITE_OBJECTIVE_PAIR_NOT_POLICY_ACCEPTANCE", selected_labels=dict(A=alabel, B=blabel),
        warmup_pair_equivalence=warmup_physical_equivalence(output),
        selected_B_vs_selected_A=against_selected, selected_B_vs_same_phase_A=against_phase,
        selected_B_no_worse_e0aa_mean_geometry=source,
        all_three_matched_diagnostics={label: v15pair.compare_records(records[ARMS[0]][label], records[ARMS[1]][label]) for label in LABELS},
        eligible_for_root_review_only=against_selected["eligible_for_root_review_only"] and against_phase["eligible_for_root_review_only"] and source["passed"],
        B_selected_hard_guards_pass=records[ARMS[1]][blabel]["fixed_development_guards_pass"],
        early_matched_failures_are_visible_not_an_AND_gate=True, selection_independent_of_pair_gains=True,
        common_gamma=1., common_true_task_training_deadline=True, new_baseline_episodes=0,
        actual_Gazebo80_acceptance="NOT_EVALUATED", automatic_extension_authorized=False)


def pair_plan(a_plan, b_plan):
    rows = []
    for arm, supplied in zip(ARMS, (a_plan, b_plan)):
        path = preflight.local_path(ROOT, supplied); plan = read(path); preflight.validate(plan)
        assert plan["stage"] == "bounded_pilot" and plan["budget"] == BUDGET and plan["sources"]["driver"]["path"] == DRIVER
        argv = plan["argv"]
        assert option(argv, "--variant") == arm and option(argv, "--research-plan") == supplied
        assert option(argv, "--initial") == INITIAL and plan["sources"]["initial_policy"] == dict(path=INITIAL, sha256=INITIAL_SHA)
        assert option(argv, "--baseline") == BASELINE and option(argv, "--retention-reference") == RETENTION
        assert preflight.sha(ROOT / RETENTION) == RETENTION_SHA
        assert option(argv, "--seed") == "9611" and option(argv, "--warmup-seed") == "9511"
        output = preflight.local_path(ROOT, option(argv, "--output")); assert output.name == arm
        filtered, index = [], 0
        while index < len(argv):
            if argv[index] in ("--variant", "--output", "--research-plan"):
                index += 2; continue
            filtered.append(argv[index]); index += 1
        rows.append(dict(variant=arm, path=supplied, output=output, common_argv=filtered,
            command=[sys.executable, "-B", str(ROOT / "scripts/joint_research_preflight_v1.py"), "--plan", supplied, "--run"]))
    assert rows[0]["common_argv"] == rows[1]["common_argv"]
    assert rows[0]["output"].parent == rows[1]["output"].parent and rows[0]["output"] != rows[1]["output"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plan"); parser.add_argument("--metric-plan"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.control_plan and not args.metric_plan and not args.run:
        print(json.dumps(dict(status="PLAN_ONLY_NO_EXECUTION", arms=list(ARMS), budget=BUDGET,
            common_gamma=1., true_training_deadlines=True, reused_baseline=BASELINE, new_baseline_episodes=0)))
        return 0
    if not args.control_plan or not args.metric_plan: parser.error("Both bound arm plans required")
    try: rows = pair_plan(args.control_plan, args.metric_plan)
    except (preflight.PlanError, OSError, ValueError, TypeError, AssertionError) as exc:
        print(json.dumps(dict(status="BLOCKED", reason=str(exc)))); return 2
    print(json.dumps(dict(status="PAIRED_PLAN_CHECK_PASS_NOT_ACCEPTANCE", will_execute=args.run)), flush=True)
    if not args.run: return 0
    if os.name != "posix": raise RuntimeError("Use WSL/Linux official preflight")
    output = rows[0]["output"].parent; output.mkdir(parents=True, exist_ok=False)
    jobs, streams = [], []
    try:
        for row in rows:
            stream = (output / (row["variant"] + ".stdout.log")).open("x", encoding="utf-8"); streams.append(stream)
            jobs.append((row, subprocess.Popen(row["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)))
        results = []
        for row, process in jobs:
            code = process.wait(); path = row["output"] / "complete.json"
            results.append(dict(variant=row["variant"], returncode=code, complete=read(path) if path.exists() else None))
        status = "complete" if all(row["returncode"] == 0 and row["complete"] and row["complete"]["status"] == "complete" for row in results) else "error"
        receipt = dict(status=status, arms=results, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if status == "complete":
            try:
                comparison = pair_comparison(output)
                (output / "pair_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
                receipt.update(comparison_sha256=preflight.sha(output / "pair_comparison.json"), eligible_for_root_review_only=comparison["eligible_for_root_review_only"])
            except Exception as exc:
                status = receipt["status"] = "error"; receipt["pair_validation_error"] = repr(exc)
        (output / "paired_complete.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 0 if status == "complete" else 1
    finally:
        for row, process in jobs:
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        for row, process in jobs:
            if process.poll() is None:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        for stream in streams: stream.close()


if __name__ == "__main__":
    raise SystemExit(main())

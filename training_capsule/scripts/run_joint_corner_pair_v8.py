"""Read-only plan validation by default; run only two bound official preflights.

Both arms use the same learned77 initializer, actual77 actor/critic inputs,
tracker and coverage hold. The only arm factor is effective corner reward.
"""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import joint_research_preflight_v1 as preflight

INITIAL = "artifacts/refinement/joint_v2/path_pilot111/path/final.zip"
INITIAL_SHA = "7df5dd0a382525d08bfb4b2d8e4514e7269438bbd44a63598eff90ac0a5f3088"
RETENTION = "artifacts/refinement/joint_v2/path_v7_baseline111/path/baseline.json"
RETENTION_SHA = "b6d244d7c36165ee0340ac3118f8df98d4e6322011c24987c455d4d069c9074f"
DRIVER = "scripts/train_joint_corner_v8.py"
EXPECTED_BUDGET = dict(ppo_steps=262144, critic_steps=131072, eval_every=131072, candidate_count=3)
SCENARIOS = ("nominal", "force_strong", "gust")
SEEDS = (84001, 84002, 84003, 84004)
ARMS = ("w8_control", "corner")
LABELS = ("checkpoint_131072", "checkpoint_262144", "final")


def compare_records(control, corner):
    """Prospective eligibility for root review only; hard acceptance is separate."""
    def ratio(value, reference):
        if value is None or reference is None or not math.isfinite(value) or not math.isfinite(reference) or reference <= 0:
            return None
        return float(value / reference)
    def tail_case(record):
        found = [r for r in record["episodes"] if r["mission"] == "route" and
                 r["scenario"] == "force_strong" and int(r["eval_seed"]) == 84001]
        return found[0] if len(found) == 1 else None
    required = {"nominal/hover/initial_relative_precision", "force_strong/hover/wind_retention", "gust/hover/wind_retention"}
    required |= {scenario + "/route/" + field for scenario in SCENARIOS for field in ("restricted_time_s", "goal_rmse_m")}
    required |= {scenario + "/route/seed" + str(seed) + "/finite_segment_peak_m" for scenario in SCENARIOS for seed in SEEDS}
    constraints = {row["name"]: row for row in corner["constraints"]}
    missing = sorted(required - set(constraints))
    failed = [row for name, row in constraints.items() if name in required and not row["pass"]]
    safety = {key: corner["aggregate"][key] for key in
              ("flight_failures", "route_incomplete", "heading_violations", "geometry_ineligible")}
    retained = not missing and not failed and all(value == 0 for value in safety.values())
    cells = {}
    for scenario in SCENARIOS:
        key = scenario + "/route"
        left, right = control["aggregate"]["cells"][key], corner["aggregate"]["cells"][key]
        cells[scenario] = dict(w8_control=left, corner=right,
            ratios={field: ratio(right[field], left[field]) for field in
                    ("segment_rmse_m", "mean_corner_peak_m", "restricted_time_s", "goal_rmse_m")},
            hover_w8_control=control["aggregate"]["cells"][scenario + "/hover"],
            hover_corner=corner["aggregate"]["cells"][scenario + "/hover"])
    left, right = tail_case(control), tail_case(corner)
    tail_finite = ratio(right["geometry"]["segment_rmse_m"], left["geometry"]["segment_rmse_m"]) if left and right else None
    tail_corner = ratio(right["geometry"]["mean_corner_post_peak_m"], left["geometry"]["mean_corner_post_peak_m"]) if left and right else None
    checks = dict(retains_corner_non_geometry_guards=bool(retained),
        nominal_finite_ratio_le_0_97=cells["nominal"]["ratios"]["segment_rmse_m"] is not None and cells["nominal"]["ratios"]["segment_rmse_m"] <= .97,
        strong84001_finite_ratio_le_0_95=tail_finite is not None and tail_finite <= .95,
        each_scenario_mean_finite_nonincreasing=all(cells[s]["ratios"]["segment_rmse_m"] is not None and cells[s]["ratios"]["segment_rmse_m"] <= 1. for s in SCENARIOS),
        nominal_corner_nonincreasing=cells["nominal"]["ratios"]["mean_corner_peak_m"] is not None and cells["nominal"]["ratios"]["mean_corner_peak_m"] <= 1.,
        strong84001_corner_nonincreasing=tail_corner is not None and tail_corner <= 1.)
    return dict(status="PROSPECTIVE_DEVELOPMENT_COMPARISON_NOT_ACCEPTANCE", cells=cells,
        strong84001=dict(w8_control=left, corner=right, finite_ratio=tail_finite, corner_ratio=tail_corner),
        checks=checks, missing_non_geometry_constraints=missing, failed_non_geometry_constraints=failed,
        corner_safety_counts=safety, eligible_for_root_review_only=all(checks.values()),
        w8_control_fast_development_accepted=bool(control["fixed_development_guards_pass"]),
        corner_fast_development_accepted=bool(corner["fixed_development_guards_pass"]),
        actual_Gazebo_acceptance="NOT_EVALUATED_FOR_THESE_NEW_POLICY_BYTES",
        interpretation="Eligibility cannot authorize a longer run; hard finite/corner acceptance stays mandatory")


def pair_comparison(output):
    records, selections = {}, {}
    for arm in ARMS:
        selections[arm] = json.loads((output / arm / "selection.json").read_text())
        records[arm] = {label: json.loads((output / arm / (label + "_validation.json")).read_text()) for label in LABELS}
    selected = compare_records(records[ARMS[0]][selections[ARMS[0]]["checkpoint"]],
                               records[ARMS[1]][selections[ARMS[1]]["checkpoint"]])
    return dict(status="MEASURED_PAIRED_PILOT_COMPARISON_NOT_GAZEBO_ACCEPTANCE",
        selected_labels={arm: selections[arm]["checkpoint"] for arm in ARMS}, selected_comparison=selected,
        matched_steps={label: compare_records(records[ARMS[0]][label], records[ARMS[1]][label]) for label in LABELS},
        candidate_timing={arm: {label: dict(timesteps=record["timesteps"], completed_ppo_epochs=record["completed_ppo_epochs"],
            ppo_kl_gradient_steps=record["ppo_kl_gradient_steps"], timing=record["checkpoint_timing"])
            for label, record in records[arm].items()} for arm in ARMS},
        independent_replication_warning="All3 snapshots belong to the same trajectory; final is after the last update, not an independent model",
        automatic_long_run_or_demo_authorized=False)


def option(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise preflight.PlanError("Expected exactly one " + flag)
    return argv[argv.index(flag) + 1]


def baseline_equivalence(left_path, right_path):
    if Path(left_path).resolve() != Path(right_path).resolve():
        raise preflight.PlanError("Both identical Path77 arms must bind the SAME actual24 baseline")
    record = json.loads(Path(left_path).read_text(encoding="utf-8"))
    if record.get("schema") != "joint-corner-metric-ppo-v8" or record.get("initial_sha256") != INITIAL_SHA:
        raise preflight.PlanError("Fresh exact learned v7 initial/source baseline required")
    if record.get("actor_observations") != 77 or record.get("context_variant") != "path" or record.get("training_corner_overlay") is not False:
        raise preflight.PlanError("Baseline must use actual Path77 and no training corner overlay")
    cases = {(r["scenario"], r["mission"], int(r["eval_seed"])) for r in record["episodes"]}
    if len(record["episodes"]) != 24 or cases != {(s, m, k) for s in SCENARIOS for m in ("hover", "route") for k in SEEDS}:
        raise preflight.PlanError("Complete fixed24 case baseline required")
    for episode in record["episodes"]:
        folder = Path(episode["trace_path"]).parent
        for name, digest in episode["hashes"].items():
            if preflight.sha(folder / name) != digest:
                raise preflight.PlanError("Actual baseline evidence changed")
        if episode["metrics"]["survived"] and episode["steps"] != (1500 if episode["mission"] == "hover" else 2000):
            raise preflight.PlanError("Baseline full30/40 horizon changed")
    return dict(status="PASS_SHARED_ACTUAL24_BASELINE_SAME77_INPUTS", receipt=str(left_path),
        sha256=preflight.sha(left_path), separate_actual_flights=24, shared_between_arms=True)


def pair_plan(control_path, corner_path):
    rows = []
    for variant, supplied in zip(ARMS, (control_path, corner_path)):
        path = preflight.local_path(ROOT, supplied)
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        preflight.validate(plan)
        if plan["stage"] != "bounded_pilot" or plan["budget"] != EXPECTED_BUDGET:
            raise preflight.PlanError("Only fixed262144 bounded pilot is supported")
        if plan["sources"]["driver"]["path"] != DRIVER:
            raise preflight.PlanError("Unexpected paired driver")
        argv = plan["argv"]
        if option(argv, "--variant") != variant or option(argv, "--research-plan") != supplied:
            raise preflight.PlanError("Arm or own bound plan differs")
        if plan["sources"]["initial_policy"] != dict(path=INITIAL, sha256=INITIAL_SHA) or option(argv, "--initial") != INITIAL:
            raise preflight.PlanError("Both arms require same learned v7 Path77 ZIP")
        if option(argv, "--retention-reference") != RETENTION or preflight.sha(ROOT / RETENTION) != RETENTION_SHA:
            raise preflight.PlanError("Original Gamma77 retention anchor must remain frozen")
        if option(argv, "--seed") != "9611" or option(argv, "--warmup-seed") != "9511":
            raise preflight.PlanError("Paired RNG seeds differ")
        baseline = preflight.local_path(ROOT, option(argv, "--baseline"))
        command = [sys.executable, "-B", str(ROOT / "scripts/joint_research_preflight_v1.py"), "--plan", supplied, "--run"]
        rows.append(dict(variant=variant, path=supplied, plan_sha256=preflight.sha(path), argv=argv,
            command=command, output=preflight.local_path(ROOT, option(argv, "--output")), baseline=baseline))
    proof = baseline_equivalence(rows[0]["baseline"], rows[1]["baseline"])
    for row in rows:
        row["baseline_equivalence"] = proof
        if row["output"].name != row["variant"]:
            raise preflight.PlanError("Output basename MUST equal canonical arm name")
        filtered, index = [], 0
        while index < len(row["argv"]):
            flag = row["argv"][index]
            if flag in ("--variant", "--output", "--research-plan"):
                index += 2
                continue
            filtered.append(flag); index += 1
        row["common_argv"] = filtered
    if rows[0]["common_argv"] != rows[1]["common_argv"]:
        raise preflight.PlanError("Arguments other than reward arm/output/own plan differ")
    if rows[0]["output"].parent != rows[1]["output"].parent or rows[0]["output"] == rows[1]["output"]:
        raise preflight.PlanError("Distinct fresh canonical arms under one new pair directory required")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plan")
    parser.add_argument("--corner-plan")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.control_plan and not args.corner_plan and not args.run:
        print(json.dumps(dict(status="PLAN_ONLY_NO_EXECUTION", driver=DRIVER, budget=EXPECTED_BUDGET,
            initial=INITIAL, initial_sha256=INITIAL_SHA, retention_reference=RETENTION,
            retention_sha256=RETENTION_SHA, variants=list(ARMS),
            next="Bind both root-reviewed13-metric plans; default remains read-only")))
        return 0
    if not args.control_plan or not args.corner_plan:
        parser.error("Both bound arm plans required")
    try:
        rows = pair_plan(args.control_plan, args.corner_plan)
    except (preflight.PlanError, OSError, ValueError, TypeError) as exc:
        print(json.dumps(dict(status="BLOCKED", reason=str(exc))))
        return 2
    print(json.dumps(dict(status="PAIRED_PLAN_CHECK_PASS_NOT_POLICY_ACCEPTANCE", will_execute=args.run,
        plans=[dict(variant=row["variant"], plan=row["path"], sha256=row["plan_sha256"], actual_preflight_command=row["command"])
               for row in rows])), flush=True)
    if not args.run:
        return 0
    if os.name != "posix":
        raise RuntimeError("Execute guarded pilot inside WSL/Linux")
    output = rows[0]["output"].parent
    output.mkdir(parents=True, exist_ok=False)
    jobs, streams = [], []
    try:
        for row in rows:
            stream = (output / (row["variant"] + ".stdout.log")).open("x", encoding="utf-8")
            streams.append(stream)
            process = subprocess.Popen(row["command"], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((row, process))
        records = []
        for row, process in jobs:
            code = process.wait()
            complete_path = row["output"] / "complete.json"
            complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
            records.append(dict(variant=row["variant"], returncode=code, complete=complete, plan_sha256=row["plan_sha256"]))
        status = "complete" if all(r["returncode"] == 0 and r["complete"] and r["complete"]["status"] == "complete" for r in records) else "error"
        receipt = dict(status=status, arms=records, policy_acceptance_is_separate=True,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if status == "complete":
            comparison = pair_comparison(output)
            (output / "pair_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            receipt["comparison_sha256"] = preflight.sha(output / "pair_comparison.json")
            receipt["prospective_eligibility_for_root_review_only"] = comparison["selected_comparison"]["eligible_for_root_review_only"]
        (output / "paired_complete.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 0 if status == "complete" else 1
    finally:
        for row, process in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for row, process in jobs:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())

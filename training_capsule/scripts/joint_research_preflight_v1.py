"""Check a prospective metric-aligned plan; execute only with explicit --run.

This validates evidence bindings and completeness, not scientific semantics.
Frozen historical training drivers and their results are never edited here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/joint_applicability_contract_v2.json"
CONTRACT_SHA = "41c524713d346eb09b480d904373d053363f83dcd682daf0d1ac4c700bbae9f4"
RESERVED = {93001, 93002, 93003}
METRICS = (
    "position_goal_rmse", "height_max_down", "recovery_time",
    "stability_full_flight", "route_completion_time", "heading",
    "nominal_hover_accuracy", "wind_hover_gain", "wind_specialist_retention",
    "route_finite_segment_rmse", "route_corner_peak", "route_absolute_peak",
    "thesis_six_cell_macro",
)
ROLES = (
    "driver", "reward", "selection", "observation", "fast_plant",
    "gazebo_runtime", "metric_judge", "native_pid", "initial_policy",
)


class PlanError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PlanError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root, relative):
    require(isinstance(relative, str) and relative, "Expected a project-relative path")
    given = Path(relative)
    require(not given.is_absolute() and "\\" not in relative and ":" not in relative,
            "Use a relative path with forward slashes")
    result = (root / given).resolve()
    require(result.is_relative_to(root.resolve()), "Path escapes project")
    return result


def bound_file(root, binding):
    require(isinstance(binding, dict), "Expected a path/SHA binding")
    path = local_path(root, binding.get("path"))
    require(path.is_file(), f"Missing file: {binding.get('path')}")
    require(sha(path) == binding.get("sha256"), f"SHA mismatch: {binding.get('path')}")
    return path


def explanation(value, label):
    require(isinstance(value, str) and len(value.strip()) >= 12,
            f"Missing substantive explanation: {label}")
    require(not any(token in value.casefold() for token in ("todo", "fill_me", "заполнить")),
            f"Unfilled explanation: {label}")


def json_pointer(document, pointer):
    require(isinstance(pointer, str) and pointer.startswith("/") and len(pointer) > 1,
            "Evidence needs a nonempty JSON pointer")
    value = document
    try:
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                require(key.isdigit(), "Invalid list index in evidence pointer")
                value = value[int(key)]
            else:
                value = value[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlanError(f"Missing evidence field: {pointer}") from exc
    return value


def validate(plan, root=ROOT):
    require(isinstance(plan, dict), "Plan must be a JSON object")
    require(plan.get("schema") == "joint-research-plan-v1", "Wrong plan schema")
    require(plan.get("status") == "READY_FOR_RUN", "Plan is still a draft")
    require(plan.get("stage") in ("bounded_pilot", "long_training"), "Unknown stage")
    contract = plan.get("contract", {})
    require(contract == {"path": CONTRACT, "sha256": CONTRACT_SHA},
            "Fixed contract identity changed")
    bound_file(root, contract)
    for key in ("hypothesis", "changed_factors", "expected_effect", "reconsider_if"):
        explanation(plan.get(key), key)
    sources = plan.get("sources", {})
    require(set(sources) == set(ROLES), "Bind all required source roles")
    for role in ROLES:
        bound_file(root, sources[role])
    contract_data = json.loads((root / CONTRACT).read_text(encoding="utf-8-sig"))
    require(sources["native_pid"] == {
        "path": contract_data["frozen_pid_profile"]["path"],
        "sha256": contract_data["frozen_pid_profile"]["sha256"]},
        "Competent PID must retain the frozen contract identity")
    driver = sources["driver"]["path"]
    require(driver.startswith("scripts/train_joint_") and driver.endswith(".py"),
            "Expected a versioned joint training driver")
    alignment = plan.get("metric_alignment", {})
    require(set(alignment) == set(METRICS), "Every required metric must be mapped")
    for metric in METRICS:
        row = alignment[metric]
        require(isinstance(row, dict), f"Invalid mapping: {metric}")
        for key in ("reward", "checkpoint", "evaluation", "known_gap"):
            explanation(row.get(key), f"{metric}.{key}")
    seeds = plan.get("development_environment_seeds")
    require(isinstance(seeds, list) and seeds and
            all(type(seed) is int and seed >= 0 for seed in seeds), "List development seeds")
    require(not RESERVED.intersection(seeds), "Reserved final seed used for development")
    required_evidence = {"baseline", "diagnosis", "reward_semantics"}
    if plan["stage"] == "long_training":
        required_evidence.add("short_pilot")
    evidence = plan.get("evidence", {})
    require(set(evidence) == required_evidence, "Missing prerequisite evidence")
    for role, binding in evidence.items():
        path = bound_file(root, binding)
        require(path.suffix == ".json", "Evidence must be inspectable JSON")
        explanation(binding.get("interpretation"), f"evidence.{role}.interpretation")
        expected = binding.get("equals")
        require(expected not in (None, False, "", "FAIL", "ERROR", "INDETERMINATE"),
                f"Evidence does not assert a successful prerequisite: {role}")
        actual = json_pointer(json.loads(path.read_text(encoding="utf-8-sig")), binding.get("pointer"))
        require(type(actual) is type(expected) and actual == expected,
                f"Evidence assertion failed: {role}")
    command = plan.get("argv")
    require(isinstance(command, list) and all(isinstance(arg, str) for arg in command),
            "argv must contain separate literal arguments")
    require(all("\x00" not in arg for arg in command), "Invalid argument")
    def option(name):
        require(command.count(name) == 1, f"Exactly one {name} is required")
        index = command.index(name) + 1
        require(index < len(command), f"Missing value for {name}")
        require(not any(arg.startswith(name + "=") for arg in command), f"Duplicate {name}")
        return command[index]
    output = local_path(root, option("--output"))
    require(output.is_relative_to((root / "artifacts/refinement/joint_v2").resolve()),
            "Use a new joint_v2 output namespace")
    require(not output.exists(), "Output already exists; preserve earlier work")
    budget = plan.get("budget", {})
    for name, flag in (("ppo_steps", "--steps"), ("critic_steps", "--warmup-steps"),
                       ("eval_every", "--eval-every")):
        value = budget.get(name)
        require(type(value) is int and value > 0 and str(value) == option(flag),
                f"Budget/command mismatch: {name}")
    require(budget["ppo_steps"] % budget["eval_every"] == 0, "Partial checkpoint budget")
    require(budget.get("candidate_count") == budget["ppo_steps"] // budget["eval_every"] + 1,
            "Freeze the candidate count, including final")
    initial = local_path(root, option("--initial"))
    require(initial == local_path(root, sources["initial_policy"]["path"]),
            "Command initial policy differs from frozen input")
    require(not any(arg in ("--verify", "--baseline-only", "--help") for arg in command),
            "Verification-only commands are not training plans")
    return [sys.executable, "-B", str(root / driver), *command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Project-relative prospective JSON plan")
    parser.add_argument("--run", action="store_true", help="Execute only after all checks pass")
    args = parser.parse_args()
    try:
        path = local_path(ROOT, args.plan)
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        command = validate(plan)
    except (PlanError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "PLAN_CHECK_PASS_NOT_POLICY_ACCEPTANCE", "plan_sha256": sha(path),
                      "metric_count": len(METRICS), "will_execute": args.run}, ensure_ascii=False),
          flush=True)
    if args.run:
        # Recheck bound bytes immediately before launching the frozen driver.
        command = validate(plan)
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

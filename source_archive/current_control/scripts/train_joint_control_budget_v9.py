"""Guarded duration study of unchanged W8/Path77 control, using frozen v8.

No arguments or --help is read-only. --verify-only uses archived inputs and a
tiny synthetic Adam fixture; it never constructs or steps an environment.
Training requires the official preflight --run parent and a long_training plan.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import joint_research_preflight_v1 as preflight
from scripts import train_joint_corner_v8 as frozen
from stable_baselines3.common.save_util import load_from_zip_file
from torch.optim.optimizer import register_optimizer_step_post_hook
import torch

DRIVER = "scripts/train_joint_control_budget_v9.py"
DOCS = ROOT / "docs/joint_control_budget_v9"
ENGINE_SHA = "32625b344255328b78249a0ebf4b85852f29e504fbef316285dc0f8a933f1aba"
AUDIT = ROOT / "audit/joint_corner_pilot_v8_root01.json"
AUDIT_SHA = "ba68e8c3669269eedaf98cdf4918e664f9402f8bc0f454bc5a978570853bc61c"
DECISION = ROOT / "audit/joint_control_budget_decision_v9_root01.json"
DECISION_SHA = "6846597880d097b6ea7b7ba9dec553182db40a1f98978466d47b12bce02f9cf9"
BASELINE = ROOT / "artifacts/refinement/joint_v2/corner_v8_baseline111/baseline.json"
BASELINE_SHA = "414f1ceeaf3ef9c777143300cd57ed979432d3d79fb61a4a8dfd4bf3e5fdb756"
OLD = ROOT / "artifacts/refinement/joint_v2/corner_pilot111/w8_control"
OLD_CONFIG_SHA = "8bf82409e7de955ac4abaf0628595eac2d4fb1e5182890a9a6049a74e1110e8d"
OLD_ZIPS = {
    "checkpoint_131072": "cfa30cd0b090435c87d093e46bc0b54589fedc425a5662360cb9fbb7de110319",
    "checkpoint_262144": "0371463a3545622e69a30ee82fd0d3c0dfe5fb87f444fff3e0e1820aa9a4e131",
    "final": "e81c587177832377307bd94c59be3bcb68245dfba64efad140df54a667257302",
}
BUDGET = dict(ppo_steps=1048576, critic_steps=131072, eval_every=131072, candidate_count=9)
PREFIX_TOTAL_STEPS = 3072
FINAL_TOTAL_STEPS = 9216  # 1024 new critic-only + 8192 new PPO Adam steps.


class PrefixReproductionError(AssertionError):
    pass


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bind(path):
    return dict(path=Path(path).relative_to(ROOT).as_posix(), sha256=frozen.sha(path))


def archived_inputs():
    """Read evidence only; do not rerun physics, baseline flights or reward probes."""
    for path, digest in ((ROOT / "scripts/train_joint_corner_v8.py", ENGINE_SHA),
                         (AUDIT, AUDIT_SHA), (BASELINE, BASELINE_SHA),
                         (DECISION, DECISION_SHA),
                         (OLD / "config.json", OLD_CONFIG_SHA),
                         (frozen.INITIAL, frozen.INITIAL_SHA256)):
        require(frozen.sha(path) == digest, "Frozen input changed: " + str(path))
    audit = read_json(AUDIT)
    require(audit["status"] == "PASS", "Independent pilot archive audit must pass")
    require(audit["selected_comparison"]["eligible_for_root_review_only"] is False,
            "Failed K40 eligibility must remain false; this is a separate control study")
    decision = read_json(DECISION)
    require(decision["status"] == "CONTROL_LEARNING_PROGRESS_VERIFIED" and
            decision["rejected_K40_branch"]["selected_eligibility"] is False and
            decision["rejected_K40_branch"]["extend"] is False,
            "Root decision must justify control progress and retain the rejected K40 branch")
    hashes = {path.relative_to(ROOT).as_posix(): frozen.sha(path) for path in frozen.source_files()}
    config = read_json(OLD / "config.json")
    require(hashes == config["source_hashes"],
            "Frozen producer source map changed; do not patch v8 or its dependencies")
    require(hashlib.sha256(inspect.getsource(frozen.PPO.train).encode()).hexdigest() == config["installed_sb3_ppo_train_sha256"],
            "Installed SB3 PPO.train differs from the archived short control")
    for label, digest in OLD_ZIPS.items():
        require(frozen.sha(OLD / (label + ".zip")) == digest, "Old prefix ZIP changed: " + label)
    frozen.physics_evidence()
    frozen.retention_reference()
    return hashes


def model_contract(model):
    require(model.gamma == .995 and model.gae_lambda == .97, "Frozen gamma/GAE required")
    require((model.n_steps, model.n_epochs, model.batch_size) == (512, 8, 1024),
            "Inherited rollout, epoch and minibatch sizes changed")
    require([model.lr_schedule(p) for p in (0., .5, 1.)] == [3e-5] * 3,
            "Prefix reproduction requires constant learning rate at progress 0/.5/1")
    require([model.clip_range(p) for p in (0., .5, 1.)] == [.15] * 3,
            "Prefix reproduction requires constant clipping at progress 0/.5/1")
    named = dict(model.policy.named_parameters())
    order = [next(name for name, value in named.items() if value is parameter)
             for group in model.policy.optimizer.param_groups for parameter in group["params"]]
    require(len(order) == 13 and len(set(order)) == 13 and set(named) == set(model.policy.state_dict()),
            "Exactly all 13 named policy tensors must be aligned with the Adam parameter order")
    require(model.observation_space.shape == (77,), "Unchanged actual Path77 input required")
    return order


def validate_arguments(args):
    require(args.variant == "w8_control", "Only unchanged W8 control is authorized")
    require((args.steps, args.warmup_steps, args.eval_every, args.seed, args.warmup_seed) ==
            (1048576, 131072, 131072, 9611, 9511), "Only fixed 1M/131072/131072 and seeds 9611/9511")
    require(args.initial == frozen.INITIAL.resolve(), "Exact learned v7 initializer required")
    require(args.baseline == BASELINE.resolve(), "Reuse the identical actual 414f initial baseline")
    require(args.retention_reference == frozen.RETENTION_REFERENCE.resolve(),
            "Original Gamma77 anchor is separate and cannot be moved")
    require(args.output is not None and args.output.name == "w8_control" and not args.output.exists(),
            "Fresh output with basename w8_control required")
    require(args.output.is_relative_to((ROOT / "artifacts/refinement/joint_v2").resolve()),
            "Use a fresh joint_v2 study namespace")


def require_bound_preflight(args):
    """Own long-training guard; frozen v8 bounded-pilot main is never called."""
    require(args.research_plan is not None, "Use joint_research_preflight_v1.py --plan <bound> --run")
    path = args.research_plan.resolve()
    require(path.is_relative_to(ROOT), "Bind a project-local prospective plan")
    plan = read_json(path)
    command = preflight.validate(plan)
    require(Path(command[2]).resolve() == Path(__file__).resolve() and sys.argv[1:] == plan["argv"],
            "Invocation differs from this driver's exact bound plan")
    require(plan["stage"] == "long_training" and plan["budget"] == BUDGET,
            "This study needs its own 13-metric long_training plan with the fixed nine candidates")
    for role, relative, digest in (
            ("reward", "scripts/train_joint_corner_v8.py", ENGINE_SHA),
            ("selection", "scripts/train_joint_corner_v8.py", ENGINE_SHA),
            ("observation", "quadrl_path_v7/policy.py", read_json(OLD / "config.json")["source_hashes"]["quadrl_path_v7/policy.py"]),
            ("fast_plant", "quadrl_corner_v8/env.py", read_json(OLD / "config.json")["source_hashes"]["quadrl_corner_v8/env.py"])):
        require(all(plan["sources"][role].get(k) == v for k, v in dict(path=relative, sha256=digest).items()),
                "Plan changed frozen " + role)
    try:
        parent = (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().decode("utf-8").split("\0")
        supplied = parent[parent.index("--plan") + 1]
        require(any(Path(arg).name == "joint_research_preflight_v1.py" for arg in parent) and
                "--run" in parent and (ROOT / supplied).resolve() == path,
                "Official preflight --run parent with this same plan is required")
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError("Direct training blocked; use the official bound preflight --run parent") from exc
    return plan


def snapshot(path):
    # load_data=False avoids constructing a policy or deserializing saved schedules.
    _, parameters, _ = load_from_zip_file(path, device="cpu", load_data=False)
    return parameters["policy"], parameters["policy.optimizer"]


def tensor_comparison(actual, expected):
    same_layout = actual.shape == expected.shape and actual.dtype == expected.dtype
    exact = same_layout and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    error = float((actual.detach().cpu().double() - expected.detach().cpu().double()).abs().max()) if same_layout else None
    return dict(exact=bool(exact), max_abs_error=error if error is not None and math.isfinite(error) else None,
                numerical_error_finite=error is not None and math.isfinite(error),
                shape=list(actual.shape), dtype=str(actual.dtype))


def compare_snapshot(actual, expected, names, actor_steps, critic_steps):
    """Compare every tensor and every Adam field, including groups and real step counters."""
    policy, optimizer = actual
    old_policy, old_optimizer = expected
    require(set(policy) == set(old_policy) == set(names) and len(names) == 13,
            "All 13 policy tensors required; partial comparisons are forbidden")
    policy_rows = {name: tensor_comparison(policy[name], old_policy[name]) for name in names}
    ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    old_ids = [index for group in old_optimizer["param_groups"] for index in group["params"]]
    require(len(ids) == len(old_ids) == 13 and len(set(ids)) == len(set(old_ids)) == 13,
            "Adam parameter order is incomplete")
    require(set(optimizer["state"]) == set(ids) and set(old_optimizer["state"]) == set(old_ids),
            "Every actor and critic Adam state is required")
    rows, step_mapping = {}, {}
    for name, index, old_index in zip(names, ids, old_ids):
        state, old_state = optimizer["state"][index], old_optimizer["state"][old_index]
        require(set(state) == set(old_state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam fields")
        rows[name] = {field: tensor_comparison(state[field], old_state[field]) for field in state}
        expected_steps = actor_steps if frozen.actor_name(name) else critic_steps
        step_mapping[name] = dict(actual=int(state["step"]), reference=int(old_state["step"]), expected=expected_steps)
    groups_equal = optimizer["param_groups"] == old_optimizer["param_groups"]
    exact = (all(row["exact"] for row in policy_rows.values()) and groups_equal and
             all(row["exact"] for fields in rows.values() for row in fields.values()) and
             all(row["actual"] == row["reference"] == row["expected"] for row in step_mapping.values()))
    return dict(exact=bool(exact), policy_tensor_count=13, policy=policy_rows, adam=rows,
                adam_parameter_groups_exact=groups_equal, adam_step_mapping=step_mapping)


class ScopedStepObserver:
    """Global post-step API, filtered to one live optimizer; never updates it."""
    def __init__(self, resolve_model, on_step):
        self.resolve_model, self.on_step = resolve_model, on_step
        self.optimizer = None
        self.actual_steps = 0

    def __call__(self, optimizer, unused_args, unused_kwargs):
        model = self.resolve_model(optimizer)
        if model is None:
            return
        if self.optimizer is None:
            self.optimizer = optimizer
        require(self.optimizer is optimizer, "Active training optimizer unexpectedly replaced")
        self.actual_steps += 1
        self.on_step(model, self.actual_steps)


class PrefixProof:
    def __init__(self, output, names, references):
        self.output, self.names, self.references = output, names, references
        self.checks = {}
        self.live_snapshot = None
        self.observer = ScopedStepObserver(self.active_model, self.on_step)

    def active_model(self, optimizer):
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code is frozen.run_arm.__code__:
                    model = frame.f_locals.get("model")
                    if frame.f_locals.get("output") == self.output and model is not None and model.policy.optimizer is optimizer:
                        return model
                frame = frame.f_back
            return None
        finally:
            del frame

    def on_step(self, model, count):
        if count == 1:
            require(model_contract(model) == self.names, "Live optimizer name/order differs from fresh initial")
        if count in (1, 1024, 1025, 1985, 3009, PREFIX_TOTAL_STEPS):
            require(all(group["lr"] == 3e-5 for group in model.policy.optimizer.param_groups), "Live LR changed")
        targets = {1985: ("checkpoint_131072", 960, 1984, 131072),
                   3009: ("checkpoint_262144", 1984, 3008, 262144),
                   PREFIX_TOTAL_STEPS: ("final", 2048, 3072, 262144)}
        if count not in targets:
            return
        label, actor_steps, critic_steps, timesteps = targets[count]
        require(model.num_timesteps == timesteps, "Actual prefix timestep/optimizer mapping changed")
        if label == "final":
            self.live_snapshot = ({name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()},
                                  copy.deepcopy(model.policy.optimizer.state_dict()))
            actual = self.live_snapshot
        else:
            actual = snapshot(self.output / (label + ".zip"))
        report = compare_snapshot(actual, self.references[label], self.names, actor_steps, critic_steps)
        self.checks[label] = dict(observed_actual_optimizer_steps=count, ppo_num_timesteps=model.num_timesteps,
                                 compared_actor_steps=actor_steps, compared_critic_steps=critic_steps,
                                 source="live post-step model" if label == "final" else "new frozen-producer saved checkpoint",
                                 reference=bind(OLD / (label + ".zip")), **report)
        if not report["exact"]:
            raise PrefixReproductionError("Exact old-control prefix reproduction FAILED at " + label +
                                          "; stopped immediately without retry or budget extension")

    def record(self):
        return dict(schema="joint-control-budget-prefix-proof-v9.0", checks=self.checks,
                    exact_prefix_verified=len(self.checks) == 3 and all(row["exact"] for row in self.checks.values()),
                    actual_optimizer_steps_observed=self.observer.actual_steps,
                    expected_prefix_steps=dict(critic_only=1024, new_ppo=2048, total=3072),
                    inherited_epochs_are_not_new_optimizer_steps=True,
                    observation_only=True, automatic_retry=False)


def synthetic_observer_test():
    """Tiny constant-tensor Adam fixture. These 3072 test steps are NOT PPO."""
    parameters = [torch.nn.Parameter(torch.full((1,), .125)) for _ in range(13)]
    optimizer = torch.optim.Adam(parameters, lr=3e-5, eps=1e-5)
    decoy_parameter = torch.nn.Parameter(torch.zeros(1))
    decoy = torch.optim.Adam([decoy_parameter], lr=3e-5)
    marks = {}
    local_steps = [0]
    def local_mark(*unused):
        local_steps[0] += 1
    def mark(unused_model, count):
        if count in (1024, 3072):
            marks[count] = {index: int(state["step"]) for index, state in optimizer.state_dict()["state"].items()}
    observer = ScopedStepObserver(lambda current: parameters if current is optimizer else None, mark)
    rng_before = torch.random.get_rng_state().clone()
    handle = register_optimizer_step_post_hook(observer)
    local_handle = optimizer.register_step_post_hook(local_mark)
    try:
        for count in range(1, 3073):
            for index, parameter in enumerate(parameters):
                parameter.grad = torch.ones_like(parameter) if index >= 7 or count > 1024 else None
            optimizer.step()
        decoy_parameter.grad = torch.ones_like(decoy_parameter)
        decoy.step()
    finally:
        handle.remove()
        local_handle.remove()
    require(observer.actual_steps == 3072 and len(marks[1024]) == 6,
            "Synthetic scope/count/critic-only check failed")
    require(local_steps[0] == 3072, "Observer interfered with the producer-style local proof hook")
    require(all(marks[3072][index] == (2048 if index < 7 else 3072) for index in range(13)),
            "Synthetic actor/critic step mapping failed")
    require(torch.equal(rng_before, torch.random.get_rng_state()), "Observer fixture changed torch RNG")
    def fail(unused_model, unused_count):
        raise PrefixReproductionError("synthetic deliberate reproduction failure")
    failing = ScopedStepObserver(lambda current: parameters if current is optimizer else None, fail)
    handle = register_optimizer_step_post_hook(failing)
    failure_propagated = False
    try:
        optimizer.step()
    except PrefixReproductionError:
        failure_propagated = True
    finally:
        handle.remove()
    require(failure_propagated and failing.actual_steps == 1, "Observer failure did not stop immediately")
    return dict(status="PASS", fixture="13 tiny constant tensors; no policy/env/rollout/PPO",
                synthetic_prefix_Adam_steps=3072, synthetic_failure_test_Adam_steps=1,
                ignored_decoy_steps=1, local_proof_hook_coexists=True,
                failure_propagated_immediately=True, torch_rng_unchanged=True)


def verify_only():
    hashes = archived_inputs()
    source = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == "learn" for node in ast.walk(source)), "Wrapper must not call learn")
    require(list(inspect.signature(frozen.run_arm).parameters) == ["args", "variant", "output", "baseline", "receipt"],
            "Frozen run_arm API changed")
    model = frozen.BCTrustPPO.load(frozen.INITIAL, device="cpu")
    names = model_contract(model)
    refs = {label: snapshot(OLD / (label + ".zip")) for label in OLD_ZIPS}
    counts = {"checkpoint_131072": (960, 1984), "checkpoint_262144": (1984, 3008), "final": (2048, 3072)}
    for label, (actor_steps, critic_steps) in counts.items():
        require(compare_snapshot(refs[label], refs[label], names, actor_steps, critic_steps)["exact"],
                "Archived prefix Adam mapping invalid")
    # Prove that tiny errors remain failures; no numerical tolerance admits a prefix.
    changed = {name: value.clone() for name, value in refs["final"][0].items()}
    changed[names[0]].view(-1)[0] += 1e-6
    require(not compare_snapshot((changed, refs["final"][1]), refs["final"], names, 2048, 3072)["exact"],
            "Exact comparator accepted a perturbed policy")
    old_optimizer = refs["final"][1]
    altered_optimizer = {**old_optimizer, "state": {key: {field: value.clone() for field, value in state.items()}
                                                  for key, state in old_optimizer["state"].items()}}
    altered_optimizer["state"][0]["exp_avg"].view(-1)[0] += 1e-9
    require(not compare_snapshot((refs["final"][0], altered_optimizer), refs["final"], names, 2048, 3072)["exact"],
            "Exact comparator accepted perturbed Adam state")
    frozen.baseline_receipt(frozen.INITIAL, ROOT / "unused_v9_baseline", BASELINE, variant="path")
    return dict(status="PASS_STATIC_AND_SYNTHETIC_ONLY", driver=bind(Path(__file__)), frozen_source_count=len(hashes),
                policy_parameter_order=names, all_13_policy_and_Adam_fields_checked=True,
                archived_prefix_step_mapping_valid=True, one_micro_policy_perturbation_rejected=True,
                tiny_Adam_state_perturbation_rejected=True, wrapper_has_no_learn_call=True,
                root_control_progress_decision=bind(DECISION),
                constant_lr_and_clip_at_progress_0_half_1=True, synthetic=synthetic_observer_test(),
                old_failed_K40_eligible_for_root_review_only=False, training_executed=False, Gazebo_executed=False)


def run(args, plan):
    hashes = archived_inputs()
    wrapper_binding = bind(Path(__file__))
    model = frozen.BCTrustPPO.load(args.initial, device="cpu")
    names = model_contract(model)
    del model
    baseline, receipt = frozen.baseline_receipt(args.initial, args.output.with_name("unused_baseline"), args.baseline, variant="path")
    proof = PrefixProof(args.output, names, {label: snapshot(OLD / (label + ".zip")) for label in OLD_ZIPS})
    handle = register_optimizer_step_post_hook(proof.observer)
    completed, error = None, None
    try:
        completed = frozen.run_arm(args, "w8_control", args.output, baseline, receipt)
        require(proof.record()["exact_prefix_verified"], "All three prefix reproductions are required")
        require(proof.observer.actual_steps == FINAL_TOTAL_STEPS, "Fixed full-budget optimizer count changed")
        require(completed["ppo_timesteps"] == BUDGET["ppo_steps"] and len(completed["candidate_records"]) == 9,
                "Full fixed budget must retain all nine candidates")
        require(hashes == archived_inputs(), "Frozen source map changed during the study")
        require(bind(Path(__file__)) == wrapper_binding, "Wrapper changed during the study")
        return completed
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        handle.remove()
        # Separate wrapper files never rewrite producer v8 config/schema/manifests.
        if args.output.is_dir():
            prefix_record = proof.record()
            if proof.live_snapshot is not None:
                tensors_path = args.output / "wrapper_v9_live_short_final.pt"
                torch.save(dict(policy=proof.live_snapshot[0], optimizer=proof.live_snapshot[1],
                                parameter_order=names, actual_new_optimizer_steps=3072,
                                new_actor_steps=2048, new_critic_steps=3072, ppo_num_timesteps=262144), tensors_path)
                prefix_record["captured_live_short_final_tensors"] = bind(tensors_path)
            frozen.write_json(args.output / "wrapper_v9_prefix_proof.json", prefix_record)
            frozen.write_json(args.output / "wrapper_v9_provenance.json", dict(
                schema="joint-control-budget-duration-study-v9.0", status="COMPLETE" if completed and error is None else "FAILED",
                error=error, wrapper=wrapper_binding, frozen_engine=bind(ROOT / "scripts/train_joint_corner_v8.py"),
                study_change="Prospective fixed duration only; exact unchanged w8_control engine/reward/selection",
                root_control_progress_decision=bind(DECISION), independent_old_pilot_audit=bind(AUDIT),
                old_failed_K40_eligible_for_root_review_only=False, effective_corner_reward_weight=0.,
                budget=BUDGET, plan=bind(args.research_plan), evidence=plan["evidence"],
                initial=bind(frozen.INITIAL), prior=bind(frozen.INITIAL), shared_initial_baseline=bind(BASELINE),
                separate_original_Gamma77_retention=bind(frozen.RETENTION_REFERENCE),
                producer_schema=frozen.SCHEMA,
                producer_source_map_unchanged=hashes == {path.relative_to(ROOT).as_posix(): frozen.sha(path) for path in frozen.source_files()},
                actual_Gazebo_acceptance="NOT_EVALUATED_FOR_THESE_NEW_POLICY_BYTES",
                native_full_hard_gate_required_before_any_77_deployment_claim=True,
                automatic_2M_or_promotion_authorized=False))
            with zipfile.ZipFile(args.output / "wrapper_v9_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for path in (Path(__file__), DOCS / "freeze.json", DOCS / "trace_proof_plan.json", DOCS / "README.md", args.research_plan):
                    archive.write(path, path.relative_to(ROOT).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("w8_control",), default="w8_control")
    parser.add_argument("--initial", type=Path, default=frozen.INITIAL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--retention-reference", type=Path, default=frozen.RETENTION_REFERENCE)
    parser.add_argument("--seed", type=int, default=9611)
    parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--steps", type=int, default=BUDGET["ppo_steps"])
    parser.add_argument("--warmup-steps", type=int, default=BUDGET["critic_steps"])
    parser.add_argument("--eval-every", type=int, default=BUDGET["eval_every"])
    parser.add_argument("--research-plan", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Read-only archived checks plus tiny synthetic Adam; never an environment")
    if not sys.argv[1:]:
        parser.print_help()
        return 0
    args = parser.parse_args()
    for name in ("initial", "output", "baseline", "retention_reference", "research_plan"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    torch.set_num_threads(1)
    if args.verify_only:
        print(json.dumps(verify_only(), allow_nan=False), flush=True)
        return 0
    try:
        validate_arguments(args)
        plan = require_bound_preflight(args)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps(dict(status="BLOCKED_BEFORE_ENVIRONMENT", reason=str(exc))), flush=True)
        return 2
    run(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

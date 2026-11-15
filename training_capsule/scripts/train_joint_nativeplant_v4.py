"""Prospective native-plant development PPO with explicit v4 selection budgets.

Uses the separate native400Hz RP shaper branch for both training and evaluation.
The actor, reward W8, mission judge and disturbances stay unchanged. Physics and
development selection BOTH differ from the old W8 experiment, so this is not a
pure causal PPO physics ablation. Actual Gazebo quality gates stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from audit.verify_joint_metrics_v2 import distances_to_leg, fixture, geometry_metrics
from quadrl_joint.env_v3 import QuadEnvJointV3
from quadrl_joint.native_rate_model_v3 import NativeRateModelV3, MODEL_VERSION
from quadrl.policy import EncoderConfig, NumpyPolicy, export_policy
from quadrl.protocol import Mission, ROUTE_POINTS
from quadrl.reward import REWARD_VERSION
from quadrl.rollout import _run_episode
from scripts.bc_trust_region_probe_v1 import BCTrustPPO, actor_name, sha

INITIAL = ROOT / "artifacts/refinement/wind_specialist_v1/best_model.zip"
SEEDS = (84001, 84002, 84003, 84004)
SCENARIOS = ("nominal", "force_strong", "gust")
CASES = tuple((s, m, seed) for s in SCENARIOS for m in ("hover", "route") for seed in SEEDS)
GEOMETRY_WEIGHT = 8.0  # Prospective W8: fixed before the first training run.
KL_WEIGHT = .1
N_ENVS = 16
SCHEMA = "joint-nativeplant-ppo-v4"
MODEL_FAMILY = "fast_native_shaper_v3"
SELECTION_VERSION = "nativeplant-development-v4.0"
ROUTE_TIME_LIMIT_S = 9.3
ROUTE_GOAL_RMSE_LIMIT_M = .62
ABSOLUTE_SEGMENT_PEAK_LIMIT_M = .50
FINITE_TARGETS_M = dict(nominal=.11, force_strong=.15, gust=.15)
CORNER_TARGETS_M = dict(nominal=.20, force_strong=.25, gust=.25)
FROZEN_PHYSICS_SHA256 = {
    "quadrl_joint/__init__.py": "852cd639ca0ff568c812feeff13086b4e6c6833b17b70bba7400e702b1bde614",
    "quadrl_joint/env_v3.py": "8f801f8d0e01d15a3e23cf5022b48da0d8cb4d146cb8c28b7784f247b70beb15",
    "quadrl_joint/native_rate_model_v3.py": "68f62995c66372aaaac8670f36125ee6a5de00d340da3dea956e5f2d11a08e52",
    "quadrl_joint/native_rate_fit_v3.json": "64e2ad8bccbbb26ad9cab7eb61ffb2a2eefa5899fd22e4fee33c3d6bd0800078",
    "audit/joint_native_rate_physics_replay_v3.py": "eae497a7ec5ad65b64016aad1fb55185244c9a3d0532a442e5e6e7ceccd223b8",
    "audit/joint_native_rate_physics_replay_v3.json": "bbea80ac9e3da5aee6eeee4dd468b230ccb504b0e418f3d3814aa0d7ee2239c4",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, record):
    """Atomic replacement lets the read-only viewer inspect a running job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def phase(output, name, **values):
    write_json(Path(output) / "phase.json", dict(phase=name, pid=os.getpid(), heartbeat_utc=utc(), **values))


def source_files():
    return sorted(set([Path(__file__), ROOT / "scripts/bc_trust_region_probe_v1.py",
        ROOT / "scripts/train_joint_geometry_v2.py", ROOT / "audit/joint_nonlinear_rate_diagnosis_v2.py",
        ROOT / "audit/joint_nonlinear_rate_diagnosis_v2.json", ROOT / "audit/joint_native_rate_physics_replay_v3.py",
        ROOT / "audit/joint_native_rate_physics_replay_v3.json", *ROOT.glob("quadrl_joint/*.py"),
        *ROOT.glob("quadrl_joint/*.json"),
        ROOT / "scripts/train_wind_specialist_v1.py", ROOT / "scripts/train.py",
        ROOT / "scripts/refine_precision.py", ROOT / "audit/verify_joint_metrics_v2.py",
        ROOT / "audit/turns_corners.py", ROOT / "sim/metrics.py",
        ROOT / "docs/protocol.md", ROOT / "docs/joint_applicability_contract_v2.json",
        *ROOT.glob("quadrl/*.py"), *ROOT.glob("quadrl/*.json"), *ROOT.glob("reference/*.py")]))


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in source_files()}


def eval_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in source_files()
            if path.parent.name in ("quadrl", "quadrl_joint", "reference") or path.name in
            ("train_joint_nativeplant_v4.py", "verify_joint_metrics_v2.py", "metrics.py", "turns_corners.py")}


def physics_evidence():
    for name, digest in FROZEN_PHYSICS_SHA256.items():
        if sha(ROOT / name) != digest:
            raise AssertionError("Frozen native physics source/evidence changed: " + name)
    fit = json.loads((ROOT / "quadrl_joint/native_rate_fit_v3.json").read_text(encoding="utf-8"))
    replay_path = ROOT / "audit/joint_native_rate_physics_replay_v3.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay.get("status") != "PASS":
        raise AssertionError("Native plant must have a frozen PASS physical replay receipt")
    if sha(ROOT / fit["diagnosis_receipt"]) != fit["diagnosis_receipt_sha256"]:
        raise AssertionError("Native fit diagnosis receipt changed")
    return dict(model_version=MODEL_VERSION, model_family=MODEL_FAMILY,
                fit_sha256=sha(ROOT / "quadrl_joint/native_rate_fit_v3.json"),
                physical_replay_receipt_sha256=sha(replay_path),
                interpretation="Independent recorded-input physics diagnosis; policy quality still requires actual evaluation")


def segment_adjustment(position, goal_index, kind, control_dt, variant):
    if variant != "geometry" or kind != "route" or not 1 <= goal_index <= 4:
        return 0.0
    distance, _ = distances_to_leg(position, ROUTE_POINTS[goal_index - 1], ROUTE_POINTS[goal_index])
    return -float(control_dt) * GEOMETRY_WEIGHT * float(distance) ** 2


class JointTaskEnv(QuadEnvJointV3):
    """Training wrapper; eval uses unwrapped native full-horizon QuadEnvJointV3."""

    def __init__(self, variant="geometry", **kwargs):
        if variant != "geometry":
            raise ValueError(variant)
        self.joint_variant = variant
        super().__init__(**kwargs)

    def _training_parameters(self, rng):
        # Parent reset's provisional mission is replaced before target and
        # initial state construction. Both arms consume the same RNG stream.
        self.mission = Mission("route" if rng.random() < .8 else "hover")
        self.max_steps = round((self.seconds if self.seconds is not None else self.mission.horizon_s) / self.control_dt)
        spec = {"name": "joint_nominal_vehicle", "force_events": []}
        if rng.random() < .2:
            return spec
        magnitude = float(rng.uniform(.3, 3.5))
        angle = float(rng.uniform(-np.pi, np.pi))
        if rng.random() < .5:
            spec["force_events"] = [dict(on_s=float(rng.uniform(0, 4)),
                off_s=float(self.max_steps * self.control_dt + 1),
                force_enu_N=[magnitude * math.cos(angle), magnitude * math.sin(angle), 0.])]
        else:
            for on in (3., 10., 17.):
                direction = angle + float(rng.uniform(-np.pi, np.pi))
                spec["force_events"].append(dict(on_s=on, off_s=on + float(rng.uniform(2, 5)),
                    force_enu_N=[magnitude * math.cos(direction), magnitude * math.sin(direction), 0.]))
        return spec

    def step(self, action):
        # QuadEnv updates Mission inside step, so retain the judged index first.
        judged_index, kind = self.mission.index, self.mission.kind
        obs, reward, terminated, truncated, info = super().step(action)
        addition = segment_adjustment(info["position"], judged_index, kind,
                                      self.control_dt, self.joint_variant)
        info.update(judged_goal_index=judged_index, geometry_reward_adjustment=addition)
        successful = bool(self.training and kind == "route" and
                          self.mission.completion_time is not None and not info["failed"])
        if successful:
            # Natural task terminal, not a TimeLimit truncation; no bootstrap
            # across a completed mission and no relabeling as flight failure.
            terminated, truncated = True, False
            info["is_success"] = True
        return obs, reward + addition, terminated, truncated, info


def aggregate(episodes):
    if len(episodes) != len(CASES):
        raise AssertionError("Complete 24-case development grid required")
    keys = {(r["scenario"], r["mission"], int(r["eval_seed"])) for r in episodes}
    if keys != set(CASES):
        raise AssertionError("Unexpected/missing development case")
    result = dict(flight_failures=sum(not r["metrics"]["survived"] for r in episodes),
                  route_incomplete=sum(not r["metrics"]["route_completed"] for r in episodes if r["mission"] == "route"),
                  heading_violations=sum(not r["metrics"]["heading_guard_pass"] for r in episodes),
                  geometry_ineligible=sum(not r["geometry"]["geometry_eligible"] for r in episodes if r["mission"] == "route"),
                  active_segment_peak_failures=sum(r["geometry"]["active_segment_peak_m"] is None or
                    r["geometry"]["active_segment_peak_m"] > ABSOLUTE_SEGMENT_PEAK_LIMIT_M
                    for r in episodes if r["mission"] == "route"),
                  active_segment_peak_checks=[dict(scenario=r["scenario"], seed=r["eval_seed"],
                    peak_m=r["geometry"]["active_segment_peak_m"],
                    pass_peak=bool(r["geometry"]["active_segment_peak_m"] is not None and
                                   r["geometry"]["active_segment_peak_m"] <= ABSOLUTE_SEGMENT_PEAK_LIMIT_M))
                    for r in episodes if r["mission"] == "route"],
                  cells={})
    for scenario in SCENARIOS:
        for mission in ("hover", "route"):
            records = [r for r in episodes if (r["scenario"], r["mission"]) == (scenario, mission)]
            cell = dict(episodes=len(records), survived=sum(r["metrics"]["survived"] for r in records),
                        heading_pass=sum(r["metrics"]["heading_guard_pass"] for r in records),
                        goal_rmse_m=float(np.mean([r["metrics"]["penalized_rmse_position_m"] for r in records])))
            if mission == "route":
                eligible = all(r["geometry"]["geometry_eligible"] for r in records)
                cell.update(completed=sum(r["metrics"]["route_completed"] for r in records),
                    restricted_time_s=float(np.mean([r["metrics"]["route_completion_time_s"] if r["metrics"]["route_completed"]
                                                     else 40. for r in records])), geometry_eligible=eligible,
                    segment_rmse_m=float(np.mean([r["geometry"]["segment_rmse_m"] for r in records])) if eligible else None,
                    mean_corner_peak_m=float(np.mean([r["geometry"]["mean_corner_post_peak_m"] for r in records])) if eligible else None,
                    absolute_active_segment_peak_m=max(r["geometry"]["active_segment_peak_m"] for r in records) if eligible else None,
                    active_segment_peak_failed_seeds=[r["eval_seed"] for r in records if r["geometry"]["active_segment_peak_m"] is None or
                                                     r["geometry"]["active_segment_peak_m"] > ABSOLUTE_SEGMENT_PEAK_LIMIT_M])
            result["cells"][scenario + "/" + mission] = cell
    return result


def selection_score(metrics, baseline):
    """New explicit development budgets; actual Gazebo contract is untouched.

    Route budgets are absolute engineering allowances on this model family,
    not values claimed to be measured PID performance. Every route's finite
    segment peak must pass; a scenario average cannot conceal one failed seed.
    """
    checks = []
    initial_nominal_hover = baseline["cells"]["nominal/hover"]["goal_rmse_m"]
    checks.append(dict(name="nominal/hover/initial_relative_precision", reference="initial wind specialist, not PID",
                       value=metrics["cells"]["nominal/hover"]["goal_rmse_m"],
                       limit=min(.05, max(1.2 * initial_nominal_hover, initial_nominal_hover + .02))))
    for scenario in ("force_strong", "gust"):
        key = scenario + "/hover"
        checks.append(dict(name=key + "/wind_retention", value=metrics["cells"][key]["goal_rmse_m"],
                           limit=1.10 * baseline["cells"][key]["goal_rmse_m"]))
    for scenario in SCENARIOS:
        key = scenario + "/route"
        for field, limit in (("restricted_time_s", ROUTE_TIME_LIMIT_S), ("goal_rmse_m", ROUTE_GOAL_RMSE_LIMIT_M)):
            checks.append(dict(name=key + "/" + field, value=metrics["cells"][key][field],
                               limit=limit))
    for peak in metrics["active_segment_peak_checks"]:
        checks.append(dict(name=peak["scenario"] + "/route/seed" + str(peak["seed"]) + "/finite_segment_peak_m",
                           value=peak["peak_m"] if peak["peak_m"] is not None else 1e9,
                           measured_value=peak["peak_m"], limit=ABSOLUTE_SEGMENT_PEAK_LIMIT_M, strict=True))
    for check in checks:
        check["excess_ratio"] = max(0., check["value"] / max(check["limit"], 1e-12) - 1.)
        check["pass"] = bool(check["value"] <= check["limit"] + (0. if check.get("strict") else 1e-12))
    ratios, finite_values, geometry_checks = [], [], []
    for scenario in SCENARIOS:
        key = scenario + "/route"
        for field, target in (("segment_rmse_m", FINITE_TARGETS_M[scenario]), ("mean_corner_peak_m", CORNER_TARGETS_M[scenario])):
            value = metrics["cells"][key][field]
            ratio = value / target if value is not None else 1e9
            ratios.append(ratio)
            geometry_checks.append(dict(scenario=scenario, metric=field, measured_m=value,
                                        fixed_target_m=target, ratio=ratio))
            if field == "segment_rmse_m":
                finite_values.append(value if value is not None else 1e9)
    failed = sum(not check["pass"] for check in checks)
    score = [metrics["flight_failures"], metrics["route_incomplete"], metrics["heading_violations"],
             metrics["geometry_ineligible"], failed,
             max(check["excess_ratio"] for check in checks),
             float(max(ratios)), float(np.mean(finite_values))]
    return dict(objective=score, constraints=checks, fixed_development_guards_pass=all(v == 0 for v in score[:5]),
                selection_version=SELECTION_VERSION, geometry_target_checks=geometry_checks,
                worst_geometry_target_ratio=float(max(ratios)), mean_finite_segment_rmse_m=float(np.mean(finite_values)),
                active_segment_peak_failed_routes=metrics["active_segment_peak_failures"],
                interpretation="New absolute engineering development budgets on fast_native_shaper_v3; no cross-backend PID comparison or Gazebo applicability claim")


def evaluate(policy_path, output, phase_output=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    controller = NumpyPolicy(policy_path)
    records = []
    for index, (scenario, mission, seed) in enumerate(CASES):
        if phase_output:
            phase(phase_output, "validation", completed_cases=index, total_cases=len(CASES),
                  evaluation=str(output), scenario=scenario, mission=mission, eval_seed=seed)
        destination = output / f"{mission}_{scenario}_{seed}"
        # Native plant is shared by train/eval; evaluation keeps the full horizon
        # and base reward, without training-only W8 or success termination.
        env = QuadEnvJointV3("hybrid", training=False, scenario=scenario, mission=mission)
        try:
            record = _run_episode(env, controller, "hybrid", True, scenario, mission, seed, destination)
        finally:
            env.close()
        record.update(model_family=MODEL_FAMILY, plant_branch="native_RP_jerk_shaper_400Hz_physical_feedback",
                      joint_rate_model_version=MODEL_VERSION)
        if mission == "route":
            with (destination / "trace.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            record["geometry"] = geometry_metrics(
                [float(r["time"]) for r in rows],
                [[float(r[k]) for k in ("x", "y", "z")] for r in rows],
                [[float(r[k]) for k in ("target_x", "target_y", "target_z")] for r in rows],
                record["mission_events"], record["metrics"]["route_completion_time_s"])
            write_json(destination / "geometry.json", record["geometry"])
            record["geometry_sha256"] = sha(destination / "geometry.json")
        if record["metrics"]["survived"] and record["steps"] != (1500 if mission == "hover" else 2000):
            raise AssertionError("Evaluation horizon changed")
        records.append(record)
        write_json(destination / "summary.json", record)
    result = dict(schema=SCHEMA, grid="development24_full_horizon", seeds=list(SEEDS),
                  episodes=records, aggregate=aggregate(records), eval_source_sha256=eval_hashes(),
                  policy_sha256=sha(policy_path), evaluated_utc=utc(),
                  horizon_s=dict(hover=30., route=40.), source="actual archived trace.csv and mission_events",
                  model_family=MODEL_FAMILY, native_physics=physics_evidence())
    write_json(output / "evaluation.json", result)
    return result


def baseline_receipt(initial, output, existing=None, phase_output=None):
    if existing is not None:
        receipt = Path(existing).resolve()
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("model_family") != MODEL_FAMILY or record.get("initial_sha256") != sha(initial):
            raise AssertionError("Baseline is not the identical initial wind specialist")
        if record.get("eval_source_sha256") != eval_hashes() or record.get("seeds") != list(SEEDS):
            raise AssertionError("Baseline evaluator or grid differs")
        if record.get("aggregate") != aggregate(record["episodes"]):
            raise AssertionError("Baseline summary differs from its episodes")
        for episode in record["episodes"]:
            for filename, digest in episode["hashes"].items():
                if sha(Path(episode["trace_path"]).parent / filename) != digest:
                    raise AssertionError("Baseline measured evidence changed")
        return record, receipt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phase_output = phase_output or output
    phase(phase_output, "validation", role="initial_wind_baseline", completed_cases=0, total_cases=len(CASES))
    model = BCTrustPPO.load(initial, device="cpu")
    export_policy(model, output / "initial_policy.npz", "hybrid", EncoderConfig())
    record = evaluate(output / "initial_policy.npz", output / "traces", phase_output)
    record["initial_sha256"] = sha(initial)
    record["initial_model"] = str(initial)
    receipt = output / "baseline.json"
    write_json(receipt, record)
    phase(phase_output, "complete", role="initial_wind_baseline", completed_cases=len(CASES))
    return record, receipt


class Heartbeat(BaseCallback):
    def __init__(self, output, name):
        super().__init__()
        self.output, self.name, self.last = output, name, 0.

    def _on_step(self):
        now = time.monotonic()
        if now - self.last >= 10.:
            phase(self.output, self.name, environment_steps=self.num_timesteps,
                  completed_epochs=self.model._n_updates,
                  ppo_kl_gradient_steps=getattr(self.model, "bc_kl_gradient_steps", 0))
            self.last = now
        return True


class JointSelection(Heartbeat):
    def __init__(self, output, interval, baseline, starting_updates):
        super().__init__(output, "ppo")
        self.output = Path(output)
        self.interval = self.next_eval = interval
        self.baseline = baseline
        self.starting_updates = starting_updates
        self.best = None
        self.records = []

    def evaluate(self, label):
        self.model.save(self.output / label)
        policy = self.output / f"{label}.npz"
        export_policy(self.model, policy, "hybrid", EncoderConfig())
        record = evaluate(policy, self.output / "validation" / label, self.output)
        score = selection_score(record["aggregate"], self.baseline["aggregate"])
        record.update(score, checkpoint=label, timesteps=self.model.num_timesteps,
                      completed_ppo_epochs=self.model._n_updates - self.starting_updates,
                      ppo_kl_gradient_steps=self.model.bc_kl_gradient_steps,
                      checkpoint_timing="After final update" if label == "final" else "Before current rollout update",
                      initial_sha256=self.baseline["initial_sha256"])
        write_json(self.output / f"{label}_validation.json", record)
        self.records.append(dict(checkpoint=label, objective=score["objective"],
                                 policy_sha256=sha(policy), guards_pass=score["fixed_development_guards_pass"]))
        if self.best is None or tuple(score["objective"]) < tuple(self.best):
            self.best = score["objective"]
            self.model.save(self.output / "best_model")
            export_policy(self.model, self.output / "best_policy.npz", "hybrid", EncoderConfig())
            write_json(self.output / "selection.json", {k: v for k, v in record.items() if k != "episodes"})
        print(json.dumps(dict(event="DEVELOPMENT_VALIDATION", checkpoint=label,
                              timesteps=self.model.num_timesteps, **score)), flush=True)
        phase(self.output, "ppo", environment_steps=self.model.num_timesteps,
              completed_epochs=self.model._n_updates - self.starting_updates)

    def _on_step(self):
        super()._on_step()
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.interval
            self.evaluate(f"checkpoint_{self.num_timesteps}")
        return True


def compare_states(policy, before, actor):
    return [name for name, value in policy.state_dict().items()
            if actor_name(name) == actor and not torch.equal(value, before[name])]


def run_arm(args, variant, output, baseline, receipt):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = source_hashes()
    initial_hash = sha(args.initial)
    config = dict(schema=SCHEMA, task="joint_nativeplant", variant=variant,
        algorithm="Wind specialist + critic warmup + fixed initial-actor KL PPO",
        initial=str(args.initial), initial_sha256=initial_hash, prior_sha256=initial_hash,
        prior_direction="current Normal || frozen SAME initial wind actor Normal",
        fixed_kl_weight=KL_WEIGHT, geometry_weight=GEOMETRY_WEIGHT if variant == "geometry" else 0.,
        prospective_geometry_weight=GEOMETRY_WEIGHT, reward_base=REWARD_VERSION,
        model_family=MODEL_FAMILY, native_physics=physics_evidence(), selection_version=SELECTION_VERSION,
        changed_factors=["native400Hz jerk-shaped RP plant replaces old aggregate RP PI/delay",
                         "new explicit absolute engineering development budgets and worst-target geometry selection"],
        attribution="Not a pure causal physics or PPO ablation; old Fast and new native-shaper model families are separate",
        development_route_time_limit_s=ROUTE_TIME_LIMIT_S, development_route_goal_rmse_limit_m=ROUTE_GOAL_RMSE_LIMIT_M,
        development_each_route_segment_peak_limit_m=ABSOLUTE_SEGMENT_PEAK_LIMIT_M,
        finite_segment_targets_m=FINITE_TARGETS_M, corner_peak_targets_m=CORNER_TARGETS_M,
        actual_gazebo_contract_unchanged="docs/joint_applicability_contract_v2.json",
        geometry_reward="-dt * 8 * distance_to_preupdate_active_finite_3D_segment_squared for route goals1..4",
        envs=N_ENVS, cpu_threads=1, seed=args.seed, ppo_seed=args.seed, warmup_seed=args.warmup_seed,
        warmup_steps=args.warmup_steps, ppo_steps=args.steps, eval_every=args.eval_every,
        candidate_count=args.steps // args.eval_every + 1,
        training_missions=dict(route=.8, hover=.2),
        training_disturbances=dict(nominal=.2, constant_force=.4, gust=.4),
        training_force_magnitude_N=[.3, 3.5], nominal_vehicle=True,
        training_success_terminal="confirmed goal4 completion: terminated=True truncated=False failed=False",
        evaluation="Unwrapped QuadEnvJointV3; full30s hover and40s route; 24 fixed dev cases; archived traces",
        validation_seeds=list(SEEDS), reserved_confirmation_seeds_not_opened=[93001, 93002, 93003],
        baseline_receipt=str(receipt), baseline_receipt_sha256=sha(receipt),
        actor_observations=74, no_future_goals_or_force_inputs=True,
        selection_order="survival, completion, heading, geometry coverage; fixed constraint count/maxexcess; worst finite/corner target ratio; meanfinite",
        nominal_hover_development_limit="min(.05,max(1.2*initial_nominal_hover_RMSE,initial_nominal_hover_RMSE+.02)); initial wind specialist, not PID",
        installed_sb3_ppo_train_sha256=hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest(),
        source_hashes=hashes, started_utc=utc(), pid=os.getpid())
    write_json(output / "config.json", config)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(receipt, "baseline_evidence/baseline.json")
    write_json(output / "manifest.json", dict(source_sha256=hashes, initial_sha256=initial_hash,
                source_snapshot_sha256=sha(output / "source_snapshot.zip"), baseline_sha256=sha(receipt)))
    phase(output, "critic", environment_steps=0)
    vector = VecMonitor(DummyVecEnv([lambda: JointTaskEnv(variant=variant, architecture="hybrid", training=True)
                                    for _ in range(N_ENVS)]))
    try:
        vector.set_attr("training_progress", 1.)
        vector.seed(args.warmup_seed)
        model = BCTrustPPO.load(args.initial, env=vector, device="cpu", seed=args.warmup_seed)
        model.anchor_steps = 0
        model.configure_bc_prior(args.initial, 0., output / "on_policy_prior_kl.jsonl")
        model.bc_kl_enabled = False
        initial = copy.deepcopy(model.policy.state_dict())
        prior = copy.deepcopy(model.bc_prior.state_dict())
        np.savez(output / "initial_parameters.npz", **{k: v.detach().cpu().numpy() for k, v in initial.items()})
        for name, parameter in model.policy.named_parameters():
            if actor_name(name):
                parameter.requires_grad_(False)
        model.set_logger(configure(str(output / "warmup"), ["csv"]))
        before = model._n_updates
        model.learn(args.warmup_steps, reset_num_timesteps=True, callback=Heartbeat(output, "critic"), progress_bar=False)
        critic_changes = compare_states(model.policy, initial, False)
        if compare_states(model.policy, initial, True) or not critic_changes:
            raise AssertionError("Critic warmup must change critic and preserve actor including log_std")
        warmup = dict(environment_steps=model.num_timesteps, completed_critic_only_epochs=model._n_updates - before,
                      actor_bit_identical=True, changed_critic_parameters=critic_changes)
        write_json(output / "warmup.json", warmup)
        model.save(output / "critic_warmed_model")
        ppo_initial = copy.deepcopy(model.policy.state_dict())
        np.savez(output / "ppo_start_parameters.npz", **{k: v.detach().cpu().numpy() for k, v in ppo_initial.items()})
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
        model.bc_kl_weight, model.bc_kl_enabled = KL_WEIGHT, True
        model.set_random_seed(args.seed)
        # Reset env/RNG for an identical PPO-start distribution in both arms.
        model._last_obs = None
        model.set_logger(configure(str(output), ["csv"]))
        model.bc_kl_gradient_steps = model.bc_kl_minibatch_sample_uses = 0
        model.ppo_gradient_steps = 0
        before = model._n_updates
        selection = JointSelection(output, args.eval_every, baseline, before)
        phase(output, "ppo", environment_steps=0)
        model.learn(args.steps, reset_num_timesteps=True, callback=selection, progress_bar=False)
        selection.evaluate("final")
        phase(output, "verification", environment_steps=model.num_timesteps)
        probe = np.random.default_rng(17491).normal(size=(200, 74)).astype(np.float32)
        parity = float(np.max(np.abs(model.predict(probe, deterministic=True)[0] - NumpyPolicy(output / "final.npz").predict(probe))))
        if parity >= 1e-5:
            raise AssertionError("NumPy export parity must be <1e-5")
        if not compare_states(model.policy, ppo_initial, True):
            raise AssertionError("PPO did not change the actor")
        if any(not torch.equal(value, prior[name]) for name, value in model.bc_prior.state_dict().items()):
            raise AssertionError("Fixed initial wind prior changed")
        if source_hashes() != hashes or sha(args.initial) != initial_hash or sha(receipt) != config["baseline_receipt_sha256"]:
            raise AssertionError("Sources, initialization or baseline changed during run")
        log = [json.loads(line) for line in (output / "on_policy_prior_kl.jsonl").read_text().splitlines()]
        actual_updates = sum(row["optimizer_steps"] for row in log)
        sample_uses = sum(row["minibatch_sample_uses"] for row in log)
        if not actual_updates or actual_updates != model.bc_kl_gradient_steps or actual_updates != model.ppo_gradient_steps:
            raise AssertionError("Actual PPO optimizer-step counts disagree")
        if sample_uses != model.bc_kl_minibatch_sample_uses:
            raise AssertionError("Actual on-policy sample-use counts disagree")
        if len(selection.records) != config["candidate_count"]:
            raise AssertionError("Prospective candidate budget changed")
        selected = json.loads((output / "selection.json").read_text())
        model_hashes = {name: sha(output / name) for name in
                        ("critic_warmed_model.zip", "best_model.zip", "best_policy.npz", "final.zip", "final.npz")}
        write_json(output / "model_manifest.json", dict(initial_sha256=initial_hash,
                   prior_sha256=initial_hash, output_sha256=model_hashes, completed_utc=utc()))
        complete = dict(status="complete", task="joint_nativeplant", variant=variant,
            model_family=MODEL_FAMILY, selection_version=SELECTION_VERSION,
            critic_only_timesteps=warmup["environment_steps"], ppo_timesteps=model.num_timesteps,
            completed_critic_only_epochs=warmup["completed_critic_only_epochs"],
            additional_actor_critic_ppo_epochs=model._n_updates - before,
            actor_critic_ppo_gradient_steps=actual_updates, on_policy_prior_kl_gradient_steps=model.bc_kl_gradient_steps,
            on_policy_prior_kl_minibatch_sample_uses=sample_uses, rollout_updates=len(log),
            offline_supervised_anchor_gradient_steps=0, actor_unchanged_by_critic_warmup=True,
            actor_changed_by_ppo=True, prior_bit_identical=True, source_hashes_unchanged=True,
            numpy_export_max_error=parity, selected_policy_sha256=sha(output / "best_policy.npz"),
            final_policy_sha256=sha(output / "final.npz"), selected_objective=selection.best,
            output_model_sha256=model_hashes,
            selected_metrics=selected["aggregate"], fixed_development_guards_pass=selected["fixed_development_guards_pass"],
            candidate_records=selection.records, wall_seconds=time.monotonic() - started, completed_utc=utc(),
            interpretation="Development fast_native_shaper_v3 evidence with new selection budgets; no Gazebo applicability or pure causal PPO claim")
        write_json(output / "complete.json", complete)
        phase(output, "complete", environment_steps=model.num_timesteps)
        print(json.dumps(complete), flush=True)
        return complete
    except Exception as exc:
        write_json(output / "error.json", dict(status="error", error=repr(exc), traceback=traceback.format_exc(), utc=utc()))
        phase(output, "error", error=repr(exc))
        raise
    finally:
        vector.close()


def verify(output, initial):
    """Small real optimizer probe and analytic checks; never a full training run."""
    output.mkdir(parents=True, exist_ok=False)
    phase(output, "verification")
    checks = []
    physics = physics_evidence()
    rate = NativeRateModelV3()
    bounds = rate.fit["parameter_bounds"]
    for field, values in (("gain", rate.gain), ("tau_s", rate.tau),
                          ("delay_s", np.asarray(rate.fit["RP_remainder"]["delay_s"]))):
        low, high = bounds[field]
        assert np.all((values >= low) & (values <= high))
    for tick in range(1200):
        previous_accel = rate.target_accel.copy()
        command = np.array([2., -2., .4]) * (1. if tick < 600 else -1.)
        delayed = rate.step(command)
        assert np.isfinite(delayed).all()
        assert np.all(np.abs(rate.target_accel) <= rate.accel_limit + 1e-12)
        assert np.all(np.abs(rate.target_accel - previous_accel) <= rate.jerk_limit * rate.dt + 1e-12)
    checks.append("Frozen native physics evidence PASS; fitted bounds and actual acceleration/jerk caps verified")
    p = np.array([3., 0., 3.])
    assert np.isclose(segment_adjustment(p, 1, "route", .02, "geometry"), -.16)
    assert segment_adjustment(p, 1, "route", .02, "control") == 0.
    assert segment_adjustment(p, 0, "route", .02, "geometry") == 0.
    assert segment_adjustment(p, 1, "hover", .02, "geometry") == 0.
    checks.append("W8 finite-segment overshoot penalty, control and non-transit zero")
    t, p, target, events = fixture()
    assert geometry_metrics(t, p, target, events, 8.5)["geometry_eligible"]
    checks.append("Imported pre-update-target geometry judge accepts full-horizon fixture")
    pair = [JointTaskEnv(variant="geometry", training=True) for _ in range(2)]
    try:
        for env in pair:
            env.training_progress = 1.
            assert isinstance(env, QuadEnvJointV3) and env.dt == .0025 and env.decimation == 8
            assert env.control_dt == .02 and env.rate_model is not None
        checks.append("Training wraps actual QuadEnvJointV3 at400Hz physics/50Hz actor with unchanged74 observations")
        distribution = dict(route=0, nominal=0, constant_force=0, gust=0)
        for seed in range(1200, 1400):
            observations = [env.reset(seed=seed)[0] for env in pair]
            assert np.array_equal(observations[0], observations[1]) and observations[0].shape == (74,)
            assert pair[0].parameters == pair[1].parameters
            spec = pair[0].parameters
            assert set(spec) == {"name", "force_events"}
            distribution["route"] += pair[0].mission.kind == "route"
            n = len(spec["force_events"])
            distribution[{0: "nominal", 1: "constant_force", 3: "gust"}[n]] += 1
            for event in spec["force_events"]:
                assert .3 - 1e-12 <= np.linalg.norm(event["force_enu_N"]) <= 3.5 + 1e-12
        checks.append("200 repeated native-plant resets: deterministic74 observations, missions/wind specs; nominal vehicle only")
        # Manufacture a valid in-band final dwell, then execute real physics.
        env = pair[1]
        env.reset(seed=1401)
        env.mission = Mission("route")
        env.mission.index = 4
        env.mission.inside_since = -.5
        env.mission.last_time = 0.
        env.target = ROUTE_POINTS[4].copy()
        import pybullet as pb
        pb.resetBasePositionAndOrientation(env.body, env.target, [0, 0, 0, 1], physicsClientId=env.cid)
        pb.resetBaseVelocity(env.body, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
        _, _, terminal, truncated, info = env.step(np.zeros(4))
        assert terminal and not truncated and not info["failed"] and info["is_success"]
        checks.append("Confirmed training route success is terminal, nontruncated and not failed")
        env.reset(seed=1402)
        env.mission = Mission("route")
        env.mission.index = 1
        env.mission.inside_since = -.5
        env.mission.last_time = 0.
        env.target = ROUTE_POINTS[1].copy()
        pb.resetBasePositionAndOrientation(env.body, [1.9, 0., 3.], [0, 0, 0, 1], physicsClientId=env.cid)
        pb.resetBaseVelocity(env.body, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["judged_goal_index"] == 1 and info["goal_index"] == 2
        assert info["geometry_reward_adjustment"] == segment_adjustment(info["position"], 1, "route", .02, "geometry")
        assert abs(info["geometry_reward_adjustment"]) < abs(segment_adjustment(info["position"], 2, "route", .02, "geometry"))
        checks.append("Actual waypoint transition charges previous finite leg, not post-update next leg")
    finally:
        for env in pair:
            env.close()
    vector = VecMonitor(DummyVecEnv([lambda: JointTaskEnv(variant="geometry", training=True)]))
    try:
        vector.set_attr("training_progress", 1.)
        model = BCTrustPPO.load(initial, env=vector, device="cpu", seed=9511)
        model.anchor_steps = 0
        # Verification deliberately shrinks a fresh in-memory rollout buffer.
        model.n_steps, model.batch_size, model.n_epochs = 16, 16, 1
        model.rollout_buffer = model.rollout_buffer_class(model.n_steps, model.observation_space,
            model.action_space, device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
        model.configure_bc_prior(initial, 0., output / "verify_prior_kl.jsonl")
        model.bc_kl_enabled = False
        original = copy.deepcopy(model.policy.state_dict())
        prior = copy.deepcopy(model.bc_prior.state_dict())
        for name, parameter in model.policy.named_parameters():
            if actor_name(name):
                parameter.requires_grad_(False)
        model.set_logger(configure(str(output / "warmup"), ["csv"]))
        model.learn(16, reset_num_timesteps=True, progress_bar=False)
        assert not compare_states(model.policy, original, True) and compare_states(model.policy, original, False)
        checks.append("Real16-step critic optimizer probe preserves actor including log_std")
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
        model.bc_kl_weight, model.bc_kl_enabled = KL_WEIGHT, True
        model.set_logger(configure(str(output), ["csv"]))
        model.ppo_gradient_steps = 0
        model.learn(32, reset_num_timesteps=True, progress_bar=False)
        assert compare_states(model.policy, original, True)
        assert all(torch.equal(v, prior[k]) for k, v in model.bc_prior.state_dict().items())
        assert model.bc_kl_gradient_steps == model.ppo_gradient_steps == 2
        export_policy(model, output / "verify_policy.npz", "hybrid", EncoderConfig())
        probe = np.random.default_rng(17491).normal(size=(200, 74)).astype(np.float32)
        parity = float(np.max(np.abs(model.predict(probe, deterministic=True)[0] - NumpyPolicy(output / "verify_policy.npz").predict(probe))))
        assert parity < 1e-5
        checks.append("Real32-step PPO: changed actor, frozen same wind prior, two logged optimizer steps and NumPy parity")
    finally:
        vector.close()
    # The ordinary eval environment must retain final-goal hover until40s.
    env = QuadEnvJointV3(training=False, mission="route")
    try:
        assert type(env) is QuadEnvJointV3 and env.max_steps == 2000 and env.dt == .0025 and env.decimation == 8
        env.mission.completion_time = 1.
        _, _, terminated, truncated, _ = env.step(np.zeros(4))
        assert not terminated and not truncated
        checks.append("Unwrapped native evaluation plant continues after route completion to fixed40s horizon")
    finally:
        env.close()
    synthetic = dict(flight_failures=0, route_incomplete=0, heading_violations=0, geometry_ineligible=0,
                     active_segment_peak_failures=0,
                     active_segment_peak_checks=[dict(scenario=scenario, seed=seed, peak_m=.2, pass_peak=True)
                                                  for scenario in SCENARIOS for seed in SEEDS],
                     cells={scenario + "/" + mission: dict(goal_rmse_m=.55, restricted_time_s=8.5,
                            segment_rmse_m=.10, mean_corner_peak_m=.18)
                            for scenario in SCENARIOS for mission in ("hover", "route")})
    synthetic["cells"]["nominal/hover"]["goal_rmse_m"] = .02
    retained = selection_score(synthetic, synthetic)
    wind_loss = copy.deepcopy(synthetic)
    wind_loss["cells"]["gust/hover"]["goal_rmse_m"] = 1.11
    for scenario in SCENARIOS:
        wind_loss["cells"][scenario + "/route"]["segment_rmse_m"] = 0.
    assert retained["fixed_development_guards_pass"]
    assert not selection_score(wind_loss, synthetic)["fixed_development_guards_pass"]
    assert tuple(retained["objective"]) < tuple(selection_score(wind_loss, synthetic)["objective"])
    checks.append("Fixed wind-retention constraints precede even perfect route geometry in selection")
    nominal_loss = copy.deepcopy(synthetic)
    nominal_loss["cells"]["nominal/hover"]["goal_rmse_m"] = .041
    assert not selection_score(nominal_loss, synthetic)["fixed_development_guards_pass"]
    assert tuple(retained["objective"]) < tuple(selection_score(nominal_loss, synthetic)["objective"])
    checks.append("Nominal hover precision guard against initial specialist precedes geometry; fixed .04m fixture limit")
    braking = copy.deepcopy(synthetic)
    for scenario in SCENARIOS:
        braking["cells"][scenario + "/route"].update(goal_rmse_m=.60, restricted_time_s=9.0)
    assert selection_score(braking, synthetic)["fixed_development_guards_pass"]
    checks.append("New absolute9.3s/.62m route budgets permit braking tradeoff beyond old initial-relative limits")
    peak_loss = copy.deepcopy(synthetic)
    peak_loss["active_segment_peak_checks"][0]["peak_m"] = .50001
    peak_loss["active_segment_peak_failures"] = 1
    for scenario in SCENARIOS:
        peak_loss["cells"][scenario + "/route"]["segment_rmse_m"] = 0.
    peak_score = selection_score(peak_loss, synthetic)
    assert not peak_score["fixed_development_guards_pass"] and peak_score["active_segment_peak_failed_routes"] == 1
    assert tuple(retained["objective"]) < tuple(peak_score["objective"])
    for field, value in (("restricted_time_s", 9.30001), ("goal_rmse_m", .62001)):
        failed = copy.deepcopy(synthetic)
        failed["cells"]["nominal/route"][field] = value
        assert not selection_score(failed, synthetic)["fixed_development_guards_pass"]
    assert np.isclose(retained["worst_geometry_target_ratio"], max(.10/.11, .10/.15, .18/.20, .18/.25))
    checks.append("Every route peak.50 guard outranks perfect mean geometry; exact route limits and worst fixed-target objective verified")
    result = dict(status="PASS", scope="Analytic and48 optimizer-probe environment steps plus physics semantics checks; no full training or Gazebo",
                  checks=checks, sampled_distribution=distribution, numpy_export_max_error=parity,
                  initial_sha256=sha(initial), source_sha256=sha(Path(__file__)), source_hashes=source_hashes(),
                  model_family=MODEL_FAMILY, native_physics=physics, selection_version=SELECTION_VERSION,
                  actual_ppo_optimizer_steps=2, reserved_confirmation_seeds_not_opened=[93001, 93002, 93003], utc=utc())
    write_json(output / "verify.json", result)
    phase(output, "complete", scope="verification only")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("geometry",), default="geometry", help="W8 reward, without arrival-v3 bonus")
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path, required=True, help="New unique output directory; existing directories are rejected")
    parser.add_argument("--seed", type=int, default=9611)
    parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--warmup-steps", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=2097152)
    parser.add_argument("--eval-every", type=int, default=1048576)
    parser.add_argument("--baseline", type=Path, help="Reuse an identical hashed initial-model development baseline.json")
    parser.add_argument("--baseline-only", action="store_true", help="Create shared24-case initial baseline without training")
    parser.add_argument("--verify", action="store_true", help="Only short analytic and real optimizer probes")
    args = parser.parse_args()
    args.initial = (args.initial if args.initial.is_absolute() else ROOT / args.initial).resolve()
    args.output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    if args.baseline:
        args.baseline = (args.baseline if args.baseline.is_absolute() else ROOT / args.baseline).resolve()
    if not args.initial.is_file():
        parser.error("Initial wind-specialist ZIP does not exist")
    if args.output.exists():
        parser.error("Output already exists; preserve previous run and choose a new directory")
    if REWARD_VERSION != "precision-arrival-v3":
        parser.error("Frozen base reward differs from precision-arrival-v3")
    torch.set_num_threads(1)
    physics_evidence()
    if args.verify:
        verify(args.output, args.initial)
        return
    if args.baseline_only:
        if args.baseline:
            parser.error("--baseline-only cannot reuse an existing baseline")
        baseline, receipt = baseline_receipt(args.initial, args.output)
        print(json.dumps(dict(status="BASELINE_COMPLETE", receipt=str(receipt), receipt_sha256=sha(receipt),
                              initial_sha256=sha(args.initial), aggregate=baseline["aggregate"])), flush=True)
        return
    # Require exact whole rollout budgets; SB3 otherwise silently rounds up.
    initial_model = BCTrustPPO.load(args.initial, device="cpu")
    rollout = initial_model.n_steps * N_ENVS
    if min(args.steps, args.warmup_steps, args.eval_every) < rollout or any(
            count % rollout for count in (args.steps, args.warmup_steps, args.eval_every)):
        parser.error(f"Step budgets and eval interval must be positive multiples of inherited rollout size{rollout}")
    if args.steps % args.eval_every:
        parser.error("PPO steps must divide exactly by eval interval for the prospective candidate budget")
    if args.variant:
        # Reserve the run path first for its baseline phase, then run_arm creates
        # it after baseline generation in a separate unique sibling directory.
        baseline_output = args.output.with_name(args.output.name + "_baseline")
        baseline, receipt = baseline_receipt(args.initial, baseline_output, args.baseline)
        run_arm(args, args.variant, args.output, baseline, receipt)


if __name__ == "__main__":
    main()

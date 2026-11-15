"""Bounded corner-metric reward pair; both arms use the SAME learned Path77 model.

Both arms receive identical measured causal77 inputs. Only K40 corner reward
differs. Both keep frozen W8/v5 wind/native physics and KL.1.
Training requires a bound plan and official preflight --run parent. Verification
and full-horizon baseline-only modes precede that guard and never run a pilot.
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
from quadrl.policy import ObservationEncoder
from quadrl_path_v7.policy import (EncoderConfigPath77 as EncoderConfig, NumpyPolicyPath77 as NumpyPolicy, export_policy_path77 as export_policy, ObservationEncoderPath77, FiniteLegContext, CONTRACT_VERSION)
from quadrl_path_v7.env import PathNativeEnv
from quadrl_corner_v8.env import JointTaskEnv, CORNER_WEIGHT
from quadrl_corner_v8.reward import CornerPeakReward, ConfirmedTransition
from quadrl.protocol import Mission, ROUTE_POINTS
from quadrl.reward import REWARD_VERSION
from dataclasses import asdict
from sim.metrics import compute_metrics
from scripts.bc_trust_region_probe_v1 import BCTrustPPO, actor_name, sha
from scripts.train_joint_nativeplant_v4 import JointTaskEnv as V4TaskEnv
from scripts import train_joint_nativeplant_v4 as v4_reference
from scripts import train_joint_tailwind_v5 as v5_reference

INITIAL = ROOT / "artifacts/refinement/joint_v2/path_pilot111/path/final.zip"
INITIAL_SHA256 = "7df5dd0a382525d08bfb4b2d8e4514e7269438bbd44a63598eff90ac0a5f3088"
INITIAL_NPZ = ROOT / "artifacts/refinement/joint_v2/path_pilot111/path/final.npz"
INITIAL_NPZ_SHA256 = "e5bbd5c9178618388a77c7ca4923ba9bb6f1c7118d36afd8fdf36ad9c5559f38"
RETENTION_REFERENCE = ROOT / "artifacts/refinement/joint_v2/path_v7_baseline111/path/baseline.json"
RETENTION_SHA256 = "b6d244d7c36165ee0340ac3118f8df98d4e6322011c24987c455d4d069c9074f"
TRACKER_SHA256 = "0291d4c643d1aa0e613f26d46b7394f0da884da17dc9af2cd454f409d81c9964"
PROTOTYPE_REPORT = ROOT / "artifacts/refinement/joint_v2/corner_reward_probe_v8/20261113T200517512Z_K40/20261113T200751508730Z/report.json"
PROTOTYPE_SHA256 = "9d8a75b79a29662c50cdece1402c4b5ee55b759b947c29ea164b9b7bbb8d7baa"
SEEDS = (84001, 84002, 84003, 84004)
SCENARIOS = ("nominal", "force_strong", "gust")
CASES = tuple((s, m, seed) for s in SCENARIOS for m in ("hover", "route") for seed in SEEDS)
GEOMETRY_WEIGHT = 8.0  # Prospective W8: fixed before the first training run.
KL_WEIGHT = .1
N_ENVS = 16
SCHEMA = "joint-corner-metric-ppo-v8"
MODEL_FAMILY = "fast_native_shaper_v3"
SELECTION_VERSION = "corner-hard-geometry-development-v8.0"
ROUTE_TIME_LIMIT_S = 9.3
ROUTE_GOAL_RMSE_LIMIT_M = .62
ABSOLUTE_SEGMENT_PEAK_LIMIT_M = .50
FINITE_TARGETS_M = dict(nominal=.11, force_strong=.15, gust=.15)
CORNER_TARGETS_M = dict(nominal=.20, force_strong=.25, gust=.25)
CURRICULUM_VERSION = "route-tailwind-phase-v5.0"
FROZEN_V4_DRIVER_SHA256 = "5f8a538497d2d3d57f70f4985f79f2bc4947f7fae768da17c27f28d8593ccf9f"
FROZEN_V5_DRIVER_SHA256 = "c4c5a37b8cd6f731765a27e15ce879c28097638f8b61a0c35cb7483ef6fdf0cf"
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
        RETENTION_REFERENCE, PROTOTYPE_REPORT, PROTOTYPE_REPORT.with_name("final_diagnostic_summary.json"),
        ROOT / "scripts/probe_joint_corner_reward_v8.py",
        ROOT / "scripts/train_joint_geometry_v2.py", ROOT / "audit/joint_nonlinear_rate_diagnosis_v2.py",
        ROOT / "scripts/train_joint_nativeplant_v4.py", ROOT / "audit/joint_nativeplant_peak_seed84001_v4.py",
        ROOT / "scripts/train_joint_tailwind_v5.py", ROOT / "scripts/run_joint_corner_pair_v8.py",
        ROOT / "scripts/joint_research_preflight_v1.py", ROOT / "scripts/probe_joint_reward_scale_v1.py",
        ROOT / "audit/joint_reward_scale_diagnosis_v1.json",
        ROOT / "artifacts/refinement/joint_v2/conditioned_initializer_v1/lineage_receipt.json",
        ROOT / "audit/joint_nativeplant_peak_seed84001_v4.json",
        ROOT / "audit/joint_nonlinear_rate_diagnosis_v2.json", ROOT / "audit/joint_native_rate_physics_replay_v3.py",
        ROOT / "audit/joint_native_rate_physics_replay_v3.json", *ROOT.glob("quadrl_path_v7/*.py"), *ROOT.glob("quadrl_corner_v8/*.py"), ROOT / "audit/joint_path_context_prototype_v7.json", *ROOT.glob("quadrl_joint/*.py"),
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
            if path.parent.name in ("quadrl", "quadrl_joint", "quadrl_path_v7", "quadrl_corner_v8", "reference") or path.name in
            ("train_joint_corner_v8.py", "verify_joint_metrics_v2.py", "metrics.py", "turns_corners.py")}


def physics_evidence():
    if sha(ROOT / "quadrl_corner_v8/reward.py") != TRACKER_SHA256:
        raise AssertionError("Frozen pure corner tracker changed")
    if sha(PROTOTYPE_REPORT) != PROTOTYPE_SHA256:
        raise AssertionError("Frozen fixedK40 offline72-route evidence changed")
    if sha(ROOT / "scripts/train_joint_nativeplant_v4.py") != FROZEN_V4_DRIVER_SHA256:
        raise AssertionError("Frozen v4 training reference changed")
    if sha(ROOT / "scripts/train_joint_tailwind_v5.py") != FROZEN_V5_DRIVER_SHA256:
        raise AssertionError("Frozen v5 curriculum reference changed")
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


def retention_reference():
    """Original Gamma77 scale is immutable; new initial baseline is provenance."""
    if sha(RETENTION_REFERENCE) != RETENTION_SHA256:
        raise AssertionError("Frozen original Gamma77 retention reference changed")
    record = json.loads(RETENTION_REFERENCE.read_text(encoding="utf-8"))
    return dict(aggregate=record["aggregate"], source=dict(
        path=str(RETENTION_REFERENCE.relative_to(ROOT)), sha256=RETENTION_SHA256,
        role="Frozen original Gamma77 native reference; not new learned v7 initial baseline",
        prevents_cumulative_drift=True))


def segment_adjustment(position, goal_index, kind, control_dt, variant):
    if variant != "geometry" or kind != "route" or not 1 <= goal_index <= 4:
        return 0.0
    distance, _ = distances_to_leg(position, ROUTE_POINTS[goal_index - 1], ROUTE_POINTS[goal_index])
    return -float(control_dt) * GEOMETRY_WEIGHT * float(distance) ** 2


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
    for summary in (metrics, baseline):
        for key, cell in summary["cells"].items():
            for field in ("goal_rmse_m", "restricted_time_s", "segment_rmse_m", "mean_corner_peak_m"):
                value = cell.get(field)
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise ValueError("Invalid selection metric: " + key + "/" + field)
        for peak in summary["active_segment_peak_checks"]:
            value = peak["peak_m"]
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Invalid finite-segment peak")
    retention = retention_reference()
    baseline = retention["aggregate"]
    checks = []
    initial_nominal_hover = baseline["cells"]["nominal/hover"]["goal_rmse_m"]
    checks.append(dict(name="nominal/hover/initial_relative_precision", reference="frozen original Gamma77 native baseline, not newly learned v7 initial or PID",
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
        for field, limit in (("segment_rmse_m", FINITE_TARGETS_M[scenario]),
                             ("mean_corner_peak_m", CORNER_TARGETS_M[scenario])):
            value = metrics["cells"][key][field]
            checks.append(dict(name=key + "/" + field, measured_value=value,
                               value=value if value is not None else 1e9, limit=limit, strict=True))
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
                selection_version=SELECTION_VERSION, retention_reference=retention["source"], geometry_target_checks=geometry_checks,
                worst_geometry_target_ratio=float(max(ratios)), mean_finite_segment_rmse_m=float(np.mean(finite_values)),
                active_segment_peak_failed_routes=metrics["active_segment_peak_failures"],
                acceptance_scope="Hard Fast development engineering proxies only; not paired Gazebo PID acceptance",
                interpretation="Finite/corner targets are hard guards plus ranking; all actual Gazebo gates remain separate and unchanged")


def _run_episode(env,controller,architecture,history,scenario,mission,seed,save_path):
    obs,initial=env.reset(seed=seed)
    if hasattr(controller,"reset"):controller.reset()
    rows=[];io={key:[] for key in ("time","dt","position","velocity","rotation","body_rates","target","previous_action","observation","action","yaw_reference","path_feature","path_valid_leg","path_target_changes","path_feature_enabled")}
    force_changes=[];previous_force=np.zeros(3)
    failure=None
    for index in range(env.max_steps):
        measured=env.measured_state
        try:
            if hasattr(controller,"policy") and hasattr(controller,"num_timesteps"):
                action=controller.predict(obs,deterministic=True)
            else:action=controller.predict(obs)
            if isinstance(action,tuple):action=action[0]
            action=np.asarray(action,dtype=float).reshape(4)
            if not np.all(np.isfinite(action)):raise ValueError("non-finite controller action")
        except Exception as exc:
            failure=f"controller_error:{type(exc).__name__}:{exc}"
            if not rows:
                p,v,rotation,omega=env.state();target=env.target
                yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
                rows.append(dict(time=0.,x=p[0],y=p[1],z=p[2],vx=v[0],vy=v[1],vz=v[2],
                                 target_x=target[0],target_y=target[1],target_z=target[2],
                                 next_target_x=target[0],next_target_y=target[1],next_target_z=target[2],
                                 error=float(np.linalg.norm(p-target)),altitude_error=float(p[2]-target[2]),yaw=yaw,
                                 yaw_error=float(np.arctan2(np.sin(env.yaw_reference-yaw),np.cos(env.yaw_reference-yaw))),
                                 tilt_cos=float(rotation[2,2]),goal_index=0,reward=0.,failed=True,
                                 force_x=0.,force_y=0.,force_z=0.,action0=0.,action1=0.,action2=0.,action3=0.))
            break
        if save_path is not None:
            io["time"].append(env.steps*env.control_dt);io["dt"].append(env.control_dt)
            for key in ("position","velocity","rotation","body_rates"):io[key].append(measured[key].copy())
            io["target"].append(env.target.copy());io["previous_action"].append(env.previous_action.copy())
            io["path_feature"].append(env.path_feature.copy())
            io["path_valid_leg"].append(env.path_observation_flags["valid_leg"])
            io["path_target_changes"].append(env.path_observation_flags["target_changes"])
            io["path_feature_enabled"].append(env.path_observation_flags["feature_enabled"])
            io["observation"].append(obs.copy());io["action"].append(action.copy());io["yaw_reference"].append(env.yaw_reference)
        obs,reward,terminated,truncated,info=env.step(action)
        force=np.asarray(info["force_enu_N"])
        if not np.allclose(force,previous_force,atol=1e-10):
            force_changes.append({"time_s":round(info["time"]-env.control_dt,9),"force_enu_N":force.tolist(),
                                  "type":"off" if np.linalg.norm(force)<1e-9 else "on_or_change"})
            previous_force=force.copy()
        p,v,target,next_target=info["position"],info["velocity"],info["target"],info["next_target"]
        row=dict(time=info["time"],x=p[0],y=p[1],z=p[2],vx=v[0],vy=v[1],vz=v[2],
                 target_x=target[0],target_y=target[1],target_z=target[2],
                 next_target_x=next_target[0],next_target_y=next_target[1],next_target_z=next_target[2],
                 error=info["error"],altitude_error=info["altitude_error"],yaw=info["yaw"],yaw_error=info["yaw_error"],
                 tilt_cos=info["tilt_cos"],goal_index=info["goal_index"],reward=reward,failed=terminated,
                 force_x=force[0],force_y=force[1],force_z=force[2],
                 action0=float(action[0]),action1=float(action[1]),action2=float(action[2]),action3=float(action[3]))
        rows.append(row)
        if terminated:failure="flight_envelope"
        if terminated or truncated:break
    off=[e["time_s"] for e in force_changes if e["type"]=="off"]
    actual_off=off[-1] if off else None
    metrics=compute_metrics([r["time"] for r in rows],[[r[k] for k in ("x","y","z")] for r in rows],
                            [[r[k] for k in ("target_x","target_y","target_z")] for r in rows],
                            [r["yaw_error"] for r in rows],env.max_steps*env.control_dt,failure=failure,mission=mission,
                            completion_time=env.mission.completion_time,force_events=env.parameters["force_events"],
                            force_off_time=actual_off,force_time_source="physics_step")
    actions=np.array([[r[f"action{i}"] for i in range(4)] for r in rows])
    metrics.update(action_delta_rms=float(np.sqrt(np.mean(np.diff(np.vstack([np.zeros(4),actions]),axis=0)**2))),
                   action_saturation_fraction=float(np.mean(np.abs(actions)>=.999)),goals_reached=len(env.mission.events))
    record={"observation_contract":CONTRACT_VERSION,"actor_observations":77,"context_variant":env.context_variant,
            "actor_io_contract":"path77-measured-preaction-v7.0","base74_encoder_calls_per_sample":1,
            "actor_target_semantics":"POST mission-update current target at observation time; trace target judges PRE-update action interval",
            "architecture":architecture,"scenario":scenario,"mission":mission,"eval_seed":int(seed),"environment_seed":int(seed),
            "initial_state":initial["initial_state"],"yaw_reference":env.yaw_reference,
            "scenario_spec":env.parameters,"mission_events":env.mission.events,"force_changes":force_changes,
            "plant_config":env.plant_config,"metrics":metrics,"steps":len(rows),
            "plant_branch":"identified_emulated_rate_PI" if architecture=="hybrid" else "physical_rotors_without_flight_PID",
            "model_family":"fast_identified_rate_proxy" if architecture=="hybrid" else "fast_physical_rotor_model",
            "direct_motor_config":asdict(env.direct_model.config) if architecture=="direct" else None,
            "direct_runtime_provenance":env.direct_model.runtime_provenance() if architecture=="direct" else None,
            "reward_version":env.reward_version,
            "controller_error":failure if failure and failure.startswith("controller_error") else None,
            "zero_time_row_has_no_command":bool(rows[0]["time"]==0 and failure and failure.startswith("controller_error"))}
    if save_path is not None:
        path=Path(save_path);path.mkdir(parents=True,exist_ok=False)
        with (path/"trace.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        # Binary float arrays preserve actual measured inputs and the actor's
        # observations/actions; row i acts over trace row i's preceding interval.
        np.savez_compressed(path/"actor_io.npz",**{k:np.asarray(v) for k,v in io.items()})
        record["trace_path"]=str(path/"trace.csv")
        record["actor_io_path"]=str(path/"actor_io.npz")
        record["hashes"]={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ("trace.csv","actor_io.npz")}
        (path/"summary.json").write_text(json.dumps(record,indent=2))
    env.close()
    return record



def evaluate(policy_path, output, phase_output=None, variant="path"):
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
        env = PathNativeEnv(variant="path", architecture="hybrid", training=False, scenario=scenario, mission=mission)
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
                  policy_sha256=sha(policy_path), evaluated_utc=utc(), observation_contract=CONTRACT_VERSION, actor_observations=77, context_variant="path", training_corner_overlay=False,
                  horizon_s=dict(hover=30., route=40.), source="actual archived trace.csv and mission_events",
                  model_family=MODEL_FAMILY, native_physics=physics_evidence())
    write_json(output / "evaluation.json", result)
    return result


def baseline_receipt(initial, output, existing=None, phase_output=None, variant="path"):
    if existing is not None:
        receipt = Path(existing).resolve()
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record.get("context_variant") != "path" or record.get("schema") != SCHEMA or record.get("model_family") != MODEL_FAMILY or record.get("initial_sha256") != sha(initial):
            raise AssertionError("Baseline is not the identical learned v7 Path77 initial policy")
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
    phase(phase_output, "validation", role="initial_learned_v7_path_baseline", completed_cases=0, total_cases=len(CASES))
    model = BCTrustPPO.load(initial, device="cpu")
    export_policy(model, output / "initial_policy.npz", "hybrid", EncoderConfig())
    record = evaluate(output / "initial_policy.npz", output / "traces", phase_output, variant)
    record["initial_sha256"] = sha(initial)
    record["initial_model"] = str(initial)
    record["baseline_role"] = "Actual learned v7 initial24 provenance; not the retention anchor"
    record["retention_reference"] = retention_reference()["source"]
    record["status"] = "BASELINE_MEASURED_NOT_ACCEPTANCE"
    receipt = output / "baseline.json"
    write_json(receipt, record)
    phase(phase_output, "complete", role="initial_learned_v7_path_baseline", completed_cases=len(CASES))
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


class CurriculumProof(BaseCallback):
    """Audit actual VecEnv infos/dones; no extra policy features or loss terms."""

    def __init__(self, output, stage):
        super().__init__()
        self.output, self.stage = Path(output), stage
        self.counts = dict(completed_episodes=0, failed_episodes=0, successful_routes=0,
                          hard_tailwind_triggers=0, actual_force_onsets=0, completed_by_regime={},
                          complete_corner_windows=0, censored_corner_windows=0,
                          unreached_corners=0, coverage_hold_steps=0)
        self.last_write = 0.

    def _on_step(self):
        rows = []
        for index, (info, done) in enumerate(zip(self.locals.get("infos", []), self.locals.get("dones", []))):
            base = dict(stage=self.stage, environment_steps=self.num_timesteps, vector_env_index=index,
                        episode_id=info.get("curriculum_episode_id"), regime=info.get("curriculum_regime"))
            if info.get("curriculum_trigger_event") is not None:
                self.counts["hard_tailwind_triggers"] += 1
                rows.append(dict(base, event="hard_tailwind_triggered", trigger=info["curriculum_trigger_event"],
                                 actual_force_enu_N=info["force_enu_N"]))
            if info.get("curriculum_force_onset") is not None:
                self.counts["actual_force_onsets"] += 1
                rows.append(dict(base, event="physical_force_onset", onset=info["curriculum_force_onset"]))
            if done:
                summary = info.get("curriculum_episode_summary")
                if summary is None:
                    raise AssertionError("Every completed training episode requires actual curriculum evidence")
                self.counts["completed_episodes"] += 1
                self.counts["failed_episodes"] += int(summary["failed"])
                self.counts["successful_routes"] += int(summary["successful_route"])
                regime = summary["regime"]
                by_regime = self.counts["completed_by_regime"]
                by_regime[regime] = by_regime.get(regime, 0) + 1
                corner_summary = info.get("corner_episode_summary")
                if corner_summary is None or corner_summary["weight_reward_per_m"] != 40.:
                    raise AssertionError("Every completed episode requires sameK40 tracker/censoring evidence")
                for count in ("complete_corner_windows", "censored_corner_windows", "unreached_corners", "coverage_hold_steps"):
                    self.counts[count] += corner_summary[count]
                rows.append(dict(base, event="episode_complete", summary=summary, corner_summary=corner_summary))
        if rows:
            with (self.output / "curriculum_events.jsonl").open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        if time.monotonic() - self.last_write >= 10.:
            self._write_counts()
        return True

    def _write_counts(self):
        write_json(self.output / ("curriculum_counts_" + self.stage + ".json"),
                   dict(version=CURRICULUM_VERSION, stage=self.stage, pid=os.getpid(),
                        environment_steps=self.num_timesteps, heartbeat_utc=utc(), **self.counts))
        self.last_write = time.monotonic()

    def _on_training_end(self):
        self._write_counts()


class JointSelection(Heartbeat):
    def __init__(self, output, interval, baseline, starting_updates, variant):
        super().__init__(output, "ppo")
        self.output = Path(output)
        self.interval = self.next_eval = interval
        self.baseline = baseline
        self.starting_updates = starting_updates
        self.variant = variant
        self.best = None
        self.records = []

    def evaluate(self, label):
        self.model.save(self.output / label)
        policy = self.output / f"{label}.npz"
        export_policy(self.model, policy, "hybrid", EncoderConfig())
        record = evaluate(policy, self.output / "validation" / label, self.output, self.variant)
        score = selection_score(record["aggregate"], self.baseline["aggregate"])
        record.update(score, checkpoint=label, timesteps=self.model.num_timesteps,
                      training_reward_arm=self.variant,
                      accepted=score["fixed_development_guards_pass"],
                      diagnostic_candidate=not score["fixed_development_guards_pass"],
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


def require_bound_preflight(args):
    """A valid plan alone cannot bypass the official preflight --run launcher."""
    from scripts import joint_research_preflight_v1 as preflight
    if args.research_plan is None:
        raise ValueError("Training requires a bound --research-plan launched through joint_research_preflight_v1.py --run")
    plan_path = args.research_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    command = preflight.validate(plan)
    if Path(command[2]).resolve() != Path(__file__).resolve() or sys.argv[1:] != plan["argv"]:
        raise ValueError("Current invocation differs from its bound preflight plan")
    if plan["stage"] != "bounded_pilot":
        raise ValueError("Only bounded pilot authorization exists for this driver")
    try:
        parent = Path("/proc") / str(os.getppid()) / "cmdline"
        argv = parent.read_bytes().decode("utf-8").split("\0")
        supplied = argv[argv.index("--plan") + 1]
        is_preflight = any(Path(item).name == "joint_research_preflight_v1.py" for item in argv)
        if not is_preflight or "--run" not in argv or (ROOT / supplied).resolve() != plan_path:
            raise ValueError("Official preflight --run parent with the same plan is required")
    except (OSError, ValueError, IndexError) as exc:
        raise ValueError("Direct training is blocked; use the bound preflight --run command") from exc
    return plan


def run_arm(args, variant, output, baseline, receipt):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = source_hashes()
    initial_hash = sha(args.initial)
    config = dict(schema=SCHEMA, task="joint_corner_metric", variant=variant,
        algorithm="Learned v7 Path77 final + fresh Adam critic warmup + fixed same-initial actor KL PPO",
        initial=str(args.initial), initial_sha256=initial_hash, prior_sha256=initial_hash,
        prior_direction="current Normal || frozen SAME learned v7 initial Path77 actor Normal",
        fixed_kl_weight=KL_WEIGHT, geometry_weight=GEOMETRY_WEIGHT,
        corner_reward=dict(enabled=variant == "corner", configured_weight_reward_per_m=CORNER_WEIGHT, formula="-40*(new_peak-old_peak), no dt multiplier; three closed1s outgoing finite3D windows"),
        retention_reference=retention_reference()["source"],
        prospective_geometry_weight=GEOMETRY_WEIGHT, reward_base=REWARD_VERSION,
        model_family=MODEL_FAMILY, native_physics=physics_evidence(), selection_version=SELECTION_VERSION,
        changed_factor="ONLY effective K40 corner reward: w8_control ignores increments; corner adds them. Same actual77 inputs, tracker, coverage hold, ZIP/prior/W8/nativeplant/v5curriculum/reset/budgets/selection",
        observation_contract=CONTRACT_VERSION, feature_scale_m=.25, feature_clipping=False,
        stopping_reward=dict(enabled=False, weight=0., included_in_either_arm=False),
        common_changes_against_v7="Both learned v7 Path77 final; fresh Adam reset13-to0; common successful-route corner-coverage hold; no observation changes",
        curriculum_version=CURRICULUM_VERSION,
        route_wind_curriculum=dict(hard_state_triggered_fraction=.5, broad_clock_fraction=.5,
            hard=dict(trigger="actual goal2 confirmation and current public goal_index3, then delay", delay_s=[0., .8],
                      force_N=[2.5, 3.5], incoming_leg="goal2 togoal3 horizontal -X", rotation_deg=[-20., 20.], duration_s=[2., 6.]),
            broad=dict(constant_fraction=.5, constant_on_s=[0., 7.], gust_fraction=.5,
                       gust_onset_intervals_s=[[0., 2.], [3., 5.], [6., 7.8]], gust_duration_s=[.8, 2.],
                       cap_off_before_next_on_s=.02, magnitude_N=[.3, 3.5])),
        hover_training_wind="EXACT frozen v4 sampler distribution; nominal and initial-condition RNG draws also preserved",
        curriculum_rng="SHA256 of frozen v4 sampler post-draw RNG state; separate generator does not advance parent RNG",
        curriculum_proof="Actual physical onset/trigger/episode VecEnv infos logged to curriculum_events.jsonl and phase counts",
        attribution="Paired short pilot; ONLY effective corner reward differs; common coverage hold and actual77 inputs; no automatic model/demo promotion",
        research_plan=str(args.research_plan) if args.research_plan else None,
        research_plan_sha256=sha(args.research_plan) if args.research_plan else None,
        development_route_time_limit_s=ROUTE_TIME_LIMIT_S, development_route_goal_rmse_limit_m=ROUTE_GOAL_RMSE_LIMIT_M,
        development_each_route_segment_peak_limit_m=ABSOLUTE_SEGMENT_PEAK_LIMIT_M,
        finite_segment_targets_m=FINITE_TARGETS_M, corner_peak_targets_m=CORNER_TARGETS_M,
        actual_gazebo_contract_unchanged="docs/joint_applicability_contract_v2.json",
        geometry_reward="-dt * 8 * distance_to_preupdate_active_finite_3D_segment_squared for route goals1..4",
        envs=N_ENVS, cpu_threads=1, seed=args.seed, ppo_seed=args.seed, warmup_seed=args.warmup_seed,
        warmup_steps=args.warmup_steps, ppo_steps=args.steps, eval_every=args.eval_every,
        candidate_count=args.steps // args.eval_every + 1,
        checkpoint_independence="Same learning trajectory;131072/262144/final are matched-step diagnostics, not three independent trained policies",
        training_missions=dict(route=.8, hover=.2),
        training_disturbances=dict(nominal=.2, wind=.8),
        training_force_magnitude_N=[.3, 3.5], nominal_vehicle=True,
        training_success_terminal="confirmed goal4, delayed only until already-open corner windows complete; immediate flight failure and actual max_steps override hold; completion20 once",
        evaluation="Observation-only PathNativeEnv over frozen QuadEnvJointV3; full30s hover and40s route; 24 fixed dev cases; archived traces",
        validation_seeds=list(SEEDS), reserved_confirmation_seeds_not_opened=[93001, 93002, 93003],
        baseline_receipt=str(receipt), baseline_receipt_sha256=sha(receipt),
        actor_observations=77, no_future_goals_or_force_inputs=True,
        selection_order="survival, completion, heading, geometry coverage; fixed constraint count/maxexcess; worst finite/corner target ratio; meanfinite",
        nominal_hover_development_limit="Frozen original Gamma77 anchor: min(.05,max(1.2*original_nominal_hover_RMSE,original_nominal_hover_RMSE+.02)); no cumulative-drift reset",
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
        old_optimizer_states = len(model.policy.optimizer.state)
        if old_optimizer_states != 13:
            raise AssertionError("Learned initial ZIP must contain the expected13 Adam states before reset")
        model.policy.optimizer.state.clear()
        if model.policy.optimizer.state:
            raise AssertionError("Optimizer reset failed")
        optimizer_audit = dict(original_parameter_states=old_optimizer_states, after_reset_parameter_states=0,
                               same_optimizer_type_and_hyperparameters=True, reset_before_critic_warmup=True)
        write_json(output / "optimizer_reset.json", optimizer_audit)
        actual_optimizer_steps = dict(critic=0, ppo=0)
        optimizer_stage = ["critic"]
        def count_optimizer_step(*unused):
            actual_optimizer_steps[optimizer_stage[0]] += 1
        step_hook = model.policy.optimizer.register_step_post_hook(count_optimizer_step)
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
        warm_proof = CurriculumProof(output, "critic")
        model.learn(args.warmup_steps, reset_num_timesteps=True,
                    callback=[Heartbeat(output, "critic"), warm_proof], progress_bar=False)
        critic_changes = compare_states(model.policy, initial, False)
        if compare_states(model.policy, initial, True) or not critic_changes:
            raise AssertionError("Critic warmup must change critic and preserve actor including log_std")
        if any(not torch.equal(value, prior[name]) for name, value in model.bc_prior.state_dict().items()):
            raise AssertionError("Fixed calibrated prior changed during critic warmup")
        warmup = dict(environment_steps=model.num_timesteps, completed_critic_only_epochs=model._n_updates - before,
                      actor_bit_identical=True, prior_bit_identical=True, changed_critic_parameters=critic_changes,
                      actual_optimizer_steps=actual_optimizer_steps["critic"], optimizer_reset=optimizer_audit,
                      optimizer_parameter_states_after_warmup=len(model.policy.optimizer.state))
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
        optimizer_stage[0] = "ppo"
        before = model._n_updates
        selection = JointSelection(output, args.eval_every, baseline, before, variant)
        ppo_proof = CurriculumProof(output, "ppo")
        phase(output, "ppo", environment_steps=0)
        model.learn(args.steps, reset_num_timesteps=True, callback=[selection, ppo_proof], progress_bar=False)
        selection.evaluate("final")
        phase(output, "verification", environment_steps=model.num_timesteps)
        probe = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
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
        if actual_updates != actual_optimizer_steps["ppo"]:
            raise AssertionError("Independent optimizer.step hook disagrees with PPO KL log")
        if sample_uses != model.bc_kl_minibatch_sample_uses:
            raise AssertionError("Actual on-policy sample-use counts disagree")
        if not ppo_proof.counts["completed_episodes"] or not ppo_proof.counts["hard_tailwind_triggers"]:
            raise AssertionError("Actual PPO curriculum evidence must include completed episodes and triggered hard wind")
        if len(selection.records) != config["candidate_count"]:
            raise AssertionError("Prospective candidate budget changed")
        selected = json.loads((output / "selection.json").read_text())
        model_hashes = {name: sha(output / name) for name in
                        ("critic_warmed_model.zip", "best_model.zip", "best_policy.npz", "final.zip", "final.npz")}
        write_json(output / "model_manifest.json", dict(initial_sha256=initial_hash,
                   prior_sha256=initial_hash, output_sha256=model_hashes, completed_utc=utc()))
        complete = dict(status="complete", task="joint_corner_metric", variant=variant,
            model_family=MODEL_FAMILY, selection_version=SELECTION_VERSION,
            critic_only_timesteps=warmup["environment_steps"], ppo_timesteps=model.num_timesteps,
            completed_critic_only_epochs=warmup["completed_critic_only_epochs"],
            additional_actor_critic_ppo_epochs=model._n_updates - before,
            actor_critic_ppo_gradient_steps=actual_updates, on_policy_prior_kl_gradient_steps=model.bc_kl_gradient_steps,
            actual_optimizer_step_hook_counts=actual_optimizer_steps, optimizer_reset=optimizer_audit,
            on_policy_prior_kl_minibatch_sample_uses=sample_uses, rollout_updates=len(log),
            offline_supervised_anchor_gradient_steps=0, actor_unchanged_by_critic_warmup=True,
            actor_changed_by_ppo=True, prior_bit_identical=True, source_hashes_unchanged=True,
            numpy_export_max_error=parity, selected_policy_sha256=sha(output / "best_policy.npz"),
            final_policy_sha256=sha(output / "final.npz"), selected_objective=selection.best,
            output_model_sha256=model_hashes,
            curriculum_version=CURRICULUM_VERSION, actual_curriculum_counts=dict(critic=warm_proof.counts, ppo=ppo_proof.counts),
            curriculum_events_sha256=sha(output / "curriculum_events.jsonl"),
            selected_metrics=selected["aggregate"], fixed_development_guards_pass=selected["fixed_development_guards_pass"],
            accepted=selected["fixed_development_guards_pass"],
            diagnostic_candidate=not selected["fixed_development_guards_pass"],
            all_candidates_failed_development_guards=not any(row["guards_pass"] for row in selection.records),
            acceptance_scope="Fast development engineering proxy only; no automatic demo/registry/actualGazebo acceptance",
            candidate_records=selection.records, wall_seconds=time.monotonic() - started, completed_utc=utc(),
            interpretation="Bounded paired corner-reward pilot; least-bad model remains diagnostic if every candidate fails hard guards")
        write_json(output / "complete.json", complete)
        phase(output, "complete", environment_steps=model.num_timesteps)
        print(json.dumps(complete), flush=True)
        return complete
    except Exception as exc:
        write_json(output / "error.json", dict(status="error", error=repr(exc), traceback=traceback.format_exc(), utc=utc()))
        phase(output, "error", error=repr(exc))
        raise
    finally:
        if "step_hook" in locals():
            step_hook.remove()
        vector.close()


def replay_actor_io(path, variant):
    """Independent stateful encoder replay of ACTUAL measured actor inputs."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    encoder = ObservationEncoderPath77()
    encoder.reset(float(data["yaw_reference"][0]))
    base_error = path_error = 0.
    for index in range(len(data["time"])):
        encoded = encoder.observe(*(data[key][index] for key in
            ("position", "velocity", "rotation", "body_rates", "target", "previous_action", "dt")))
        base_error = max(base_error, float(np.max(np.abs(encoded[:74] - data["observation"][index, :74]))))
        raw = encoded[-3:].copy()
        path_error = max(path_error, float(np.max(np.abs(raw - data["path_feature"][index]))))
        assert bool(data["path_valid_leg"][index]) == encoder.path_context.valid_leg
        assert int(data["path_target_changes"][index]) == encoder.path_context.target_changes
        assert bool(data["path_feature_enabled"][index])
        expected = raw
        assert np.array_equal(expected, data["observation"][index, -3:])
    assert base_error == 0. and path_error == 0.
    return dict(rows=len(data["time"]), base74_max_error=base_error, causal3_max_error=path_error,
                target_changes=encoder.path_context.target_changes,
                last_leg_retained=encoder.path_context.valid_leg,
                nonzero_path_samples=int(np.count_nonzero(np.linalg.norm(data["path_feature"], axis=1))))


def optimizer_probe(output, initial, variant):
    """128 critic +128 PPO real transitions per arm; probe outputs never reused."""
    output.mkdir(parents=True, exist_ok=False)
    vector = VecMonitor(DummyVecEnv([lambda: JointTaskEnv(variant=variant, architecture="hybrid", training=True)
                                    for _ in range(2)]))
    try:
        vector.set_attr("training_progress", 1.)
        vector.seed(9511)
        model = BCTrustPPO.load(initial, env=vector, device="cpu", seed=9511,
                               n_steps=64, batch_size=64, n_epochs=1)
        model.anchor_steps = 0
        assert model.gamma == .995 and model.gae_lambda == .97
        assert model.lr_schedule(1.) == 3e-5
        original_states = len(model.policy.optimizer.state)
        assert original_states == 13
        model.policy.optimizer.state.clear()
        assert not model.policy.optimizer.state
        assert all(group.get("weight_decay", 0.) == 0. for group in model.policy.optimizer.param_groups)
        model.configure_bc_prior(initial, 0., output / "prior_kl.jsonl")
        model.bc_kl_enabled = False
        prior = copy.deepcopy(model.bc_prior.state_dict())
        before = copy.deepcopy(model.policy.state_dict())
        first = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
        assert all(torch.count_nonzero(before[name][:, -3:]).item() > 0 for name in first)
        counts, stage = dict(critic=0, ppo=0), ["critic"]
        gradient_max = {key: dict(actor_path_columns=0., critic_path_columns=0.) for key in counts}
        def pre_hook(optimizer, unused_args, unused_kwargs):
            for name, parameter in model.policy.named_parameters():
                if name in first and parameter.grad is not None:
                    key = "actor_path_columns" if name == first[0] else "critic_path_columns"
                    gradient_max[stage[0]][key] = max(gradient_max[stage[0]][key],
                        float(torch.linalg.vector_norm(parameter.grad[:, -3:]).detach()))
        def post_hook(*unused):
            counts[stage[0]] += 1
        pre = model.policy.optimizer.register_step_pre_hook(pre_hook)
        post = model.policy.optimizer.register_step_post_hook(post_hook)
        for name, parameter in model.policy.named_parameters():
            if actor_name(name):
                parameter.requires_grad_(False)
        model.set_logger(configure(str(output / "critic"), ["csv"]))
        model.learn(128, reset_num_timesteps=True, progress_bar=False)
        assert not compare_states(model.policy, before, True)
        assert compare_states(model.policy, before, False)
        warm = copy.deepcopy(model.policy.state_dict())
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
        model.bc_kl_weight, model.bc_kl_enabled = KL_WEIGHT, True
        model.set_random_seed(9611)
        model._last_obs = None
        model.set_logger(configure(str(output / "ppo"), ["csv"]))
        stage[0] = "ppo"
        model.learn(128, reset_num_timesteps=True, progress_bar=False)
        assert compare_states(model.policy, warm, True)
        assert all(torch.equal(value, prior[name]) for name, value in model.bc_prior.state_dict().items())
        observations = model.rollout_buffer.observations
        feature_nonzero = int(np.count_nonzero(np.linalg.norm(observations[..., -3:], axis=-1)))
        assert feature_nonzero > 0
        assert gradient_max["ppo"]["actor_path_columns"] > 0.
        assert gradient_max["ppo"]["critic_path_columns"] > 0.
        export_policy(model, output / "probe.npz", "hybrid", EncoderConfig())
        probe = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
        parity = float(np.max(np.abs(model.predict(probe, deterministic=True)[0] - NumpyPolicy(output / "probe.npz").predict(probe))))
        assert parity < 1e-5 and counts == dict(critic=2, ppo=2)
        pre.remove(); post.remove()
        result = dict(status="PASS", actual_environment_transitions=256,
            critic_transitions=128, ppo_transitions=128, optimizer_steps=counts,
            original_adam_parameter_states=original_states, adam_states_after_clear=0,
            no_weight_decay=True, path_column_gradient_max=gradient_max,
            ppo_nonzero_feature_samples=feature_nonzero, numpy_torch_parity1000=parity,
            actor_warmup_unchanged=True, actor_ppo_changed=True, prior_unchanged=True,
            prior_sha256=sha(initial), KL_weight=.1, actor_and_critic_observations=77,
            learning_rate=3e-5, gamma=.995, gae_lambda=.97,
            probe_only_not_pretraining=True, probe_model_not_used_by_training=True)
        write_json(output / "receipt.json", result)
        return result
    finally:
        vector.close()


def coverage_fixtures():
    """Real frozen Mission/reward with synthetic measured position, not flight evidence."""
    from unittest.mock import patch
    from quadrl_corner_v8.env import FrozenPathTaskEnv, coverage_termination
    from quadrl.reward import reward_value
    outcomes = {}
    assert coverage_termination(training=True, mission="route", successful=True,
        failed=False, horizon=False, pending_windows=1, parent_terminated=True,
        parent_truncated=True) == (False, True, False)
    for variant in ("w8_control", "corner"):
        for ending in ("complete", "failure", "horizon"):
            env = JointTaskEnv(variant=variant, architecture="hybrid", training=True)
            try:
                env.mission = Mission("route")
                env.steps = 0
                env.max_steps = 135 if ending == "horizon" else 2000
                env.corner_tracker = CornerPeakReward("route", weight=40.)
                env.corner_tracker.observe(0., env.mission.target, env.mission.target)
                env.coverage_hold_steps = env.completion_event_count = 0
                bonuses, holds, summaries = [], [], []
                def parent_step(current, action):
                    current.steps += 1
                    stamp = current.steps * current.control_dt
                    target = current.mission.target.copy()
                    position = target + [.1, 0., 0.]
                    event = current.mission.update(stamp, position)
                    completed_event = event is not None and current.mission.completion_time is not None
                    failed = bool(ending == "failure" and current.mission.completion_time is not None
                                  and stamp >= current.mission.completion_time + .12 - 1e-9)
                    base = reward_value(.1, 0., 1., 0., 0., 0., 0., event is not None,
                                        completed_event, failed, current.control_dt)
                    bonuses.append((10. if event else 0.) + (20. if completed_event else 0.))
                    success = current.mission.completion_time is not None and not failed
                    info = dict(time=stamp, position=position.tolist(), next_target=current.mission.target.tolist(),
                        target=target.tolist(), mission_event=event, failed=failed,
                        is_success=success, force_enu_N=[0., 0., 0.])
                    return np.zeros(77, np.float32), base, failed or success, current.steps >= current.max_steps, info
                with patch.object(FrozenPathTaskEnv, "step", parent_step):
                    for unused in range(2000):
                        obs, reward, terminal, truncated, info = env.step(np.zeros(4))
                        assert reward == info["base_w8_reward"] + info["corner_reward_adjustment"]
                        if info["corner_coverage_hold"]:
                            holds.append(info["time"])
                            assert bonuses[-1] == (30. if info["mission_event"] is not None else 0.)
                            assert "curriculum_episode_summary" not in info
                        if terminal or truncated:
                            summaries.append(info["corner_episode_summary"])
                            break
                assert len(summaries) == 1 and env.completion_event_count == 1
                assert sum(bonuses) == 70. and sum(b == 30. for b in bonuses) == 1
                report = summaries[0]
                assert holds and report["coverage_hold_steps"] == len(holds)
                assert report["enabled"] == (variant == "corner")
                if ending == "complete":
                    assert info["is_success"]
                    assert terminal and not truncated and report["complete_corner_windows"] == 3
                    assert report["censored_corner_windows"] == 0
                    assert report["finish_reason"] == "completed_training_route"
                else:
                    assert not info["is_success"]
                    assert report["censored_corner_windows"] > 0
                    assert any(window["full_window_peak_m"] is None for window in report["windows"])
                    assert report["finish_reason"] == ("failure" if ending == "failure" else "full_horizon")
                    assert terminal if ending == "failure" else truncated
                outcomes[variant + "/" + ending] = dict(last_time_s=info["time"], terminated=terminal,
                    truncated=truncated, total_arrival_completion_bonus=sum(bonuses),
                    completion20_emissions=1, no_extra_hold_bonus=True, report=report)
            finally:
                env.close()
    for ending in ("complete", "failure", "horizon"):
        a, b = outcomes["w8_control/" + ending], outcomes["corner/" + ending]
        assert (a["last_time_s"], a["terminated"], a["truncated"]) == (b["last_time_s"], b["terminated"], b["truncated"])
        for key in ("windows", "reward", "coverage_hold_steps", "complete_corner_windows", "censored_corner_windows"):
            assert a["report"][key] == b["report"][key]
    return outcomes


def actual_lockstep(initial_npz):
    from quadrl_path_v7.env import JointTaskEnv as ParentTaskEnv
    controller = NumpyPolicy(initial_npz)
    results = []
    for seed in (9511, 9512, 9513):
        envs = [ParentTaskEnv(variant="path", architecture="hybrid", training=True)]
        envs += [JointTaskEnv(variant=arm, architecture="hybrid", training=True) for arm in ("w8_control", "corner")]
        try:
            for env in envs:
                env.training_progress = 1.
            observations = [env.reset(seed=seed)[0] for env in envs]
            replay = ObservationEncoderPath77(); replay.reset(envs[1].yaw_reference)
            events = interior = extra_steps = samples = 0
            corner_nonzero = False
            initial_mission = envs[0].mission.kind
            for unused in range(envs[0].max_steps):
                assert all(np.array_equal(observations[0], observation) for observation in observations[1:])
                measured = envs[1].measured_state
                encoded = replay.observe(*(measured[key] for key in ("position", "velocity", "rotation", "body_rates")),
                    envs[1].target, envs[1].previous_action, envs[1].control_dt)
                assert np.array_equal(encoded, observations[1])
                action = controller.predict(observations[0])
                transitions = [env.step(action) for env in envs]
                observations = [row[0] for row in transitions]
                parent, control, corner = transitions
                pi, ci, ki = parent[4], control[4], corner[4]
                assert all(np.array_equal(envs[0].state()[0], env.state()[0]) for env in envs[1:])
                assert control[1] == parent[1] == ci["base_w8_reward"] == ki["base_w8_reward"]
                assert corner[1] == control[1] + ki["corner_tracker_reward_adjustment"]
                assert ci["corner_tracker_reward_adjustment"] == ki["corner_tracker_reward_adjustment"]
                assert ci["corner_reward_adjustment"] == 0.
                assert control[2:4] == corner[2:4]
                assert envs[1].corner_tracker.report() == envs[2].corner_tracker.report()
                for key in ("position", "velocity", "target", "next_target", "mission_event", "failed", "force_enu_N", "judged_goal_index", "geometry_reward_adjustment"):
                    assert pi[key] == ci[key] == ki[key]
                event = ci["mission_event"]
                if event is not None:
                    events += 1
                    index = event["goal_index"]
                    if index in (1, 2, 3):
                        interior += 1
                        window = envs[1].corner_tracker.windows[-1]
                        assert window.transition_s == ci["time"] and window.samples == 1
                        assert list(window.a) == event["target"] and list(window.b) == ci["next_target"]
                        assert ci["judged_goal_index"] == index
                corner_nonzero |= ci["corner_tracker_reward_adjustment"] != 0.
                if parent[2:4] != control[2:4]:
                    assert ci["corner_coverage_hold"] and not pi["failed"] and pi["is_success"]
                    extra_steps += 1
                samples += 1
                if control[2] or control[3]:
                    break
            assert initial_mission != "route" or (events == 5 and interior == 3 and corner_nonzero)
            results.append(dict(seed=seed, mission=initial_mission, samples=samples, mission_events=events,
                same_sample_interior_transitions=interior, observed_nonzero_corner_penalty=corner_nonzero,
                common_coverage_hold_steps=extra_steps, full77_observations_exact=True,
                base_w8_pre_target_reward_exact=True, physical_state_exact=True,
                actor77_independent_replay_exact=True, reward_only_between_arm_difference=True))
        finally:
            for env in envs:
                env.close()
    assert any(result["mission"] == "route" for result in results)
    return results


def verify(output, initial, semantic_receipt):
    if semantic_receipt.exists():
        raise FileExistsError("Fresh semantic receipt required; keep previous proof")
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    physics = physics_evidence()
    assert sha(INITIAL_NPZ) == INITIAL_NPZ_SHA256
    retention = retention_reference()
    model = BCTrustPPO.load(initial, device="cpu")
    assert len(model.policy.optimizer.state) == 13
    export_policy(model, output / "initial77.npz", "hybrid", EncoderConfig())
    saved = NumpyPolicy(INITIAL_NPZ)
    exported = NumpyPolicy(output / "initial77.npz")
    x = np.random.default_rng(17491).normal(size=(1000, 77)).astype(np.float32)
    parity = dict(numpy_torch=float(np.max(np.abs(model.predict(x, deterministic=True)[0] - exported.predict(x)))),
                  original_v7_npz=float(np.max(np.abs(saved.predict(x) - exported.predict(x)))))
    assert parity["numpy_torch"] < 1e-5 and parity["original_v7_npz"] == 0.
    from scripts import probe_joint_corner_reward_v8 as pure_probe
    fixtures = pure_probe.fixture_checks()
    assert not pure_probe.FAILURES
    coverage = coverage_fixtures()
    lockstep = actual_lockstep(output / "initial77.npz")
    # Every geometry guard, plus frozen retention independent of new baseline.
    metrics = dict(flight_failures=0, route_incomplete=0, heading_violations=0, geometry_ineligible=0,
        active_segment_peak_failures=0, active_segment_peak_checks=[dict(scenario=s, seed=k, peak_m=.1) for s in SCENARIOS for k in SEEDS], cells={})
    for scenario in SCENARIOS:
        metrics["cells"][scenario + "/hover"] = dict(goal_rmse_m=.03)
        metrics["cells"][scenario + "/route"] = dict(goal_rmse_m=.4, restricted_time_s=9., segment_rmse_m=.1, mean_corner_peak_m=.19)
    assert selection_score(metrics, metrics)["fixed_development_guards_pass"]
    drifted = copy.deepcopy(metrics)
    for scenario in SCENARIOS:
        drifted["cells"][scenario + "/hover"]["goal_rmse_m"] = 100.
    assert selection_score(metrics, metrics) == selection_score(metrics, drifted)
    rejected = []
    for scenario in SCENARIOS:
        for field, limit in (("segment_rmse_m", FINITE_TARGETS_M[scenario]), ("mean_corner_peak_m", CORNER_TARGETS_M[scenario])):
            bad = copy.deepcopy(metrics); bad["cells"][scenario + "/route"][field] = limit + 1e-8
            assert not selection_score(bad, metrics)["fixed_development_guards_pass"]
            rejected.append(scenario + "/" + field)
    bad = copy.deepcopy(metrics); bad["active_segment_peak_checks"][0]["peak_m"] = .50000001
    assert not selection_score(bad, metrics)["fixed_development_guards_pass"]
    for scenario in ("force_strong", "gust"):
        bad = copy.deepcopy(metrics)
        bad["cells"][scenario + "/hover"]["goal_rmse_m"] = 1.1 * retention["aggregate"]["cells"][scenario + "/hover"]["goal_rmse_m"] + 1e-7
        assert not selection_score(bad, drifted)["fixed_development_guards_pass"]
    probes = {arm: optimizer_probe(output / (arm + "_optimizer_probe"), initial, arm) for arm in ("w8_control", "corner")}
    assert source_hashes() == hashes and sha(initial) == INITIAL_SHA256
    result = dict(schema="joint-corner-training-semantics-v8.0", status="PASS", utc=utc(),
        scope="Analytic corner/coverage fixtures, actual native/W8 lockstep,512 real optimizer-probe transitions; no bounded pilot or Gazebo",
        initializer_sha256=sha(initial), initializer_npz_sha256=INITIAL_NPZ_SHA256,
        initializer_adam_states_before=13, initializer_adam_states_after_clear=0,
        initial_parity1000=parity, pure_tracker_fixtures=fixtures,
        common_coverage_termination_fixtures=coverage, actual_lockstep_fixtures=lockstep,
        hard_geometry_rejections=rejected, retention_reference=retention["source"],
        retention_anchor_independent_of_new_initial_baseline=True,
        actual_optimizer_probes=probes, source_hashes=hashes, native_physics=physics,
        selection_version=SELECTION_VERSION, observation_contract=CONTRACT_VERSION,
        actor_and_critic_observations=77, both_actual_path_inputs=True,
        configured_corner_weight_reward_per_m=40., control_tracks_same40_but_effective_reward_zero=True,
        tracker_sha256=TRACKER_SHA256, frozen_offline72route_report_sha256=PROTOTYPE_SHA256,
        known_alignment_gap="FixedK40 rejects tail finite regression in selected/final discounted score; other individual/aggregate orderings still conflict. Not global metric alignment or quality guarantee.",
        paired_training_executed=False, Gazebo_executed=False,
        reserved_confirmation_seeds_not_opened=[93001, 93002, 93003])
    write_json(semantic_receipt, result); write_json(output / "verify.json", result)
    phase(output, "complete", scope="verification only")
    print(json.dumps(dict(status="PASS", receipt=str(semantic_receipt), sha256=sha(semantic_receipt))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("w8_control", "corner"), default="corner")
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9611)
    parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--warmup-steps", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--eval-every", type=int, default=131072)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--retention-reference", type=Path, default=RETENTION_REFERENCE,
                        help="Fixed original Gamma77 native baseline; must not move to new v7 initial")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--research-plan", type=Path)
    parser.add_argument("--semantic-receipt", type=Path, default=ROOT / "audit/joint_corner_training_semantics_v8_20261113_final.json")
    args = parser.parse_args()
    for name in ("initial", "output", "baseline", "retention_reference", "research_plan", "semantic_receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (value if value.is_absolute() else ROOT / value).resolve())
    if args.initial != INITIAL.resolve() or sha(args.initial) != INITIAL_SHA256:
        parser.error("Exact learned v7 Path final ZIP is required")
    if args.retention_reference != RETENTION_REFERENCE.resolve() or sha(args.retention_reference) != RETENTION_SHA256:
        parser.error("Frozen original Gamma77 retention baseline path and bytes required")
    if args.output.exists():
        parser.error("Fresh output required; preserve earlier results")
    if REWARD_VERSION != "precision-arrival-v3":
        parser.error("Frozen precision-arrival-v3 reward required")
    torch.set_num_threads(1)
    physics_evidence(); retention_reference()
    if args.verify:
        verify(args.output, args.initial, args.semantic_receipt)
        return
    if args.baseline_only:
        if args.baseline:
            parser.error("--baseline-only requires a fresh actual baseline")
        baseline, receipt = baseline_receipt(args.initial, args.output, variant="path")
        print(json.dumps(dict(status="BASELINE_COMPLETE", receipt=str(receipt), receipt_sha256=sha(receipt),
                              initial_sha256=sha(args.initial), aggregate=baseline["aggregate"])), flush=True)
        return
    if (args.steps, args.warmup_steps, args.eval_every, args.seed, args.warmup_seed) != (262144, 131072, 131072, 9611, 9511):
        parser.error("Only fixed262144 PPO/131072 critic/eval, seeds9611/9511 authorized")
    if "--retention-reference" not in sys.argv:
        parser.error("Training plan must explicitly bind --retention-reference")
    require_bound_preflight(args)
    initial_model = BCTrustPPO.load(args.initial, device="cpu")
    assert initial_model.gamma == .995 and initial_model.gae_lambda == .97 and initial_model.lr_schedule(1.) == 3e-5
    rollout = initial_model.n_steps * N_ENVS
    if min(args.steps, args.warmup_steps, args.eval_every) < rollout or any(
            count % rollout for count in (args.steps, args.warmup_steps, args.eval_every)):
        parser.error("Budgets must be exact inherited rollout multiples")
    if args.output.name != args.variant:
        parser.error("Output basename must equal w8_control/corner arm")
    if not args.baseline:
        parser.error("Actual shared initial24 baseline must be measured before training")
    baseline, receipt = baseline_receipt(args.initial, args.output.with_name("unused_baseline"), args.baseline, variant="path")
    run_arm(args, args.variant, args.output, baseline, receipt)


if __name__ == "__main__":
    main()



"""Conditioned trained actors; unchanged frozen native physics and metrics."""
from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scripts import train_joint_route_basis_v15 as frozen
from scripts import run_joint_route_basis_pair_v15 as comparison
from .policy import CONTRACT_VERSION,RAW_CONTRACT,SOURCE_SHA,NumpyPolicyRouteMemoryTrain

ROOT=Path(__file__).resolve().parents[1]
COMMON_BASELINE=ROOT/"artifacts/refinement/joint_v2/route_basis_v15_baseline111/velocity_residual/baseline.json"
compute_metrics=frozen.compute_metrics
read,write_json=frozen.read,frozen.write_json


def run_episode(env,controller,architecture,history,scenario,mission,seed,save_path):
    obs,initial=env.reset(seed=seed)
    controller.reset(env.yaw_reference)
    rows=[];io={key:[] for key in ("time","dt","position","velocity","rotation","body_rates","target","previous_action","observation","action","yaw_reference","path_feature","path_valid_leg","path_target_changes","path_feature_enabled","basis_feature","basis_gate","raw_observation","effective_observation","raw_integral_world","conditioned_integral_world","memory_target_changed","memory_raw_encoder_calls")}
    force_changes=[];previous_force=np.zeros(3)
    failure=None
    for index in range(env.max_steps):
        measured=env.measured_state
        try:
            effective=controller.encoder.observe(measured["position"],measured["velocity"],measured["rotation"],
                measured["body_rates"],env.target,env.previous_action,env.control_dt)
            diagnostic=controller.encoder.diagnostics()
            assert np.array_equal(diagnostic["raw_observation"],obs),"Original80 env/controller measured replay differs"
            action=controller.predict(effective)
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
            io["basis_feature"].append(env.basis_feature.copy());io["basis_gate"].append(env.basis_gate)
            io["path_valid_leg"].append(env.path_observation_flags["valid_leg"])
            io["path_target_changes"].append(env.path_observation_flags["target_changes"])
            io["path_feature_enabled"].append(env.path_observation_flags["feature_enabled"])
            io["observation"].append(effective.copy())
            io["raw_observation"].append(obs.copy());io["effective_observation"].append(effective.copy())
            for key in ("raw_integral_world","conditioned_integral_world"):io[key].append(diagnostic[key].copy())
            io["memory_target_changed"].append(diagnostic["target_changed"])
            io["memory_raw_encoder_calls"].append(diagnostic["raw_encoder_calls"])
            io["action"].append(action.copy());io["yaw_reference"].append(env.yaw_reference)
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
    record={"observation_contract":CONTRACT_VERSION,"actor_observations":80,"basis_kind":env.basis_kind,"context_variant":env.context_variant,
            "actor_io_contract":"route-memory-trained-measured-preaction-v21.0",
            "conditioning_mode":controller.mode,"initial_actor_npz_sha256":SOURCE_SHA,"current_policy_contract":controller.contract,
            "raw_observation_contract":RAW_CONTRACT,"actual_NN_input":"effective_observation; observation is same actual effective80",
            "environment_raw_encoder_calls_per_sample":1,"controller_raw_encoder_calls_per_sample":1,
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
        np.savez_compressed(path/"actor_io.npz",basis_kind=np.array(env.basis_kind),**{k:np.asarray(v) for k,v in io.items()})
        record["trace_path"]=str(path/"trace.csv")
        record["actor_io_path"]=str(path/"actor_io.npz")
        record["hashes"]={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ("trace.csv","actor_io.npz")}
        (path/"summary.json").write_text(json.dumps(record,indent=2))
    env.close()
    return record

def evaluate(mode,policy_path,output,root_output):
    output.mkdir(parents=True,exist_ok=False)
    controller=NumpyPolicyRouteMemoryTrain(policy_path,expected_mode=mode);records=[]
    for index,(scenario,mission,seed) in enumerate(frozen.CASES):
        frozen.phase(root_output,"validation",conditioning_mode=mode,completed_cases=index,
            total_cases=24,scenario=scenario,mission=mission,eval_seed=seed,
            training_transitions=0,optimizer_steps=0)
        destination=output/f"{mission}_{scenario}_{seed}"
        env=frozen.NativeEvalEnv(basis_kind="velocity_residual",architecture="hybrid",training=False,scenario=scenario,mission=mission)
        try:record=run_episode(env,controller,"hybrid",True,scenario,mission,seed,destination)
        finally:env.close()
        record.update(model_family=frozen.MODEL_FAMILY,plant_branch="native_RP_jerk_shaper_400Hz_physical_feedback",
            joint_rate_model_version=frozen.MODEL_VERSION,conditioning_mode=mode,current_policy_sha256=sha(policy_path))
        if mission=="route":
            with (destination/"trace.csv").open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
            record["geometry"]=frozen.geometry_metrics([float(r["time"]) for r in rows],
                [[float(r[k]) for k in ("x","y","z")] for r in rows],
                [[float(r[k]) for k in ("target_x","target_y","target_z")] for r in rows],
                record["mission_events"],record["metrics"]["route_completion_time_s"])
            write_json(destination/"geometry.json",record["geometry"]);record["geometry_sha256"]=sha(destination/"geometry.json")
        if record["metrics"]["survived"]:assert record["steps"]==(1500 if mission=="hover" else 2000)
        records.append(record);write_json(destination/"summary.json",record)
    result=dict(schema="joint-route-memory-trained-native24-v21.0",grid="development24_full_horizon",conditioning_mode=mode,
        episodes=records,aggregate=frozen.aggregate(records),policy_sha256=sha(policy_path),source_policy_sha256=SOURCE_SHA,
        observation_contract=CONTRACT_VERSION,actor_observations=80,raw_effective_input_separately_archived=True,
        horizon_s=dict(hover=30.,route=40.),evaluated_utc=utc(),training_transitions=0,optimizer_steps=0)
    result.update(frozen.selection_score(result["aggregate"],read(COMMON_BASELINE)["aggregate"]))
    result["source77_nonregression"]=comparison.source77_nonregression(result)
    write_json(output/"evaluation.json",result)
    return result

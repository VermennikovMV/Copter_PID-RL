"""Shared fast rollout/evaluation and lexicographic validation selection."""
import csv,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .env import QuadEnv
from .protocol import SCENARIOS
from sim.metrics import compute_metrics

VALIDATION_SEEDS=tuple(range(82001,82009))
QUICK_VALIDATION=(
    ("nominal","hover",82001),("nominal","route",82002),
    ("force_strong","hover",82003),("force_strong","route",82004),
    ("gust","hover",82005),("mass_40","route",82006),
    ("delay_40","hover",82007),("sensor_noise","route",82008),
    ("motor_20","hover",82002),("gust","route",82003),
)


def run_episode(controller,architecture="hybrid",history=True,scenario="nominal",mission="hover",seed=101,
                seconds=None,save_path=None):
    env=QuadEnv(architecture,history=history,scenario=scenario,mission=mission,seconds=seconds)
    try:
        return _run_episode(env,controller,architecture,history,scenario,mission,seed,save_path)
    finally:
        env.close()


def _run_episode(env,controller,architecture,history,scenario,mission,seed,save_path):
    obs,initial=env.reset(seed=seed)
    if hasattr(controller,"reset"):controller.reset()
    rows=[];io={key:[] for key in ("time","dt","position","velocity","rotation","body_rates","target","previous_action","observation","action","yaw_reference")}
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
    record={"architecture":architecture,"scenario":scenario,"mission":mission,"eval_seed":int(seed),"environment_seed":int(seed),
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


def objective(records):
    metrics=[r["metrics"] for r in records]
    route=[m for r,m in zip(records,metrics) if r["mission"]=="route"]
    recovery=[m for m in metrics if m["recovery_applicable"]]
    # Explicit hierarchy: survival, completion, heading gate, then accuracy,
    # recovery and route time. No successful-only dropping of censored episodes.
    return [sum(not m["survived"] for m in metrics)/len(metrics),
            sum(not m["route_completed"] for m in route)/max(1,len(route)),
            sum(not m["heading_guard_pass"] for m in metrics)/len(metrics),
            float(np.mean([m["penalized_rmse_position_m"] for m in metrics])),
            float(np.mean([m["recovery_time_s"] if m["recovery_time_s"] is not None else m["horizon_s"]-(m["force_off_time_s"] or 0) for m in recovery])) if recovery else 0.,
            float(np.mean([m["route_completion_time_s"] if m["route_completed"] else m["horizon_s"] for m in route])) if route else 0.]


def validate(controller,architecture="hybrid",history=True,full=False):
    cases=[(s,m,seed) for s in SCENARIOS for m in ("hover","route") for seed in VALIDATION_SEEDS] if full else QUICK_VALIDATION
    records=[run_episode(controller,architecture,history,s,m,seed) for s,m,seed in cases]
    return {"grid":"full176" if full else "quick10","objective":objective(records),"episodes":records}

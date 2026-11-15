"""Original24 scores unchanged; five frozen matched profiles judged separately."""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import numpy as np
from quadrl_route_memory_train_v21p1.eval import run_episode,evaluate as evaluate_original
from quadrl_route_memory_train_v21.policy import NumpyPolicyRouteMemoryTrain
from quadrl_route_transfer_v22.env import TransferNativeEnv
from scripts import train_joint_route_basis_v15 as frozen
ROOT=Path(__file__).resolve().parents[1]
PROFILE_PLAN=ROOT/"docs/joint_route_transfer_v22/physical_plan.root01.ready.json"
read,write_json,sha,utc=frozen.read,frozen.write_json,frozen.sha,frozen.utc


def profile_plan():return read(PROFILE_PLAN)


def profiles():return [copy.deepcopy(c) for c in profile_plan()["cases"] if c["mode"]=="zero_route_integral"]


def fixed_yaw():return float(profile_plan()["source_measured_Gazebo_initial"]["state"]["rpy"][2])


def profile_admission(record):
    m,g=record["metrics"],record["geometry"]
    nominal=record["scenario"]=="nominal"
    def limit(value,maximum):return value is not None and np.isfinite(value) and value<=maximum
    checks=dict(survived=bool(m["survived"]),complete40s=record["steps"]==2000 and m["duration_s"]==40.,
        route_complete=bool(m["route_completed"]),heading_full=bool(m["heading_full_flight_eligible"]),
        heading_rms=limit(m["heading_rmse_deg"],5.),heading_peak=limit(m["heading_peak_deg"],15.),
        geometry_complete=bool(g["geometry_eligible"]),all3corners=g["complete_corner_windows"]==3,
        finite=limit(g["segment_rmse_m"],.11 if nominal else .15),corner=limit(g["mean_corner_post_peak_m"],.20 if nominal else .25),
        peak=limit(g["active_segment_peak_m"],.5),time=limit(m["route_completion_time_s"],9.3),goal=limit(m["penalized_rmse_position_m"],.62))
    return dict(passed=all(checks.values()),checks=checks,
        scenario=record["scenario"],intervention=record["intervention"],eval_seed=record["eval_seed"])


def evaluate_profiles(policy_path,output,phase_output=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=profile_plan();records=[];counts=dict(control_steps=0,NN_forwards=0)
    class CountedController:
        def __init__(self,policy):self.policy=policy
        def __getattr__(self,name):return getattr(self.policy,name)
        def predict(self,*args,**kwargs):counts["NN_forwards"]+=1;return self.policy.predict(*args,**kwargs)
    for index,case in enumerate(profiles()):
        if phase_output:frozen.phase(phase_output,"matched_profile_validation",completed_cases=index,total_cases=5,case=case["name"])
        local=dict(control_steps=0)
        controller=CountedController(NumpyPolicyRouteMemoryTrain(policy_path,expected_mode="zero_route_integral"))
        env=TransferNativeEnv(intervention=case["intervention"],measured_initial=plan["source_measured_Gazebo_initial"]["state"],
            physical_pulse=plan["source_physical_Gazebo_pulse"]["event"],budget=local,basis_kind="velocity_residual",
            architecture="hybrid",training=False,scenario=case["scenario"],mission="route",seconds=40.)
        destination=output/case["name"]
        try:
            record=run_episode(env,controller,"hybrid",True,case["scenario"],"route",case["seed"],destination)
            assert len(env.transfer_reset_receipts)==1
            record["initial_intervention_proof"]=copy.deepcopy(env.transfer_reset_receipts[0])
        finally:env.close()
        counts["control_steps"]+=local["control_steps"]
        with (destination/"trace.csv").open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
        record["geometry"]=frozen.geometry_metrics([float(r["time"]) for r in rows],
            [[float(r[k]) for k in ("x","y","z")] for r in rows],[[float(r[k]) for k in ("target_x","target_y","target_z")] for r in rows],
            record["mission_events"],record["metrics"]["route_completion_time_s"])
        record.update(intervention=case["intervention"],profile_key=case["name"],current_policy_sha256=sha(policy_path),
            model_family=frozen.MODEL_FAMILY,plant_branch="native_RP_jerk_shaper_400Hz_physical_feedback",joint_rate_model_version=frozen.MODEL_VERSION,
            native_original24_mean_inclusion=False)
        record["profile_admission"]=profile_admission(record)
        # run_episode owns summary.json. Additions have a separate file.
        write_json(destination/"profile_summary.json",record)
        records.append(record)
    assert counts["control_steps"]<=10000 and counts["NN_forwards"]==counts["control_steps"]
    result=dict(schema="joint-heading-five-matched-profiles-v23.0",episodes=records,profile_count=5,
        all_profiles_pass=all(r["profile_admission"]["passed"] for r in records),actual_counts=counts,
        policy_sha256=sha(policy_path),profile_plan_sha256=sha(PROFILE_PLAN),evaluated_utc=utc(),
        not_averaged_into_original24=True,extra_training_or_optimizer_steps=0)
    write_json(output/"profiles.json",result)
    return result

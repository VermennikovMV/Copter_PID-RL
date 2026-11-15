"""Causal memory conditioning of exact source68 weights; no learning or oracle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from quadrl.policy import NumpyPolicy,heading_rotation
from quadrl_route_basis_v14.policy import (CONTRACT_VERSION as RAW_CONTRACT,
    EncoderConfigRoute80,ObservationEncoderRoute80,NumpyPolicyRoute80)

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATH=Path("artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz")
SOURCE_SHA="68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5"
CONTRACT_VERSION="measured-memory-route-integral-v5.0"
MODES=("raw","zero_route_integral","reset_on_target")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditioning_config(mode):
    if mode not in MODES:raise ValueError("Declared conditioning mode required")
    return dict(mode=mode,integral_input_columns=[22,23,24],raw_old_contract=RAW_CONTRACT,
        raw_encoder_calls_per_observation=1,gating="own observed-target history valid_leg",
        reset_timing="observed target change before current dt integration",
        world_integral_rule="original first-sample skip; error*dt; XYnorm and separate Z clamp",
        other77_inputs_bit_exact=True,hidden_or_future_inputs=False)


def source_values():
    path=ROOT/SOURCE_PATH
    if sha(path)!=SOURCE_SHA:raise ValueError("Exact archived source68 policy required")
    NumpyPolicyRoute80(path,expected_kind="velocity_residual")
    with np.load(path,allow_pickle=False) as archive:return {name:archive[name].copy() for name in archive.files}


def raw_config(values):
    base=json.loads(str(values["encoder_config"]));base["history_times_s"]=tuple(base["history_times_s"])
    return EncoderConfigRoute80(**base,**json.loads(str(values["path_context_config"])),**json.loads(str(values["route_basis_config"])))


class ObservationEncoderRouteMemory:
    """Run the original measured80 encoder once, then replace only integral3."""
    def __init__(self,config,mode):
        conditioning_config(mode)
        self.config,self.mode=config,mode
        self.raw_encoder=ObservationEncoderRoute80(config)
        self.reset()

    @property
    def path_context(self):return self.raw_encoder.path_context

    def reset(self,yaw_reference=None):
        self.raw_encoder.reset(yaw_reference)
        self.integral_world=np.zeros(3)
        self.previous_target=None;self.has_previous_sample=False
        self.raw_observation=None;self.effective_observation=None;self.observation_calls=0
        self.target_changed=False

    def observe(self,position,velocity,rotation,body_rates,target,previous_action=None,dt=.02):
        # Validation and the original history update occur only in this call.
        raw=self.raw_encoder.observe(position,velocity,rotation,body_rates,target,previous_action,dt)
        pos,goal=np.asarray(position),np.asarray(target)
        self.target_changed=self.previous_target is not None and not np.array_equal(goal,self.previous_target)
        if self.mode=="reset_on_target" and self.target_changed:self.integral_world.fill(0.)
        if self.has_previous_sample:
            self.integral_world+=(goal-pos)*dt
            length=np.linalg.norm(self.integral_world[:2])
            if length>self.config.integral_limit:self.integral_world[:2]*=self.config.integral_limit/length
            self.integral_world[2]=np.clip(self.integral_world[2],-self.config.integral_limit,self.config.integral_limit)
        self.previous_target=goal.copy();self.has_previous_sample=True
        effective=raw.copy()
        if self.mode=="zero_route_integral" and self.path_context.valid_leg:effective[22:25]=0.
        elif self.mode=="reset_on_target" and self.path_context.valid_leg:
            unused,heading=heading_rotation(rotation)
            effective[22:25]=(heading@self.integral_world/self.config.integral_limit).astype(np.float32)
        untouched=np.r_[np.arange(22),np.arange(25,80)]
        assert np.array_equal(raw[untouched],effective[untouched])
        self.raw_observation,self.effective_observation=raw,effective
        self.observation_calls+=1
        assert self.raw_encoder.base77_observe_calls==self.observation_calls
        return effective

    def diagnostics(self):
        return dict(raw_observation=self.raw_observation.copy(),effective_observation=self.effective_observation.copy(),
            raw_integral_world=self.raw_encoder.base77.base_encoder.integral_world.copy(),
            conditioned_integral_world=self.integral_world.copy(),valid_leg=bool(self.path_context.valid_leg),
            target_changes=int(self.path_context.target_changes),target_changed=bool(self.target_changed),
            raw_encoder_calls=self.observation_calls)


class NumpyPolicyRouteMemory(NumpyPolicy):
    def __init__(self,path,expected_mode=None):
        with np.load(path,allow_pickle=False) as archive:values={name:archive[name].copy() for name in archive.files}
        if str(values.get("contract"))!=CONTRACT_VERSION:raise ValueError("Explicit v20 memory contract required")
        config=json.loads(str(values["conditioning_config"]))
        self.mode=config["mode"]
        if expected_mode is not None and self.mode!=expected_mode:raise ValueError("Conditioning mode mismatch")
        if config!=conditioning_config(self.mode):raise ValueError("Unexpected conditioning parameters")
        original=source_values()
        additions={"conditioning_config","raw_old_contract","source_policy_path","source_policy_sha256"}
        if set(values)!=set(original)|additions:raise ValueError("Unexpected NPZ schema")
        if str(values["raw_old_contract"])!=RAW_CONTRACT:raise ValueError("Raw contract mismatch")
        if str(values["source_policy_path"])!=SOURCE_PATH.as_posix() or str(values["source_policy_sha256"])!=SOURCE_SHA:
            raise ValueError("Source68 provenance required")
        for name,value in original.items():
            if name!="contract" and not np.array_equal(values[name],value):raise ValueError("Changed original actor or encoder metadata: "+name)
        self.weights=[values[f"w{i}"].copy() for i in range(int(values["layers"]))]
        self.biases=[values[f"b{i}"].copy() for i in range(int(values["layers"]))]
        self.architecture,self.contract=str(values["architecture"]),CONTRACT_VERSION
        self.encoder=ObservationEncoderRouteMemory(raw_config(values),self.mode)
        self.reset()


def export_conditioned_policy(destination,mode):
    destination=Path(destination)
    if destination.exists():raise FileExistsError("Fresh conditioned policy path required")
    values=source_values();values["raw_old_contract"]=values["contract"].copy()
    values["contract"]=np.array(CONTRACT_VERSION)
    values["conditioning_config"]=np.array(json.dumps(conditioning_config(mode),sort_keys=True))
    values["source_policy_path"]=np.array(SOURCE_PATH.as_posix());values["source_policy_sha256"]=np.array(SOURCE_SHA)
    np.savez(destination,**values)
    NumpyPolicyRouteMemory(destination,expected_mode=mode)
    return destination

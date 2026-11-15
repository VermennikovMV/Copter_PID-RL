"""One shared encoder, current observed-leg selector, one stateless mean call."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from quadrl.policy import decode_hybrid
from quadrl_route_memory_v20.policy import ObservationEncoderRouteMemory,conditioning_config,raw_config

ROOT=Path(__file__).resolve().parents[1]
CONTRACT_VERSION="measured-shared-phase-expert-v29.0"
BUNDLE_SCHEMA="joint-causal-two-expert-bundle-v29.0"
F0_PATH="artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/zero_route_integral/policy.npz"
F0_SHA="f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461"
ROUTE_PATH="artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.npz"
ROUTE_SHA="ce86bb56d7e0f5c1775266d4a45cd3417e979303d27cc662a58e8070bf34e345"
ZIP_PATH="artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.zip"
ZIP_SHA="62eaf3fb179f915a91cb15686720c06c5dd8cc80595d1308279ee1895ce13c11"
PROOF_PATH="audit/review_joint_prior_ablation_v28_pilot_01.json"
PROOF_SHA="8b9936e55e286587ae8ee66d9c487bbc08471a1d0abef9fcad53b8d858e1e2a4"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bundle_payload():
    return dict(schema=BUNDLE_SCHEMA,observation_contract=CONTRACT_VERSION,architecture="hybrid",
        conditioning=conditioning_config("zero_route_integral"),shared_encoder=True,
        selector=dict(predicate="CURRENT shared encoder valid_leg AFTER observe",false_expert=0,true_expert=1,
            latched=False,reset_on_switch=False,uses_mission_kind_or_future_waypoint=False),
        experts=[dict(index=0,role="F0_without_observed_leg",policy=dict(path=F0_PATH,sha256=F0_SHA)),
            dict(index=1,role="prior0_FINAL_with_observed_leg",policy=dict(path=ROUTE_PATH,sha256=ROUTE_SHA),
                model_zip=dict(path=ZIP_PATH,sha256=ZIP_SHA),training_proof=dict(path=PROOF_PATH,sha256=PROOF_SHA,
                    pointer="/candidate_claims/prior0/final"))],
        new_optimizer_steps=0,new_training_steps=0,
        development_choice="Manual causal composition after known v28 quality; no new PPO training or inherited cell acceptance")


def validate_sources():
    for name,digest in ((F0_PATH,F0_SHA),(ROUTE_PATH,ROUTE_SHA),(ZIP_PATH,ZIP_SHA),(PROOF_PATH,PROOF_SHA)):
        if sha(ROOT/name)!=digest:raise ValueError("Frozen expert/proof SHA differs: "+name)
    claim=read(ROOT/PROOF_PATH)["candidate_claims"]["prior0"]["final"]
    if not (claim["policy_npz_sha256"]==ROUTE_SHA and claim["model_zip_sha256"]==ZIP_SHA and
            claim["actual_checkpoint_PPO_updates"]==2048 and claim["all6_actor_arrays_changed"] and
            claim["log_std_exact_source68"]):raise ValueError("Exact prior0 FINAL training proof required")
    arrays=[]
    for name in (F0_PATH,ROUTE_PATH):
        with np.load(ROOT/name,allow_pickle=False) as archive:arrays.append({k:archive[k].copy() for k in archive.files})
    for values in arrays:
        if str(values["architecture"])!="hybrid" or int(values["layers"])!=3:raise ValueError("Frozen hybrid topology required")
        if json.loads(str(values["conditioning_config"]))!=conditioning_config("zero_route_integral"):
            raise ValueError("Shared zero-route-integral preprocessing required")
        if [values[f"w{i}"].shape for i in range(3)]!=[(128,80),(128,128),(4,128)]:raise ValueError("Actor80 shape differs")
        if [values[f"b{i}"].shape for i in range(3)]!=[(128,),(128,),(4,)]:raise ValueError("Actor biases differ")
        if not all(values[f"{p}{i}"].dtype==np.float32 and np.isfinite(values[f"{p}{i}"]).all()
                   for p in ("w","b") for i in range(3)):raise ValueError("Finite float32 actor arrays required")
    for key in ("encoder_config","architecture","layers"):
        if not np.array_equal(arrays[0][key],arrays[1][key]):raise ValueError("Expert encoder/topology metadata differs")
    if raw_config(arrays[0])!=raw_config(arrays[1]):raise ValueError("Expert preprocessing differs")
    return arrays


class StatelessMean80:
    """The original three-GEMM clipped mean, with no encoder/action memory."""
    def __init__(self,values):
        self.weights=[values[f"w{i}"].copy() for i in range(3)]
        self.biases=[values[f"b{i}"].copy() for i in range(3)]
        for array in self.weights+self.biases:array.setflags(write=False)
    def predict(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(80,) or not np.isfinite(x).all():raise ValueError("One finite effective80 sample required")
        for index,(weights,bias) in enumerate(zip(self.weights,self.biases)):
            x=x@weights.T+bias
            if index<len(self.weights)-1:x=np.tanh(x)
        return np.clip(x,-1,1)


class NumpyPhaseExpertPolicy:
    def __init__(self,path):
        path=Path(path)
        if read(path)!=bundle_payload():raise ValueError("Strict v29 bundle and fixed two experts required")
        arrays=validate_sources()
        self.experts=[StatelessMean80(value) for value in arrays]
        self.encoder=ObservationEncoderRouteMemory(raw_config(arrays[0]),"zero_route_integral")
        self.architecture="hybrid";self.mode="zero_route_integral";self.contract=CONTRACT_VERSION
        self.bundle_sha256=sha(path);self.reset()
    def reset(self,yaw_reference=None):
        self.encoder.reset(yaw_reference);self.previous_action=np.zeros(4,np.float32)
        self.last_prediction=None;self.expert_forwards=[0,0];self.switches=0
    @property
    def expert_index(self):return int(bool(self.encoder.path_context.valid_leg))
    def predict(self,observation):
        x=np.asarray(observation,np.float32);calls=self.encoder.observation_calls
        expected=1 if self.last_prediction is None else self.last_prediction["encoder_calls"]+1
        if calls!=expected or not np.array_equal(x,self.encoder.effective_observation):
            raise ValueError("Predict requires the single current shared-encoder observation")
        if self.last_prediction is not None and calls!=self.last_prediction["encoder_calls"]+1:
            raise ValueError("Exactly one selected expert forward per observed sample")
        expert=self.expert_index
        previous=self.previous_action.copy()
        action=self.experts[expert].predict(x)
        self.expert_forwards[expert]+=1
        if self.last_prediction is not None:self.switches+=int(expert!=self.last_prediction["expert_index"])
        self.last_prediction=dict(expert_index=expert,valid_leg=bool(self.encoder.path_context.valid_leg),
            target_changed=bool(self.encoder.target_changed),target_changes=int(self.encoder.path_context.target_changes),
            encoder_calls=calls,previous_action=previous,actual_previous_action=previous.copy(),emitted_action=action.copy(),
            selected_source_sha256=F0_SHA if expert==0 else ROUTE_SHA)
        self.previous_action=action.copy()
        return action
    def diagnostics(self):
        if self.last_prediction is None or self.last_prediction["encoder_calls"]!=self.encoder.observation_calls:
            raise ValueError("Diagnostics require the current emitted action")
        return {**self.encoder.diagnostics(),**self.last_prediction,"expert_forwards":self.expert_forwards.copy(),
            "expert_switches":self.switches,"shared_encoder":True,"new_optimizer_steps":0}
    def predict_enu(self,position,velocity,rotation,body_rates,target,dt=.02):
        observation=self.encoder.observe(position,velocity,rotation,body_rates,target,self.previous_action,dt)
        action=self.predict(observation);thrust,rates=decode_hybrid(action)
        return dict(architecture="hybrid",action=action,thrust_fraction=thrust,rates_flu=rates,
            rates_frd=rates*np.array([1.,-1.,-1.]),observation=observation)

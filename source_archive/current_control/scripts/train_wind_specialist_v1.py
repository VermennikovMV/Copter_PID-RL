"""Explicitly limited wind-hover specialist; original benchmark stays frozen.

The previous mixed-perturbation/route training is the fixed reference. Only the
actor-training task distribution is narrowed here. Critic warmup, reward, PPO
settings, KL0.1 and seeds are unchanged. Selection is prospectively limited to
nominal/strong/gust hover; full176 side effects remain recorded and reported.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from quadrl.env import QuadEnv
from quadrl.protocol import Mission
from quadrl.policy import EncoderConfig,NumpyPolicy,export_policy
from quadrl.rollout import validate
from scripts.bc_trust_region_probe_v1 import BCTrustPPO,actor_name,sha
from scripts.refine_precision import Selection

OUT=ROOT/"artifacts/refinement/wind_specialist_v1"
INITIAL=ROOT/"artifacts/training/main_hybrid_seed111_v1/bc_only_model.zip"


class WindTaskEnv(QuadEnv):
    def __init__(self,**kwargs):
        self.wind_specialist=False
        super().__init__(**kwargs)

    def _training_parameters(self,rng):
        if not self.wind_specialist:
            return super()._training_parameters(rng)
        # Parent reset may have randomly selected route. Specialist episodes
        # explicitly replace it BEFORE target/initial state are constructed.
        self.mission=Mission("hover")
        self.max_steps=round((self.seconds if self.seconds is not None else 30.)/self.control_dt)
        spec={"name":"wind_hover_nominal_vehicle","force_events":[]}
        if rng.random()<.2:
            return spec
        magnitude=float(rng.uniform(.3,3.5))
        angle=float(rng.uniform(-np.pi,np.pi))
        if rng.random()<.5:
            spec["force_events"]=[dict(on_s=float(rng.uniform(0,4)),off_s=float(self.max_steps*self.control_dt+1),
                                       force_enu_N=[magnitude*math.cos(angle),magnitude*math.sin(angle),0.])]
        else:
            for on in (3.,10.,17.):
                direction=angle+float(rng.uniform(-np.pi,np.pi))
                spec["force_events"].append(dict(on_s=on,off_s=on+float(rng.uniform(2,5)),
                    force_enu_N=[magnitude*math.cos(direction),magnitude*math.sin(direction),0.]))
        return spec


def target_objective(validation):
    target=[r for r in validation["episodes"] if r["mission"]=="hover" and
            r["scenario"] in ("nominal","force_strong","gust")]
    wind=[r for r in target if r["scenario"]!="nominal"]
    nominal=[r for r in target if r["scenario"]=="nominal"]
    if len(target)!=24 or len(wind)!=16 or len(nominal)!=8:
        raise AssertionError("Expected complete development target-domain grid")
    return [sum(not r["metrics"]["survived"] for r in target),
            sum(not r["metrics"]["heading_guard_pass"] for r in target),
            float(np.mean([r["metrics"]["penalized_rmse_position_m"] for r in wind])),
            float(np.mean([r["metrics"]["penalized_rmse_position_m"] for r in nominal]))]


class WindSelection(Selection):
    def evaluate(self,label):
        policy=self.output/f"{label}.npz"
        self.model.save(self.output/label)
        export_policy(self.model,policy,"hybrid",EncoderConfig())
        validation=validate(NumpyPolicy(policy),"hybrid",True,full=True)
        score=target_objective(validation)
        validation.update(selection_objective=score,timesteps=self.model.num_timesteps,
                          checkpoint=label,completed_ppo_epochs=self.model._n_updates-self.starting_updates,
                          checkpoint_timing="After final update" if label=="final" else "Before current rollout update")
        (self.output/f"{label}_validation.json").write_text(json.dumps(validation,indent=2))
        if self.best is None or tuple(score)<tuple(self.best):
            self.best=score
            self.model.save(self.output/"best_model")
            export_policy(self.model,self.output/"best_policy.npz","hybrid",EncoderConfig())
            (self.output/"selection.json").write_text(json.dumps(
                {k:v for k,v in validation.items() if k!="episodes"},indent=2))
        print(json.dumps(dict(event="TARGET_VALIDATION",checkpoint=label,target_objective=score,
                              full176_objective=validation["objective"])),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    # Reference selection receives the SAME target objective and candidate count.
    # All three snapshots already exist; no new reference training or held-out.
    reference=ROOT/"artifacts/refinement/bc_trust_v1/kl01"
    reference_scores={label:target_objective(json.loads((reference/f"{label}_validation.json").read_text()))
                      for label in ("checkpoint_524288","checkpoint_1048576","final")}
    best_reference=min(reference_scores,key=lambda label:tuple(reference_scores[label]))
    bc_path=ROOT/"artifacts/refinement/critic_warmup_v1/bc_full_validation.json"
    bc=json.loads(bc_path.read_text())
    bc_score=target_objective(bc)
    wind_limit=.95*min(bc_score[2],reference_scores[best_reference][2])
    files=sorted(set([Path(__file__),ROOT/"scripts/bc_trust_region_probe_v1.py",
        ROOT/"scripts/refine_precision.py",ROOT/"scripts/train.py",*ROOT.glob("quadrl/*.py"),
        *ROOT.glob("quadrl/*.json"),*ROOT.glob("reference/*.py"),ROOT/"sim/metrics.py"]))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    reference_files=[bc_path]
    for label in reference_scores:
        reference_files.extend(reference/f"{label}{suffix}" for suffix in ("_validation.json",".npz",".zip"))
    reference_hashes={str(path.relative_to(ROOT)):sha(path) for path in reference_files}
    plan=dict(stage="Separate development specialist; not a general replacement or held-out result",
              intended_domain="Hover at nominal vehicle mass/motors; horizontal force/gusts0.3..3.5N",
              initial=str(INITIAL),initial_sha256=sha(INITIAL),
              changed_factor="Actor-training task distribution only; target-domain selection applied equally to reference snapshots",
              warmup_steps=131072,ppo_steps=1048576,warmup_seed=9171,ppo_seed=9111,
              fixed_kl_weight=.1,reward_and_PPO_hyperparameters_unchanged=True,
              training_mixture="20% nominal /80% force or gust hover; mass/motors/delay/noise nominal",
              full176_results_all_reported=True,
              target_selection="Fail count, heading count, wind hover RMSE, nominal hover RMSE",
              reference_target_objectives=reference_scores,best_reference_target_checkpoint=best_reference,
              reference_evidence_sha256=reference_hashes,
              bc_target_objective=bc_score,
              gate=dict(target_failures=0,target_heading_violations=0,
                        wind_hover_rmse_limit_m=wind_limit,
                        wind_hover_requirement="At least5% better than BOTH BC and target-selected full-DR KL reference",
                        nominal_hover_limit_m=max(1.2*bc_score[3],bc_score[3]+.01)),
              validation_seeds=list(range(82001,82009)),
              confirmation_seeds_not_opened=[92011,92012,92013],
              interpretation="Report route/mass/motor/delay regressions. Specialist success cannot be called broad success.",
              source_hashes=hashes,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (OUT/"plan.json").write_text(json.dumps(plan,indent=2))
    (OUT/"config.json").write_text(json.dumps(plan,indent=2))
    with zipfile.ZipFile(OUT/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,str(path.relative_to(ROOT)))
        for path in reference_files:archive.write(path,"reference_evidence/"+str(path.relative_to(ROOT)))
    torch.set_num_threads(1)
    vector=VecMonitor(DummyVecEnv([lambda:WindTaskEnv(architecture="hybrid",training=True) for _ in range(16)]))
    try:
        vector.set_attr("training_progress",1.)
        vector.seed(9171)
        model=BCTrustPPO.load(INITIAL,env=vector,device="cpu",seed=9171)
        model.anchor_steps=0
        model.configure_bc_prior(INITIAL,0.,OUT/"on_policy_prior_kl.jsonl")
        model.bc_kl_enabled=False
        initial=copy.deepcopy(model.policy.state_dict())
        prior=copy.deepcopy(model.bc_prior.state_dict())
        model.set_logger(configure(str(OUT/"warmup"),["csv"]))
        started=time.monotonic()
        for name,parameter in model.policy.named_parameters():
            if actor_name(name):parameter.requires_grad_(False)
        model.learn(131072,reset_num_timesteps=True,progress_bar=False)
        if any(not torch.equal(value,initial[name]) for name,value in model.policy.state_dict().items()
               if actor_name(name)):
            raise AssertionError("Critic warmup changed actor")
        np.savez(OUT/"ppo_start_parameters.npz",**{
            k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()})
        model.save(OUT/"critic_warmed_model")
        vector.set_attr("wind_specialist",True)
        for parameter in model.policy.parameters():parameter.requires_grad_(True)
        model.bc_kl_weight=.1
        model.bc_kl_enabled=True
        model.set_random_seed(9111)
        model.set_logger(configure(str(OUT),["csv"]))
        before=model._n_updates
        selection=WindSelection(OUT,524288,before)
        model.learn(1048576,reset_num_timesteps=True,callback=selection,progress_bar=False)
        selection.evaluate("final")
        probe=np.random.default_rng(17491).normal(size=(200,74)).astype(np.float32)
        parity=float(np.max(np.abs(model.predict(probe,deterministic=True)[0]
                                   -NumpyPolicy(OUT/"final.npz").predict(probe))))
        if parity>1e-5:raise AssertionError("NumPy parity")
        if not all(torch.equal(v,prior[k]) for k,v in model.bc_prior.state_dict().items()):
            raise AssertionError("Prior changed")
        if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):
            raise AssertionError("Source changed")
        if any(sha(ROOT/name)!=digest for name,digest in reference_hashes.items()):
            raise AssertionError("Reference evidence changed")
        epochs=model._n_updates-before
        complete=dict(status="complete",ppo_timesteps=model.num_timesteps,
                      completed_actor_critic_epochs=epochs,ppo_kl_gradient_steps=model.bc_kl_gradient_steps,
                      expected_gradient_steps=epochs*math.ceil(model.n_steps*model.n_envs/model.batch_size),
                      source_hashes_unchanged=True,prior_bit_identical=True,numpy_export_max_error=parity,
                      selected_target_objective=selection.best,selected_policy_sha256=sha(OUT/"best_policy.npz"),
                      wall_seconds=time.monotonic()-started)
        (OUT/"complete.json").write_text(json.dumps(complete,indent=2))
        print(json.dumps(complete),flush=True)
    finally:vector.close()


if __name__=="__main__":main()

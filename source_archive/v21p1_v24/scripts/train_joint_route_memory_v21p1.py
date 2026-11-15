"""True PPO gains measured against each mode's own zero-update v20 baseline."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from scripts import train_joint_critic_calibration_v18 as v18
from scripts import train_joint_route_memory_v21 as original_v21
from scripts import prototype_joint_route_memory_v20 as v20
from scripts import run_joint_route_basis_pair_v15 as relative
from scripts.bc_trust_region_probe_v1 import BCTrustPPO,actor_name,sha
from quadrl_route_memory_train_v21.policy import (MODES,INITIAL_ZIP,INITIAL_ZIP_SHA,SOURCE_SHA,
    CONTRACT_VERSION,export_policy,NumpyPolicyRouteMemoryTrain,source_values)
from quadrl_route_memory_train_v21.env import JointTaskEnv
from quadrl_route_memory_train_v21p1.eval import evaluate,COMMON_BASELINE
v16=v18.v16
ARMS=MODES
INITIAL,INITIAL_SHA=ROOT/INITIAL_ZIP,INITIAL_ZIP_SHA
BUDGET,PROBE_BUDGET=v18.BUDGET.copy(),v18.PROBE_BUDGET.copy()
BASIS_KIND="velocity_residual"
SCHEMA="joint-conditioned-memory-ppo-v21.0"
BASELINES={mode:ROOT/"artifacts/refinement/joint_v2/route_memory_baseline_v20_root01"/mode/"evaluation.json" for mode in MODES}
read,write_json,utc,phase=v18.read,v18.write_json,v18.utc,v18.phase
is_mean_actor=v18.is_mean_actor


def source_hashes():
    previous=original_v21.source_hashes();original_v21.assert_preparation_frozen(previous)
    paths=[Path(__file__),ROOT/"scripts/run_joint_route_memory_pair_v21p1.py",
        ROOT/"quadrl_route_memory_train_v21p1/__init__.py",ROOT/"quadrl_route_memory_train_v21p1/eval.py"]
    return {**previous,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def assert_preparation_frozen(sources):
    record=read(ROOT/"docs/joint_route_memory_train_v21p1/preparation_freeze.json")
    assert record["source_sha256"]==sources,"Changed source after prospective conditioning training freeze"
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name


def own_baseline(mode):
    record=read(BASELINES[mode])
    assert len(record["episodes"])==24 and record["conditioning_mode"]==mode
    assert record["source_policy_sha256"]==SOURCE_SHA
    return record


def new_model(env,seed,output,mode,probe=False):
    options=dict(gamma=1.)
    if probe:options.update(n_steps=128,batch_size=64,n_epochs=2)
    model=BCTrustPPO.load(INITIAL,env=env,device="cpu",seed=seed,**options)
    v16.assert_hyperparameters(model,probe)
    before=len(model.policy.optimizer.state)
    original_epochs=model._n_updates
    model.policy.optimizer.state.clear();model._n_updates=0
    assert not model.policy.optimizer.state
    model.anchor_steps=0
    model.configure_bc_prior(INITIAL,0.,output/"on_policy_prior_kl.jsonl")
    model.bc_kl_enabled=False
    model.conditioning_mode=mode;model.conditioning_contract=CONTRACT_VERSION
    model.v21_loaded_optimizer_states=before
    write_json(output/"optimizer_reset.json",dict(initializer_path=INITIAL.relative_to(ROOT).as_posix(),
        initializer_sha256=INITIAL_SHA,initializer_states=before,after_clear=0,reset=[before,0],
        source_epoch_counter=original_epochs,continuation_epoch_counter_reset_to=0,
        actor_weights_unchanged=True,source_actor_npz_sha256=SOURCE_SHA,conditioning_mode=mode,
        prior_same_initial_source68=True,prior_receives_same_effective_input_within_mode=True))
    return model


class ContinuationGuard(v18.CalibrationGuard):
    def __init__(self,model,mode,output):
        self.conditioning_mode=mode
        super().__init__(model,"critic_lr3e4",output)

    def before_step(self,optimizer,unused_args,unused_kwargs):
        super().before_step(optimizer,unused_args,unused_kwargs)
        self.pending["conditioning_mode"]=self.conditioning_mode

    def report(self):
        report=super().report()
        report.update(conditioning_mode=self.conditioning_mode,
            initializer_Adam_reset=[self.model.v21_loaded_optimizer_states,0],
            changed_factor="Input conditioning raw versus exactv20zero_route_integral; SAME source68 initialization/PPO/reward",
            critic_coupling="Conditioning changes actions/physical trajectories/reward/critic/GAE; no cross-mode warm trajectory parity claimed",
            prior_same_initial_actor_within_mode=True,warm_physical_or_critic_equality_required=False)
        return report


def vector(count,seed,mode):
    env=VecMonitor(DummyVecEnv([lambda:JointTaskEnv(conditioning_mode=mode,architecture="hybrid",training=True) for unused in range(count)]))
    env.set_attr("training_progress",1.);env.seed(seed)
    return env


class FirstPPORollout(BaseCallback):
    def __init__(self,output,guard):
        super().__init__();self.output,self.guard=Path(output),guard;self.saved=False;self.route=[];self.samples=[]

    def _on_step(self):
        if not self.saved:
            self.route.append(np.asarray([i["metric_cost_ledger"]["mission_kind"]=="route" for i in self.locals["infos"]]))
            self.samples.append([{name:np.asarray(value).copy() for name,value in env.last_step_actor_sample.items()}
                for env in self.training_env.venv.envs])
        return True

    def _on_rollout_end(self):
        if self.saved:return
        b=self.model.rollout_buffer
        assert self.guard.counts["ppo"]==0 and b.full and not b.generator_ready
        arrays=dict(observations=b.observations,actions=b.actions,old_values=b.values,log_probs=b.log_probs,
            advantages=b.advantages,returns=b.returns,episode_starts=b.episode_starts,rewards=b.rewards,
            last_values=self.locals["values"].detach().cpu().numpy().reshape(-1).copy(),last_dones=np.asarray(self.locals["dones"]).copy(),
            mission_is_route=np.stack(self.route),route_context_nonzero=np.any(b.observations[...,74:80]!=0.,axis=-1))
        keys=("raw_observation","effective_observation","position","velocity","rotation","body_rates","target","previous_action","dt",
            "raw_integral_world","conditioned_integral_world","memory_valid_leg","memory_target_changes","memory_target_changed","yaw_reference")
        for name in keys:arrays["sample_"+name]=np.stack([np.stack([row[name] for row in batch]) for batch in self.samples])
        assert np.array_equal(arrays["sample_effective_observation"],b.observations)
        assert np.all(b.rewards[arrays["mission_is_route"]]<=1e-6)
        np.savez_compressed(self.output/"actual_first_ppo_rollout.npz",**arrays)
        write_json(self.output/"actual_first_ppo_rollout.json",dict(status="ACTUAL_FIRST_PPO_ROLLOUT_ARCHIVED_BEFORE_UPDATE",
            conditioning_mode=self.guard.conditioning_mode,environment_steps=self.model.num_timesteps,
            actual_ppo_optimizer_steps_before_update=0,policy_before_rollout=dict(path=(self.output/"critic_warmed_model.zip").relative_to(ROOT).as_posix(),sha256=sha(self.output/"critic_warmed_model.zip")),
            binding_scope="Own-mode warmed ZIP weights identical to actor before own first rollout; warm Adam carried within mode; cross-mode critic weights/Adam/physics may differ",
            contract=CONTRACT_VERSION,actual_NN_inputs="observations == sample_effective_observation",raw_original80_separate=True,
            calibration=v18.calibration_stats(b.values,b.advantages,b.rewards,arrays["mission_is_route"],arrays["route_context_nonzero"]),
            after_original_compute_returns_and_advantage=True,before_any_PPO_update=True,raw_distribution_actions=True,
            shapes={n:list(a.shape) for n,a in arrays.items()},payload_sha256=sha(self.output/"actual_first_ppo_rollout.npz"),
            extra_environment_steps=0,extra_forward_calls=0,extra_backward_calls=0,extra_optimizer_steps=0))
        self.saved=True;self.route.clear();self.samples.clear()


def admission(record,mode):
    baseline=own_baseline(mode)
    old=relative.compare_records(baseline,record)
    nominal=old["cells"]["nominal"]["ratios"]["segment_rmse_m"]
    hover={s:record["aggregate"]["cells"][s+"/hover"]["goal_rmse_m"]/baseline["aggregate"]["cells"][s+"/hover"]["goal_rmse_m"] for s in relative.SCENARIOS}
    source77=relative.source77_nonregression(record)
    checks=dict(all_original_fixed_native_hard_pass=bool(record["fixed_development_guards_pass"]),
        nominal_finite_le_0_97_own_zero_update=nominal is not None and nominal<=.97,
        tail84001_finite_nonincrease=old["tail84001"]["finite_ratio"] is not None and old["tail84001"]["finite_ratio"]<=1.,
        tail84001_corner_nonincrease=old["tail84001"]["corner_ratio"] is not None and old["tail84001"]["corner_ratio"]<=1.,
        each_hover_le_1_10_own_zero_update=all(np.isfinite(r) and r<=1.10 for r in hover.values()),source77_nonregression=bool(source77["passed"]))
    return dict(passed=all(checks.values()),checks=checks,nominal_finite_ratio=nominal,hover_ratios=hover,
        tail84001=old["tail84001"],source77=source77,baseline_path=BASELINES[mode].relative_to(ROOT).as_posix(),
        baseline_sha256=sha(BASELINES[mode]),interpretation="True PPO gain versus OWN fixed zero-update conditioning baseline; not raw/conditioned manual gain")


class MemorySelection(v16.v15.frozen.Heartbeat):
    def __init__(self,output,mode,starting_updates):
        super().__init__(output,"ppo")
        self.output,self.mode,self.starting_updates=Path(output),mode,starting_updates
        self.next_eval=131072;self.best=None;self.records=[]

    def evaluate(self,label):
        self.model.save(self.output/label)
        path=self.output/(label+".npz");export_policy(self.model,path,self.mode)
        record=evaluate(self.mode,path,self.output/"validation"/label,self.output)
        rule=admission(record,self.mode)
        record.update(checkpoint=label,timesteps=self.model.num_timesteps,completed_ppo_epochs=self.model._n_updates-self.starting_updates,
            ppo_kl_gradient_steps=self.model.bc_kl_gradient_steps,initial_sha256=INITIAL_SHA,own_baseline_admission=rule,
            accepted=rule["passed"],diagnostic_candidate=not rule["passed"],
            checkpoint_timing="After final update" if label=="final" else "Before current rollout update",
            selection_rule="Prospective own-baseline admission first, unchanged native objective second; stable phase order")
        write_json(self.output/(label+"_validation.json"),record)
        self.records.append(dict(checkpoint=label,objective=record["objective"],policy_sha256=sha(path),
            guards_pass=record["fixed_development_guards_pass"],own_baseline_admission_pass=rule["passed"]))
        key=(not rule["passed"],*record["objective"])
        if self.best is None or key<self.best:
            self.best=key;self.model.save(self.output/"best_model");shutil.copyfile(path,self.output/"best_policy.npz")
            write_json(self.output/"selection.json",{name:value for name,value in record.items() if name!="episodes"})
        print(json.dumps(dict(event="CONDITIONED_MEMORY_VALIDATION",conditioning_mode=self.mode,checkpoint=label,
            fixed_development_guards_pass=record["fixed_development_guards_pass"],own_baseline_admission=rule["passed"],objective=record["objective"])),flush=True)

    def _on_step(self):
        super()._on_step()
        if self.num_timesteps>=self.next_eval:
            self.next_eval+=131072;self.evaluate("checkpoint_"+str(self.num_timesteps))
        return True


def require_physical_plan(args,sources):
    from scripts import joint_research_preflight_v1 as gate
    plan=read(args.physical_plan)
    assert plan["schema"]=="joint-route-memory-training-physical-preparation-plan-v21.0"
    assert plan["status"]=="READY_FOR_ROOT_AUTHORIZED_PHYSICAL_PREPARATION"
    assert plan["mode"]=="semantic_pair512" and plan["budget"]==PROBE_BUDGET
    assert plan["source_sha256"]==sources and plan["output"]==args.output.relative_to(ROOT).as_posix()
    binding=plan["root_physical_authorization"];path=gate.bound_file(ROOT,binding)
    assert binding["equals"] is True and gate.json_pointer(read(path),binding["pointer"]) is True
    return plan


def train_one(args,sources,arm,probe=False):
    output=args.output/arm if probe else args.output
    output.mkdir(parents=probe,exist_ok=False)
    started=time.monotonic()
    warm_steps,ppo_steps=(128,128) if probe else (131072,262144)
    write_json(output/"config.json",dict(schema=SCHEMA,variant=arm,conditioning_mode=arm,training_reward_arm="metric_cost",basis_kind=BASIS_KIND,
        initial_sha256=INITIAL_SHA,prior_sha256=INITIAL_SHA,budget=PROBE_BUDGET if probe else BUDGET,
        actor_trainable_mean_elements=27396,log_std_frozen=True,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
        gamma=1.,true_finite_training_deadlines=True,reward_and_plant="EXACT v16 metric_cost unchanged",
        critic="Original77 input columns/all other critic tensors; new3 hardzero before sharedclip",
        reused_baseline=dict(path=BASELINES[arm].relative_to(ROOT).as_posix(),sha256=sha(BASELINES[arm]),new_episodes=0),
        source_sha256=sources,started_utc=utc(),actual_Gazebo_acceptance=False))
    with zipfile.ZipFile(output/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in sources: archive.write(ROOT/name,name)
    write_json(output/"manifest.json",dict(source_sha256=sources,initial_sha256=INITIAL_SHA,source_snapshot_sha256=sha(output/"source_snapshot.zip")))
    env,guard=vector(1 if probe else 16,9511,arm),None
    try:
        model=new_model(env,9511,output,arm,probe)
        guard=ContinuationGuard(model,arm,output)
        trace=v18.CalibrationTrace()
        warm_proof=None if probe else v16.v15.frozen.CurriculumProof(output,"critic")
        callbacks=[trace] if probe else [v16.v15.frozen.Heartbeat(output,"critic"),warm_proof,trace]
        model.set_logger(configure(str(output/"warmup"),["csv"]))
        model.learn(warm_steps,reset_num_timesteps=True,callback=callbacks,progress_bar=False)
        warm=guard.report()
        assert warm["actual_optimizer_steps"]==dict(critic=4 if probe else 1024,ppo=0)
        assert warm["Adam_parameter_states"]==6 and not warm["changed_actor_tensors"]
        model.save(output/"critic_warmed_model"); trace.save(output/"warm_trace.npz",BASIS_KIND)
        trace.rows.clear();write_json(output/"warmup.json",warm)
        v16.v15.start_ppo(model,guard)
        model.set_logger(configure(str(output),["csv"]))
        ppo_proof=None if probe else v16.v15.frozen.CurriculumProof(output,"ppo")
        selection=None if probe else MemorySelection(output,arm,model._n_updates)
        ppo_trace=v18.CalibrationTrace() if probe else None
        capture=FirstPPORollout(output,guard)
        model.learn(ppo_steps,reset_num_timesteps=True,callback=[ppo_trace,capture] if probe else [selection,ppo_proof,capture],progress_bar=False)
        assert capture.saved
        if not probe: selection.evaluate("final")
        report=guard.report();v16.assert_hyperparameters(model,probe)
        assert report["actual_optimizer_steps"]==dict(critic=4 if probe else 1024,ppo=4 if probe else 2048)
        assert report["Adam_parameter_states"]==12
        assert model.bc_kl_gradient_steps==model.ppo_gradient_steps==(4 if probe else 2048)
        assert report["changed_actor_tensors"] and "log_std" not in report["changed_actor_tensors"]
        if probe:
            model.save(output/"probe_model");ppo_trace.save(output/"ppo_trace.npz",BASIS_KIND)
            export_policy(model,output/"probe_policy.npz",arm)
            files=("critic_warmed_model.zip","warm_trace.npz","probe_model.zip","ppo_trace.npz","probe_policy.npz")
            selected=None
        else:
            assert len(selection.records)==3 and model.num_timesteps==262144
            assert ppo_proof.counts["completed_episodes"] and ppo_proof.counts["hard_tailwind_triggers"]
            files=("critic_warmed_model.zip","warm_trace.npz","best_model.zip","best_policy.npz","checkpoint_131072.zip","checkpoint_131072.npz","checkpoint_262144.zip","checkpoint_262144.npz","final.zip","final.npz")
            selected=read(output/"selection.json")
        assert source_hashes()==sources and sha(INITIAL)==INITIAL_SHA
        hashes={name:sha(output/name) for name in files}
        hashes["actual_gradient_norms.jsonl"]=sha(output/"actual_gradient_norms.jsonl")
        hashes["actual_first_ppo_rollout.npz"]=sha(output/"actual_first_ppo_rollout.npz")
        hashes["actual_first_ppo_rollout.json"]=sha(output/"actual_first_ppo_rollout.json")
        write_json(output/"guard_summary.json",report)
        write_json(output/"model_manifest.json",dict(initial_sha256=INITIAL_SHA,output_sha256=hashes))
        complete=dict(status="complete",schema=SCHEMA,variant=arm,conditioning_mode=arm,training_reward_arm="metric_cost",basis_kind=BASIS_KIND,
            critic_transitions=warm_steps,ppo_transitions=ppo_steps,optimizer=report,warmup=warm,
            output_model_sha256=hashes,source_sha256=sources,source_unchanged=True,
            actual_Gazebo80_acceptance=False,automatic_extension_authorized=False,new_baseline_episodes=0,
            wall_seconds=time.monotonic()-started,completed_utc=utc())
        if not probe:
            complete.update(candidate_records=selection.records,selected_policy_sha256=sha(output/"best_policy.npz"),
                selected_metrics=selected["aggregate"],fixed_development_guards_pass=selected["fixed_development_guards_pass"],
                diagnostic_candidate=not selected["own_baseline_admission"]["passed"],own_baseline_admission=selected["own_baseline_admission"],
                all_candidates_failed_development_guards=not any(row["guards_pass"] for row in selection.records),
                all_candidates_failed_own_baseline_admission=not any(row["own_baseline_admission_pass"] for row in selection.records),
                actual_curriculum_counts=dict(critic=warm_proof.counts,ppo=ppo_proof.counts))
        write_json(output/"complete.json",complete);phase(output,"complete",environment_steps=ppo_steps)
        return complete
    except Exception as exc:
        write_json(output/"error.json",dict(status="error",error=repr(exc),traceback=traceback.format_exc(),utc=utc()))
        raise
    finally:
        if guard is not None: guard.close()
        env.close()


def require_bound_preflight(args):
    from scripts import joint_research_preflight_v1 as gate
    assert args.research_plan is not None
    plan=read(args.research_plan);command=gate.validate(plan)
    assert plan["stage"]=="bounded_pilot" and plan["budget"]==BUDGET
    assert Path(command[2]).resolve()==Path(__file__).resolve() and sys.argv[1:]==plan["argv"]
    parent=(Path("/proc")/str(os.getppid())/"cmdline").read_bytes().decode().split("\0")
    assert any(Path(item).name=="joint_research_preflight_v1.py" for item in parent) and "--run" in parent
    assert (ROOT/parent[parent.index("--plan")+1]).resolve()==args.research_plan


def algorithm_static_check(args,sources):
    args.output.mkdir(parents=True,exist_ok=False);rows={};all13=None
    arrays=source_values();record,archive_path=v20.archived_sample()
    from quadrl_route_memory_v20.policy import NumpyPolicyRouteMemory
    for mode in ARMS:
        output=args.output/mode;output.mkdir()
        model=new_model(None,9511,output,mode)
        original=copy.deepcopy(model.policy.state_dict())
        if all13 is None:all13=original
        else:assert all(torch.equal(value,all13[name]) for name,value in original.items())
        layers=[layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer,torch.nn.Linear)]+[model.policy.action_net]
        for index,layer in enumerate(layers):
            np.testing.assert_array_equal(layer.weight.detach().cpu().numpy(),arrays[f"w{index}"])
            np.testing.assert_array_equal(layer.bias.detach().cpu().numpy(),arrays[f"b{index}"])
        guard=ContinuationGuard(model,mode,output)
        try:
            assert model.get_env() is None and guard.stage_lr==3e-4
            assert sum(p.requires_grad for p in model.policy.parameters())==6
            guard.set_stage("ppo")
            assert sum(p.requires_grad for p in model.policy.parameters())==12
            assert sum(p.numel() for n,p in model.policy.named_parameters() if is_mean_actor(n))==27396
            export_policy(model,output/"initial_policy.npz",mode)
            policy=NumpyPolicyRouteMemoryTrain(output/"initial_policy.npz",expected_mode=mode)
            reference=NumpyPolicyRouteMemory(ROOT/f"artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/{mode}/policy.npz",expected_mode=mode)
            policy.reset(record["yaw_reference"]);reference.reset(record["yaw_reference"])
            effective=[]
            with np.load(archive_path,allow_pickle=False) as io:
                for index in range(1000):
                    sample=tuple(io[name][index] for name in ("position","velocity","rotation","body_rates","target","previous_action","dt"))
                    current=policy.encoder.observe(*sample)
                    assert np.array_equal(current,reference.encoder.observe(*sample))
                    assert np.array_equal(policy.encoder.raw_observation,io["observation"][index])
                    effective.append(current)
            x=np.stack(effective)
            with torch.no_grad():assert float(model.prior_kl(torch.as_tensor(x)))==0.
            parity=float(np.max(np.abs(model.predict(x,deterministic=True)[0]-policy.predict(x))))
            assert parity<1e-5
            assert all(torch.equal(value,original[name]) for name,value in model.policy.state_dict().items())
            assert not model.policy.optimizer.state and all(p.grad is None for p in model.policy.parameters())
            v16.assert_hyperparameters(model,False)
            own=own_baseline(mode);no_gain=admission(own,mode)
            assert not no_gain["checks"]["nominal_finite_le_0_97_own_zero_update"]
            rows[mode]=dict(source68_ZIP_NPZ_all6_actor_arrays_exact=True,source13_weights_unchanged=True,
                optimizer_reset=[model.v21_loaded_optimizer_states,0],mean_actor_trainable_elements=27396,
                mean_actor_trainable_tensors=6,PPO_trainable_tensors=12,log_std_frozen=True,
                prior_source68_same_effective_inputs_initial_KL=0.,v20_conditioning_replay1000_exact=True,
                Torch_NumPy_effective_input_parity1000=parity,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
                own_zero_update_baseline=dict(path=BASELINES[mode].relative_to(ROOT).as_posix(),sha256=sha(BASELINES[mode]),episodes=24),
                baseline_itself_not_misreported_as_new_PPO_gain=True,initial_policy_sha256=sha(output/"initial_policy.npz"),
                actual_optimizer_steps=0,gradient_log_bytes=(output/"actual_gradient_norms.jsonl").stat().st_size)
        finally:guard.close()
    assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS_STATIC_ZERO_ENV_UPDATES",utc=utc(),source_sha256=sources,arms=rows,
        initial_zip_sha256=INITIAL_SHA,initial_actor_npz_sha256=SOURCE_SHA,both_initial_all13_weights_exact=True,
        prospective_semantic_budget=PROBE_BUDGET,reused_zero_update_baseline_episodes=48,new_baseline_episodes=0,
        environment_constructions=0,physical_steps=0,backward_calls=0,optimizer_steps=0,
        main_PPO_executed=False,policy_quality_accepted=False,warm_cross_mode_physical_parity_claimed=False,
        selection_rule="All3 candidates: ownbaseline admission filter, then unchanged nativeobjective; noneeligible diagnostic only")
    write_json(args.output/"static_receipt.json",report);assert not args.static_receipt.exists()
    write_json(args.static_receipt,report)
    print(json.dumps(dict(status=report["status"],receipt=str(args.static_receipt),sha256=sha(args.static_receipt))),flush=True)


def verify_pair(args,sources):
    plan=require_physical_plan(args,sources)
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/"prospective_plan.json",plan)
    arms={mode:train_one(args,sources,mode,True) for mode in ARMS}
    from scripts.run_joint_route_memory_pair_v21p1 import semantic_evidence
    evidence=semantic_evidence(args.output)
    for mode in ARMS:
        for name in ("warm_trace.npz","ppo_trace.npz"):
            with np.load(args.output/mode/name,allow_pickle=False) as trace:
                np.testing.assert_allclose(trace["reward"],trace["metric_reward"],rtol=1e-6,atol=2e-6)
                assert np.all(~trace["TimeLimit.truncated"])
                assert np.array_equal(trace["observation"],trace["effective_observation"])
                unchanged=np.r_[np.arange(22),np.arange(25,80)]
                assert np.array_equal(trace["raw_observation"][:,unchanged],trace["effective_observation"][:,unchanged])
                if mode=="zero_route_integral":assert np.all(trace["effective_observation"][trace["memory_valid_leg"].astype(bool),22:25]==0.)
    assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS",utc=utc(),source_sha256=sources,arms=arms,
        prospective_plan_sha256=sha(args.physical_plan),total_training_physical_steps=512,additional_nontraining_physical_steps=0,
        total_optimizer_steps=dict(critic=8,PPO=8),actual_conditioning_warm_prior_LR_and_first_PPO_evidence=evidence,
        common_metric_cost_reward=True,common_gamma=1.,common_true_finite_training_terminals=True,
        warm_learning_rate=3e-4,PPO_learning_rate=3e-5,reused_zero_update_baseline_episodes=48,new_baseline_episodes=0,
        cross_mode_physics_critic_reward_GAE_equality_required=False,actual_pilot_executed=False,
        conditioning_gain_is_not_new_PPO_gain=True,extra_forward_backward_optimizer_or_physics_calls=0)
    write_json(args.output/"semantic_receipt.json",report);assert not args.semantic_receipt.exists()
    write_json(args.semantic_receipt,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=ARMS)
    parser.add_argument("--eval-qa",action="store_true")
    parser.add_argument("--qa-plan",type=Path,default=ROOT/"docs/joint_route_memory_train_v21p1/eval_qa_plan.DRAFT.json")
    parser.add_argument("--qa-policy",type=Path)
    parser.add_argument("--qa-receipt",type=Path,default=ROOT/"audit/joint_route_memory_train_eval_qa_v21p1_root01.json")
    parser.add_argument("--initial",type=Path,default=INITIAL)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--retention-reference",type=Path,default=v16.v15.frozen.RETENTION_REFERENCE)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=9611);parser.add_argument("--warmup-seed",type=int,default=9511)
    parser.add_argument("--steps",type=int,default=262144);parser.add_argument("--warmup-steps",type=int,default=131072)
    parser.add_argument("--eval-every",type=int,default=131072);parser.add_argument("--research-plan",type=Path)
    parser.add_argument("--static-check",action="store_true");parser.add_argument("--verify",action="store_true")
    parser.add_argument("--physical-plan",type=Path,default=ROOT/"docs/joint_route_memory_train_v21p1/semantic_plan.DRAFT.json")
    parser.add_argument("--static-receipt",type=Path,default=ROOT/"audit/joint_route_memory_train_static_v21p1_root01.json")
    parser.add_argument("--semantic-receipt",type=Path,default=ROOT/"audit/joint_route_memory_train_semantics_v21p1_root01.json")
    args=parser.parse_args()
    for name in ("initial","baseline","retention_reference","output","research_plan","physical_plan","static_receipt","semantic_receipt","qa_plan","qa_policy","qa_receipt"):
        value=getattr(args,name)
        if value is not None:setattr(args,name,(value if value.is_absolute() else ROOT/value).resolve())
    if args.output.exists():parser.error("Fresh output required")
    if args.static_check and args.verify:parser.error("One preparation mode required")
    if (args.steps,args.warmup_steps,args.eval_every,args.seed,args.warmup_seed)!=(262144,131072,131072,9611,9511):parser.error("Fixed pilot budgets/seeds only")
    assert args.initial==INITIAL.resolve() and sha(INITIAL)==INITIAL_SHA
    if args.variant is not None:assert args.baseline==BASELINES[args.variant].resolve()
    assert args.retention_reference==v16.v15.frozen.RETENTION_REFERENCE.resolve() and sha(args.retention_reference)==v16.v15.frozen.RETENTION_SHA256
    torch.set_num_threads(1);sources=source_hashes()
    if args.static_check:static_check(args,sources);return
    assert_preparation_frozen(sources)
    if args.eval_qa:evaluator_qa(args,sources);return
    if args.verify:verify_pair(args,sources);return
    if args.variant is None or args.output.name!=args.variant:parser.error("Canonical conditioning arm/output required")
    for flag in ("--initial","--baseline","--retention-reference"):
        if flag not in sys.argv:parser.error("Explicit binding required: "+flag)
    require_bound_preflight(args);train_one(args,sources,args.variant)


def undefined_globals(path):
    import builtins
    import symtable
    table=symtable.symtable(Path(path).read_text(encoding="utf-8"),str(path),"exec")
    bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    available=bound|set(dir(builtins))|{"__file__","__name__","__package__","__spec__","__loader__","__cached__","__builtins__","__doc__"};missing=set()
    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in available:missing.add(symbol.get_name())
        for child in scope.get_children():visit(child)
    visit(table)
    return sorted(missing)


def patch_proof():
    import ast
    old_eval=ROOT/"quadrl_route_memory_train_v21/eval.py"
    new_eval=ROOT/"quadrl_route_memory_train_v21p1/eval.py"
    oldtext=old_eval.read_text(encoding="utf-8")
    corrected=new_eval.read_text(encoding="utf-8")
    reversed_text=corrected.replace("from quadrl_route_memory_train_v21.policy import ","from .policy import ")
    reversed_text=reversed_text.replace("\nsha,utc=frozen.sha,frozen.utc","")
    assert ast.dump(ast.parse(oldtext),include_attributes=False)==ast.dump(ast.parse(reversed_text),include_attributes=False)
    assert undefined_globals(old_eval)==["sha","utc"]
    assert not undefined_globals(new_eval)
    def definitions(path):
        tree=ast.parse(Path(path).read_text(encoding="utf-8"))
        return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=definitions(ROOT/"scripts/train_joint_route_memory_v21.py");current=definitions(Path(__file__))
    names=("own_baseline","new_model","ContinuationGuard","vector","FirstPPORollout","admission","MemorySelection","train_one","require_bound_preflight")
    for name in names:assert ast.dump(old[name],include_attributes=False)==ast.dump(current[name],include_attributes=False),name
    static=current["algorithm_static_check"];static=copy.deepcopy(static);static.name="static_check"
    assert ast.dump(static,include_attributes=False)==ast.dump(old["static_check"],include_attributes=False)
    before=(ROOT/"scripts/run_joint_route_memory_pair_v21.py").read_text(encoding="utf-8")
    after=(ROOT/"scripts/run_joint_route_memory_pair_v21p1.py").read_text(encoding="utf-8").replace("scripts/train_joint_route_memory_v21p1.py","scripts/train_joint_route_memory_v21.py")
    assert ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False)
    files=[Path(__file__),ROOT/"scripts/run_joint_route_memory_pair_v21p1.py",new_eval,ROOT/"quadrl_route_memory_train_v21p1/__init__.py"]
    lint={p.relative_to(ROOT).as_posix():undefined_globals(p) for p in files}
    assert all(not names for names in lint.values()),lint
    semantic=ROOT/"audit/joint_route_memory_train_semantics_v21_root01.json"
    original=read(semantic);assert original["status"]=="PASS" and original["total_training_physical_steps"]==512
    return dict(status="PASS_METADATA_ONLY_EVALUATOR_FIX_AND_ORIGINAL_TRAINING_ALGORITHM_IDENTITY",
        evaluator_reverse_AST_exact=True,old_evaluator_undefined_globals=["sha","utc"],new_all_module_undefined_globals=lint,
        evaluator_globals_fixed=True,normalized_algorithm_identity_exact=True,unchanged_training_functions=list(names),
        pair_reverse_driver_path_AST_exact=True,old_static_procedure_AST_exact=True,
        original_v21_semantics_scope_preserved=True,carried_original512=dict(path=semantic.relative_to(ROOT).as_posix(),sha256=sha(semantic)),
        original_policy_env_conditioning_contract_unchanged=True,no_new_512_executed=True,
        new_physical_steps=0,new_backward_calls=0,new_optimizer_steps=0)


def static_check(args,sources):
    proof=patch_proof()
    inner=copy.copy(args);inner.static_receipt=args.output/"algorithm_static_receipt.json"
    algorithm_static_check(inner,sources)
    report=read(inner.static_receipt)
    report.update(software_patch="v21p1 sha+utc evaluator globals only; explicit new source/path bindings",patch_proof=proof)
    assert source_hashes()==sources and not args.static_receipt.exists()
    write_json(args.output/"patch_static_receipt.json",report);write_json(args.static_receipt,report)
    print(json.dumps(dict(status=report["status"],receipt=str(args.static_receipt),sha256=sha(args.static_receipt))),flush=True)


def evaluator_qa(args,sources):
    from scripts import joint_research_preflight_v1 as gate
    from scripts.run_joint_route_memory_baseline_v20 import exact_episode_pair
    plan=read(args.qa_plan)
    budget=dict(evaluation_episodes=24,evaluation_control_steps=42000,training_transitions=0,
        optimizer_steps=0,backward_calls=0,additional_physical_steps=0)
    assert plan["schema"]=="joint-route-memory-evaluator-qa-plan-v21p1.0"
    assert plan["status"]=="READY_FOR_ROOT_AUTHORIZED_EVALUATOR_QA"
    assert plan["mode"]=="initial_zero_policy_native24_evaluation_qa" and plan["budget"]==budget
    assert plan["source_sha256"]==sources and plan["output"]==args.output.relative_to(ROOT).as_posix()
    assert args.qa_policy is not None and gate.bound_file(ROOT,plan["initial_policy"])==args.qa_policy
    authorization=plan["root_physical_authorization"]
    assert authorization["equals"] is True and gate.json_pointer(read(gate.bound_file(ROOT,authorization)),authorization["pointer"]) is True
    controller=NumpyPolicyRouteMemoryTrain(args.qa_policy,expected_mode="zero_route_integral")
    original=source_values()
    assert all(np.array_equal(w,original[f"w{i}"]) and np.array_equal(b,original[f"b{i}"]) for i,(w,b) in enumerate(zip(controller.weights,controller.biases)))
    assert not args.output.exists() and not args.qa_receipt.exists()
    proof=patch_proof();assert_preparation_frozen(sources)
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/"prospective_plan.json",plan)
    with zipfile.ZipFile(args.output/"qa_source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    result=evaluate("zero_route_integral",args.qa_policy,args.output/"evaluation",args.output)
    baseline=own_baseline("zero_route_integral")
    pairs=[exact_episode_pair(a,b) for a,b in zip(result["episodes"],baseline["episodes"])]
    assert len(pairs)==24 and all(row["passed"] for row in pairs)
    assert result["aggregate"]==baseline["aggregate"]
    assert all(row["controller_error"] is None for row in result["episodes"])
    actual=sum(row["steps"] for row in result["episodes"]);assert actual==42000
    assert result["policy_sha256"]==sha(args.qa_policy) and result["evaluated_utc"]
    assert source_hashes()==sources
    report=dict(schema="joint-route-memory-evaluator-qa-receipt-v21p1.0",status="PASS_EVALUATOR_COMPLETION_AND_V20_ZERO_PORTABILITY",
        utc=utc(),output=args.output.relative_to(ROOT).as_posix(),source_sha256=sources,prospective_plan_sha256=sha(args.qa_plan),
        budget=budget,evaluation_episodes=24,actual_evaluation_control_steps=actual,new_physical_control_steps=actual,
        training_transitions=0,optimizer_steps=0,backward_calls=0,additional_physical_steps=0,
        all24_exact_trace_and_actual_NN_IO=True,aggregate_exact_v20_zero=True,case_exact_comparisons=pairs,
        current_policy_sha256=sha(args.qa_policy),evaluator_aggregate_completed_UTC=result["evaluated_utc"],
        evaluation_path=(args.output/"evaluation/evaluation.json").relative_to(ROOT).as_posix(),
        evaluation_sha256=sha(args.output/"evaluation/evaluation.json"),patch_proof=proof,
        metadata_only_reverse_AST=True,original_v21_semantics_scope_preserved=True,
        interpretation="24 NEW physical software-portability QA episodes; not new scientific replication or new PPO gain",
        automatic_main_or_Gazebo_authorized=False)
    write_json(args.output/"complete.json",report);write_json(args.qa_receipt,report)
    phase(args.output,"complete",evaluation_episodes=24,evaluation_control_steps=actual,training_transitions=0,optimizer_steps=0)
    print(json.dumps(dict(status=report["status"],receipt=str(args.qa_receipt),sha256=sha(args.qa_receipt))),flush=True)


if __name__=="__main__":main()




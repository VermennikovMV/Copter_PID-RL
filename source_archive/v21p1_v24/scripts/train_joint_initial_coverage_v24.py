"""Matched full startup bundle; unchanged heading schedule, PPO and all29 gates."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import zipfile
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from scripts import train_joint_route_memory_v21p1 as base
from scripts import train_joint_heading_coverage_v23 as heading
from scripts import run_joint_route_memory_pair_v21p1 as old_pair
from scripts import run_joint_memory_transfer_initial_v22 as transfer
from quadrl_route_initial_v24.env import JointTaskEnv,wrapped
from quadrl_route_heading_v23.eval import (evaluate_original,evaluate_profiles,profiles,fixed_yaw,PROFILE_PLAN)
from quadrl_route_memory_train_v21.policy import CONTRACT_VERSION,export_policy,NumpyPolicyRouteMemoryTrain
v18,v16=base.v18,base.v16
INITIAL,INITIAL_SHA=base.INITIAL,base.INITIAL_SHA
BUDGET,PROBE_BUDGET=base.BUDGET.copy(),base.PROBE_BUDGET.copy()
BASELINES=base.BASELINES
BASIS_KIND=base.BASIS_KIND
ARMS=("zero_route_integral",)
PROBE_ARMS=("control_probe","initial_probe")
SCHEMA="joint-initial-coverage-ppo-v24.0"
read,write_json,utc,phase,sha=base.read,base.write_json,base.utc,base.phase,base.sha
new_model,ContinuationGuard=base.new_model,base.ContinuationGuard
CONTROL=ROOT/"artifacts/refinement/joint_v2/route_heading_pilot_v23_111/zero_route_integral"
OLD_SEM=ROOT/"artifacts/refinement/joint_v2/route_heading_semantic_v23_root01/heading_probe"
FIXTURE_BUDGET=dict(environments=2,explicit_reset_calls=36,maximum_constructor_reset_calls=4,maximum_total_reset_calls=40,
    control_steps=0,NN_forwards=0,backward_calls=0,optimizer_steps=0)


def source_hashes():
    previous=heading.source_hashes();heading.assert_preparation_frozen(previous)
    paths=[Path(__file__),*sorted((ROOT/"quadrl_route_initial_v24").glob("*.py"))]
    for label in old_pair.LABELS:
        paths.extend(CONTROL/(label+suffix) for suffix in (".npz",".zip","_validation.json"))
    paths.extend(CONTROL/name for name in ("complete.json","selection.json"))
    paths.extend(OLD_SEM/name for name in ("critic_warmed_model.zip","probe_model.zip","warm_trace.npz","ppo_trace.npz","actual_first_ppo_rollout.npz","actual_gradient_norms.jsonl"))
    return {**previous,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def assert_preparation_frozen(sources):
    record=read(ROOT/"docs/joint_route_initial_v24/preparation_freeze.json")
    assert record["source_sha256"]==sources
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name


def vector(count,seed,enabled):
    env=VecMonitor(DummyVecEnv([lambda:JointTaskEnv(heading_enabled=True,matched_yaw=fixed_yaw(),full_initial_enabled=enabled,deployment_state=read(PROFILE_PLAN)["source_measured_Gazebo_initial"]["state"],architecture="hybrid",training=True) for unused in range(count)]))
    env.set_attr("training_progress",1.);env.seed(seed)
    return env


class HeadingGuard(heading.HeadingGuard):
    def __init__(self,model,mode,output,enabled):
        self.full_initial_enabled=enabled
        super().__init__(model,mode,output,True)

    def report(self):
        report=super().report()
        report.update(full_initial_enabled=self.full_initial_enabled,
            changed_factor="ONLY matched-route FULL deployment p/v/roll/pitch/bodyrates and reference-error-zero bundle; heading strata/source68/PPO/reward unchanged",
            critic_coupling="Bundled initial physical/reference state changes on-policy trajectories/mission ends; components not separately identified")
        return report


def materialize(path):
    with np.load(path,allow_pickle=False) as archive:return {name:archive[name].copy() for name in archive.files}


def global_rng():return (torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),copy.deepcopy(random.getstate()))


def assert_rng(before):
    current=global_rng()
    assert torch.equal(before[0],current[0]) and before[2]==current[2]
    assert before[1][0]==current[1][0] and np.array_equal(before[1][1],current[1][1]) and before[1][2:]==current[1][2:]


HeadingSelection=heading.HeadingSelection


def require_physical_plan(args,sources):
    from scripts import joint_research_preflight_v1 as gate
    plan=read(args.physical_plan)
    assert plan["schema"]=="joint-route-initial-preparation-plan-v24.0"
    assert plan["status"]=="READY_FOR_ROOT_AUTHORIZED_PHYSICAL_PREPARATION"
    assert plan["mode"]=="semantic_pair512_with_reset_fixtures" and plan["budget"]==PROBE_BUDGET
    assert plan["reset_fixture_budget"]==FIXTURE_BUDGET
    assert plan["source_sha256"]==sources and plan["output"]==args.output.relative_to(ROOT).as_posix()
    binding=plan["root_physical_authorization"]
    assert binding["equals"] is True and gate.json_pointer(read(gate.bound_file(ROOT,binding)),binding["pointer"]) is True
    return plan


def reset_fixtures():
    from quadrl_route_heading_v23.env import JointTaskEnv as Reference
    from quadrl_route_transfer_v22.initial import rotation
    class CountedReference(Reference):
        def __init__(self,**kwargs):self.reset_calls=0;super().__init__(**kwargs)
        def reset(self,*args,**kwargs):self.reset_calls+=1;return super().reset(*args,**kwargs)
    deployment=read(PROFILE_PLAN)["source_measured_Gazebo_initial"]["state"]
    a=CountedReference(heading_enabled=True,matched_yaw=fixed_yaw(),architecture="hybrid",training=True)
    b=JointTaskEnv(heading_enabled=True,matched_yaw=fixed_yaw(),full_initial_enabled=True,deployment_state=deployment,architecture="hybrid",training=True)
    constructor=a.reset_calls+b.heading_constructor_resets
    rows=[]
    try:
        a.training_progress=b.training_progress=1.
        for index in range(17):
            seed=9511 if index==0 else None
            sentinel_a={"fixture_saved_action_sample":index};sentinel_b={"fixture_saved_action_sample":index}
            a.last_step_actor_sample=sentinel_a;b.last_step_actor_sample=sentinel_b
            oa,ia=a.reset(seed=seed);ob,ib=b.reset(seed=seed);r=ib["heading_reset"]
            assert a.last_step_actor_sample is sentinel_a and b.last_step_actor_sample is sentinel_b
            assert a.mission.kind==b.mission.kind and json.dumps(ia["parameters"],sort_keys=True)==json.dumps(ib["parameters"],sort_keys=True)
            assert json.dumps(a.np_random.bit_generator.state,sort_keys=True)==json.dumps(b.np_random.bit_generator.state,sort_keys=True)
            assert json.dumps(a.sensor_rng.bit_generator.state,sort_keys=True)==json.dumps(b.sensor_rng.bit_generator.state,sort_keys=True)
            if not r["full_initial_bundle"]:
                assert np.array_equal(oa,ob)
                assert ia["initial_state"]==ib["initial_state"]
            else:
                actual=b.state();expected=(deployment["position"],deployment["velocity"],rotation(deployment["rpy"]),deployment["body_rates"])
                assert all(np.max(np.abs(value-np.asarray(wanted)))<1e-11 for value,wanted in zip(actual,expected))
                assert b.yaw_reference==deployment["rpy"][2] and abs(r["relative_error_after"])<1e-12
                assert b.mission.kind=="route" and b.mission.index==0 and b.mission.inside_since==0. and b.mission.last_time==0.
                assert b.conditioner.observation_calls==1 and len(b.encoder.history)==1 and b.encoder.time==0.
                assert np.count_nonzero(b.encoder.integral_world)==0 and not b.path_context.valid_leg
                assert np.count_nonzero(b.rate_model.target_rate)==np.count_nonzero(b.rate_model.target_accel)==0
                assert all(np.count_nonzero(list(q))==0 for q in [b.rate_model.yaw_queue,*[d.history for d in b.rate_model.delays]])
            assert a.steps==b.steps==0
            rows.append(copy.deepcopy(r))
        first=b.heading_snapshot();assert first["route_counter"]>=2
        assert abs(first["matched_routes"]-first["original_routes"])<=1
        oa,ia=a.reset(seed=9611);ob,ib=b.reset(seed=9611)
        assert b.heading_stage_seed==9611 and b.heading_route_counter in (0,1)
        if not ib["heading_reset"]["full_initial_bundle"]:assert np.array_equal(oa,ob)
        assert a.reset_calls+b.heading_constructor_resets+18<=40 and constructor<=4
        return dict(status="PASS_BOUNDED_RESET_ONLY_NO_CONTROL_OR_NN",budget=FIXTURE_BUDGET,
            actual_environments=2,actual_explicit_reset_calls=36,actual_constructor_reset_calls=constructor,
            first_stage_snapshot=first,next_stage_snapshot=b.heading_snapshot(),reset_records=rows,
            original_branch_and_hover_observation_exact=True,force_parameters_and_sampler_RNG_unchanged=True,
            full_matched_physical_state_and_reference_zero_exact=True,new_position_Mission_dwell_reinitialized=True,
            sensor_RNG_one_observation_state_exact=True,stage_seed_counter_reset=True,
            terminal_action_audit_cache_survives_automatic_reset=True,
            control_steps=0,NN_forwards=0,backward_calls=0,optimizer_steps=0)
    finally:a.close();b.close()



def same_structure(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_structure(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same_structure(x,y) for x,y in zip(a,b))
    return a==b


def semantic_evidence(output):
    from stable_baselines3.common.save_util import load_from_zip_file
    control=Path(output)/"control_probe"
    for name in ("critic_warmed_model.zip","probe_model.zip"):
        old=load_from_zip_file(OLD_SEM/name,device="cpu")[1]
        actual=load_from_zip_file(control/name,device="cpu")[1]
        assert same_structure(old["policy"],actual["policy"]),name+" ALL13"
        assert same_structure(old["policy.optimizer"],actual["policy.optimizer"]),name+" entireAdam"
    exact={}
    for name in ("warm_trace.npz","ppo_trace.npz","actual_first_ppo_rollout.npz"):
        old,actual=materialize(OLD_SEM/name),materialize(control/name)
        assert old.keys()<=actual.keys()
        for key,value in old.items():assert np.array_equal(value,actual[key]),name+":"+key
        exact[name]=dict(old_fields_exact=len(old),new_fields=sorted(actual.keys()-old.keys()))
    oldgrad=[json.loads(row) for row in (OLD_SEM/"actual_gradient_norms.jsonl").read_text().splitlines()]
    newgrad=[json.loads(row) for row in (control/"actual_gradient_norms.jsonl").read_text().splitlines()]
    assert oldgrad==newgrad
    per_arm={}
    for arm in PROBE_ARMS:
        folder=Path(output)/arm
        arrays=materialize(folder/"actual_first_ppo_rollout.npz")
        per_arm[arm]=dict(GAE=old_pair.previous.gae_readback(arrays),effective_memory=old_pair.memory_replay(arrays,"zero_route_integral"),
            critic_heading=read(folder/"heading_counts_critic.json"),PPO_heading=read(folder/"heading_counts_ppo.json"))
    return dict(status="PASS_CONTROL_V23_HEADING256_EXACT_AND_C_FULL_INITIAL_SEMANTICS",control_all13_and_entire_Adam_exact_old_v23_heading256=True,
        control_all_old_IO_trace_and_first_rollout_fields_exact=exact,control_actual_gradient_and_Adam_rows_exact=True,
        per_arm=per_arm,NPZ_materialized_once_per_archive=True,full_initial_C_physical_or_critic_parity_required=False)


def verify_pair(args,sources):
    plan=require_physical_plan(args,sources)
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/"prospective_plan.json",plan)
    fixtures=reset_fixtures();write_json(args.output/"reset_fixture_proof.json",fixtures)
    arms={arm:train_one(args,sources,arm,True) for arm in PROBE_ARMS}
    evidence=semantic_evidence(args.output)
    for arm in PROBE_ARMS:
        for name in ("warm_trace.npz","ppo_trace.npz"):
            trace=materialize(args.output/arm/name)
            np.testing.assert_allclose(trace["reward"],trace["metric_reward"],rtol=1e-6,atol=2e-6)
            assert np.all(~trace["TimeLimit.truncated"])
    assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS",utc=utc(),source_sha256=sources,arms=arms,reset_fixture_proof=fixtures,
        actual_semantic_evidence=evidence,prospective_plan_sha256=sha(args.physical_plan),
        total_training_physical_steps=512,additional_nontraining_physical_steps=0,total_optimizer_steps=dict(critic=8,PPO=8),
        new_evaluation_episodes=0,reset_only_fixture_environments=2,reset_only_fixture_control_steps=0,
        actual_main_C_executed=False,automatic_main_or_Gazebo_authorized=False)
    write_json(args.output/"semantic_receipt.json",report);assert not args.semantic_receipt.exists();write_json(args.semantic_receipt,report)


def static_check(args,sources):
    args.output.mkdir(parents=True,exist_ok=False)
    oldargs=copy.copy(args);oldargs.output=args.output/"unchanged_core";oldargs.static_receipt=args.output/"unchanged_core_static.json"
    base.static_check(oldargs,base.source_hashes())
    paths=[Path(__file__),*sorted((ROOT/"quadrl_route_initial_v24").glob("*.py"))]
    lint={p.relative_to(ROOT).as_posix():base.undefined_globals(p) for p in paths};assert not any(lint.values()),lint
    yaw=fixed_yaw();deployment=read(PROFILE_PLAN)["source_measured_Gazebo_initial"]["state"]
    from quadrl_route_transfer_v22.initial import rotation
    assert deployment["rpy"][2]==yaw
    assert np.max(np.abs(rotation(deployment["rpy"])-np.asarray(deployment["rotation"])))<1e-12
    assert HeadingSelection is heading.HeadingSelection
    assert evaluate_profiles is heading.evaluate_profiles
    assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS_STATIC_ZERO_ENV_OR_UPDATES",utc=utc(),source_sha256=sources,
        unchanged_initialization_core=dict(path=oldargs.static_receipt.relative_to(ROOT).as_posix(),sha256=sha(oldargs.static_receipt)),
        source68_initialization=read(oldargs.static_receipt)["arms"]["zero_route_integral"],
        all_new_module_undefined_globals=lint,exact_matched_Gazebo_yaw=yaw,matched_full_bundle=deployment,matched_reference_equals_actual_yaw_error_zero=True,
        evaluator_and_scientific_selector_reused_unchanged=True,
        prospective_semantic_budget=PROBE_BUDGET,prospective_reset_fixture_budget=FIXTURE_BUDGET,
        exact_five_profiles=profiles(),control_A_v23_training_reused_not_rerun=True,main_C_arms=1,
        new_evaluation_episodes=87,reused_coverage_episodes=116,coverage_episodes=203,
        new_evaluation_control_steps=156000,reused_control_steps=208000,coverage_control_steps=364000,
        environment_constructions=0,physical_steps=0,backward_calls=0,optimizer_steps=0,
        reset_runtime_physically_tested=False,automatic_main_or_Gazebo_authorized=False)
    write_json(args.output/"static_receipt.json",report);assert not args.static_receipt.exists();write_json(args.static_receipt,report)
    print(json.dumps(dict(status=report["status"],receipt=str(args.static_receipt),sha256=sha(args.static_receipt))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=ARMS);parser.add_argument("--initial",type=Path,default=INITIAL)
    parser.add_argument("--baseline",type=Path,default=BASELINES["zero_route_integral"])
    parser.add_argument("--retention-reference",type=Path,default=base.v16.v15.frozen.RETENTION_REFERENCE)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--research-plan",type=Path)
    parser.add_argument("--seed",type=int,default=9611);parser.add_argument("--warmup-seed",type=int,default=9511)
    parser.add_argument("--steps",type=int,default=262144);parser.add_argument("--warmup-steps",type=int,default=131072)
    parser.add_argument("--eval-every",type=int,default=131072)
    parser.add_argument("--static-check",action="store_true");parser.add_argument("--verify",action="store_true")
    parser.add_argument("--physical-plan",type=Path,default=ROOT/"docs/joint_route_initial_v24/preparation_plan.DRAFT.json")
    parser.add_argument("--static-receipt",type=Path,default=ROOT/"audit/joint_initial_coverage_static_v24_root01.json")
    parser.add_argument("--semantic-receipt",type=Path,default=ROOT/"audit/joint_initial_coverage_semantics_v24_root01.json")
    args=parser.parse_args()
    for name in ("initial","baseline","retention_reference","output","research_plan","physical_plan","static_receipt","semantic_receipt"):
        value=getattr(args,name)
        if value is not None:setattr(args,name,(value if value.is_absolute() else ROOT/value).resolve())
    assert not args.output.exists() and not (args.static_check and args.verify)
    assert (args.steps,args.warmup_steps,args.eval_every,args.seed,args.warmup_seed)==(262144,131072,131072,9611,9511)
    assert args.initial==INITIAL.resolve() and sha(INITIAL)==INITIAL_SHA
    assert args.baseline==BASELINES["zero_route_integral"].resolve()
    assert args.retention_reference==base.v16.v15.frozen.RETENTION_REFERENCE.resolve()
    torch.set_num_threads(1);sources=source_hashes()
    if args.static_check:static_check(args,sources);return
    assert_preparation_frozen(sources)
    if args.verify:verify_pair(args,sources);return
    assert args.variant=="zero_route_integral" and args.output.name==args.variant
    assert all(flag in sys.argv for flag in ("--initial","--baseline","--retention-reference"))
    require_bound_preflight(args);train_one(args,sources,args.variant)


class FirstPPORollout(base.BaseCallback):
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
            "raw_integral_world","conditioned_integral_world","memory_valid_leg","memory_target_changes","memory_target_changed","yaw_reference","heading_matched","heading_route_ordinal","heading_source_actual_yaw","heading_source_yaw_reference","heading_actual_yaw","heading_yaw_reference","heading_delta","heading_relative_error_before","heading_relative_error_after")
        for name in keys:arrays["sample_"+name]=np.stack([np.stack([row[name] for row in batch]) for batch in self.samples])
        assert np.array_equal(arrays["sample_effective_observation"],b.observations)
        assert np.all(b.rewards[arrays["mission_is_route"]]<=1e-6)
        np.savez_compressed(self.output/"actual_first_ppo_rollout.npz",**arrays)
        write_json(self.output/"actual_first_ppo_rollout.json",dict(status="ACTUAL_FIRST_PPO_ROLLOUT_ARCHIVED_BEFORE_UPDATE",
            conditioning_mode=self.guard.conditioning_mode,environment_steps=self.model.num_timesteps,
            heading_coverage_enabled=True,full_initial_enabled=self.guard.full_initial_enabled,
            actual_ppo_optimizer_steps_before_update=0,policy_before_rollout=dict(path=(self.output/"critic_warmed_model.zip").relative_to(ROOT).as_posix(),sha256=sha(self.output/"critic_warmed_model.zip")),
            binding_scope="Own-mode warmed ZIP weights identical to actor before own first rollout; warm Adam carried within mode; cross-mode critic weights/Adam/physics may differ",
            contract=CONTRACT_VERSION,actual_NN_inputs="observations == sample_effective_observation",raw_original80_separate=True,
            calibration=v18.calibration_stats(b.values,b.advantages,b.rewards,arrays["mission_is_route"],arrays["route_context_nonzero"]),
            after_original_compute_returns_and_advantage=True,before_any_PPO_update=True,raw_distribution_actions=True,
            shapes={n:list(a.shape) for n,a in arrays.items()},payload_sha256=sha(self.output/"actual_first_ppo_rollout.npz"),
            extra_environment_steps=0,extra_forward_calls=0,extra_backward_calls=0,extra_optimizer_steps=0))
        self.saved=True;self.route.clear();self.samples.clear()


def train_one(args,sources,arm,probe=False):
    mode="zero_route_integral";enabled=arm!="control_probe"
    output=args.output/arm if probe else args.output
    output.mkdir(parents=probe,exist_ok=False)
    started=time.monotonic()
    warm_steps,ppo_steps=(128,128) if probe else (131072,262144)
    write_json(output/"config.json",dict(schema=SCHEMA,variant=arm,conditioning_mode=mode,heading_coverage_enabled=True,full_initial_enabled=enabled,training_reward_arm="metric_cost",basis_kind=BASIS_KIND,
        initial_sha256=INITIAL_SHA,prior_sha256=INITIAL_SHA,budget=PROBE_BUDGET if probe else BUDGET,
        actor_trainable_mean_elements=27396,log_std_frozen=True,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
        gamma=1.,true_finite_training_deadlines=True,reward_and_plant="EXACT v16 metric_cost unchanged",
        critic="Original77 input columns/all other critic tensors; new3 hardzero before sharedclip",
        reused_baseline=dict(path=BASELINES[mode].relative_to(ROOT).as_posix(),sha256=sha(BASELINES[mode]),new_episodes=0),
        source_sha256=sources,started_utc=utc(),actual_Gazebo_acceptance=False))
    with zipfile.ZipFile(output/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in sources: archive.write(ROOT/name,name)
    write_json(output/"manifest.json",dict(source_sha256=sources,initial_sha256=INITIAL_SHA,source_snapshot_sha256=sha(output/"source_snapshot.zip"),
        heading_coverage_enabled=True,full_initial_enabled=enabled,initializer_restart_source68=True))
    env,guard=vector(1 if probe else 16,9511,enabled),None
    try:
        model=new_model(env,9511,output,mode,probe)
        guard=HeadingGuard(model,mode,output,enabled)
        trace=v18.CalibrationTrace()
        warm_proof=None if probe else v16.v15.frozen.CurriculumProof(output,"critic")
        callbacks=[trace] if probe else [v16.v15.frozen.Heartbeat(output,"critic"),warm_proof,trace]
        model.set_logger(configure(str(output/"warmup"),["csv"]))
        model.learn(warm_steps,reset_num_timesteps=True,callback=callbacks,progress_bar=False)
        write_json(output/"heading_counts_critic.json",env.env_method("heading_snapshot"))
        warm=guard.report()
        assert warm["actual_optimizer_steps"]==dict(critic=4 if probe else 1024,ppo=0)
        assert warm["Adam_parameter_states"]==6 and not warm["changed_actor_tensors"]
        model.save(output/"critic_warmed_model"); trace.save(output/"warm_trace.npz",BASIS_KIND)
        trace.rows.clear();write_json(output/"warmup.json",warm)
        v16.v15.start_ppo(model,guard)
        model.set_logger(configure(str(output),["csv"]))
        ppo_proof=None if probe else v16.v15.frozen.CurriculumProof(output,"ppo")
        selection=None if probe else HeadingSelection(output,mode,model._n_updates)
        ppo_trace=v18.CalibrationTrace() if probe else None
        capture=FirstPPORollout(output,guard)
        model.learn(ppo_steps,reset_num_timesteps=True,callback=[ppo_trace,capture] if probe else [selection,ppo_proof,capture],progress_bar=False)
        assert capture.saved
        if not probe: selection.evaluate("final")
        write_json(output/"heading_counts_ppo.json",env.env_method("heading_snapshot"))
        report=guard.report();v16.assert_hyperparameters(model,probe)
        assert report["actual_optimizer_steps"]==dict(critic=4 if probe else 1024,ppo=4 if probe else 2048)
        assert report["Adam_parameter_states"]==12
        assert model.bc_kl_gradient_steps==model.ppo_gradient_steps==(4 if probe else 2048)
        assert report["changed_actor_tensors"] and "log_std" not in report["changed_actor_tensors"]
        if probe:
            model.save(output/"probe_model");ppo_trace.save(output/"ppo_trace.npz",BASIS_KIND)
            export_policy(model,output/"probe_policy.npz",mode)
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
        write_json(output/"model_manifest.json",dict(initial_sha256=INITIAL_SHA,output_sha256=hashes,
            heading_coverage_enabled=True,full_initial_enabled=enabled,initializer_restart_source68=True))
        complete=dict(status="complete",schema=SCHEMA,variant=arm,conditioning_mode=mode,heading_coverage_enabled=True,full_initial_enabled=enabled,training_reward_arm="metric_cost",basis_kind=BASIS_KIND,
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


if __name__=="__main__":main()

"""Three fixed route-normal objectives; common calibrated full-mean PPO."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from scripts import train_joint_critic_calibration_v18 as v18
from scripts.bc_trust_region_probe_v1 import actor_name,sha
from quadrl_path_damping_v19.env import JointTaskEnv,BASIS_KIND
from quadrl_path_damping_v19.reward import ARMS,synthetic_fixtures
v16=v18.v16
INITIAL,INITIAL_SHA,BASELINE=v18.INITIAL,v18.INITIAL_SHA,v18.BASELINE
BUDGET=v18.BUDGET.copy()
PROBE_BUDGET={**v18.PROBE_BUDGET,"arms":3,"total_training_physical_steps":768,"total_physical_steps":768}
SCHEMA="joint-path-damping-ppo-v19.0"
read,write_json,utc,phase=v18.read,v18.write_json,v18.utc,v18.phase
is_mean_actor=v18.is_mean_actor


def source_hashes():
    previous=v18.source_hashes();v18.assert_preparation_frozen(previous)
    paths=[Path(__file__),ROOT/"scripts/run_joint_path_damping_triplet_v19.py"]
    paths.extend(sorted((ROOT/"quadrl_path_damping_v19").glob("*.py")))
    return {**previous,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def assert_preparation_frozen(sources):
    record=read(ROOT/"docs/joint_path_damping_v19/preparation_freeze.json")
    assert record["source_sha256"]==sources,"Changed source after prospective freeze"
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name


def vector(count,seed,arm):
    env=VecMonitor(DummyVecEnv([lambda:JointTaskEnv(variant=arm,architecture="hybrid",training=True) for unused in range(count)]))
    env.set_attr("training_progress",1.);env.seed(seed)
    return env


class DampingGuard(v18.CalibrationGuard):
    def __init__(self,model,arm,output):
        self.objective_arm=arm
        super().__init__(model,"critic_lr3e4",output)

    def before_step(self,optimizer,unused_args,unused_kwargs):
        super().before_step(optimizer,unused_args,unused_kwargs)
        self.pending["objective_arm"]=self.objective_arm

    def report(self):
        report=super().report()
        report.update(objective_arm=self.objective_arm,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
            changed_factor="ONLY route objective: exact metric_cost / +4vperpSq / +4(vperp+1.5enormal)Sq",
            critic_coupling="Different route rewards alter warm critic/Adam and subsequent sharedclip/GAE; full actor capacity and learning rates common",
            warm_critic_Adam_equality_required=False)
        return report


class DampingTrace(v18.CalibrationTrace):
    def _on_step(self):
        result=super()._on_step()
        fields=("post_position","post_velocity","leg_start","leg_end","leg_unit","closest_finite","closest_fraction",
            "post_error_normal","post_velocity_perp","normal_target_residual","post_velocity_perp_sq",
            "normal_target_residual_sq","eligible_interval","geometry_valid","added_cost_rate","reward_adjustment",
            "effective_reward")
        for name in fields:
            self.rows.setdefault("damping_"+name,[]).append(np.stack([np.asarray(i["path_damping_ledger"][name]) for i in self.locals["infos"]]))
        return result


class FirstPPORollout(v18.FirstPPORollout):
    def _on_rollout_end(self):
        already=self.saved
        super()._on_rollout_end()
        if not already:
            path=self.output/"actual_first_ppo_rollout.json"
            record=read(path)
            record.update(objective_arm=self.guard.objective_arm,
                comparison="Three objectives, common warm LR3e-4 and PPO LR3e-5; critic/rewards/GAE may differ",
                critic_calibration_arm="critic_lr3e4 (shared validated regime, not the v19 treatment)")
            write_json(path,record)


def require_physical_plan(args,sources):
    from scripts import joint_research_preflight_v1 as gate
    plan=read(args.physical_plan)
    assert plan["schema"]=="joint-path-damping-physical-preparation-plan-v19.0"
    assert plan["status"]=="READY_FOR_ROOT_AUTHORIZED_PHYSICAL_PREPARATION"
    assert plan["mode"]=="semantic_triplet768" and plan["budget"]==PROBE_BUDGET
    assert plan["source_sha256"]==sources and plan["output"]==args.output.relative_to(ROOT).as_posix()
    binding=plan["root_physical_authorization"];path=gate.bound_file(ROOT,binding)
    assert binding["equals"] is True and gate.json_pointer(read(path),binding["pointer"]) is True
    return plan


def train_one(args,sources,arm,probe=False):
    output=args.output/arm if probe else args.output
    output.mkdir(parents=probe,exist_ok=False)
    started=time.monotonic()
    warm_steps,ppo_steps=(128,128) if probe else (131072,262144)
    write_json(output/"config.json",dict(schema=SCHEMA,variant=arm,training_reward_arm=arm,basis_kind=BASIS_KIND,
        initial_sha256=INITIAL_SHA,prior_sha256=INITIAL_SHA,budget=PROBE_BUDGET if probe else BUDGET,
        actor_trainable_mean_elements=27396,log_std_frozen=True,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
        gamma=1.,true_finite_training_deadlines=True,reward_and_plant="Frozen v16 physical dynamics, common metric cost plus preregistered route-normal objective",
        critic="Original77 input columns/all other critic tensors; new3 hardzero before sharedclip",
        reused_baseline=dict(path=BASELINE.relative_to(ROOT).as_posix(),sha256=sha(BASELINE),new_episodes=0),
        source_sha256=sources,started_utc=utc(),actual_Gazebo_acceptance=False))
    with zipfile.ZipFile(output/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in sources: archive.write(ROOT/name,name)
    write_json(output/"manifest.json",dict(source_sha256=sources,initial_sha256=INITIAL_SHA,source_snapshot_sha256=sha(output/"source_snapshot.zip")))
    env,guard=vector(1 if probe else 16,9511,arm),None
    try:
        model=v16.new_model(env,9511,output,probe)
        guard=DampingGuard(model,arm,output)
        trace=DampingTrace()
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
        selection=None if probe else v16.v15.RouteBasisSelection(output,v16.baseline(),BASIS_KIND,model._n_updates)
        ppo_trace=DampingTrace() if probe else None
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
            v16.v15.export(model,output/"probe_policy.npz",BASIS_KIND)
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
        complete=dict(status="complete",schema=SCHEMA,variant=arm,training_reward_arm=arm,basis_kind=BASIS_KIND,
            critic_transitions=warm_steps,ppo_transitions=ppo_steps,optimizer=report,warmup=warm,
            output_model_sha256=hashes,source_sha256=sources,source_unchanged=True,
            actual_Gazebo80_acceptance=False,automatic_extension_authorized=False,new_baseline_episodes=0,
            wall_seconds=time.monotonic()-started,completed_utc=utc())
        if not probe:
            complete.update(candidate_records=selection.records,selected_policy_sha256=sha(output/"best_policy.npz"),
                selected_metrics=selected["aggregate"],fixed_development_guards_pass=selected["fixed_development_guards_pass"],
                diagnostic_candidate=not selected["fixed_development_guards_pass"],
                all_candidates_failed_development_guards=not any(row["guards_pass"] for row in selection.records),
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


def static_check(args,sources):
    args.output.mkdir(parents=True,exist_ok=False);rows={}
    fixtures=synthetic_fixtures();base_fixtures=v16.synthetic_fixtures()
    for arm in ARMS:
        output=args.output/arm;output.mkdir()
        model=v16.new_model(None,9511,output)
        initial=copy.deepcopy(model.policy.state_dict())
        guard=DampingGuard(model,arm,output)
        try:
            assert model.get_env() is None
            assert model.learning_rate==3e-4 and all(model.lr_schedule(p)==3e-4 for p in (0.,.5,1.))
            assert all(g["lr"]==3e-4 for g in model.policy.optimizer.param_groups)
            assert sum(p.requires_grad for p in model.policy.parameters())==6
            assert all(not p.requires_grad for n,p in model.policy.named_parameters() if actor_name(n))
            guard.set_stage("ppo")
            trainable={n for n,p in model.policy.named_parameters() if p.requires_grad}
            assert len(trainable)==12 and "log_std" not in trainable
            assert sum(is_mean_actor(n) for n in trainable)==6
            assert sum(p.numel() for n,p in model.policy.named_parameters() if is_mean_actor(n))==27396
            assert model.learning_rate==3e-5 and all(model.lr_schedule(p)==3e-5 for p in (0.,.5,1.))
            assert all(g["lr"]==3e-5 for g in model.policy.optimizer.param_groups)
            x=np.random.default_rng(174919).normal(size=(1000,80)).astype(np.float32)
            with torch.no_grad():assert float(model.prior_kl(torch.as_tensor(x)))==0.
            v16.v15.export(model,output/"initial_policy.npz",BASIS_KIND)
            policy=v16.v15.NumpyPolicyRoute80(output/"initial_policy.npz",expected_kind=BASIS_KIND)
            error=float(np.max(np.abs(model.predict(x,deterministic=True)[0]-policy.predict(x))))
            assert error<1e-5
            assert all(torch.equal(value,initial[name]) for name,value in model.policy.state_dict().items())
            assert not model.policy.optimizer.state and all(p.grad is None for p in model.policy.parameters())
            v16.assert_hyperparameters(model,False)
            rows[arm]=dict(mean_actor_trainable_elements=27396,mean_actor_trainable_tensors=6,
                warm_critic_tensors=6,PPO_trainable_tensors=12,log_std_frozen=True,
                initial_current_prior_KL=0.,Torch_NumPy_parity1000=error,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
                actual_optimizer_steps=0,gradient_log_bytes=(output/"actual_gradient_norms.jsonl").stat().st_size,
                initial_policy_sha256=sha(output/"initial_policy.npz"))
        finally:guard.close()
    record=v16.baseline();assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS_STATIC_NO_ENV_OR_UPDATES",utc=utc(),source_sha256=sources,arms=rows,
        initial_sha256=INITIAL_SHA,pure_reward_fixtures=fixtures,inherited_base_reward_fixtures=base_fixtures,
        prospective_semantic_budget=PROBE_BUDGET,
        reused_baseline=dict(path=BASELINE.relative_to(ROOT).as_posix(),sha256=sha(BASELINE),episodes=len(record["episodes"])),
        environment_constructions=0,physical_steps=0,backward_calls=0,optimizer_steps=0,new_baseline_episodes=0,
        policy_quality_accepted=False,universal_metric_alignment_claimed=False,
        known_gap="Recorded84002 C cost can slightly prefer worse geometry; not an RMSE equivalence or guarantee")
    write_json(args.output/"static_receipt.json",report);assert not args.static_receipt.exists()
    write_json(args.static_receipt,report)
    print(json.dumps(dict(status=report["status"],receipt=str(args.static_receipt),sha256=sha(args.static_receipt))),flush=True)


def verify_triplet(args,sources):
    plan=require_physical_plan(args,sources)
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/"prospective_plan.json",plan)
    arms={arm:train_one(args,sources,arm,True) for arm in ARMS}
    from scripts.run_joint_path_damping_triplet_v19 import warmup_equivalence,first_rollout_equivalence
    warm,first=warmup_equivalence(args.output),first_rollout_equivalence(args.output)
    replay={}
    for arm in ARMS:
        replay[arm]={}
        for name in ("warm_trace.npz","ppo_trace.npz"):
            with np.load(args.output/arm/name,allow_pickle=False) as trace:
                actual,base,adjustment=trace["reward"],trace["metric_reward"],trace["damping_reward_adjustment"]
                np.testing.assert_allclose(actual,trace["damping_effective_reward"],rtol=1e-6,atol=2e-6)
                np.testing.assert_allclose(actual,base+adjustment,rtol=1e-6,atol=2e-6)
                assert np.all(~trace["TimeLimit.truncated"])
                eligible=trace["damping_eligible_interval"].astype(bool)
                square=trace["damping_post_velocity_perp_sq"] if arm=="damping" else trace["damping_normal_target_residual_sq"]
                expected=np.where(eligible & (arm!="metric_cost"),-4.*trace["dt"]*square,0.)
                np.testing.assert_allclose(adjustment,expected,rtol=1e-12,atol=1e-12)
                replay[arm][name]=dict(rows=len(actual),eligible_intervals=int(eligible.sum()),
                    max_reward_sum_error=float(np.max(np.abs(actual-base-adjustment))),
                    added_penalty_total=float(-adjustment.sum()),extra_physical_steps=0)
    assert source_hashes()==sources
    report=dict(schema=SCHEMA,status="PASS",utc=utc(),source_sha256=sources,arms=arms,
        prospective_plan_sha256=sha(args.physical_plan),total_training_physical_steps=768,total_physical_steps=768,
        additional_nontraining_physical_steps=0,total_optimizer_steps=dict(critic=12,PPO=12),
        warmup_actor7_physics_basecost_exact=warm,first_PPO_actor_physics_exact=first,actual_reward_ledger_replay=replay,
        common_gamma=1.,common_true_finite_training_terminals=True,warm_learning_rate=3e-4,PPO_learning_rate=3e-5,
        new_baseline_episodes=0,actual_pilot_executed=False,full_hover_invariance_claimed=False,
        actual_gradient_capture_extra_backward_or_optimizer=0,universal_metric_alignment_claimed=False)
    write_json(args.output/"semantic_receipt.json",report);assert not args.semantic_receipt.exists()
    write_json(args.semantic_receipt,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=ARMS)
    parser.add_argument("--initial",type=Path,default=INITIAL)
    parser.add_argument("--baseline",type=Path,default=BASELINE)
    parser.add_argument("--retention-reference",type=Path,default=v16.v15.frozen.RETENTION_REFERENCE)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=9611);parser.add_argument("--warmup-seed",type=int,default=9511)
    parser.add_argument("--steps",type=int,default=262144);parser.add_argument("--warmup-steps",type=int,default=131072)
    parser.add_argument("--eval-every",type=int,default=131072);parser.add_argument("--research-plan",type=Path)
    parser.add_argument("--static-check",action="store_true");parser.add_argument("--verify",action="store_true")
    parser.add_argument("--physical-plan",type=Path,default=ROOT/"docs/joint_path_damping_v19/semantic_plan.DRAFT.json")
    parser.add_argument("--static-receipt",type=Path,default=ROOT/"audit/joint_path_damping_static_v19_root01.json")
    parser.add_argument("--semantic-receipt",type=Path,default=ROOT/"audit/joint_path_damping_semantics_v19_root01.json")
    args=parser.parse_args()
    for name in ("initial","baseline","retention_reference","output","research_plan","physical_plan","static_receipt","semantic_receipt"):
        value=getattr(args,name)
        if value is not None:setattr(args,name,(value if value.is_absolute() else ROOT/value).resolve())
    if args.output.exists():parser.error("Fresh output required")
    if args.static_check and args.verify:parser.error("One preparation mode required")
    if (args.steps,args.warmup_steps,args.eval_every,args.seed,args.warmup_seed)!=(262144,131072,131072,9611,9511):parser.error("Fixed pilot budgets/seeds only")
    assert args.initial==INITIAL.resolve() and sha(INITIAL)==INITIAL_SHA
    assert args.baseline==BASELINE.resolve()
    assert args.retention_reference==v16.v15.frozen.RETENTION_REFERENCE.resolve() and sha(args.retention_reference)==v16.v15.frozen.RETENTION_SHA256
    torch.set_num_threads(1);sources=source_hashes()
    if args.static_check:static_check(args,sources);return
    assert_preparation_frozen(sources)
    if args.verify:verify_triplet(args,sources);return
    if args.variant is None or args.output.name!=args.variant:parser.error("Canonical damping objective arm/output required")
    for flag in ("--initial","--baseline","--retention-reference"):
        if flag not in sys.argv:parser.error("Explicit binding required: "+flag)
    require_bound_preflight(args);train_one(args,sources,args.variant)


if __name__=="__main__":main()

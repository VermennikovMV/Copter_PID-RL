"""Independent completed v16 audit using frozen v15 numerical kernels.

Normal function composition only: no producer judges/encoder imports, global
patches, environment construction, additional evaluation, or optimization.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit import audit_joint_route_basis_pilot_v15 as v

ARMS=("w8_control","metric_cost")
PAIR=ROOT/"artifacts/refinement/joint_v2/metric_cost_pilot_v16_111"
BASE=ROOT/"artifacts/refinement/joint_v2/route_basis_v15_baseline111/velocity_residual/baseline.json"
OUTPUT=ROOT/"audit/joint_metric_cost_pilot_v16_root01.json"
KIND="velocity_residual"


def warm_readback(folder,arm):
    np=v.np
    path=folder/"warm_trace.npz"
    v.file_hash(path,v.sha(path),arm+"/warm rows")
    with np.load(path,allow_pickle=False) as z: t={k:z[k].copy() for k in z.files}
    n=len(t["time"]);v.check(n==131072,arm+"/actual warm131072 rows")
    v.check(t["observation"].shape==(n,80) and str(t["basis_kind"])==KIND,arm+"/warm80 shape and kind")
    wanted=t["parent_reward"] if arm=="w8_control" else t["metric_reward"]
    reward_error=float(np.max(np.abs(t["reward"]-wanted)))
    v.check(np.allclose(t["reward"],wanted,rtol=1e-6,atol=2e-6),arm+"/actual returned reward")
    route=(t["cost_rate"]>0)|t["active_PRE_leg"]|t["completed_before"]|(t["terminal_penalty"]>0)
    expected=np.where(route,-t["dt"]*t["cost_rate"]-t["terminal_penalty"],t["parent_reward"])
    v.check(np.allclose(t["metric_reward"],expected,rtol=0,atol=1e-10),arm+"/metric accounting and hover parent identity")
    d2=np.zeros(n)
    for j in range(1,5):
        m=t["active_PRE_leg"]&(t["judged_goal_index"]==j)
        a,b=v.POINTS[j-1],v.POINTS[j];leg=b-a
        q=np.clip((t["post_position"][m]-a)@leg/(leg@leg),0,1)
        residual=t["post_position"][m]-(a+q[:,None]*leg)
        d2[m]=np.sum(residual**2,axis=1)
    finite_error=float(np.max(np.abs(d2-t["finite_distance_sq"])))
    v.check(finite_error<1e-10,arm+"/independent warm PRE finite geometry")
    action_delta=t["action_clipped"]-t["previous_action"]
    known=1+80*d2+.003*np.sum(t["post_body_rates"]**2,axis=1)+.008*np.sum(action_delta**2,axis=1)
    running=route&~t["completed_before"]
    v.check(np.all(t["cost_rate"][running]>=known[running]-1e-9),arm+"/nonnegative yaw-tilt cost remainder")
    v.check(np.all(t["cost_rate"][route&t["completed_before"]]==0),arm+"/postcompletion hold running cost zero")
    v.check(np.array_equal(t["action_clipped"],np.clip(t["action_raw"],-1,1)),arm+"/actual clipped applied actions")
    v.check(np.all(~t["TimeLimit.truncated"]),arm+"/actual no TimeLimit bootstrap")
    deadline=t["finite_task_deadline"]
    v.check(np.all(t["learning_terminated"][deadline]) and np.all(~t["learning_truncated"][deadline]),arm+"/actual finite-task deadline flags")
    logs=[json.loads(line) for line in (folder/"curriculum_events.jsonl").read_text().splitlines()]
    endings={(int(e["environment_steps"]),int(e["vector_env_index"])):e["summary"] for e in logs if e["stage"]=="critic" and e["event"]=="episode_complete"}
    unfinished=0
    for i in np.flatnonzero(deadline|t["failed"]):
        summary=endings[((int(i)//16+1)*16,int(t["vector_env_index"][i]))]
        v.check(bool(summary["failed"])==bool(t["failed"][i]),arm+"/deadline is not physical failure")
        is_unfinished=summary["mission"]=="route" and bool(deadline[i]) and int(summary["goals_reached"])<5
        unfinished+=int(is_unfinished)
        penalty=400. if summary["mission"]=="route" and (bool(summary["failed"]) or is_unfinished) else 0.
        v.check(t["terminal_penalty"][i]==penalty,arm+"/actual failure-or-unfinished400 once")
    # Recover exact total stability cost from independently expanded old W8
    # reward on interior intervals. Post rotations are not archived in warm IO.
    i=np.arange(16,n);p=i-16
    keep=(~t["done"][p])&(~t["done"][i])&np.all(t["target"][i]==t["next_target"][i],axis=1)&(~t["completed_before"][i])
    keep&=np.abs(t["time"][i]-t["time"][p]-t["dt"][i])<1e-8
    i,p=i[keep],p[keep]
    err=np.linalg.norm(t["post_position"][i]-t["judged_target"][i],axis=1)
    before=np.linalg.norm(t["post_position"][p]-t["judged_target"][i],axis=1)
    core=t["dt"][i]*(1.5*np.exp(-(err/.2)**2)-.35*err**2-.04*np.sum(t["post_velocity"][i]**2,axis=1))+.8*(before-err)-t["dt"][i]*8*d2[i]
    stability=core-t["parent_reward"][i]
    m=route[i]
    predicted=-t["dt"][i[m]]*(1+80*d2[i[m]])-stability[m]
    expanded_error=float(np.max(np.abs(predicted-t["metric_reward"][i[m]]))) if np.any(m) else 0.
    v.check(expanded_error<1e-8,arm+"/expanded W8-to-cost interior ledger")
    return t,dict(rows=n,actual_deadline_rows=int(deadline.sum()),actual_unfinished_route_deadlines=unfinished,
        reward_float32_error=reward_error,independent_finite_square_error=finite_error,
        expanded_parent_to_metric_cost_rows=int(m.sum()),expanded_parent_to_metric_cost_error=expanded_error,
        stability_scope="Total stability recovered from exact expanded old W8 reward on nonterminal interior intervals; post rotation absent from warm archive. Whole initial24 cost ledger separately replayed yaw/tilt explicitly.")


def audit(output):
    paired=v.read(PAIR/"paired_complete.json")
    v.check(paired["status"]=="complete","both actual arms completed")
    if paired["status"]!="complete": raise RuntimeError("Completed pair required")
    for arm in ARMS:
        if v.read(PAIR/arm/"complete.json")["status"]!="complete": raise RuntimeError("Completed arm required")
    if output.exists(): raise FileExistsError("Preserve previous final audit")
    v._load_numeric_for_completed_audit()
    initial_data,initial_policy,initial_optimizer=v.archive_for_completed_audit(v.INITIAL)
    v.file_hash(v.INITIAL,v.INITIAL_SHA,"initial")
    v.check(len(initial_optimizer["state"])==0,"initializer Adam0")
    baseline=v.read(BASE);np_initial=v.load_export(BASE.parent/"initial_policy.npz")
    v.future_export_parity(np_initial,initial_policy,np_initial,KIND,"shared initial80")
    base_episodes=[v.episode80(e,np_initial,"shared reused baseline/"+str(i)) for i,e in enumerate(baseline["episodes"])]
    v.same(baseline["aggregate"],v.aggregate(base_episodes),"shared24 initial aggregate")
    retention=v.read(v.RETENTION);v.file_hash(v.RETENTION,v.RETENTION_SHA,"original Gamma reference")
    records={};selected={};evidence={};warms={};warm_states={}
    for arm in ARMS:
        folder=PAIR/arm;complete=v.read(folder/"complete.json");config=v.read(folder/"config.json");warm=v.read(folder/"warmup.json")
        v.same(config,dict(schema="joint-finite-metric-cost-ppo-v16.0",variant=arm,basis_kind=KIND,common_gamma=1.,training_deadlines_true_terminal=True,new_baseline_episodes=0),arm+"/config")
        v.check(config["reused_baseline"]["path"]==BASE.relative_to(ROOT).as_posix(),arm+"/same reused24 baseline")
        v.file_hash(BASE,config["reused_baseline"]["sha256"],arm+"/reused baseline bytes")
        plan_path=ROOT/"docs/joint_metric_cost_plans_v16"/(arm+".bound.json")
        v.file_hash(plan_path,config["research_plan_sha256"],arm+"/official bound plan")
        manifest=v.read(folder/"manifest.json");v.file_hash(folder/"source_snapshot.zip",manifest["source_snapshot_sha256"],arm+"/source snapshot")
        with zipfile.ZipFile(folder/"source_snapshot.zip") as z:
            for name,digest in config["source_sha256"].items():
                v.file_hash(ROOT/name,digest,arm+"/source/"+name)
                v.check(hashlib.sha256(z.read(name)).hexdigest()==digest,arm+"/archived source/"+name)
        v.same(v.read(folder/"optimizer_reset.json"),dict(initializer_states=0,after_clear=0,reset=[0,0],gamma=1.,rollout_buffer_gamma=1.,source13to0_reset_belongs_v14=True),arm+"/fresh actual reset")
        wd,wp,wo=v.archive_for_completed_audit(folder/"critic_warmed_model.zip")
        v.check(wd["gamma"]==1. and wd["num_timesteps"]==131072 and wd["_n_updates"]-initial_data["_n_updates"]==128,arm+"/actual warm counters gamma")
        masks={"warm":v.mask_state_proof(wp,initial_policy,wo,0,arm+"/warm")};warm_states[arm]=wp
        v.check_guard(warm,0,arm+"/warm guard");v.check_guard(complete["optimizer"],2048,arm+"/PPO guard")
        v.same(complete["warmup"],warm,arm+"/complete warm agreement")
        v.same(complete,dict(critic_transitions=131072,ppo_transitions=262144,source_unchanged=True,reused_baseline_new_episodes=0,actual_Gazebo80_acceptance=False,automatic_extension_authorized=False),arm+"/actual declared budget")
        v.same(v.read(folder/"guard_summary.json"),complete["optimizer"],arm+"/guard summary")
        for report in (warm,complete["optimizer"]):v.same(report,dict(gamma=1.,rollout_buffer_gamma=1.,common_training_deadlines_true_finite_task_terminals=True,warm_critic_Adam_equality_required=False),arm+"/actual gamma and deadline configuration")
        logs_path=folder/"on_policy_prior_kl.jsonl";v.file_hash(logs_path,v.sha(logs_path),arm+"/onpolicy log")
        logs=[json.loads(line) for line in logs_path.read_text().splitlines()]
        v.check(len(logs)==32,arm+"/32 genuine PPO rollouts")
        for i,row in enumerate(logs):v.same(row,dict(timesteps=(i+1)*8192,completed_epochs=8,optimizer_steps=64,minibatch_sample_uses=65536,cumulative_kl_gradient_steps=(i+1)*64,cumulative_kl_minibatch_sample_uses=(i+1)*65536,weight=.1,prior_sha256=v.INITIAL_SHA),arm+"/KL/"+str(i))
        training=v.curriculum(folder,"w8_control",complete)
        for name,digest in complete["output_model_sha256"].items():v.file_hash(folder/name,digest,arm+"/model/"+name)
        warms[arm],warm_proof=warm_readback(folder,arm)
        records[arm]={};candidates={}
        for label,steps,epochs,updates in zip(v.LABELS,(131072,262144,262144),(120,248,256),(960,1984,2048)):
            context=arm+"/"+label;r=v.read(folder/(label+"_validation.json"));data,policy,opt=v.archive_for_completed_audit(folder/(label+".zip"))
            v.same(r,dict(timesteps=steps,completed_ppo_epochs=epochs,ppo_kl_gradient_steps=updates,actor_observations=80,basis_kind=KIND,checkpoint_timing="After final update" if label=="final" else "Before current rollout update",selection_independent_of_pair_gains=True),context+"/timing")
            v.check(data["gamma"]==1. and data["num_timesteps"]==steps and data["_n_updates"]-wd["_n_updates"]==epochs and data["bc_kl_gradient_steps"]==data["ppo_gradient_steps"]==updates,context+"/saved actual counters/gamma")
            masks[label]=v.mask_state_proof(policy,initial_policy,opt,updates,context)
            npz=v.load_export(folder/(label+".npz"));parity=v.future_export_parity(npz,policy,np_initial,KIND,context)
            v.MAXIMA["tensor_numpy_error"]=max(v.MAXIMA["tensor_numpy_error"],parity["maximum_Torch_NPZ_error"])
            v.file_hash(folder/(label+".npz"),r["policy_sha256"],context+"/actual evaluated bytes")
            raw=v.read(folder/"validation"/label/"evaluation.json")
            for key in ("episodes","aggregate","policy_sha256","eval_source_sha256"):v.same(r[key],raw[key],context+"/raw/"+key)
            v.same(r["eval_source_sha256"],baseline["eval_source_sha256"],context+"/EXACT frozen v15 evaluator closure")
            episodes=[v.episode80(e,npz,context+"/"+str(i)) for i,e in enumerate(r["episodes"])]
            agg=v.aggregate(episodes);v.same(r["aggregate"],agg,context+"/independent metrics")
            score=v.selector(agg,retention["aggregate"]);v.same(r,score,context+"/independent fixed selector")
            records[arm][label]=dict(aggregate=agg,episodes=episodes,**score)
            for e in r["episodes"]:
                if e["mission"]=="hover":
                    b=next(x for x in baseline["episodes"] if (x["mission"],x["scenario"],x["eval_seed"])==(e["mission"],e["scenario"],e["eval_seed"]))
                    v.check(e["hashes"]["trace.csv"]==b["hashes"]["trace.csv"],context+"/isolated80 hover exact shared baseline")
            candidates[label]=dict(objective=score["objective"],hard_guards_pass=score["fixed_development_guards_pass"],failed_constraints=[c for c in score["constraints"] if not c["pass"]],aggregate=agg)
            print(json.dumps(dict(event="AUDITED_V16_CANDIDATE",arm=arm,label=label,integrity_failures=len(v.FAILURES))),flush=True)
        selected[arm]=min(v.LABELS,key=lambda k:tuple(records[arm][k]["objective"]))
        chosen=records[arm][selected[arm]]
        v.same(v.read(folder/"selection.json"),dict(checkpoint=selected[arm],objective=chosen["objective"],aggregate=chosen["aggregate"]),arm+"/independent selection")
        v.check(v.sha(folder/"best_policy.npz")==v.sha(folder/(selected[arm]+".npz")),arm+"/selected NPZ exact")
        unused,best,unused2=v.archive_for_completed_audit(folder/"best_model.zip");unused,actual,unused2=v.archive_for_completed_audit(folder/(selected[arm]+".zip"))
        v.exact_state(best,actual,arm+"/selected all13 exact")
        v.same(complete,dict(selected_policy_sha256=v.sha(folder/"best_policy.npz"),selected_metrics=chosen["aggregate"],fixed_development_guards_pass=chosen["fixed_development_guards_pass"],diagnostic_candidate=not chosen["fixed_development_guards_pass"],all_candidates_failed_development_guards=not any(x["fixed_development_guards_pass"] for x in records[arm].values())),arm+"/complete independent quality")
        evidence[arm]=dict(selected=selected[arm],selected_policy_sha256=v.sha(folder/"best_policy.npz"),masks=masks,warm_objective_ledger=warm_proof,curriculum=training,candidates=candidates)
    for name in warm_states[ARMS[0]]:
        if v.actor(name):v.exact_state(warm_states[ARMS[0]][name],warm_states[ARMS[1]][name],"warm actor A/B exact/"+name)
    for key in warms[ARMS[0]]:
        if key!="reward":v.check(v.np.array_equal(warms[ARMS[0]][key],warms[ARMS[1]][key]),"all131072 warm physical/action/input/objective array A/B/"+key)
    v.check(not v.np.array_equal(warms[ARMS[0]]["reward"],warms[ARMS[1]]["reward"]),"warm reward actually differs")
    e0aa=v.read(ROOT/"artifacts/refinement/joint_v2/corner_pilot111/w8_control/final_validation.json")
    aliases={old:records[new] for old,new in zip(v.ARMS,ARMS)};picks={old:selected[new] for old,new in zip(v.ARMS,ARMS)}
    relative=v.prospective_selected_eligibility(aliases,picks,e0aa)
    pair=v.read(PAIR/"pair_comparison.json");v.file_hash(PAIR/"pair_comparison.json",paired["comparison_sha256"],"sealed pair comparison")
    v.same(pair["selected_labels"],dict(A=selected[ARMS[0]],B=selected[ARMS[1]]),"independent selected labels")
    v.check_pair_comparison(pair["selected_B_vs_selected_A"],relative["selected_B_vs_selected_A"],"selectedB versus selectedA")
    v.check_pair_comparison(pair["selected_B_vs_same_phase_A"],relative["selected_B_vs_A_same_phase"],"selectedB versus samephaseA")
    for label in v.LABELS:v.check_pair_comparison(pair["all_three_matched_diagnostics"][label],relative["all3_matched_diagnostic"][label],"matched/"+label)
    for row in pair["selected_B_no_worse_e0aa_mean_geometry"]["checks"]:v.same(row["passed"],relative["selected_B_vs_e0aa_mean_finite_and_corner_nonregression"][row["scenario"]+"/"+row["metric"]],"old e0aa nonregression")
    v.same(pair["eligible_for_root_review_only"],relative["eligible_for_root_resource_review_only"],"overall relative eligibility")
    v.check(v.OBSERVATION_PROOF["rows"]==294000,"144new candidate+24REUSED baseline=168 coverage/294000rows")
    for name,digest in list(v.FILES.items()):v.check(v.sha(ROOT/name)==digest,"all audit inputs unchanged/"+name)
    result=dict(schema="joint-metric-cost-pilot-independent-audit-v16.0",status="PASS" if not v.FAILURES else "FAIL",checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=v.sha(__file__),helper_source_sha256=v.sha(Path(v.__file__)),checks=v.CHECKS,failures=v.FAILURES,input_sha256=v.FILES,
        candidate_new_episodes=144,baseline_REUSED_episodes=24,new_baseline_episodes=0,total_coverage_episodes=168,actor_input_rows=v.OBSERVATION_PROOF["rows"],maximum_errors=v.MAXIMA,observation_proof=v.OBSERVATION_PROOF,arms=evidence,selected_pair_eligibility=relative,
        common_changes="gamma1 and true finite-task training deadlines/zero bootstrap; native physical/evaluation horizons unchanged",between_arm_factor="Whole route reward bundle",warm_actor_physics_exact_but_reward_critic_Adam_may_differ=True,
        historical_scope="Raw saved tensor/moments/counters and warm131072rows each are replayed; PPO perstep reward/gradient/transientbuffer history not fully archived, immutable source/runtime guards and episode logs provide that scope.",
        policy_quality="PASS_FAST_DEVELOPMENT_ONLY" if relative["B_absolute_hard_guards_pass"] else "FAIL_HARD_DEVELOPMENT_GUARDS",actual_Gazebo80_acceptance="NOT_EVALUATED",automatic_extension_authorized=False,new_environment_or_optimizer_or_evaluation_steps=0)
    path=output if not v.FAILURES else output.with_name(output.stem+"_review_error_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    with path.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],checks=v.CHECKS,failures=v.FAILURES,receipt=str(path),receipt_sha256=v.sha(path),selected=selected,relative_eligible=relative["eligible_for_root_resource_review_only"])),flush=True)
    return 0 if not v.FAILURES else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--audit-completed",action="store_true");parser.add_argument("--output",type=Path,default=OUTPUT);args=parser.parse_args()
    if not args.audit_completed:print(json.dumps(dict(status="PREPARATION_ONLY_NO_NUMERIC_REPLAY",new_environment_steps=0,new_optimizer_steps=0,final_receipt_created=False)));raise SystemExit(0)
    try:raise SystemExit(audit(args.output))
    except Exception as exc:
        path=args.output.with_name(args.output.stem+"_review_error_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
        with path.open("x",encoding="utf-8") as f:json.dump(dict(status="AUDIT_REVIEW_ERROR",error=repr(exc),traceback=traceback.format_exc(),failures=v.FAILURES,input_sha256=v.FILES,source_sha256=v.sha(__file__)),f,indent=2);f.write("\n")
        print(json.dumps(dict(status="AUDIT_REVIEW_ERROR",path=str(path),error=repr(exc))),flush=True);raise

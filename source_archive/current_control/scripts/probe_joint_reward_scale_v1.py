"""Read-only reward ledger for existing native conditioning probe recordings.

Creates one fresh JSON receipt. Does not instantiate a simulator, call policy
inference, change a policy, run training, or relabel the observed probe return
as an on-policy PPO return. Velocity conditioning kappa differs from PPO gamma.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys,math
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from audit.verify_joint_metrics_v2 import distances_to_leg
from quadrl.protocol import ROUTE_POINTS
from scripts.bc_trust_region_probe_v1 import BCTrustPPO
import torch
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'audit/joint_reward_scale_diagnosis_v1.json')
args=parser.parse_args()
output=args.output if args.output.is_absolute() else root/args.output
if output.exists():raise FileExistsError('Preserve existing reward-scale evidence: '+str(output))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
model=BCTrustPPO.load(root/'artifacts/refinement/joint_v2/training_pair111/geometry/best_model.zip',device='cpu')
discount=model.gamma
group=model.policy.optimizer.param_groups[0]
settings={field:getattr(model,field) for field in ('gamma','gae_lambda','n_steps','batch_size','n_epochs','ent_coef','vf_coef','max_grad_norm','target_kl','normalize_advantage','learning_rate')}
settings.update(clip_range=float(model.clip_range(1.)),optimizer=type(model.policy.optimizer).__name__,optimizer_lr=group['lr'],optimizer_betas=group['betas'],optimizer_eps=group['eps'],optimizer_weight_decay=group['weight_decay'],optimizer_parameter_state_count=len(model.policy.optimizer.state),optimizer_step_range=[min(float(s['step']) for s in model.policy.optimizer.state.values()),max(float(s['step']) for s in model.policy.optimizer.state.values())],actor_std=torch.exp(model.policy.log_std).detach().numpy().tolist(),PPO_discount_half_life_s=.02*math.log(.5)/math.log(discount),GAE_trace_half_life_s=.02*math.log(.5)/math.log(discount*model.gae_lambda))
print('ACTUAL_SAVED_PPO_SETTINGS',json.dumps(settings))
base=root/'artifacts/refinement/joint_v2/velocity_conditioning_probe_v1'
records=[]
cell_means=[]
input_hashes={}
for condition in ('gamma_1p00','gamma_1p15'):
    evaluation=json.loads((base/condition/'evaluation/evaluation.json').read_text())
    diagnostic=json.loads((base/condition/'diagnostic_result.json').read_text())
    assert sha(base/condition/'evaluation/evaluation.json')==diagnostic['evaluation_sha256']
    assert sha(base/condition/'policy.npz')==diagnostic['policy_sha256']
    input_hashes[str((base/condition/'evaluation/evaluation.json').relative_to(root))]=diagnostic['evaluation_sha256']
    assert len(evaluation['episodes'])==24
    for episode in evaluation['episodes']:
        assert episode['model_family']=='fast_native_shaper_v3'
        for name,digest in episode['hashes'].items():
            path=Path(episode['trace_path']).parent/name
            assert sha(path)==digest
        if episode['mission']!='route':continue
        rows=list(csv.DictReader(Path(episode['trace_path']).open()))
        times=np.asarray([float(r['time']) for r in rows])
        completion=episode['metrics']['route_completion_time_s']
        assert completion is not None
        count=int(np.flatnonzero(abs(times-completion)<1e-8)[0])+1
        assert count<len(rows)
        t=times[:count];dt=np.diff(np.r_[0.,t])
        position=np.asarray([[float(r[k]) for k in ('x','y','z')] for r in rows[:count]])
        velocity=np.asarray([[float(r[k]) for k in ('vx','vy','vz')] for r in rows[:count]])
        target=np.asarray([[float(r[k]) for k in ('target_x','target_y','target_z')] for r in rows[:count]])
        action=np.asarray([[float(r['action'+str(i)]) for i in range(4)] for r in rows[:count]])
        error=np.asarray([float(r['error']) for r in rows[:count]])
        yaw=np.asarray([float(r['yaw_error']) for r in rows[:count]])
        tilt=np.asarray([float(r['tilt_cos']) for r in rows[:count]])
        with np.load(episode['actor_io_path']) as archive:
            before_position=archive['position'][:count].copy()
            post_rates=archive['body_rates'][1:count+1].copy()
            assert np.array_equal(archive['target'][:count],target)
        event_times=np.asarray([e['time_s'] for e in episode['mission_events']])
        arrival=np.any(abs(t[:,None]-event_times[None,:])<1e-8,axis=1)
        completed=abs(t-completion)<1e-8
        previous_action=np.vstack([np.zeros(4),action[:-1]])
        components=dict(precision=dt*1.5*np.exp(-(error/.2)**2),goal_quadratic=-dt*.35*error**2,speed=-dt*.04*np.sum(velocity**2,axis=1),heading=-dt*10*(1-np.cos(yaw)),tilt=-dt*.05*(1-tilt),body_rates=-dt*.003*np.sum(post_rates**2,axis=1),smooth=-dt*.008*np.sum((action-previous_action)**2,axis=1),progress=.8*(np.linalg.norm(before_position-target,axis=1)-error),arrival=10.*arrival,completion=20.*completed)
        logged=np.asarray([float(r['reward']) for r in rows[:count]])
        reconstructed=sum(components.values())
        residual=float(np.max(np.abs(reconstructed-logged)))
        assert residual<1e-8,residual
        active=np.minimum(np.searchsorted(event_times,t-1e-8,side='left'),4)
        distance=np.zeros(count)
        for goal_index in range(1,5):
            mask=active==goal_index
            distance[mask]=distances_to_leg(position[mask],ROUTE_POINTS[goal_index-1],ROUTE_POINTS[goal_index])[0]
        components['W8_counterfactual']=-dt*8*distance**2
        weights=discount**np.arange(count)
        entry=[]
        for event in episode['mission_events']:
            if event['goal_index'] not in (1,2,3):continue
            index=int(np.flatnonzero(abs(t-event['time_s'])<1e-8)[0])
            entry.append(dict(goal=event['goal_index'],time=event['time_s'],distance=float(error[index]),speed=float(np.linalg.norm(velocity[index])),W8_step=float(components['W8_counterfactual'][index]),arrival_bonus=10.))
        result=dict(condition=condition,scenario=episode['scenario'],seed=episode['eval_seed'],completion_s=completion,segment_rmse_m=episode['geometry']['segment_rmse_m'],corner_peak_m=episode['geometry']['mean_corner_post_peak_m'],absolute_segment_peak_m=episode['geometry']['active_segment_peak_m'],undiscounted={key:float(values.sum()) for key,values in components.items()},discounted={key:float(weights@values) for key,values in components.items()},logged_base_sum=float(logged.sum()),reconstruction_max_error=residual,corner_entries=entry)
        result['offline_W8_trace_return']=float(sum(result['undiscounted'].values()))
        result['offline_W8_discounted_trace_return']=float(sum(result['discounted'].values()))
        records.append(result)
    for scenario in ('nominal','force_strong','gust'):
        group=[r for r in records if r['condition']==condition and r['scenario']==scenario]
        mean=dict(condition=condition,scenario=scenario,n=len(group),completion_s=float(np.mean([r['completion_s'] for r in group])),finite=float(np.mean([r['segment_rmse_m'] for r in group])),undiscounted={key:float(np.mean([r['undiscounted'][key] for r in group])) for key in group[0]['undiscounted']},discounted={key:float(np.mean([r['discounted'][key] for r in group])) for key in group[0]['discounted']},offline_W8_trace_return=float(np.mean([r['offline_W8_trace_return'] for r in group])),offline_W8_discounted_trace_return=float(np.mean([r['offline_W8_discounted_trace_return'] for r in group])))
        cell_means.append(mean)
        print('CELL_MEAN',json.dumps(mean))
print('SCOPE',json.dumps(dict(existing_probe_episodes_read=48,route_ledgers=24,terminal_scope='actual measured t0 through confirmed goal4; no postcompletion hover',recorded_trace_reward='BASE reward only; offline W8 shown separately',actual_PPO_return_reconstructed=False,why_not='No stochastic training rollouts, value estimates, bootstrap/advantages or KL minibatch losses are present in these deterministic probes',max_reward_row_reconstruction_error=max(r['reconstruction_max_error'] for r in records))))
source_paths=[Path(__file__),root/'quadrl/reward.py',root/'quadrl/protocol.py',root/'quadrl/rollout.py',root/'quadrl_joint/env_v3.py',root/'scripts/train_joint_geometry_v2.py',root/'scripts/train_joint_nativeplant_v4.py',root/'scripts/bc_trust_region_probe_v1.py',root/'scripts/probe_joint_velocity_conditioning_v1.py',root/'docs/joint_applicability_contract_v2.json']
receipt=dict(schema='joint-reward-scale-diagnosis-v1',status='RECORDED_LEDGER_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Existing48 native probe episodes read;24 complete route reward ledgers, no new simulator/inference/training/policy edits',existing_probe_root=str(base.relative_to(root)),model_family='fast_native_shaper_v3',velocity_conditioning_kappa=[1.,1.15],PPO_discount_gamma=discount,actual_saved_PPO_settings=settings,input_evaluation_sha256=input_hashes,source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths},cell_means=cell_means,route_records=records,max_base_reward_row_reconstruction_error=max(r['reconstruction_max_error'] for r in records),ledger_semantics=dict(recorded_trace_reward='Frozen BASE precision-arrival-v3 only; native evaluator never added W8',cutoff='Actual t0 through confirmed goal4, reflecting training success termination; later40s hover excluded',post_step_rates='Next recorded pre-action actor_io body_rates; completion precedes final trace row, so every route-prefix rate is measured',offline_W8='Exact -8*dt*finite-active-segment-distance-squared using pre-update event targets, added only as counterfactual score on recorded states',discounted_score='sum gamma_PPO**row_index * reconstructed_component, not a measured training return',actual_PPO_returns_or_advantages_reconstructed=False,missing_for_on_policy_claim=['Stochastic training rollout trajectories','Critic values and terminal/bootstrap values','GAE/normalized minibatch advantages','Current-rollout prior KL minibatch losses'],causal_limit='Reward-scale comparison is not a gradient attribution; fixed70 bonuses cancel in undiscounted complete-route comparisons but their timing matters under gamma.995',KL='Weight.1 * minibatch mean KL(current Normal || frozen initial wind Normal), summed across4actions; optimizer loss, not environment reward',no_Gazebo_acceptance_claim=True))
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
print('FROZEN_RECEIPT',json.dumps(dict(path=str(output),sha256=sha(output),script_sha256=sha(Path(__file__)))))

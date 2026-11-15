"""Development probe: BC + critic warmup + KL-regularized on-policy PPO.

The sole addition to installed SB3 PPO.train is weight * mean KL(current Normal
|| fixed BC Normal), summed over four actions, on each CURRENT rollout minibatch.
No offline teacher labels or PID are used. The prior is training-only; exported
NumPy inference is the existing pure 74-observation actor. Saved ZIPs support
inference; resuming training requires configure_bc_prior() explicitly.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from quadrl.env import QuadEnv
from quadrl.policy import NumpyPolicy
from scripts.refine_precision import Selection
from scripts.train import AnchoredPPO

OUT=ROOT/'artifacts/refinement/bc_trust_v1'
INITIAL=ROOT/'artifacts/training/main_hybrid_seed111_v1/bc_only_model.zip'
ARMS={'kl01':.1,'kl10':1.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actor_name(name):
    return name.startswith(('mlp_extractor.policy_net.','action_net.')) or name=='log_std'


class BCTrustPPO(AnchoredPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['bc_prior']

    def configure_bc_prior(self,path,weight,debug_log_path=None):
        if not np.isfinite(weight) or weight<0:
            raise ValueError('Nonnegative finite prior KL weight required')
        if getattr(self,'anchor_steps',0):
            raise ValueError('Offline supervised anchors are outside this experiment')
        if not isinstance(self.action_space,gym.spaces.Box) or self.action_space.shape!=(4,):
            raise ValueError('Four continuous normalized actions required')
        _,parameters,_=load_from_zip_file(path,device=self.device)
        prior=copy.deepcopy(self.policy)
        prior.load_state_dict(parameters['policy'],strict=True)
        # The prior has no optimizer and is never registered with current Adam.
        del prior.optimizer
        prior.set_training_mode(False)
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        self.bc_prior=prior
        self.bc_prior_sha256=sha(path)
        self.bc_kl_weight=float(weight)
        self.bc_kl_enabled=True
        self.bc_kl_gradient_steps=0
        self.bc_kl_minibatch_sample_uses=0
        self.bc_kl_debug_log_path=str(debug_log_path) if debug_log_path is not None else None

    def prior_kl(self,observations):
        if getattr(self,'bc_prior',None) is None:
            raise RuntimeError('Training-only BC prior missing; configure_bc_prior before learning')
        current=self.policy.get_distribution(observations)
        with torch.no_grad():
            prior=self.bc_prior.get_distribution(observations)
        if not isinstance(current,DiagGaussianDistribution) or not isinstance(prior,DiagGaussianDistribution):
            raise TypeError('Probe is defined only for unsquashed diagonal Normal policies')
        return torch.distributions.kl_divergence(current.distribution,prior.distribution).sum(dim=-1).mean()

    def train(self):
        weight=getattr(self,'bc_kl_weight',0.)
        if not getattr(self,'bc_kl_enabled',False):
            # Critic warmup is BIT-FOR-BIT the original AnchoredPPO implementation.
            return super().train()
        if getattr(self,'anchor_steps',0):
            raise RuntimeError('KL PPO must not run offline supervised updates')
        if getattr(self,'bc_prior',None) is None:
            raise RuntimeError('Reloaded ZIP needs explicit fixed BC prior before training')
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range=self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf=self.clip_range_vf(self._current_progress_remaining)
        entropy_losses=[]
        pg_losses,value_losses=[],[]
        clip_fractions=[]
        prior_kls=[]
        continue_training=True
        updates_before=self._n_updates
        optimizer_steps=0
        sample_uses=0
        rollout_sha=hashlib.sha256(self.rollout_buffer.observations.tobytes()).hexdigest()
        # This is the installed SB3 PPO.train loop, with one marked loss addition.
        for epoch in range(self.n_epochs):
            approx_kl_divs=[]
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions=rollout_data.actions
                if isinstance(self.action_space,gym.spaces.Discrete):
                    actions=rollout_data.actions.long().flatten()
                values,log_prob,entropy=self.policy.evaluate_actions(rollout_data.observations,actions)
                values=values.flatten()
                advantages=rollout_data.advantages
                if self.normalize_advantage and len(advantages)>1:
                    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio=torch.exp(log_prob-rollout_data.old_log_prob)
                policy_loss_1=advantages*ratio
                policy_loss_2=advantages*torch.clamp(ratio,1-clip_range,1+clip_range)
                policy_loss=-torch.min(policy_loss_1,policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction=torch.mean((torch.abs(ratio-1)>clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    values_pred=values
                else:
                    values_pred=rollout_data.old_values+torch.clamp(
                        values-rollout_data.old_values,-clip_range_vf,clip_range_vf)
                value_loss=F.mse_loss(rollout_data.returns,values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss=-torch.mean(-log_prob)
                else:
                    entropy_loss=-torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss=policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                # ONLY algorithmic addition: current Normal || frozen BC Normal.
                bc_kl=self.prior_kl(rollout_data.observations)
                loss=loss+weight*bc_kl
                prior_kls.append(float(bc_kl.detach()))
                with torch.no_grad():
                    log_ratio=log_prob-rollout_data.old_log_prob
                    approx_kl_div=torch.mean((torch.exp(log_ratio)-1)-log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div>1.5*self.target_kl:
                    continue_training=False
                    if self.verbose>=1:
                        print(f'Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}')
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Preserve original JOINT clipping; KL has no direct critic grad,
                # but a changed joint norm can indirectly rescale critic gradients.
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
                optimizer_steps+=1
                sample_uses+=len(rollout_data.observations)
            self._n_updates+=1
            if not continue_training:
                break
        explained_var=explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten())
        self.logger.record('train/entropy_loss',np.mean(entropy_losses))
        self.logger.record('train/policy_gradient_loss',np.mean(pg_losses))
        self.logger.record('train/value_loss',np.mean(value_losses))
        self.logger.record('train/approx_kl',np.mean(approx_kl_divs))
        self.logger.record('train/clip_fraction',np.mean(clip_fractions))
        self.logger.record('train/loss',loss.item())
        self.logger.record('train/explained_variance',explained_var)
        if hasattr(self.policy,'log_std'):
            self.logger.record('train/std',torch.exp(self.policy.log_std).mean().item())
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip_range)
        if self.clip_range_vf is not None:
            self.logger.record('train/clip_range_vf',clip_range_vf)
        self.ppo_gradient_steps=getattr(self,'ppo_gradient_steps',0)+optimizer_steps
        self.bc_kl_gradient_steps+=optimizer_steps
        self.bc_kl_minibatch_sample_uses+=sample_uses
        self.logger.record('train/bc_prior_kl',np.mean(prior_kls))
        self.logger.record('train/bc_prior_kl_weight',weight)
        self.logger.record('train/bc_kl_gradient_steps',self.bc_kl_gradient_steps)
        self.logger.record('train/bc_kl_minibatch_sample_uses',self.bc_kl_minibatch_sample_uses)
        if self.bc_kl_debug_log_path:
            row=dict(timesteps=self.num_timesteps,completed_epochs=self._n_updates-updates_before,
                     optimizer_steps=optimizer_steps,minibatch_sample_uses=sample_uses,
                     cumulative_kl_gradient_steps=self.bc_kl_gradient_steps,
                     cumulative_kl_minibatch_sample_uses=self.bc_kl_minibatch_sample_uses,
                     mean_unweighted_prior_kl=float(np.mean(prior_kls)),weight=weight,
                     prior_sha256=self.bc_prior_sha256,
                     current_on_policy_rollout_observations_sha256=rollout_sha)
            with Path(self.bc_kl_debug_log_path).open('a') as handle:
                handle.write(json.dumps(row)+'\n')


def train(arm):
    out=OUT/arm
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    files=sorted(set([Path(__file__),ROOT/'scripts/refine_precision.py',ROOT/'scripts/train.py',
                      *ROOT.glob('quadrl/*.py'),*ROOT.glob('quadrl/*.json'),
                      *ROOT.glob('reference/*.py'),ROOT/'sim/metrics.py']))
    hashes={str(path.relative_to(ROOT)):sha(path) for path in files}
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,str(path.relative_to(ROOT)))
    vector=VecMonitor(DummyVecEnv([lambda:QuadEnv('hybrid',training=True) for _ in range(16)]))
    try:
        vector.set_attr('training_progress',1.)
        vector.seed(9171)
        model=BCTrustPPO.load(INITIAL,env=vector,device='cpu',seed=9171)
        model.anchor_steps=0
        model.configure_bc_prior(INITIAL,0.,out/'on_policy_prior_kl.jsonl')
        model.bc_kl_enabled=False
        initial=copy.deepcopy(model.policy.state_dict())
        prior_initial=copy.deepcopy(model.bc_prior.state_dict())
        np.savez(out/'initial_parameters.npz',**{
            key:value.detach().cpu().numpy().copy() for key,value in initial.items()})
        config=dict(arm=arm,bc_prior_kl_weight=ARMS[arm],
                    algorithm='BC + critic warmup + KL-regularized on-policy PPO',
                    initial=str(INITIAL),initial_sha256=sha(INITIAL),prior_sha256=sha(INITIAL),
                    prior_direction='current Normal || fixed BC Normal',
                    prior_kl_reduction='sum four action dimensions then minibatch mean; pre-action-clipping Normal',
                    prior_inputs='Only current rollout minibatch observations; no offline label dataset',
                    warmup_seed=9171,ppo_seed=9111,envs=16,warmup_steps=131072,ppo_steps=1048576,
                    reward='Unchanged precision-arrival-v3',domain='Unchanged full original DR',
                    source_hashes=hashes,
                    installed_sb3_ppo_train_sha256=hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest(),
                    validation_seeds=list(range(82001,82009)),reserved_confirmation_seeds=[92001,92002,92003],
                    joint_gradient_clipping='Unchanged SB3 joint clipping can indirectly affect critic scaling',
                    runtime='Pure74-observation NumPy actor; BC prior excluded from inference')
        (out/'config.json').write_text(json.dumps(config,indent=2))
        for name,parameter in model.policy.named_parameters():
            if actor_name(name):parameter.requires_grad_(False)
        model.set_logger(configure(str(out/'warmup'),['csv']))
        started=time.monotonic()
        before=model._n_updates
        model.learn(131072,reset_num_timesteps=True,progress_bar=False)
        changed=[name for name,value in model.policy.state_dict().items() if not torch.equal(value,initial[name])]
        if not changed or any(actor_name(name) for name in changed):
            raise AssertionError('Critic warmup must change only the critic')
        warmup_record=dict(environment_steps=model.num_timesteps,
                           completed_critic_only_epochs=model._n_updates-before,
                           frozen_actor_bit_identical=True,changed_parameters=changed,
                           prior_kl_weight_during_warmup=0.,wall_seconds=time.monotonic()-started)
        (out/'warmup.json').write_text(json.dumps(warmup_record,indent=2))
        model.save(out/'pretrained_before_actor_ppo')
        for parameter in model.policy.parameters():parameter.requires_grad_(True)
        np.savez(out/'ppo_start_parameters.npz',**{
            name:value.detach().cpu().numpy().copy() for name,value in model.policy.state_dict().items()})
        model.bc_kl_weight=ARMS[arm]
        model.bc_kl_enabled=True
        model.set_random_seed(9111)
        model.set_logger(configure(str(out),['csv']))
        before=model._n_updates
        selection=Selection(out,524288,before)
        model.learn(1048576,reset_num_timesteps=True,callback=selection,progress_bar=False)
        selection.evaluate('final')
        probe=np.random.default_rng(17491).normal(size=(200,74)).astype(np.float32)
        parity=float(np.max(np.abs(model.predict(probe,deterministic=True)[0]
                                   -NumpyPolicy(out/'final.npz').predict(probe))))
        if parity>1e-5:raise AssertionError('NumPy actor parity failed')
        if not any(not torch.equal(value,initial[name]) for name,value in model.policy.state_dict().items()
                   if actor_name(name)):
            raise AssertionError('PPO must change the BC actor')
        if not all(torch.equal(value,prior_initial[name]) for name,value in model.bc_prior.state_dict().items()):
            raise AssertionError('Fixed BC prior changed during training')
        if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):
            raise AssertionError('Source changed during run')
        expected_steps=(model._n_updates-before)*math.ceil(model.n_steps*model.n_envs/model.batch_size)
        if model.bc_kl_gradient_steps!=expected_steps:
            raise AssertionError('KL gradient count differs from actual PPO minibatch updates')
        complete=dict(status='complete',algorithm=config['algorithm'],ppo_timesteps=model.num_timesteps,
                      critic_only_timesteps=warmup_record['environment_steps'],
                      completed_critic_only_epochs=warmup_record['completed_critic_only_epochs'],
                      additional_actor_critic_ppo_epochs=model._n_updates-before,
                      actor_critic_ppo_gradient_steps=expected_steps,
                      on_policy_prior_kl_gradient_steps=model.bc_kl_gradient_steps,
                      on_policy_prior_kl_minibatch_sample_uses=model.bc_kl_minibatch_sample_uses,
                      offline_supervised_anchor_gradient_steps=0,actor_changed_by_ppo=True,
                      actor_unchanged_by_critic_warmup=True,bc_prior_bit_identical=True,
                      numpy_export_max_error=parity,source_hashes_unchanged=True,
                      selected_policy_sha256=sha(out/'best_policy.npz'),final_policy_sha256=sha(out/'final.npz'),
                      selected_objective=selection.best,wall_seconds=time.monotonic()-started)
        (out/'complete.json').write_text(json.dumps(complete,indent=2))
        print(json.dumps(complete),flush=True)
    finally:vector.close()


def launch():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(stage='Single-initial-seed development probe; not held-out confirmation',
              algorithm='BC + critic warmup + KL-regularized on-policy PPO',
              initial_model=str(INITIAL.relative_to(ROOT)),initial_sha256=sha(INITIAL),
              prior_model=str(INITIAL.relative_to(ROOT)),prior_sha256=sha(INITIAL),
              arms=ARMS,common_frozen_actor_critic_pretraining_steps=131072,
              common_actor_critic_ppo_steps=1048576,warmup_seed=9171,ppo_seed=9111,
              sole_added_loss='weight * mean KL(current Normal || fixed BC Normal) on current on-policy minibatch observations',
              no_offline_supervised_updates=True,prior_frozen_and_training_only=True,
              reward_DR_and_PPO_hyperparameters_unchanged=True,
              existing_equal_budget_references='artifacts/refinement/critic_warmup_v1/{control,warm}',
              validation='Original full176 development lexicographic selection',
              validation_seeds=list(range(82001,82009)),reserved_confirmation_seeds=[92001,92002,92003],
              main_heldout_seeds_not_opened=list(range(202601,202611)),
              gate='Selected model must preserve BC full176 survival/completion/heading counts; '
                   'mean strong/gust hover RMSE <=1.05*BC; nominal hover <=max(1.2*BC,BC+.01m); '
                   'restricted mean route time <=0.95*BC (at least5% faster). '
                   'Passing permits a Gazebo pilot only; the separate thesis15% wind-macro criterion is unchanged.',
              interpretation='Compare both KL weights with BC, original PPO, cold-critic control and warm-critic PPO. '
                   'Do not attribute behavior-cloning benefit to PPO; one seed is screening only.',
              installed_sb3_ppo_train_sha256=hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest(),
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    jobs=[]
    for arm in ARMS:
        log=(OUT/f'{arm}.log').open('w')
        command=[sys.executable,str(Path(__file__)),'--arm',arm]
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((arm,process,log))
    (OUT/'processes.json').write_text(json.dumps([
        dict(arm=arm,pid=process.pid) for arm,process,_ in jobs],indent=2))
    print(json.dumps(dict(event='STARTED',pids={arm:process.pid for arm,process,_ in jobs})),flush=True)
    while any(process.poll() is None for _,process,_ in jobs):time.sleep(5)
    records=[]
    for arm,process,log in jobs:
        log.close();records.append(dict(arm=arm,returncode=process.returncode))
    (OUT/'process_completion.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records),flush=True)
    return 0 if all(row['returncode']==0 for row in records) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=tuple(ARMS))
    args=parser.parse_args()
    if args.arm:train(args.arm)
    else:raise SystemExit(launch())

#!/usr/bin/env python3
"""Explicit optional offline BC warm start followed by genuine on-policy PPO."""
import argparse,copy,hashlib,json,os,platform,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from quadrl.env import QuadEnv
from quadrl.policy import export_policy,NumpyPolicy,EncoderConfig,CONTRACT_VERSION
from quadrl.rollout import validate
from quadrl.reward import REWARD_VERSION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior_clone(model,path,out,epochs,seed,obs_size):
    with np.load(path) as z:
        x=z["observations"][:,:obs_size];y=z["actions"];episode_ids=z["episode_ids"]
    heldout=(episode_ids%5)==0
    train_x=torch.from_numpy(x[~heldout]);train_y=torch.from_numpy(y[~heldout])
    valid_x=torch.from_numpy(x[heldout]);valid_y=torch.from_numpy(y[heldout])
    params=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer=torch.optim.Adam(params,lr=8e-4)
    rng=np.random.default_rng(seed+419)
    rows=[];best=float("inf");best_state=None
    for epoch in range(epochs):
        order=rng.permutation(len(train_x));losses=[]
        for start in range(0,len(order),1024):
            indices=order[start:start+1024]
            pred=model.policy.action_net(model.policy.mlp_extractor.forward_actor(train_x[indices]))
            loss=torch.mean((pred-train_y[indices])**2)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            pred=model.policy.action_net(model.policy.mlp_extractor.forward_actor(valid_x))
            valid=float(torch.mean((pred-valid_y)**2))
        row={"epoch":epoch+1,"train_mse":float(np.mean(losses)),"development_holdout_mse":valid}
        rows.append(row)
        if valid<best:best=valid;best_state=copy.deepcopy(model.policy.state_dict())
        with (out/"bc_progress.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
    model.policy.load_state_dict(best_state)
    record={"method":"offline behavior cloning from measured-only PID+DOB actions; teacher not called at inference",
            "samples":len(x),"fit_samples":len(train_x),"development_holdout_samples":len(valid_x),
            "epochs":epochs,"gradient_steps":epochs*int(np.ceil(len(train_x)/1024)),
            "best_development_mse":best,"dataset":str(Path(path).resolve()),"dataset_sha256":sha(path)}
    (out/"bc_summary.json").write_text(json.dumps(record,indent=2))
    print("BC_COMPLETE "+json.dumps(record),flush=True)


class AnchoredPPO(PPO):
    """Genuine on-policy PPO followed by explicit offline supervised updates.

    The next rollout samples the resulting actor and stores its actual log
    probabilities. Offline labels never enter PPO advantages/rewards, and the
    teacher controller is never called by this method or by the exported actor.
    """
    def configure_anchor(self,path,obs_size,steps,learning_rate,seed,out):
        with np.load(path) as data:
            fit=(data["episode_ids"]%5)!=0
            self._anchor_x=torch.from_numpy(data["observations"][fit,:obs_size])
            self._anchor_y=torch.from_numpy(data["actions"][fit])
        self.anchor_steps=steps;self.anchor_output=str(out)
        params=list(self.policy.mlp_extractor.policy_net.parameters())+list(self.policy.action_net.parameters())
        self._anchor_optimizer=torch.optim.Adam(params,lr=learning_rate)
        self._anchor_rng=np.random.default_rng(seed+749)
        self.il_gradient_steps=0;self.ppo_gradient_steps=0

    def _excluded_save_params(self):
        return super()._excluded_save_params()+["_anchor_x","_anchor_y","_anchor_optimizer","_anchor_rng"]

    def train(self):
        before=self._n_updates
        super().train()
        self.ppo_gradient_steps=getattr(self,"ppo_gradient_steps",0)+(self._n_updates-before)*int(np.ceil(self.n_steps*self.n_envs/self.batch_size))
        if not getattr(self,"anchor_steps",0):return
        losses=[]
        params=self._anchor_optimizer.param_groups[0]["params"]
        for _ in range(self.anchor_steps):
            indices=self._anchor_rng.integers(0,len(self._anchor_x),1024)
            predictions=self.policy.action_net(self.policy.mlp_extractor.forward_actor(self._anchor_x[indices]))
            loss=((predictions-self._anchor_y[indices])**2).mean()
            self._anchor_optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.);self._anchor_optimizer.step()
            self.il_gradient_steps+=1;losses.append(float(loss.detach()))
        self.logger.record("train/anchor_mse",float(np.mean(losses)))
        self.logger.record("train/il_gradient_steps",self.il_gradient_steps)
        self.logger.record("train/ppo_gradient_steps",self.ppo_gradient_steps)
        with (Path(self.anchor_output)/"alternating_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(timesteps=self.num_timesteps,ppo_gradient_steps=self.ppo_gradient_steps,
                il_gradient_steps=self.il_gradient_steps,anchor_mse=float(np.mean(losses))))+"\n")

    def save_anchor_state(self,path):
        if getattr(self,"anchor_steps",0):
            torch.save({"optimizer":self._anchor_optimizer.state_dict(),"rng":self._anchor_rng.bit_generator.state,
                        "il_gradient_steps":self.il_gradient_steps,"ppo_gradient_steps":self.ppo_gradient_steps},path)


class EvidenceCallback(BaseCallback):
    def __init__(self,out,args,encoder):
        super().__init__();self.out,self.args,self.encoder=out,args,encoder
        self.next_eval=args.eval_every;self.best=None;self.started=time.monotonic()

    def _on_step(self):
        if self.num_timesteps%8192==0:
            progress=1. if self.args.bc_data else min(1.,self.num_timesteps/1_000_000)
            self.training_env.set_attr("training_progress",progress)
        if self.num_timesteps>=self.next_eval:
            self.next_eval+=self.args.eval_every
            checkpoint=f"checkpoint_{self.num_timesteps}"
            self.model.save(self.out/checkpoint)
            self.model.save_anchor_state(self.out/f"{checkpoint}_il_state.pt")
            export_policy(self.model,self.out/f"{checkpoint}.npz",self.args.architecture,self.encoder)
            result=validate(self.model,self.args.architecture,self.encoder.history,full=self.args.full_selection)
            result.update(timesteps=self.num_timesteps,checkpoint=checkpoint,wall_seconds=time.monotonic()-self.started,
                          ppo_gradient_steps=getattr(self.model,"ppo_gradient_steps",0),
                          offline_anchor_gradient_steps=getattr(self.model,"il_gradient_steps",0),
                          completed_ppo_epochs=self.model._n_updates,
                          checkpoint_timing="During collection; counts identify completed updates before this snapshot")
            with (self.out/"validation.jsonl").open("a") as f:f.write(json.dumps(result)+"\n")
            print("VALIDATION "+json.dumps({k:result[k] for k in ("timesteps","objective","wall_seconds")}),flush=True)
            # A callback may fire before the first rollout's PPO update. Keep
            # that diagnostic, but never call a BC-only actor a selected RL
            # checkpoint.
            eligible=result["ppo_gradient_steps"]>0
            if eligible and (self.best is None or tuple(result["objective"])<tuple(self.best)):
                self.best=result["objective"]
                self.model.save(self.out/"best_model")
                export_policy(self.model,self.out/"best_policy.npz",self.args.architecture,self.encoder)
                (self.out/"selection.json").write_text(json.dumps({k:v for k,v in result.items() if k!="episodes"},indent=2))
        return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--architecture",choices=["hybrid","direct"],default="hybrid")
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--steps",type=int,default=3_000_000)
    ap.add_argument("--envs",type=int,default=16)
    ap.add_argument("--eval-every",type=int,default=500_000)
    ap.add_argument("--no-history",action="store_true")
    ap.add_argument("--bc-data",type=Path)
    ap.add_argument("--bc-epochs",type=int,default=35)
    ap.add_argument("--learning-rate",type=float,default=3e-5)
    ap.add_argument("--log-std",type=float,default=-3.)
    ap.add_argument("--anchor-steps",type=int,default=0)
    ap.add_argument("--anchor-learning-rate",type=float,default=1e-4)
    ap.add_argument("--reward-version",choices=["initial-v2",REWARD_VERSION],default=REWARD_VERSION)
    ap.add_argument("--reward-variant",choices=["full","no_smooth"],default="full")
    ap.add_argument("--full-final",action="store_true")
    ap.add_argument("--full-selection",action="store_true",help="Select every checkpoint on all176 validation flights")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.steps<=0 or args.envs<1 or args.eval_every<1 or args.anchor_steps<0:raise ValueError("positive rollout sizes and nonnegative anchor updates required")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.anchor_steps and not args.bc_data:raise ValueError("Offline anchor requires a declared BC dataset")
    torch.set_num_threads(1)
    encoder=EncoderConfig(history=not args.no_history)
    files=sorted([*ROOT.glob("quadrl/*.py"),*ROOT.glob("quadrl/*.json"),ROOT/"sim/metrics.py",*ROOT.glob("reference/*.py"),Path(__file__)])
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(training_seed=args.seed,algorithm="BC+alternating-PPO-IL" if args.anchor_steps else "BC+PPO" if args.bc_data else "PPO-from-scratch",observation_size=encoder.size,
                  observation_contract=CONTRACT_VERSION,torch=torch.__version__,python=sys.version,platform=platform.platform(),
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                  validation_selection=("full176" if args.full_selection else "quick10")+": lexicographic failure,route noncompletion,heading violation,macro penalized RMSE,recovery,restricted completion time",
                  validation_seeds=list(range(82001,82009)),test_seeds_never_used=list(range(202601,202611)))
    (out/"config.json").write_text(json.dumps(config,indent=2))
    with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    vec=VecMonitor(DummyVecEnv([lambda:QuadEnv(args.architecture,training=True,history=encoder.history,
        reward_version=args.reward_version,reward_variant=args.reward_variant) for _ in range(args.envs)]))
    vec.seed(args.seed)
    if args.bc_data:vec.set_attr("training_progress",1.)
    model=AnchoredPPO("MlpPolicy",vec,seed=args.seed,device="cpu",verbose=0,n_steps=512,batch_size=1024,n_epochs=8,
              learning_rate=args.learning_rate,gamma=.995,gae_lambda=.97,ent_coef=0.,clip_range=.15,
              policy_kwargs={"net_arch":{"pi":[128,128],"vf":[128,128]},"log_std_init":args.log_std})
    model.set_logger(configure(str(out),["csv"]))
    start=time.monotonic()
    if args.bc_data:
        dataset_manifest=json.loads((args.bc_data.parent/'manifest.json').read_text())
        dataset_architecture=dataset_manifest.get('environment_request',{}).get('architecture')
        if dataset_architecture!=args.architecture:
            raise ValueError(f"Teacher action architecture {dataset_architecture} does not match {args.architecture}")
        behavior_clone(model,args.bc_data,out,args.bc_epochs,args.seed,encoder.size)
        model.save(out/"bc_only_model");export_policy(model,out/"bc_only_policy.npz",args.architecture,encoder)
        before=validate(model,args.architecture,encoder.history)
        (out/"bc_validation.json").write_text(json.dumps(before,indent=2))
        print("BC_VALIDATION "+json.dumps(before["objective"]),flush=True)
    if args.anchor_steps:model.configure_anchor(args.bc_data,encoder.size,args.anchor_steps,args.anchor_learning_rate,args.seed,out)
    callback=EvidenceCallback(out,args,encoder)
    model.learn(args.steps,callback=callback,progress_bar=False)
    model.save(out/"final_model");export_policy(model,out/"final_policy.npz",args.architecture,encoder)
    model.save_anchor_state(out/"final_il_state.pt")
    if not (out/"best_policy.npz").exists():
        model.save(out/"best_model");export_policy(model,out/"best_policy.npz",args.architecture,encoder)
    reference=NumpyPolicy(out/"final_policy.npz")
    check=np.random.default_rng(723).normal(size=(200,encoder.size)).astype(np.float32)
    parity=float(np.max(np.abs(model.predict(check,deterministic=True)[0]-reference.predict(check))))
    if parity>1e-5:raise AssertionError(f"NumPy export mismatch{parity}")
    if args.full_final:
        full=validate(NumpyPolicy(out/"best_policy.npz"),args.architecture,encoder.history,full=True)
        (out/"full_validation.json").write_text(json.dumps(full,indent=2))
        print("FULL_VALIDATION "+json.dumps(full["objective"]),flush=True)
    summary={"timesteps":model.num_timesteps,"wall_seconds":time.monotonic()-start,
             "ppo_gradient_steps":model.ppo_gradient_steps,"offline_anchor_gradient_steps":getattr(model,"il_gradient_steps",0),
             "best_quick_validation_objective":callback.best,"numpy_export_max_error":parity,
             "best_policy_sha256":sha(out/"best_policy.npz"),"final_policy_sha256":sha(out/"final_policy.npz"),
             "finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (out/"training_summary.json").write_text(json.dumps(summary,indent=2))
    print("COMPLETED "+json.dumps(summary),flush=True)
    vec.close()

if __name__=="__main__":main()

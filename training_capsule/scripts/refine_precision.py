"""Isolated, paired PPO refinement; does not modify the frozen benchmark.

Each treatment changes exactly one factor: Gaussian width, quadratic position
weight, or explicit clean-nominal coverage. All others retain their baseline.
Existing waypoint bonuses, dynamics, observations and optimizer are retained.
All checkpoint selection uses the original full176 development grid.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from quadrl.env import QuadEnv
from quadrl.policy import EncoderConfig, NumpyPolicy, export_policy
from quadrl.rollout import validate
from scripts.train import AnchoredPPO


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precision_adjustment(error, dt, sigma, quadratic=.35):
    return float(dt * 1.5 * (math.exp(-(error / sigma) ** 2)
                             - math.exp(-(error / .2) ** 2))
                 -dt*(quadratic-.35)*error**2)


class PrecisionEnv(QuadEnv):
    def __init__(self, sigma=.2, quadratic=.35, clean_fraction=0., **kwargs):
        self.precision_sigma = sigma
        self.position_quadratic = quadratic
        self.clean_fraction = clean_fraction
        super().__init__(**kwargs)

    def _training_parameters(self, rng):
        if self.clean_fraction and rng.random() < self.clean_fraction:
            return {"name":"training_clean_nominal", "force_events":[]}
        return super()._training_parameters(rng)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        adjustment = precision_adjustment(info["error"], self.control_dt,
                                          self.precision_sigma,self.position_quadratic)
        info["precision_reward_adjustment"] = adjustment
        return obs, reward + adjustment, terminated, truncated, info


class Selection(BaseCallback):
    def __init__(self, output, interval, starting_updates):
        super().__init__()
        self.output = output
        self.interval = interval
        self.next_eval = interval
        self.best = None
        self.started = time.monotonic()
        self.starting_updates = starting_updates

    def evaluate(self, label):
        policy_path = self.output / f"{label}.npz"
        self.model.save(self.output / label)
        export_policy(self.model, policy_path, "hybrid", EncoderConfig())
        result = validate(NumpyPolicy(policy_path), "hybrid", True, full=True)
        result.update(timesteps=self.model.num_timesteps,
                      completed_ppo_epochs=self.model._n_updates - self.starting_updates,
                      checkpoint=label, wall_seconds=time.monotonic()-self.started,
                      checkpoint_timing="After final update" if label=="final" else
                          "During collection before current rollout update; completed epochs explicitly counted")
        (self.output / f"{label}_validation.json").write_text(json.dumps(result, indent=2))
        objective = result["objective"]
        if self.best is None or tuple(objective) < tuple(self.best):
            self.best = objective
            self.model.save(self.output / "best_model")
            export_policy(self.model, self.output / "best_policy.npz", "hybrid", EncoderConfig())
            (self.output / "selection.json").write_text(json.dumps(
                {k:v for k,v in result.items() if k != "episodes"}, indent=2))
        print(json.dumps(dict(event="VALIDATION", checkpoint=label,
                              timesteps=self.model.num_timesteps, objective=objective)), flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.interval
            self.evaluate(f"checkpoint_{self.num_timesteps}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("control", "tight", "quadratic", "clean20"), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--steps", type=int, default=1048576)
    parser.add_argument("--eval-every", type=int, default=524288)
    args = parser.parse_args()
    if args.steps < 8192 or args.eval_every < 8192:
        raise ValueError("At least one complete PPO rollout is required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sigma = .08 if args.variant == "tight" else .2
    quadratic = 1.4 if args.variant == "quadratic" else .35
    clean_fraction = .2 if args.variant == "clean20" else 0.
    sources = sorted(set([Path(__file__), *ROOT.glob("quadrl/*.py"),
                          *ROOT.glob("quadrl/*.json"), *ROOT.glob("reference/*.py"),
                          ROOT/"scripts/train.py", ROOT/"sim/metrics.py"]))
    frozen = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    config = dict(variant=args.variant, position_gaussian_sigma_m=sigma,
                  position_quadratic_weight=quadratic, clean_nominal_fraction=clean_fraction,
                  initial=str(args.initial.resolve()), initial_sha256=sha(args.initial),
                  seed=args.seed, requested_steps=args.steps, envs=16,
                  eval_every=args.eval_every, observation_size=74,
                  changed_factor={"control":"none; equal-budget continuation control",
                                  "tight":"position Gaussian width only",
                                  "quadratic":"quadratic position coefficient only",
                                  "clean20":"explicit clean-nominal training mixture only"}[args.variant],
                  algorithm="PPO continuation; no offline anchor updates",
                  source_hashes=frozen, validation_seeds=list(range(82001,82009)),
                  reserved_confirmation_seeds=[92001,92002,92003],
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (out/"config.json").write_text(json.dumps(config, indent=2))
    with zipfile.ZipFile(out/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)))
    vector = VecMonitor(DummyVecEnv([
        lambda: PrecisionEnv(sigma=sigma, quadratic=quadratic,clean_fraction=clean_fraction,
                             architecture="hybrid", training=True,
                             history=True) for _ in range(16)]))
    try:
        vector.seed(args.seed)
        vector.set_attr("training_progress", 1.)
        model = AnchoredPPO.load(args.initial, env=vector, device="cpu", seed=args.seed)
        if getattr(model,"anchor_steps",0):
            raise ValueError("This experiment requires a non-anchored initial Hybrid model")
        model.anchor_steps = 0
        model.set_logger(configure(str(out), ["csv"]))
        parameters = {k:v.detach().cpu().numpy().copy() for k,v in model.policy.state_dict().items()}
        np.savez(out/"initial_parameters.npz", **parameters)
        config.update(initial_completed_ppo_epochs=model._n_updates,
                      initial_log_std=model.policy.log_std.detach().cpu().tolist(),
                      optimizer_learning_rates=[g["lr"] for g in model.policy.optimizer.param_groups],
                      n_steps=model.n_steps, batch_size=model.batch_size, n_epochs=model.n_epochs,
                      gamma=model.gamma, gae_lambda=model.gae_lambda)
        (out/"config.json").write_text(json.dumps(config,indent=2))
        start = time.monotonic()
        starting_updates = model._n_updates
        selection = Selection(out, args.eval_every, starting_updates)
        model.learn(total_timesteps=args.steps, callback=selection,
                    reset_num_timesteps=True, progress_bar=False)
        selection.evaluate("final")
        probe=np.random.default_rng(17491).normal(size=(200,74)).astype(np.float32)
        parity=float(np.max(np.abs(model.predict(probe,deterministic=True)[0]
                                   -NumpyPolicy(out/"final.npz").predict(probe))))
        if parity>1e-5:
            raise AssertionError(f"NumPy export mismatch: {parity}")
        changed={name for name,p in model.policy.state_dict().items()
                 if not np.array_equal(p.detach().cpu().numpy(),parameters[name])}
        if not any("action_net" in name or "policy_net" in name for name in changed):
            raise AssertionError("PPO did not change actor weights")
        if any(sha(ROOT/name)!=digest for name,digest in frozen.items()):
            raise AssertionError("Source files changed during experiment")
        record=dict(status="complete",timesteps=model.num_timesteps,
                    additional_completed_ppo_epochs=model._n_updates-starting_updates,
                    additional_ppo_gradient_steps=(model._n_updates-starting_updates)
                      *math.ceil(model.n_steps*model.n_envs/model.batch_size),
                    offline_anchor_gradient_steps=0,wall_seconds=time.monotonic()-start,
                    actor_changed=True,changed_parameter_names=sorted(changed),
                    numpy_export_max_error=parity,source_hashes_unchanged=True,
                    selected_policy_sha256=sha(out/"best_policy.npz"),
                    final_policy_sha256=sha(out/"final.npz"),
                    selected_objective=selection.best)
        (out/"complete.json").write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
    finally:
        vector.close()


if __name__ == "__main__":
    main()

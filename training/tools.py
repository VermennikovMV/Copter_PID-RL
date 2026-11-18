"""Model utilities.

  python -m training.tools export-actor runs/e2e/ckpt_X models/e2e_policy.npz
  python -m training.tools widen  OLD_CKPT NEW_CKPT 1                 # add zero-weight inputs at the end
  python -m training.tools narrow OLD_CKPT NEW_CKPT "74:0.5,75:0.625"  # remove inputs, fold values into bias

widen/narrow keep the behaviour of a trained policy when its inputs change, so that training can
continue instead of restarting.
"""
import argparse
import pickle
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np


def actor_arrays(model):
    import torch
    seq = list(model.policy.mlp_extractor.policy_net)
    layers = [*seq[::2], model.policy.action_net]
    assert all(isinstance(m, torch.nn.Tanh) for m in seq[1::2]), 'dense tanh actor expected'
    out = {}
    for i, layer in enumerate(layers):
        out[f'w{i}'] = layer.weight.detach().cpu().numpy().astype(np.float32)
        out[f'b{i}'] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return out


def export_actor(ckpt, out):
    from stable_baselines3 import PPO
    model = PPO.load(str(Path(ckpt) / 'model.zip'), device='cpu')
    arrays = actor_arrays(model)
    np.savez(out, **arrays)
    x = np.random.default_rng(0).normal(size=model.observation_space.shape).astype(np.float32)
    h = x.astype(np.float64)
    n = len(arrays) // 2
    for i in range(n):
        h = arrays[f'w{i}'].astype(np.float64) @ h + arrays[f'b{i}']
        if i < n - 1:
            h = np.tanh(h)
    err = float(np.abs(np.clip(h, -1, 1) - model.predict(x, deterministic=True)[0]).max())
    assert err < 1e-5, f'export parity failed: {err}'
    print('exported', out, 'inputs', model.observation_space.shape[0], 'parity', err)


def _rebuild(old, n_inputs):
    from stable_baselines3 import PPO
    new = PPO('MlpPolicy', gym.make('Pendulum-v1'), device='cpu', policy_kwargs=old.policy_kwargs,
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
              learning_rate=old.learning_rate, gamma=old.gamma, gae_lambda=old.gae_lambda,
              clip_range=old.clip_range, ent_coef=old.ent_coef, max_grad_norm=old.max_grad_norm)
    space = gym.spaces.Box(-np.inf, np.inf, (n_inputs,), np.float32)
    new.observation_space, new.action_space = space, old.action_space
    new.policy = new.policy_class(space, old.action_space, new.lr_schedule, **new.policy_kwargs).to('cpu')
    new.num_timesteps = old.num_timesteps
    return new


def _save(new, old_dir, new_dir, n_inputs):
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    new_dir.mkdir(parents=True, exist_ok=False)
    new.save(str(new_dir / 'model.zip'))
    with open(old_dir / 'vecnorm.pkl', 'rb') as f:
        vn = pickle.load(f)
    vn.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_inputs,), np.float32)
    vn.obs_rms = RunningMeanStd(shape=(n_inputs,))   # observations are not normalised; reward stats kept
    vn.venv, vn.class_attributes, vn.returns = None, {}, np.zeros(1)
    with open(new_dir / 'vecnorm.pkl', 'wb') as f:
        pickle.dump(vn, f)


def widen(old_dir, new_dir, extra):
    from stable_baselines3 import PPO
    old = PPO.load(str(old_dir / 'model.zip'), device='cpu')
    n = old.observation_space.shape[0]
    new = _rebuild(old, n + extra)
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    for k, v in sd_old.items():
        if sd_new[k].shape == v.shape:
            sd_new[k] = v.clone()
        else:
            w = sd_new[k] * 0
            w[:, :n] = v
            sd_new[k] = w
    new.policy.load_state_dict(sd_new)
    _save(new, old_dir, new_dir, n + extra)
    print('widened', n, '->', n + extra)


def narrow(old_dir, new_dir, fold):
    from stable_baselines3 import PPO
    old = PPO.load(str(old_dir / 'model.zip'), device='cpu')
    n = old.observation_space.shape[0]
    keep = [i for i in range(n) if i not in fold]
    new = _rebuild(old, len(keep))
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    folded = {}
    for k, v in sd_old.items():
        if k.endswith('weight') and v.dim() == 2 and v.shape[1] == n:
            sd_new[k] = v[:, keep].clone()
            folded[k[:-len('weight')] + 'bias'] = sum(v[:, i] * val for i, val in fold.items())
        else:
            sd_new[k] = v.clone()
    for b, extra in folded.items():   # after the copy loop so the plain bias copy cannot overwrite it
        sd_new[b] = sd_old[b].clone() + extra
    new.policy.load_state_dict(sd_new)
    _save(new, old_dir, new_dir, len(keep))
    print('narrowed', n, '->', len(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export-actor'); e.add_argument('ckpt'); e.add_argument('out')
    w = sub.add_parser('widen'); w.add_argument('old'); w.add_argument('new'); w.add_argument('extra', type=int)
    n = sub.add_parser('narrow'); n.add_argument('old'); n.add_argument('new'); n.add_argument('fold')
    a = ap.parse_args()
    if a.cmd == 'export-actor':
        export_actor(a.ckpt, a.out)
    elif a.cmd == 'widen':
        widen(Path(a.old), Path(a.new), a.extra)
    else:
        narrow(Path(a.old), Path(a.new), {int(k): float(v) for k, v in (x.split(':') for x in a.fold.split(','))})


if __name__ == '__main__':
    main()

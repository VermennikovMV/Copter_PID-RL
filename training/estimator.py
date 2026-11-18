"""Disturbance estimator: data collection and supervised training.

  # round 1: the ArduPilot cascade flies; 70 % of episodes with large smooth random corrections
  python -m training.estimator collect --out runs/estimator/round1.npz
  python -m training.estimator train runs/estimator/round1.npz --out runs/estimator/round1_model.npz
  # round 2: the round-1 compensator flies (the closed loop the estimator works in), small corrections
  python -m training.estimator collect --out runs/estimator/round2.npz --estimator runs/estimator/round1_model.npz
      --perturbation 0.7 --amplitude 0.3 --seed 9000
  python -m training.estimator train runs/estimator/round1.npz runs/estimator/round2.npz --out models/estimator.npz

Inputs (83): the 80 inputs of training.stacks.ardupilot (vehicle state, history, position-controller
targets) + the collective thrust command of the cascade (raw, 0.3 s and 1.5 s low-pass).
Labels (4), known exactly in simulation: disturbance acceleration in the heading frame (x, y),
vertical imbalance and the effective mass ratio m / (m_nominal * motor efficiency) - a heavier
airframe and weaker motors are indistinguishable and need the same correction.
Data are collected on the randomised training distribution (configs/compensator_train.json):
round 1 with the cascade alone, round 2 with the round-1 compensator in the loop, both with and
without smooth random corrections through the residual input (to cover off-nominal states).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = 9.81


def label(base):
    from experiments.scenarios import force_N
    from controllers.base import heading_rotation
    from training.physics import MASS
    t = (base.steps - 1) * base.control_dt          # force applied during the step just taken
    force = np.asarray(force_N(t, base.parameters), float)
    _, h = heading_rotation(np.asarray(base.measured_state['rotation']))
    horizontal = h @ force / base.mass
    effective = base.mass / (MASS * float(base.parameters.get('motor_scale', 1.)))
    vertical = force[2] / base.mass - G * (1. - 1. / effective)
    return np.array([horizontal[0], horizontal[1], vertical, effective], np.float32)


def collect_worker(job):
    n, seed, config, stand, perturbation, estimator, amplitude = job
    from training import stacks
    from training.wrappers import ThrustHistory
    if estimator:   # the compensator flies; the correction below is added to its output
        env = stacks.compensator(estimator, config, stand, training=True)
        thrust = env.thrust                     # updated by the wrapper after every step
    else:
        env = stacks.ardupilot(config, stand, training=True)
        thrust = ThrustHistory()
    base = env.unwrapped
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    x, y = [], []
    noise, active = np.zeros(4), rng.random() < perturbation
    while len(x) < n:
        if active:   # smooth random correction (Ornstein-Uhlenbeck, ~0.5 s correlation)
            noise += .04 * -noise + .25 * amplitude * rng.normal(size=4)
        obs, _, term, trunc, _ = env.step(np.clip(noise, -1, 1) if active else np.zeros(4))
        if not estimator:
            thrust.push(base.pid)
        x.append(np.r_[obs, thrust.features()].astype(np.float32))
        y.append(label(base))
        if term or trunc:
            obs, _ = env.reset()
            if not estimator:
                thrust.reset()
            noise, active = np.zeros(4), rng.random() < perturbation
    return np.asarray(x), np.asarray(y)


def collect(a):
    jobs = [(a.steps, a.seed + i, a.config, a.stand, a.perturbation, a.estimator, a.amplitude) for i in range(a.procs)]
    with Pool(a.procs) as pool:
        parts = pool.map(collect_worker, jobs)
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=x, Y=y)
    print('samples', x.shape, 'label mean', y.mean(0).round(3), 'std', y.std(0).round(3))


def train(a):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    parts = [np.load(f) for f in a.data]
    x = np.concatenate([d['X'] for d in parts]).astype(np.float32)
    y = np.concatenate([d['Y'] for d in parts]).astype(np.float32)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(x))
    val, tr = idx[:len(x) // 10], idx[len(x) // 10:]
    y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-6
    xt, yt = torch.from_numpy(x), torch.from_numpy((y - y_mean) / y_std)
    sizes, layers = [x.shape[1], *a.hidden], []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    net = nn.Sequential(*layers, nn.Linear(sizes[-1], y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for epoch in range(a.epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(tr))
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            loss = ((net(xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 5 == 4 or epoch == a.epochs - 1:
            with torch.no_grad():
                pv = net(xt[val]).numpy() * y_std + y_mean
            err = np.sqrt(((pv - y[val]) ** 2).mean(0))
            print(f'epoch {epoch + 1}: validation RMSE {err.round(4)}, '
                  f'explained variance {np.round(1 - (err / y[val].std(0)) ** 2, 3)}')
    lin = [m for m in net if isinstance(m, nn.Linear)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **{f'W{i}': m.weight.detach().numpy() for i, m in enumerate(lin)},
             **{f'b{i}': m.bias.detach().numpy() for i, m in enumerate(lin)}, y_mean=y_mean, y_std=y_std)
    print('saved', a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--out', required=True)
    c.add_argument('--steps', type=int, default=200000, help='samples per process')
    c.add_argument('--procs', type=int, default=20)
    c.add_argument('--perturbation', type=float, default=0.7, help='share of episodes with random corrections')
    c.add_argument('--amplitude', type=float, default=1.0, help='scale of the random corrections (1: std ~0.9 of the limit)')
    c.add_argument('--estimator', help='fly the compensator with this estimator (round 2)')
    c.add_argument('--seed', type=int, default=7000, help='seed of the first process')
    c.add_argument('--config', default=str(ROOT / 'configs/compensator_train.json'))
    c.add_argument('--stand', default=str(ROOT / 'configs/stand_params.json'))
    t = sub.add_parser('train')
    t.add_argument('data', nargs='+')
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--seed', type=int, default=0, help='weight initialisation seed')
    t.add_argument('--hidden', type=int, nargs='+', default=[128, 128], help='hidden layer sizes (onboard: 128 128)')
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else train(a)


if __name__ == '__main__':
    sys.exit(main())

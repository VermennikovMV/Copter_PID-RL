"""PPO training of the end-to-end policy (four motor commands) in PyBullet.

  python -m training.ppo --out runs/e2e --steps 30000000
  python -m training.ppo --out runs/e2e_more --steps 10000000 --resume runs/e2e/ckpt_030000000 --lr 1e-4

The task reward is training.wrappers.TaskReward (speed mode on routes by default: progress
towards the waypoint, a time penalty and an inaccuracy penalty; crash -100). Route episodes
start at random points of random tracks (training.wrappers.RandomTrackStart).
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REWARD = dict(route_speed_mode=True, route_time=0.4, action_size=0.0)


def make_env(rank, seed, config, reward, random_start):
    def thunk():
        from training import stacks
        env = stacks.end_to_end(config, training=True, random_start=random_start, reward=reward)
        env.reset(seed=seed * 1000 + rank)
        return env
    return thunk


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--config', default=str(ROOT / 'configs/e2e_train.json'))
    p.add_argument('--reward', default=json.dumps(DEFAULT_REWARD), help='JSON overrides of RewardConfig')
    p.add_argument('--start-vmax', type=float, default=8.0, help='random-start speed limit, m/s')
    p.add_argument('--max-offtrack', type=float, default=3.0, help='training crash distance from the segment, m')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--envs', type=int, default=14)
    p.add_argument('--n-steps', type=int, default=256)
    p.add_argument('--batch', type=int, default=1792)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--log-std', type=float, default=-1.6, help='initial exploration noise (log std)')
    p.add_argument('--net', type=int, nargs='+', default=[128, 128])
    p.add_argument('--resume', help='checkpoint directory (model.zip + vecnorm.pkl)')
    p.add_argument('--reset-log-std', type=float, help='on resume: set the exploration noise (log std)')
    p.add_argument('--ckpt-every', type=int, default=1792000)
    a = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reward = json.loads(a.reward)
    random_start = dict(vmax=a.start_vmax, max_off_m=a.max_offtrack)
    venv = VecMonitor(SubprocVecEnv([make_env(i, a.seed, a.config, reward, random_start) for i in range(a.envs)],
                                    start_method='forkserver'))
    if a.resume:
        venv = VecNormalize.load(str(Path(a.resume) / 'vecnorm.pkl'), venv)
        model = PPO.load(str(Path(a.resume) / 'model.zip'), env=venv, device='cpu')
        model.learning_rate = a.lr
        model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = a.lr
        if a.reset_log_std is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(a.reset_log_std)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=a.gamma, clip_reward=10.)
        model = PPO('MlpPolicy', venv, n_steps=a.n_steps, batch_size=a.batch, n_epochs=a.epochs, learning_rate=a.lr,
                    gamma=a.gamma, gae_lambda=a.lam, clip_range=a.clip, max_grad_norm=0.5, seed=a.seed, device='cpu',
                    policy_kwargs=dict(net_arch=dict(pi=a.net, vf=a.net), activation_fn=torch.nn.Tanh,
                                       log_std_init=a.log_std))
        with torch.no_grad():   # start exactly at zero action = hover PWM on every motor
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))
    (out / 'config.json').write_bytes(Path(a.config).read_bytes())
    log = (out / 'progress.csv').open('a')
    t0, start = time.time(), int(model.num_timesteps)

    class Progress(BaseCallback):
        def _on_rollout_end(self):
            buf = self.model.ep_info_buffer
            r = sum(e['r'] for e in buf) / max(len(buf), 1)
            n = sum(e['l'] for e in buf) / max(len(buf), 1)
            std = float(torch.exp(self.model.policy.log_std.detach()).mean())
            log.write(f'{self.num_timesteps},{time.time() - t0:.0f},{r:.2f},{n:.0f},{std:.3f}\n')
            log.flush()

        def _on_rollout_start(self):
            done = int(self.model.num_timesteps) - start
            if a.ckpt_every and done > 0 and done % a.ckpt_every == 0:
                d = out / f'ckpt_{self.model.num_timesteps:09d}'
                d.mkdir(exist_ok=False)
                self.model.save(str(d / 'model.zip'))
                venv.save(str(d / 'vecnorm.pkl'))

        def _on_step(self):
            return not (out / 'STOP').exists()

    model.learn(total_timesteps=a.steps, callback=Progress(), reset_num_timesteps=not a.resume)
    model.save(str(out / 'model.zip'))
    venv.save(str(out / 'vecnorm.pkl'))
    from training.tools import export_actor
    export_actor(out, out / 'policy.npz')
    venv.close()
    log.close()


if __name__ == '__main__':
    main()

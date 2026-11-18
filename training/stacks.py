"""Environment stacks used for training and evaluation.

ardupilot(...)     - the ArduPilot cascade copy in GUIDED (shaped targets), residual input available
compensator(...)   - ardupilot(...) + learned disturbance estimator + physics compensation
end_to_end(...)    - the policy drives the four motors; track-ahead inputs
"""
from pathlib import Path

from training.environment import QuadrotorEnv
from training.wrappers import (AttitudeResidual, DisturbanceCompensation, GuidedShaping, PidTargetObservation,
                               RandomTrackStart, TaskReward, TrackObservation)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'


def ardupilot(config=CONFIGS / 'compensator_eval.json', stand=CONFIGS / 'stand_params.json', training=False,
              mission=None, scenario='nominal', shaping_overrides=None, seconds=None):
    """GUIDED cascade copy with the bounded attitude/thrust residual input and the 80 inputs
    (vehicle state + position-controller targets) used by the estimator."""
    env = QuadrotorEnv(controller='residual', training=training, mission=mission, scenario=scenario,
                       config_path=str(config), seconds=seconds)
    env = GuidedShaping(env, stand, randomize=training, overrides=shaping_overrides)
    env = AttitudeResidual(env, delay_probability=0.05 if training else 0.)
    return PidTargetObservation(env)


def compensator(estimator, config=CONFIGS / 'compensator_eval.json', stand=CONFIGS / 'stand_params.json',
                training=False, mission=None, scenario='nominal', shaping_overrides=None, seconds=None):
    env = ardupilot(config, stand, training, mission, scenario, shaping_overrides, seconds)
    return DisturbanceCompensation(env, estimator)


def end_to_end(config=CONFIGS / 'e2e_eval.json', training=False, mission=None, scenario='nominal',
               random_start=None, reward=None, seconds=None):
    """random_start: dict of RandomTrackStart arguments (training only); reward: RewardConfig overrides."""
    env = QuadrotorEnv(controller='direct', training=training, mission=mission, scenario=scenario,
                       config_path=str(config), seconds=seconds)
    if random_start is not None:
        env = RandomTrackStart(env, **random_start)
    env = TrackObservation(env, target_distance=True)
    if reward is not None:
        env = TaskReward(env, reward)
    return env

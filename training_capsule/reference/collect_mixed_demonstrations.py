"""Prospective nominal/domain and hover/route mixture for offline BC only."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reference.collect_demonstrations import collect_dataset, ensure_development_seeds
from reference.dob import DOBConfig, DOBReference
from quadrl.env import QuadEnv
from quadrl.policy import EncoderConfig
from quadrl.protocol import Mission


def episode_design(seeds, design_seed=2000):
    seeds = ensure_development_seeds(seeds)
    if len(seeds) != 96:
        raise ValueError("this prospective mixture is specified for exactly96 episodes")
    rng = np.random.default_rng(design_seed)
    shuffled = rng.permutation(len(seeds))
    nominal = set(int(i) for i in shuffled[:20])
    route = set(int(i) for i in shuffled[:8]) | set(int(i) for i in shuffled[20:52])
    return [{"seed": seed, "training_domain": index not in nominal,
             "mission": "route" if index in route else "hover"}
            for index, seed in enumerate(seeds)]


class MixedDemonstrationEnv:
    """Scenario/mission scheduler, never part of the policy or teacher input.

    The base training reset's RNG calls are retained. Only the mission state
    machine is selected after reset; both kinds have the identical initial goal,
    so the returned initial measured observation remains valid without resampling.
    """
    def __init__(self, episode_specs, seconds=30.0, history=True):
        self.specs = {int(item["seed"]): dict(item) for item in episode_specs}
        self.seconds = float(seconds)
        self.architecture = "hybrid"
        self.training = "mixed_explicit_episode_specs"
        self.scenario = "nominal_or_training_domain"
        self.mission_kind = "mixed_explicit_episode_specs"
        self.control_dt = .02
        self.training_progress = 1.0
        self.encoder_config = EncoderConfig(history=history)
        self.reward_variant = "base_env_default"
        self._env = None

    @property
    def plant_config(self):
        return self._env.plant_config if self._env is not None else None

    def describe_collection(self):
        return {"factory": "reference.collect_mixed_demonstrations.MixedDemonstrationEnv",
                "episode_specs": list(self.specs.values()), "nominal_count": 20,
                "training_domain_count": 76, "route_count": 40, "hover_count": 56,
                "nominal_route_count": 8, "training_route_count": 32,
                "seconds_per_episode": self.seconds,
                "mission_override": "after base reset, same initial target; no measurement resampling"}

    def reset(self, seed):
        if seed not in self.specs:
            raise ValueError("seed is absent from the prospective mixture")
        self.close()
        spec = self.specs[seed]
        self._env = QuadEnv(architecture="hybrid", training=spec["training_domain"], scenario="nominal",
                            mission=spec["mission"], seconds=self.seconds, history=self.encoder_config.history)
        self._env.training_progress = self.training_progress
        observation, info = self._env.reset(seed=seed)
        mission = Mission(spec["mission"])
        if not np.array_equal(mission.target, self._env.target):
            raise RuntimeError("mission selection would alter the initial observation target")
        self._env.mission = mission
        self._env.target = mission.target
        info["collection_episode_spec"] = spec
        return observation, info

    def step(self, action):
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2001)
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + 96)
    specs = episode_design(seeds)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["enabled"] = True
    controller = DOBReference(config=DOBConfig(**config))
    env = MixedDemonstrationEnv(specs)
    try:
        manifest = collect_dataset(env, controller, seeds, args.output, training_progress=1.0)
        print(json.dumps({key: manifest[key] for key in ("samples", "failures", "valid", "dataset_sha256")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()

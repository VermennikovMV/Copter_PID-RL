"""Offline measured-observation/teacher-action development data for warm starts.

This collector never installs a teacher in a learned policy's execution path.
It deliberately rejects the registered validation and final-test seed ranges.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.dob import DOBConfig, DOBReference


def ensure_development_seeds(seeds):
    values = [int(value) for value in seeds]
    if not values or len(set(values)) != len(values):
        raise ValueError("nonempty, distinct development seeds required")
    # Restrict rather than merely blacklist a currently known test interval.
    if any(value < 101 or value > 60000 for value in values):
        raise ValueError("BC training seeds must be in development range 101..60000")
    return values


def source_hashes():
    result = {}
    for name in ("quadrl/__init__.py", "quadrl/env.py", "quadrl/policy.py", "quadrl/protocol.py", "quadrl/plant_config.json", "quadrl/reward.py",
                 "reference/__init__.py", "reference/direct_physics.py",
                 "reference/collect_mixed_demonstrations.py",
                 "reference/dob.py", "reference/collect_demonstrations.py",
                 "reference/evaluate_reference.py"):
        path = ROOT / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def collect_dataset(env, controller, seeds, output, training_progress=1.0):
    """Collect without access to env.state(), env.parameters or plant truth.

    Only the public reset/step observations are passed to ``predict``. Episode
    termination and scalar rewards are metadata, never additional policy inputs.
    An env created with training=True is recommended for domain randomization.
    """
    seeds = ensure_development_seeds(seeds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes_before = source_hashes()
    for name, expected_hash in hashes_before.items():
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = (ROOT / name).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != expected_hash:
            raise RuntimeError("source changed while creating collection snapshot")
        destination.write_bytes(source_bytes)
    started_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    observations, actions, episode_ids = [], [], []
    episode_records = []
    if hasattr(env, "training_progress"):
        env.training_progress = float(training_progress)
    for episode_id, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        controller.reset()
        episode_start = len(observations)
        reward_sum = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            observation = np.asarray(obs, dtype=np.float32)
            if observation.shape not in ((30,), (74,)) or not np.isfinite(observation).all():
                raise ValueError("teacher requires a finite measured V2 observation")
            action = np.asarray(controller.predict(observation), dtype=np.float32)
            if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.0 + 1e-6:
                raise ValueError("teacher emitted invalid normalized action")
            observations.append(observation.copy())
            actions.append(action.copy())
            episode_ids.append(episode_id)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
        episode_records.append({
            "episode_id": episode_id, "seed": seed, "sample_start": episode_start,
            "sample_count": len(observations) - episode_start,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "failed": bool(info.get("failed", terminated)), "return": reward_sum,
            "last_time": float(info.get("time", 0.0)),
        })
    hashes_after = source_hashes()
    changed = [name for name in set(hashes_before) | set(hashes_after)
               if hashes_before.get(name) != hashes_after.get(name)]
    arrays_path = output / "observations_actions.npz"
    np.savez_compressed(arrays_path,
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        episode_ids=np.asarray(episode_ids, dtype=np.int32))
    manifest = {
        "schema": "offline_teacher_measured_v2.0", "split": "development_training",
        "runtime_teacher_required": False, "privileged_policy_inputs": False,
        "teacher": controller.metadata(), "seeds": seeds,
        "environment_nominal_configuration": getattr(env, "plant_config", None),
        "environment_request": {name: getattr(env, name, None) for name in
                                ("architecture", "training", "scenario", "mission_kind", "seconds", "control_dt", "reward_variant", "reward_version")},
        "encoder_config": asdict(env.encoder_config) if hasattr(env, "encoder_config") else None,
        "collection_design": env.describe_collection() if hasattr(env, "describe_collection") else None,
        "training_progress": float(training_progress), "samples": len(observations),
        "observation_size": int(np.asarray(observations).shape[1]),
        "failures": sum(row["failed"] for row in episode_records),
        "all_episodes_retained": True, "episodes": episode_records,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_monotonic_s": time.monotonic() - started_monotonic,
        "source_hashes_before": hashes_before, "source_hashes_after": hashes_after,
        "source_files_changed_during_collection": changed,
        "dataset_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "valid": not changed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if changed:
        raise RuntimeError("source files changed while collecting; data preserved but invalid")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--training-progress", type=float, default=1.0)
    parser.add_argument("--observer-disabled", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    seeds = ensure_development_seeds(range(args.seed_start, args.seed_start + args.episodes))
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    cfg["enabled"] = not args.observer_disabled
    controller = DOBReference(config=DOBConfig(**cfg))
    from quadrl.env import QuadEnv
    env = QuadEnv(architecture="hybrid", training=True, scenario="nominal",
                  mission="hover", seconds=args.seconds)
    try:
        result = collect_dataset(env, controller, seeds, args.output, args.training_progress)
        print(json.dumps({key: result[key] for key in ("samples", "failures", "valid", "dataset_sha256")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()

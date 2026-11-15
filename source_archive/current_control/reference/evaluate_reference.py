"""Development-only paired flights for the analytical DOB attribution baseline."""
from pathlib import Path
from dataclasses import replace
import argparse
import csv
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reference.dob import DOBConfig, DOBReference
from reference.collect_demonstrations import source_hashes, ensure_development_seeds


def run_episode(env, controller, seed):
    observation, _ = env.reset(seed=seed)
    controller.reset()
    rows, terminated, truncated = [], False, False
    while not (terminated or truncated):
        action = controller.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append({"time": info["time"], "error": info["error"],
                     "altitude_error": info["altitude_error"], "yaw_error": info["yaw_error"],
                     "position": info["position"], "target": info["target"],
                     "action": action.tolist(),
                     "disturbance": controller.disturbance.tolist()})
    start_index = round(2.0 / env.control_dt)
    samples = rows[start_index:]
    squared_error = sum(row["error"]**2 for row in samples)
    count = max(1, env.max_steps - start_index)
    missing_samples = max(0, count - len(samples))
    result = {
        "survived": not terminated, "duration_s": info["time"],
        "engineering_rmse_m": float(np.sqrt(squared_error / len(samples))) if samples else None,
        "engineering_failure_penalized_rmse_m": float(np.sqrt((squared_error + missing_samples * 16.0) / count)),
        "altitude_drop_m": max(0.0, max(-row["altitude_error"] for row in rows)),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean([row["yaw_error"]**2 for row in rows])))),
        "route_completion_s": info.get("route_completion_s"),
    }
    return result, {name: np.asarray([row[name] for row in rows]) for name in rows[0]}


def main():
    from quadrl.env import QuadEnv
    from quadrl.protocol import SCENARIOS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--missions", nargs="+", choices=["hover", "route"], default=["hover", "route"])
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plant-config", type=Path)
    args = parser.parse_args()
    seeds = ensure_development_seeds(args.seeds)
    config = DOBConfig(**json.loads(args.config.read_text())) if args.config else DOBConfig()
    plant_config = json.loads(args.plant_config.read_text()) if args.plant_config else None
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    started = time.monotonic()
    results = []
    for scenario in args.scenarios:
        for mission in args.missions:
            for seed in seeds:
                for enabled in (False, True):
                    label = "pid_same_gains_dob" if enabled else "pid_same_gains"
                    env = QuadEnv(architecture="hybrid", scenario=scenario, mission=mission,
                                  seconds=args.seconds, plant_config=plant_config)
                    controller = DOBReference(config=replace(config, enabled=enabled))
                    try:
                        metrics, trace = run_episode(env, controller, seed)
                        name = f"{label}_{scenario}_{mission}_{seed}"
                        np.savez_compressed(args.output / f"{name}.npz", **trace)
                        results.append({"controller": label, "scenario": scenario, "mission": mission,
                                        "seed": seed, **metrics})
                        print(json.dumps(results[-1]), flush=True)
                        used_plant_config = dict(env.plant_config)
                    finally:
                        env.close()
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    after = source_hashes()
    manifest = {"split": "development", "purpose": "engineering_same_gain_DOB_ablation",
                "not_final_thesis_results": True, "controller": DOBReference(config=config).metadata(),
                "plant_config": used_plant_config, "seeds": seeds, "scenarios": args.scenarios,
                "missions": args.missions, "episodes": len(results),
                "elapsed_monotonic_s": time.monotonic() - started,
                "source_hashes_before": before, "source_hashes_after": after,
                "source_unchanged_during_run": before == after}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

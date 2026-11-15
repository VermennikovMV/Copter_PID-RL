"""Prospective bounded PID/DOB gain search on the shared validation protocol."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def freeze_sources(destination):
    names = ["quadrl/__init__.py", "quadrl/env.py", "quadrl/policy.py", "quadrl/protocol.py",
             "quadrl/plant_config.json", "quadrl/rollout.py", "quadrl/baseline.py", "quadrl/pid_config.json",
             "sim/__init__.py", "sim/metrics.py", "reference/__init__.py", "reference/dob.py",
             "reference/direct_physics.py", "reference/tune_gains.py"]
    hashes = {}
    for name in names:
        source = ROOT / name
        data = source.read_bytes() if source.exists() else b""
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def candidates():
    from reference.dob import DOBConfig
    base = DOBConfig(kp=(4., 4., 6.), kd=(3.6, 3.6, 4.), ki=(1., 1., 2.),
                     nominal_motor_tau_s=.015, nominal_drag_per_mass=(0., 0., 1.55/1.75))
    values = []
    def add(label, cfg):
        values.append({"id": f"c{len(values):02d}_{label}", "config": asdict(cfg)})
    add("prior_teacher", replace(base, kp=(2.5, 2.5, 4), kd=(2.8, 2.8, 3.2), ki=(.55, .55, 1.1)))
    add("current_pid", base)
    for p in (4., 6., 8.):
        for damping in (.75, .95):
            for integral in (.7, 1.5, 2.5):
                d = 2 * math.sqrt(p) * damping
                add(f"p{p:g}_z{damping:g}_i{integral:g}", replace(base, kp=(p,p,6), kd=(d,d,4), ki=(integral,integral,2)))
    for p in (10., 12.):
        for damping in (.75, .95):
            for integral in (1.5, 2.5):
                d = 2 * math.sqrt(p) * damping
                add(f"p{p:g}_z{damping:g}_i{integral:g}", replace(base, kp=(p,p,6), kd=(d,d,4), ki=(integral,integral,2)))
    for p in (6., 10.):
        for vertical_p in (4., 8.):
            d, dz = 2 * math.sqrt(p) * .8, 2 * math.sqrt(vertical_p) * .9
            add(f"p{p:g}_vertical{vertical_p:g}", replace(base, kp=(p,p,vertical_p), kd=(d,d,dz), ki=(1.5,1.5,.35*vertical_p)))
    for attitude in (6.5, 8.):
        add(f"p8_attitude{attitude:g}", replace(base, kp=(8,8,6), kd=(4.5,4.5,4), ki=(1.5,1.5,2), attitude_gain=attitude))
    for p in (8., 12.):
        d = 2 * math.sqrt(p) * .8
        add(f"p{p:g}_accel7", replace(base, kp=(p,p,6), kd=(d,d,4), ki=(1.5,1.5,2), horizontal_acceleration_limit=7.))
    assert len(values) == 36
    return values


def initialize_worker(snapshot):
    sys.path.insert(0, snapshot)


def trial(job):
    from reference.dob import DOBConfig, DOBReference
    from quadrl.rollout import run_episode, objective, QUICK_VALIDATION, VALIDATION_SEEDS, SCENARIOS
    candidate, enabled, stage, output = job
    label = "dob" if enabled else "pid"
    directory = Path(output) / stage / candidate["id"] / label
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(candidate["config"])
    config["enabled"] = enabled
    controller = DOBReference(config=DOBConfig(**config))
    cases = QUICK_VALIDATION if stage == "quick" else [(s,m,seed) for s in SCENARIOS for m in ("hover", "route") for seed in VALIDATION_SEEDS]
    started = time.monotonic()
    records = []
    for scenario, mission, seed in cases:
        trace_path = directory / "traces" / f"{scenario}_{mission}_{seed}" if stage.startswith("full") else None
        records.append(run_episode(controller, architecture="hybrid", history=True, scenario=scenario,
                                   mission=mission, seed=seed, save_path=trace_path))
    result = {"candidate": candidate["id"], "controller": label, "stage": stage,
              "config": config, "objective": objective(records), "episode_count": len(records),
              "elapsed_monotonic_s": time.monotonic() - started, "episodes": records}
    json_write(directory / "results.json", result)
    return {key: result[key] for key in ("candidate", "controller", "stage", "objective", "episode_count", "elapsed_monotonic_s")}


def run_stage(executor, jobs, output, stage):
    futures = [executor.submit(trial, job) for job in jobs]
    results = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)
        json_write(Path(output) / f"{stage}_progress.json", {"completed": len(results), "planned": len(jobs), "results": results})
        print(json.dumps({"stage": stage, "completed": len(results), "planned": len(jobs), **result}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--extend-controls", action="store_true")
    args = parser.parse_args()
    if args.extend_controls:
        manifest = json.loads((args.output / "manifest.json").read_text())
        snapshot = args.output / "source_snapshot"
        sys.path.insert(0, str(snapshot.resolve()))
        controls = manifest["candidates"][:2]
        extension = {"reason": "Full shortlist has incomplete routes; mandatory full current/prior controls prevent a quick-only baseline from being incorrectly declared inferior.",
                     "decision_before_final_test": True, "candidates": [row["id"] for row in controls],
                     "episodes_planned": 704, "source_snapshot_unchanged": True,
                     "extension_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "started_utc": datetime.now(timezone.utc).isoformat()}
        json_write(args.output / "controls_extension.json", extension)
        (args.output / "controls_extension_source.py").write_bytes(Path(__file__).read_bytes())
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                 initializer=initialize_worker, initargs=(str(snapshot.resolve()),)) as executor:
            jobs = [(candidate, enabled, "full_controls", str(args.output.resolve())) for candidate in controls for enabled in (False, True)]
            extra = run_stage(executor, jobs, args.output, "full_controls")
        original = json.loads((args.output / "selection.json").read_text())
        combined = original["full"] + extra
        winners = {}
        for controller in ("pid", "dob"):
            best = min((row for row in combined if row["controller"] == controller), key=lambda row: (row["objective"], row["candidate"]))
            winners[controller] = best
            config = next(candidate["config"].copy() for candidate in manifest["candidates"] if candidate["id"] == best["candidate"])
            config["enabled"] = controller == "dob"
            json_write(args.output / ("recommended_pid_config.json" if controller == "pid" else "recommended_teacher_config.json"), config)
        summary = {"full_candidates_including_controls": combined, "winners": winners,
                   "total_episodes": original["episode_count"] + sum(row["episode_count"] for row in extra),
                   "controls_extension": extension, "completed_utc": datetime.now(timezone.utc).isoformat()}
        json_write(args.output / "combined_selection.json", summary)
        print(json.dumps({"winners": winners, "total_episodes": summary["total_episodes"]}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot"
    source_hashes = freeze_sources(snapshot)
    sys.path.insert(0, str(snapshot.resolve()))
    from quadrl.rollout import QUICK_VALIDATION, VALIDATION_SEEDS, SCENARIOS
    from reference.dob import DOBConfig
    grid = candidates()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "split": "validation_tuning",
                "never_final_test": True, "source_hashes": source_hashes, "candidates": grid,
                "selection_rule": "Lexicographic shared quadrl.rollout.objective, then candidate id. Top2 per PID/DOB quick10; union evaluated as paired PID/DOB full176. Select each variant from this common full union.",
                "quick_cases": QUICK_VALIDATION, "full_seeds": VALIDATION_SEEDS,
                "full_scenarios": list(SCENARIOS), "full_missions": ["hover", "route"],
                "quick_planned_episodes": len(grid)*2*len(QUICK_VALIDATION), "workers": args.workers}
    json_write(args.output / "manifest.json", manifest)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    start = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
             initializer=initialize_worker, initargs=(str(snapshot.resolve()),)) as executor:
        quick_jobs = [(candidate, enabled, "quick", str(args.output.resolve())) for candidate in grid for enabled in (False, True)]
        quick = run_stage(executor, quick_jobs, args.output, "quick")
        chosen = set()
        for controller in ("pid", "dob"):
            ranked = sorted([row for row in quick if row["controller"] == controller], key=lambda row: (row["objective"], row["candidate"]))
            chosen.update(row["candidate"] for row in ranked[:2])
        selected = [candidate for candidate in grid if candidate["id"] in chosen]
        manifest["full_selected_candidate_ids"] = [candidate["id"] for candidate in selected]
        manifest["full_planned_episodes"] = len(selected)*2*len(SCENARIOS)*2*len(VALIDATION_SEEDS)
        json_write(args.output / "manifest.json", manifest)
        full_jobs = [(candidate, enabled, "full", str(args.output.resolve())) for candidate in selected for enabled in (False, True)]
        full = run_stage(executor, full_jobs, args.output, "full")
    winners = {}
    for controller in ("pid", "dob"):
        best = min((row for row in full if row["controller"] == controller), key=lambda row: (row["objective"], row["candidate"]))
        winners[controller] = best
        config = next(candidate["config"].copy() for candidate in grid if candidate["id"] == best["candidate"])
        config["enabled"] = controller == "dob"
        json_write(args.output / ("best_pid_config.json" if controller == "pid" else "best_teacher_config.json"), config)
    summary = {"split": "validation_tuning", "quick": quick, "full": full, "winners": winners,
               "episode_count": sum(row["episode_count"] for row in quick+full),
               "elapsed_monotonic_s": time.monotonic() - start,
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    json_write(args.output / "selection.json", summary)
    print(json.dumps({"winners": winners, "episode_count": summary["episode_count"], "elapsed_monotonic_s": summary["elapsed_monotonic_s"]}), flush=True)


if __name__ == "__main__":
    main()

"""Run the two fixed training arms against one measured initial baseline.

Both processes receive the same initialization, RNG seeds and budgets. Only
the driver variant differs. Logs and model outputs always use fresh paths.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_joint_geometry_v2 as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=training.INITIAL)
    parser.add_argument("--seed", type=int, default=9611)
    parser.add_argument("--warmup-seed", type=int, default=9511)
    parser.add_argument("--steps", type=int, default=2097152)
    parser.add_argument("--warmup-steps", type=int, default=131072)
    parser.add_argument("--eval-every", type=int, default=1048576)
    args = parser.parse_args()
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    initial = (args.initial if args.initial.is_absolute() else ROOT / args.initial).resolve()
    if not output.is_relative_to(ROOT / "artifacts/refinement/joint_v2"):
        parser.error("Pair output must remain inside artifacts/refinement/joint_v2")
    if output.exists():
        parser.error("Preserve existing experiment; select a fresh output path")
    if not initial.is_file():
        parser.error("Missing initial model")
    training.torch.set_num_threads(1)
    model = training.BCTrustPPO.load(initial, device="cpu")
    rollout = model.n_steps * training.N_ENVS
    if any(n < rollout or n % rollout for n in (args.steps, args.warmup_steps, args.eval_every)) or args.steps % args.eval_every:
        parser.error("Use full inherited rollout multiples and an exact evaluation budget")
    del model
    output.mkdir(parents=True)
    started = time.monotonic()
    frozen = training.source_hashes()
    initial_sha = training.sha(initial)
    plan = dict(status="PROSPECTIVE_PAIRED_TRAINING", pid=os.getpid(), output=str(output),
                initial=str(initial), initial_sha256=initial_sha, seed=args.seed,
                warmup_seed=args.warmup_seed, ppo_steps_per_arm=args.steps,
                warmup_steps_per_arm=args.warmup_steps, eval_every=args.eval_every,
                variants=["control", "geometry"], source_sha256=frozen,
                orchestrator_sha256=training.sha(Path(__file__)),
                utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                shared_baseline="Same24 measured development cases; no final confirmation seeds",
                changed_factor="Geometry reward W8 versus control W0; identical other training choices")
    training.write_json(output / "plan.json", plan)
    shutil.copy2(Path(__file__), output / "pair_orchestrator.py")
    training.phase(output, "validation", role="shared_initial_baseline")
    _, baseline_path = training.baseline_receipt(initial, output / "baseline", phase_output=output)
    if training.source_hashes() != frozen:
        raise RuntimeError("Training sources changed during baseline")
    jobs, streams = [], []
    try:
        for variant in plan["variants"]:
            command = [sys.executable, "-B", str(ROOT / "scripts/train_joint_geometry_v2.py"),
                       "--variant", variant, "--initial", str(initial), "--output", str(output / variant),
                       "--baseline", str(baseline_path), "--seed", str(args.seed),
                       "--warmup-seed", str(args.warmup_seed), "--warmup-steps", str(args.warmup_steps),
                       "--steps", str(args.steps), "--eval-every", str(args.eval_every)]
            stream = (output / (variant + ".stdout.log")).open("x")
            streams.append(stream)
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((variant, process))
        training.write_json(output / "children.json", {name: process.pid for name, process in jobs})
        while any(process.poll() is None for _, process in jobs):
            training.phase(output, "training", child_processes={name: dict(pid=p.pid, returncode=p.poll()) for name, p in jobs})
            time.sleep(5)
        arms = []
        for name, process in jobs:
            path = output / name / "complete.json"
            complete = json.loads(path.read_text()) if path.exists() else None
            arms.append(dict(variant=name, pid=process.pid, returncode=process.returncode,
                             complete=complete, complete_sha256=training.sha(path) if path.exists() else None))
        intact = (training.source_hashes() == frozen and training.sha(initial) == initial_sha
                  and training.sha(Path(__file__)) == plan["orchestrator_sha256"]
                  and training.sha(output / "pair_orchestrator.py") == plan["orchestrator_sha256"])
        success = intact and all(r["returncode"] == 0 and r["complete"] and r["complete"].get("status") == "complete" for r in arms)
        record = dict(status="complete" if success else "error", arms=arms,
                      sources_and_initial_unchanged=intact, baseline_sha256=training.sha(baseline_path),
                      plan_sha256=training.sha(output / "plan.json"), wall_seconds=time.monotonic()-started,
                      interpretation="Paired Fast development evidence only; actual Gazebo verification remains necessary")
        training.write_json(output / "paired_complete.json", record)
        training.phase(output, record["status"])
        print(json.dumps(dict(status=record["status"], output=str(output), arm_returncodes={n: p.returncode for n,p in jobs})), flush=True)
        return 0 if success else 1
    finally:
        # Only children created by this invocation are eligible for cleanup.
        for _, process in jobs:
            if process.poll() is None:
                process.terminate()
        for _, process in jobs:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())

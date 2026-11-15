"""Real CPU Path77 rollout; never launch Gazebo, training, HTTP or browser."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from quadrl_path_v7.policy import NumpyPolicyPath77
from quadrl_path_v7.cpu_eval import Path77CpuEnv
from quadrl.rollout import _run_episode

POLICY = ROOT / "v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz"
POLICY_SHA = "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("nominal", "force_mild", "force_strong", "gust"), default="force_strong")
    parser.add_argument("--mission", choices=("hover", "route"), default="route")
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seconds = args.seconds if args.seconds is not None else (30. if args.mission == "hover" else 40.)
    if not 100 <= args.seed < 90000 or not 0 < seconds <= 40:
        parser.error("Use a viewed/development seed[100,90000), positive horizon<=40s")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs" / (stamp + "_path77")).resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        parser.error("Choose a fresh output beneath runs/")
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == POLICY_SHA
    policy = NumpyPolicyPath77(POLICY)
    env = Path77CpuEnv(scenario=args.scenario, mission=args.mission, seconds=seconds)
    try:
        record = _run_episode(env, policy, "hybrid", True, args.scenario, args.mission, args.seed, output)
    finally:
        env.close()
    record.update(scope="PYBULLET_CPU_DEVELOPMENT_NOT_GAZEBO_VALIDATION", model="exact77_FULLCONTROL_e0aa",
        policy_sha256=POLICY_SHA, observation_size=77, no_Gazebo_or_training_started=True)
    record.pop("trace_path", None)
    record.pop("actor_io_path", None)
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(scope=record["scope"], steps=record["steps"], goals_reached=len(record["mission_events"]),
        completion_time_s=record["metrics"]["route_completion_time_s"], survived=record["metrics"]["survived"],
        output=str(output.relative_to(ROOT))), indent=2))


if __name__ == "__main__":
    main()

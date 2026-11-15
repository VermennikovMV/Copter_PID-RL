"""Create a fresh CPU reproduction workspace; never start training implicitly.

--baseline-only runs24 baseline episodes without gradients.
--prepare adds the original bounded semantic probes and seals a plan.
--run needs the prepared workspace and uses the official preflight parent.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "training_capsule"
INITIAL = "artifacts/refinement/joint_v2/path_pilot111/path/final.zip"
BASELINE = "artifacts/refinement/joint_v2/public_repro_baseline"
SEMANTICS = "audit/public_repro_semantics.json"
PLAN = "docs/public_repro_w8_control.bound.json"

def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")

def verify(folder):
    manifest = read(folder / "CAPSULE_MANIFEST.json")
    for row in manifest["files"]:
        path = folder / row["path"]
        assert path.is_file() and sha(path) == row["export_sha256"], row["path"]
    assert sha(folder / "public_plan.template.json") == manifest["template_sha256"]
    return manifest

def call(workspace, *argv):
    command = [sys.executable, "-B", *argv]
    print(json.dumps(dict(event="EXECUTE_EXPLICIT_CPU_REPRODUCTION", command=command), ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=workspace, check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline-only",action="store_true")
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--run",action="store_true")
    parser.add_argument("--workspace",type=Path)
    args=parser.parse_args()
    verify(CAPSULE)
    if args.workspace:
        workspace=(args.workspace if args.workspace.is_absolute() else ROOT/args.workspace).resolve()
        assert workspace.is_relative_to((ROOT/"runs").resolve()) and workspace.is_dir(), "Use an existing owned runs workspace"
        assert sha(workspace/"CAPSULE_MANIFEST.json")==sha(CAPSULE/"CAPSULE_MANIFEST.json")
    else:
        assert not args.run, "--run requires a prepared --workspace"
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        workspace=ROOT/"runs"/("reproduction_"+stamp+"_"+uuid.uuid4().hex[:8])
        shutil.copytree(CAPSULE,workspace)
    manifest=verify(workspace)
    if args.run:
        prepared=read(workspace/"prepared.json")
        assert prepared["status"]=="READY_FOR_EXPLICIT_CPU_REPRODUCTION"
        assert sha(workspace/PLAN)==prepared["plan_sha256"]
        call(workspace,"scripts/joint_research_preflight_v1.py","--plan",PLAN,"--run")
        return
    if not (workspace/BASELINE/"baseline.json").exists():
        call(workspace,"scripts/train_joint_corner_v8.py","--baseline-only","--initial",INITIAL,"--output",BASELINE)
    if args.baseline_only:
        print(json.dumps(dict(status="CPU_BASELINE_ONLY_NO_OPTIMIZER",workspace=str(workspace),new_baseline_episodes=24,training_started=False),ensure_ascii=False))
        return
    if not (workspace/SEMANTICS).exists():
        call(workspace,"scripts/train_joint_corner_v8.py","--verify","--initial",INITIAL,
             "--output","artifacts/refinement/joint_v2/public_repro_semantic_probe","--semantic-receipt",SEMANTICS)
    else:
        # Resume only a successful proof for these exact current source bytes.
        semantic=read(workspace/SEMANTICS)
        assert semantic["status"]=="PASS" and semantic["initializer_sha256"]==manifest["initial_export_sha256"]
        assert semantic["actor_and_critic_observations"]==77 and not semantic["paired_training_executed"]
        for name,digest in semantic["source_hashes"].items(): assert sha(workspace/name)==digest,name
    plan=read(workspace/"public_plan.template.json")
    plan.update(status="READY_FOR_RUN", hypothesis="Reproduce the measured short W8 CONTROL procedure from identical actor/critic tensors with explicit portable metadata identities; not a new Gazebo claim.",
        changed_factors="Only public local-path metadata and corresponding actual SHA bindings differ from the archived procedure. Algorithms, numeric dynamics, model tensors, RNG, budgets and fixed quality criteria are unchanged. New baseline/probe/output records are created.",
        expected_effect="A genuine CPU reproduction with the same initial numeric policy and guarded procedure. Exact final checkpoint equality is to be checked after completion, not promised by preparation.",
        reconsider_if="Stop on any source, baseline, initializer, semantic or quality mismatch. All failed checkpoints stay visible; no automatic extension or deployment.")
    baseline=BASELINE+"/baseline.json"
    plan["evidence"]["baseline"]=dict(path=baseline,sha256=sha(workspace/baseline),pointer="/status",equals="BASELINE_MEASURED_NOT_ACCEPTANCE",interpretation="Fresh24 actual full-horizon CPU baseline episodes from the portable initializer, with raw traces and current source identities. Not new Gazebo or independent robustness evidence.")
    plan["evidence"]["reward_semantics"]=dict(path=SEMANTICS,sha256=sha(workspace/SEMANTICS),pointer="/status",equals="PASS",interpretation="Fresh bounded semantic probes of the original guarded procedure, including real optimizer updates; probe policies are never used as the training initializer.")
    values={"--initial":INITIAL,"--baseline":baseline,"--output":"artifacts/refinement/joint_v2/public_reproduction/w8_control","--research-plan":PLAN}
    for flag,value in values.items():
        assert plan["argv"].count(flag)==1
        plan["argv"][plan["argv"].index(flag)+1]=value
    plan["additional_immutable_sources"]=[dict(path=row["path"],sha256=row["export_sha256"]) for row in manifest["files"]]
    plan["portability_provenance"]=dict(capsule_manifest_sha256=sha(workspace/"CAPSULE_MANIFEST.json"),initial_original_sha256=manifest["initial_original_sha256"],initial_export_sha256=manifest["initial_export_sha256"],originals_and_public_hashes_are_distinct=True)
    write_new(workspace/PLAN,plan)
    call(workspace,"scripts/joint_research_preflight_v1.py","--plan",PLAN)
    write_new(workspace/"prepared.json",dict(status="READY_FOR_EXPLICIT_CPU_REPRODUCTION",plan_sha256=sha(workspace/PLAN),baseline_episodes=24,semantic_probe_completed=True,actual_main_PPO_started=False,Gazebo_started=False))
    print(json.dumps(dict(status="READY_FOR_EXPLICIT_CPU_REPRODUCTION",workspace=str(workspace),next="python scripts/reproduce_training.py --workspace <above> --run"),ensure_ascii=False))

if __name__=="__main__": main()

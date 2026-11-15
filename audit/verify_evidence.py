#!/usr/bin/env python3
"""Independent audit of saved experiments; never imports controller/evaluator code.

Recomputes geometry and metrics from raw traces, verifies matched experimental
conditions, and fails on corruption or inconsistent summaries. Authenticity is
an evidence assessment, not something a SHA256 digest alone can establish.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line, parse_constant=reject_constant) for line in stream if line.strip()]


def digest(path):
    hash_obj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def bool_value(value):
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    raise ValueError("Unrecognized boolean: " + repr(value))


def rms(values):
    return math.sqrt(sum(x*x for x in values) / len(values)) if values else None


def same_number(a, b, atol=1e-7, rtol=1e-7):
    return a is None and b is None or (a is not None and b is not None and
           math.isfinite(float(a)) and math.isfinite(float(b)) and
           abs(float(a)-float(b)) <= atol + rtol*abs(float(b)))


def fast_metrics(rows, planned_steps):
    if not rows:
        raise ValueError("Empty trace")
    errors, alt_errors, times = [], [], []
    previous = -math.inf
    for row in rows:
        t = float(row["time"])
        geometry = [float(row[axis])-float(row["target_"+axis]) for axis in ("x", "y", "z")]
        numbers = [t, *geometry, float(row["tilt_cos"]), float(row["error"]), float(row["altitude_error"])]
        numbers += [float(row["action"+str(i)]) for i in range(4)]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Nonfinite trace values")
        if t <= previous:
            raise ValueError("Model time not strictly increasing")
        previous = t
        err = math.sqrt(sum(x*x for x in geometry))
        if not same_number(row["error"], err) or not same_number(row["altitude_error"], geometry[2]):
            raise ValueError("Stored error inconsistent with independent position/target geometry")
        errors.append(err); alt_errors.append(geometry[2]); times.append(t)
    if len(rows) > planned_steps:
        raise ValueError("Trace longer than planned horizon")
    if any(bool_value(row["failed"]) for row in rows[:-1]):
        raise ValueError("Trace continued after reported terminal failure")
    failed = bool_value(rows[-1]["failed"])
    if not failed and len(rows) != planned_steps:
        raise ValueError("Incomplete episode reported as survived")
    dt = times[1]-times[0] if len(times)>1 else times[0]
    if not same_number(dt, .02, 1e-6):
        raise ValueError("Trace is not declared 50 Hz fast experiment")
    if any(not same_number(b-a, dt, 1e-6) for a,b in zip(times,times[1:])):
        raise ValueError("Nonuniform fast-simulator time")
    tail = [error for t,error in zip(times,errors) if t > 2.0+1e-8]
    settle = None
    if not failed:
        for i in range(len(errors)-49):
            if all(error < .25 for error in errors[i:i+50]):
                settle = times[i]
                break
    return {"steps": len(rows), "survived": not failed,
            "success": not failed and bool(tail) and rms(tail)<.5,
            "rmse": rms(errors),
            "penalized_rmse": rms(errors + [4.]*(planned_steps-len(rows))),
            "rmse_after2s": rms(tail), "rmse_z": rms(alt_errors),
            "max_error": max(errors), "max_altitude_drop": max(0.,-min(alt_errors)),
            "settle_time": settle, "sim_seconds": times[-1]}


def local_path(path, root):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    candidate = (root/candidate).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Trace path outside experiment root")
    return candidate


def audit_fast(folder, root=ROOT):
    errors, warnings = [], []
    config = read_json(folder/"config.json")
    episodes = read_jsonl(folder/"episodes.jsonl")
    saved_summary = read_json(folder/"summary.json")
    snapshot_verified = 0
    if config.get("source_hashes"):
        try:
            with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
                for name, expected_hash in config["source_hashes"].items():
                    if hashlib.sha256(snapshot.read(name)).hexdigest() != expected_hash:
                        raise ValueError("Archived evaluation source hash mismatch: "+name)
                    snapshot_verified += 1
        except Exception as exc:
            errors.append(str(exc))
    else:
        warnings.append("Historical evaluation has no source hash map")
    expected_names = ["fast_pid"] + [item.split("=",1)[0] for item in config["policy"]]
    expected = {(name,scenario,mission,seed) for name in expected_names
                for scenario in config["scenarios"] for mission in config["missions"] for seed in config["seeds"]}
    seen, trace_hashes = set(), set()
    matched, groups, verified = defaultdict(list), defaultdict(list), []
    for episode in episodes:
        key = tuple(episode[k] for k in ("controller","scenario","mission","seed"))
        label = "/".join(map(str,key))
        if key in seen:
            errors.append(label+": duplicate episode")
        seen.add(key)
        try:
            path = local_path(episode["trace"], root)
            actual_hash = digest(path)
            if actual_hash != episode["trace_sha256"]:
                raise ValueError("Raw trace SHA256 mismatch")
            if actual_hash in trace_hashes:
                warnings.append(label+": raw trace duplicates another episode byte-for-byte")
            trace_hashes.add(actual_hash)
            with path.open(newline="",encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            metrics = fast_metrics(rows,int(episode["planned_steps"]))
            for field, value in metrics.items():
                if field not in episode:
                    if field == "rmse_z":
                        warnings.append(label+": rmse_z absent in old schema")
                        continue
                    raise ValueError("Missing episode metric " + field)
                if isinstance(value,bool):
                    if bool_value(episode[field]) != value:
                        raise ValueError("Boolean metric mismatch " + field)
                elif not same_number(episode[field],value):
                    raise ValueError(f"Metric {field}: stored={episode[field]}, independently recomputed={value}")
            policy = episode.get("policy")
            if policy:
                policy_path = Path(policy)
                if not policy_path.exists() and "/artifacts/" in policy:
                    policy_path=root/"artifacts"/policy.split("/artifacts/",1)[1]
                if not policy_path.exists():
                    raise ValueError("Policy referenced by episode is missing")
                policy_hash=digest(policy_path)
                if policy_hash != config["policy_hashes"].get(episode["controller"]):
                    raise ValueError("Policy hash mismatch with run config")
                if episode.get("policy_hash",policy_hash)!=policy_hash:
                    raise ValueError("Episode policy hash mismatch")
                train_config=read_json(policy_path.parent/"config.json")
                if set(config["seeds"]) & {train_config["seed"],*train_config["validation_seeds"]}:
                    raise ValueError("Held-out evaluation seeds overlap training or validation")
                if episode.get("training_seed")!=train_config["seed"]:
                    raise ValueError("Episode training seed disagrees with checkpoint configuration")
            matched[key[1:]].append((episode,rows))
            groups[key[:3]].append(metrics)
            verified.append({"controller":key[0],"scenario":key[1],"mission":key[2],"seed":key[3],**metrics})
        except Exception as exc:
            errors.append(label+": "+str(exc))
    if expected != seen:
        errors.append("Coverage mismatch: missing="+str(sorted(expected-seen))+" extra="+str(sorted(seen-expected)))
    for key, entries in matched.items():
        first_episode,first_rows=entries[0]
        for episode,rows in entries[1:]:
            for field in ("initial_state","disturbance","planned_steps"):
                if episode[field] != first_episode[field]:
                    errors.append(str(key)+": mismatched "+field+" for "+episode["controller"])
            for ref,row in zip(first_rows,rows):
                if any(not same_number(ref["target_"+axis],row["target_"+axis]) for axis in ("x","y","z")):
                    errors.append(str(key)+": mismatched target trajectory")
                    break
    for saved in saved_summary:
        key=tuple(saved[k] for k in ("controller","scenario","mission"))
        group=groups.get(key,[])
        if not group:
            errors.append(str(key)+": summary has no independently verified episodes")
            continue
        values={"n":len(group),"survived":sum(row["survived"] for row in group),
                "success":sum(row["success"] for row in group),
                "mean_penalized_rmse":statistics.mean(row["penalized_rmse"] for row in group),
                "mean_rmse":statistics.mean(row["rmse"] for row in group)}
        for name,value in values.items():
            if not same_number(saved.get(name),value):
                errors.append(str(key)+": aggregate mismatch "+name)
    return {"kind":"fast", "folder":str(folder),"status":"PASS" if not errors else "FAIL",
            "episodes":len(episodes),"independently_verified_episodes":len(verified),
            "archived_source_files_verified":snapshot_verified,
            "matched_condition_groups":len(matched),"errors":errors,"warnings":warnings,
            "recomputed":verified,
            "limits":["Trace consistency and hashes do not by themselves prove data origin.",
                      "Fast PID is not ArduPilot; no real-world inference is made.",
                      "setttle_time is first 1s window near target, not post-gust recovery."]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fast",type=Path,action="append",default=[])
    parser.add_argument("--gazebo",type=Path,action="append",default=[])
    parser.add_argument("--training",type=Path,action="append",default=[])
    parser.add_argument("--gazebo-direct",type=Path,action="append",default=[])
    parser.add_argument("--skip-export-check",action="store_true",help="For metadata-only environments; default independently loads PPO and compares portable actor")
    parser.add_argument("--all",action="store_true",help="Inspect all completed artifacts; list interrupted/infrastructure attempts separately")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    excluded=[]
    if args.all:
        for folder in sorted((ROOT/"artifacts"/"training").glob("*")):
            if not folder.is_dir():continue
            if (folder/"training_summary.json").exists():
                args.training.append(folder)
            elif (folder/"config.json").exists():
                excluded.append({"kind":"training","folder":str(folder),"reason":"No completed training_summary; interrupted/in-progress attempt, not accepted training evidence"})
        for path in sorted((ROOT/"artifacts"/"fast").rglob("config.json")):
            if (path.parent/"summary.json").exists():
                args.fast.append(path.parent)
            else:
                excluded.append({"kind":"fast","folder":str(path.parent),"reason":"Evaluation not completed; partial traces retained"})
        for path in sorted((ROOT/"artifacts"/"gazebo").rglob("summary.json")):
            try:
                saved=read_json(path)
                if "frames_observed" in saved and "controller" not in saved:
                    excluded.append({"kind":"gazebo_renderer","folder":str(path.parent),"reason":"Renderer capability smoke; no flight or controller metrics claimed","status":saved.get("status")})
                elif saved.get("status") in ("PASS","FAIL"):
                    args.gazebo.append(path.parent)
                else:
                    excluded.append({"kind":"gazebo","folder":str(path.parent),"reason":"Infrastructure/unfinished result, not accepted flight evidence","status":saved.get("status"),"error":saved.get("error")})
            except Exception as exc:
                excluded.append({"kind":"gazebo","folder":str(path.parent),"reason":"Unreadable summary: "+str(exc)})
        args.fast=list(dict.fromkeys(args.fast)); args.training=list(dict.fromkeys(args.training)); args.gazebo=list(dict.fromkeys(args.gazebo))
        for path in sorted((ROOT/"artifacts"/"gazebo_direct").rglob("summary.json")):
            saved=read_json(path)
            if saved.get("status") in ("PASS","FAIL"):
                args.gazebo_direct.append(path.parent)
            else:
                excluded.append({"kind":"gazebo_direct","folder":str(path.parent),"reason":"Infrastructure/unfinished direct attempt","status":saved.get("status"),"error":saved.get("error")})
        args.gazebo_direct=list(dict.fromkeys(args.gazebo_direct))
        for path in sorted((ROOT/"artifacts"/"user_runs").rglob("summary.json")):
            saved=read_json(path)
            if saved.get("status") in ("PASS","FAIL"):
                if (path.parent/"actuators.jsonl").exists():args.gazebo_direct.append(path.parent)
                elif (path.parent/"gazebo_manifest.json").exists():args.gazebo.append(path.parent)
                else:excluded.append({"kind":"user_run","folder":str(path.parent),"reason":"No supported simulator evidence schema"})
            else:excluded.append({"kind":"user_run","folder":str(path.parent),"reason":"Infrastructure/unfinished user launch","status":saved.get("status")})
    checks=[]
    for folder in args.fast:
        try:
            checks.append(audit_fast(folder.resolve()))
        except Exception as exc:
            checks.append({"kind":"fast","folder":str(folder),"status":"FAIL","errors":[str(exc)]})
    if args.gazebo:
        from gazebo_evidence import audit_gazebo
        for folder in args.gazebo:
            try:
                checks.append(audit_gazebo(folder.resolve()))
            except Exception as exc:
                checks.append({"kind":"gazebo","folder":str(folder),"status":"FAIL","errors":[str(exc)]})
    if args.training:
        from training_evidence import audit_training
        for folder in args.training:
            try:
                checks.append(audit_training(folder.resolve(),not args.skip_export_check))
            except Exception as exc:
                checks.append({"kind":"training","folder":str(folder),"status":"FAIL","errors":[str(exc)]})
    if args.gazebo_direct:
        from direct_evidence import audit_direct
        for folder in args.gazebo_direct:
            try:
                checks.append(audit_direct(folder.resolve()))
            except Exception as exc:
                checks.append({"kind":"gazebo_direct","folder":str(folder),"status":"FAIL","errors":[str(exc)]})
    sensitivity=ROOT/"artifacts/provenance/baseline_sensitivity"
    if args.all and (sensitivity/"summary.json").exists():
        from sensitivity_evidence import audit_sensitivity
        try:checks.append(audit_sensitivity(sensitivity))
        except Exception as exc:checks.append({"kind":"baseline_sensitivity","folder":str(sensitivity),"status":"FAIL","errors":[str(exc)]})
    result={"utc":datetime.now(timezone.utc).isoformat(),"auditor_script_sha256":digest(Path(__file__)),
            "auditor_source_hashes":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
            "scope":"Consistency of completed saved artifacts; PASS is not a claim every controller or exploratory attempt succeeded.",
            "status":"PASS" if checks and all(c["status"]=="PASS" for c in checks) else "FAIL",
            "excluded_attempts":excluded,"checks":checks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":len(checks),"output":str(args.output)},ensure_ascii=False))
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    sys.exit(main())

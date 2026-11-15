"""Replay every teacher label from saved measurements and archived source only."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = directory / "observations_actions.npz"
    checks = {"dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest() == manifest["dataset_sha256"],
              "valid_manifest": manifest["valid"],
              "development_seeds_only": all(101 <= seed <= 60000 for seed in manifest["seeds"]),
              "no_source_changes": not manifest["source_files_changed_during_collection"]}
    for name, expected_hash in manifest["source_hashes_before"].items():
        checks["snapshot:" + name] = hashlib.sha256((directory / "source_snapshot" / name).read_bytes()).hexdigest() == expected_hash
    source = directory / "source_snapshot" / "reference" / "dob.py"
    spec = importlib.util.spec_from_file_location("archived_dob_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controller = module.DOBReference(dt=manifest["teacher"]["dt"], config=module.DOBConfig(**manifest["teacher"]["config"]))
    started = time.monotonic()
    maximum_error = 0.0
    with np.load(dataset_path) as data:
        checks["only_measured_inputs_and_labels"] = set(data.files) == {"observations", "actions", "episode_ids"}
        observations, actions, episode_ids = data["observations"], data["actions"], data["episode_ids"]
        checks["sample_shapes"] = observations.shape == (manifest["samples"], manifest["observation_size"]) and actions.shape == (manifest["samples"], 4)
        checks["finite_data"] = bool(np.isfinite(observations).all() and np.isfinite(actions).all())
        checks["episode_indexing"] = True
        for episode in manifest["episodes"]:
            start, count = episode["sample_start"], episode["sample_count"]
            checks["episode_indexing"] &= bool(np.all(episode_ids[start:start + count] == episode["episode_id"]))
            controller.reset()
            for index in range(start, start + count):
                computed_action = controller.predict(observations[index]).astype(np.float32)
                maximum_error = max(maximum_error, float(np.max(np.abs(computed_action - actions[index]))))
    checks["all_labels_reproduced_from_measured_observations"] = maximum_error <= 1e-7
    result = {"pass": all(checks.values()), "checks": checks, "samples_replayed": manifest["samples"],
              "maximum_action_difference": maximum_error, "elapsed_monotonic_s": time.monotonic() - started,
              "source": "archived DOB code, archived measured observations only"}
    (directory / "audit_measured_replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.directory)
    print(json.dumps(result))
    raise SystemExit(0 if result["pass"] else 1)

"""Independent actual-force ON/OFF causality and hover-recovery audit.

The force log is corroborated against the second derivative of independent Gazebo
world position. Time windows and all ON/OFF transitions are retained.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from packet_clock_ordered import ordered_clock
from independent_metrics import compute_metrics

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def acceleration_jump(truth_time, positions, event_time, half_window):
    selected = np.abs(truth_time - event_time) <= half_window
    time = truth_time[selected] - event_time
    if len(time) < 7 or np.count_nonzero(time < 0) < 3 or np.count_nonzero(time >= 0) < 3:
        return {"valid": False, "samples": len(time)}
    scaled = time / half_window
    positive = np.maximum(scaled, 0.)
    # Continuous position/velocity, piecewise acceleration and jerk. Smooth
    # attitude-driven acceleration is represented by common polynomial terms.
    design = np.column_stack([np.ones(len(time)), scaled, scaled**2, scaled**3,
                              .5 * positive**2, positive**3])
    coeff = np.linalg.lstsq(design, positions[selected], rcond=None)[0]
    jump = coeff[4] / half_window**2
    residual = design @ coeff - positions[selected]
    return {"valid": True, "samples": len(time), "delta_acceleration_enu_m_s2": jump.tolist(),
            "position_fit_rms_m": float(np.sqrt(np.mean(residual**2)))}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE.parent / "artifacts/gazebo/dev_pid_gust_plugin_ascii_82001")
    parser.add_argument("--output", type=Path, default=HERE / "force_plugin_evidence.json")
    args = parser.parse_args()
    folder = args.run
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "gazebo_manifest.json").read_text())
    physics = read_jsonl(folder / "force_physics.jsonl")
    published = read_jsonl(folder / "gazebo_events.jsonl")
    published = {row["seq"]: row for row in published if row["event"] == "force_command_published"}
    checks = []
    checks.append({"check": "monotonic_actual_physics_ticks", "pass": bool(np.all(np.diff([row["sim_time"] for row in physics]) > 0))})
    checks.append({"check": "applied_valid_link_and_sequence", "pass": all(row["link_valid"] and row["entity_id"] == 10 and row["seq"] == i for i, row in enumerate(physics))})
    checks.append({"check": "published_commands_equal_physics_applied", "pass": all(row["seq"] in published and
                   np.max(np.abs(np.array(row["force_enu_N"]) - published[row["seq"]]["force_enu_N"])) < 1e-12 for row in physics)})
    for key, name in [("force_plugin_sha256", "libVkrForceSystem.so"), ("force_plugin_source_sha256", "ForceSystem.cc")]:
        checks.append({"check": "archived_" + name, "pass": hashlib.sha256((folder / "gazebo_assets" / name).read_bytes()).hexdigest() == manifest[key]})
    truth_rows = read_jsonl(folder / "groundtruth.jsonl")
    unique = {row["sim_time"]: row for row in truth_rows if row.get("name") == "iris"}
    truth_time = np.array(sorted(unique))
    positions = np.array([unique[t]["position"] for t in truth_time])
    transitions = []
    old_force = np.zeros(3)
    actual_offs = []
    for event in physics:
        force = np.array(event["force_enu_N"])
        delta = force - old_force
        if np.linalg.norm(delta) > 1e-9:
            expected = delta / manifest["total_link_mass_kg"]
            windows = []
            for width in (.06, .08, .10, .12):
                estimate = acceleration_jump(truth_time, positions, event["sim_time"], width)
                estimate["half_window_s"] = width
                if estimate["valid"]:
                    actual = np.array(estimate["delta_acceleration_enu_m_s2"])
                    estimate["projected_ratio_to_force_over_mass"] = float(np.dot(actual, expected) / np.dot(expected, expected))
                    estimate["direction_cosine"] = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
                windows.append(estimate)
            primary = windows[1]
            causality = primary["valid"] and .7 <= primary["projected_ratio_to_force_over_mass"] <= 1.3 and primary["direction_cosine"] > .95
            checks.append({"check": "truth_acceleration_change_seq_" + str(event["seq"]), "pass": bool(causality)})
            transitions.append({"seq": event["seq"], "event_sim_time": event["sim_time"], "delta_force_N": delta.tolist(),
                                "expected_delta_acceleration_m_s2": expected.tolist(), "estimates": windows})
            if np.linalg.norm(old_force) > 1e-9 and np.linalg.norm(force) < 1e-9:
                actual_offs.append(event["sim_time"])
        old_force = force
    with (folder / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    column = lambda key: np.array([float(row[key]) for row in rows])
    time, boot = column("t"), column("boot_s")
    clock = ordered_clock(folder / "network.pcap.gz", float(boot[0]), float(boot[-1]),
                          summary.get("tcp_port", 5760), summary.get("plugin_port", 9002))
    offset = clock["median_gazebo_minus_boot_s"]
    off_elapsed = actual_offs[-1] - offset - summary["measurement_boot_start_s"]
    position = np.column_stack([column(key) for key in ("x_e", "y_n", "z_up")])
    target = np.column_stack([column(key) for key in ("target_e", "target_n", "target_up")])
    metrics = compute_metrics(time=time, position=position, target=target, yaw=column("yaw_error_rad"), yaw_target=0.,
                              planned_seconds=summary["requested_duration_s"], failed=summary.get("failure") is not None,
                              last_force_off_s=off_elapsed)
    comparisons = {"rmse_m": "rmse_position_m", "penalized_rmse_m": "penalized_rmse_position_m",
                   "max_altitude_drop_m": "altitude_drop_m", "heading_rmse_deg": "heading_rmse_deg",
                   "heading_peak_deg": "heading_peak_deg"}
    for own, reported in comparisons.items():
        checks.append({"check": "independent_" + reported, "pass": abs(metrics[own] - summary[reported]) < 1e-9})
    checks.append({"check": "recovery_from_actual_last_OFF", "pass": abs(metrics["recovery"]["time_s"] - summary["recovery_time_s"]) < 1e-9,
                   "recomputed_off_s": off_elapsed, "recomputed_recovery_s": metrics["recovery"]["time_s"]})
    for row in checks:
        row["pass"] = bool(row["pass"])
    result = {"status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "scope": "development force plugin causality and metric audit; not heldout controller comparison",
              "run": str(folder), "checks": checks, "force_transition_truth_evidence": transitions, "capture_order_clock": clock,
              "independent_metrics": metrics,
              "causality_method": "All six ON/OFF transitions; independent world-position piecewise cubic fit. Primary half-window80ms, sensitivity60/100/120ms. Required projected acceleration ratio0.7..1.3 and direction cosine>0.95; does not measure detailed aerodynamic realism."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "recovery_s": metrics["recovery"]["time_s"],
                      "force_acceleration_ratios_80ms": [row["estimates"][1].get("projected_ratio_to_force_over_mass") for row in transitions]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

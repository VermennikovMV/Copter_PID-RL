"""Read archived three-aircraft demos; never start physics or an optimizer.

PASS describes evidence integrity. Controller failures and incomplete routes are
retained independently, including the fixed full-horizon failure penalty.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "v2" if (PROJECT / "v2/quadrl/policy.py").is_file() else PROJECT
ROLES = ("pid", "hybrid", "direct")
POINTS = np.array([[0, 0, 3], [2, 0, 3], [2, 2, 3.5], [0, 2, 3], [0, 0, 3]], float)
MODEL_SHA = {
    "direct": "5d0d19e02c30c62ed3a0efdcf003d8b6b635ae1eea9b1d32b4737836c46ff900",
    "F0": "f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461",
    "route": "ce86bb56d7e0f5c1775266d4a45cd3417e979303d27cc662a58e8070bf34e345",
}
PATHS = {
    "direct": "artifacts/training/main_direct_seed111_v2/best_policy.npz",
    "F0": "artifacts/refinement/joint_v2/route_memory_prototype_v20/root02/zero_route_integral/policy.npz",
    "route": "artifacts/refinement/joint_v2/prior_ablation_pilot_v28_111/prior0/final.npz",
}
PID_PARAMETERS = {"PSC_NE_POS_P": 1.5, "PSC_NE_VEL_P": 4., "PSC_NE_VEL_I": 2., "PSC_NE_VEL_D": .5}
PWM_HOVER = np.array([1533.216, 1533.215, 1533.214, 1533.217])
ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
# Evidence tolerances, fixed in this auditor before the archived demo readback.
NN_TOL = 2e-5
COORDINATE_TOL_M = .25
TRUTH_MAX_TIME_GAP_S = .060001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON token: " + value)


def archived_asset(out, declared):
    """Resolve the copied archive, never an old tmpfs path or another run."""
    text = str(declared).replace("\\", "/")
    marker = "gazebo_assets/"
    if marker not in text:
        raise ValueError("Expected an explicitly archived Gazebo asset: " + text)
    result = (out / "gazebo_assets" / text.split(marker, 1)[1]).resolve()
    if not result.is_relative_to((out / "gazebo_assets").resolve()):
        raise ValueError("Gazebo asset escapes this archived run")
    if not result.is_file():
        raise FileNotFoundError("Required copied Gazebo asset missing: " + str(result))
    return result


class Audit:
    def __init__(self):
        self.inputs = {}
        self.failures = []
        self.checks = 0
        self.unused_mavlink_missing_fields = Counter()

    def check(self, condition, label, detail=None):
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": label, "detail": detail})

    def read(self, path, lines=False, mavlink=False):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.inputs[str(path)] = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8-sig")
        if lines:
            if not mavlink:
                return [json.loads(line, parse_constant=reject_constant) for line in text.splitlines() if line.strip()]
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            # HOME_POSITION.q is unused by these controllers. Preserve the raw
            # standard missing quaternion instead of rewriting the MAV archive.
            # Every other nonfinite token, including any actor-state field, is
            # rejected. Commands and physical evidence always use strict JSON.
            def inspect(value, keys, row):
                if isinstance(value, dict):
                    for key, child in value.items():
                        inspect(child, keys + (key,), row)
                elif isinstance(value, list):
                    for key, child in enumerate(value):
                        inspect(child, keys + (key,), row)
                elif isinstance(value, float) and not math.isfinite(value):
                    allowed = (math.isnan(value) and len(keys) == 3 and keys[:2] == ("message", "q")
                               and keys[2] in range(4) and row.get("message", {}).get("mavpackettype") == "HOME_POSITION")
                    if not allowed:
                        raise ValueError("Nonfinite used/unsupported MAV field: " + str(keys))
                    self.unused_mavlink_missing_fields[str(path) + ":HOME_POSITION.q[" + str(keys[2]) + "]"] += 1
            for row in rows:
                inspect(row, (), row)
            return rows
        return json.loads(text, parse_constant=reject_constant)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        self.inputs[str(path)] = digest
        if expected is not None:
            self.check(digest == expected, "immutable_sha:" + str(path))
        return digest


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def max_error(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 1e30
    return float(np.max(np.abs(a - b))) if a.size else 0.


def nearest(rows, stamps, stamp):
    index = bisect_left(stamps, stamp)
    indices = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    chosen = min(indices, key=lambda i: abs(stamps[i] - stamp))
    return rows[chosen], abs(stamps[chosen] - stamp)


def sample_time(row):
    # Old archives keep their actual old mission clock. A corrected runtime
    # must explicitly log the sensor clock rather than rewriting old evidence.
    return float(row["mission_time_s"] if "mission_time_s" in row else row["measurement_t"])


def verify_mission_sample_clock(row, epoch_ms, position_message, attitude_message, audit, role):
    if epoch_ms is None:
        audit.check("mission_time_s" not in row, role + ":no_unbound_replacement_of_legacy_mission_clock")
        return
    expected = (max(int(position_message["time_boot_ms"]), int(attitude_message["time_boot_ms"])) - epoch_ms) * .001
    audit.check(row["mission_epoch_boot_ms"] == epoch_ms and abs(row["mission_epoch_boot_s"] - epoch_ms * .001) < 1e-10
                and row["mission_clock_source"] == "actual_sensor_pair_relative_to_one_common_raw_epoch"
                and math.isfinite(row["mission_time_s"]) and abs(row["mission_time_s"] - expected) < 1e-9 and expected >= 0.,
                role + ":fixed_epoch_and_raw_sensor_mission_clock_every_row")


def recompute_route(rows, mission, audit, role):
    """Independently reconstruct ordered dwell and PRE-active finite segments."""
    index, inside, previous, completion = 0, None, None, None
    events, active, distances, weights, gap = [], [], [], [], 0.
    for row in rows:
        t, p = sample_time(row), np.asarray(row["task_position"], float)
        audit.check(max_error(row["target_before_update"], POINTS[index]) < 1e-10, role + ":PRE_target")
        width = t - (previous if previous is not None else 0.)
        if previous is not None:
            gap = max(gap, width)
        if mission == "route" and index > 0 and completion is None:
            a, b = POINTS[index - 1], POINTS[index]
            q = float(np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0., 1.))
            active.append((t, index, p.copy()))
            distances.append(float(np.linalg.norm(p - (a + q * (b - a)))))
            weights.append(width)
        if mission == "route" and completion is None:
            if previous is not None and width > .03000000001:
                inside = None
            if np.linalg.norm(p - POINTS[index]) > .25:
                inside = None
            else:
                if inside is None:
                    inside = t
                if t - inside + 1e-9 >= .5:
                    events.append({"goal_index": index, "time_s": t})
                    inside = None
                    if index == 4:
                        completion = t
                    else:
                        index += 1
        audit.check(max_error(row["target_next"], POINTS[index]) < 1e-10, role + ":POST_target")
        logged = row.get("goal_event")
        expected_event = bool(events and abs(events[-1]["time_s"] - t) < 1e-10)
        audit.check(bool(logged) == expected_event, role + ":goal_event_presence")
        if logged:
            audit.check(logged["goal_index"] == events[-1]["goal_index"] and abs(logged["time_s"] - t) < 1e-8,
                        role + ":goal_event_identity")
        previous = t
    corners = []
    for corner in (1, 2, 3):
        transition = next((e["time_s"] for e in events if e["goal_index"] == corner), None)
        samples = [] if transition is None else [r for r in rows if transition - 1e-9 <= sample_time(r) <= transition + 1. + 1e-9]
        if samples and sample_time(rows[-1]) >= transition + 1. - .060001:
            a, b = POINTS[corner], POINTS[corner + 1]
            p = np.array([r["task_position"] for r in samples])
            q = np.clip((p - a) @ (b - a) / np.dot(b - a, b - a), 0., 1.)
            peak = float(np.linalg.norm(p - (a + q[:, None] * (b - a)), axis=1).max())
        else:
            peak = None
        corners.append({"corner": corner, "transition_s": transition, "finite_outgoing_peak_m": peak})
    complete_geometry = completion is not None and gap <= TRUTH_MAX_TIME_GAP_S and bool(weights)
    return {"route_completed": completion is not None if mission == "route" else None,
            "route_completion_time_s": completion, "restricted_route_time_s": completion if completion is not None else (40. if mission == "route" else None),
            "goal_events": events, "active_geometry_rows": len(active), "maximum_control_gap_s": gap,
            "finite_active_segment_rmse_m": float(np.sqrt(np.average(np.square(distances), weights=weights))) if complete_geometry else None,
            "finite_active_segment_peak_m": max(distances) if distances else None,
            "geometry_complete": complete_geometry if mission == "route" else None, "corners": corners if mission == "route" else []}


def thesis_metrics(rows, horizon, failure, mission, yaw_reference, force_off):
    t = np.array([sample_time(r) for r in rows], float)
    pos = np.array([r["task_position"] for r in rows], float)
    targets = np.array([r["target_before_update"] for r in rows], float)
    error = np.linalg.norm(pos - targets, axis=1)
    w = np.diff(np.r_[0., np.minimum(t, horizon)])
    covered = float(w.sum())
    survived = failure is None and t[-1] >= horizon - .040000001
    integral = float(np.dot(w, error ** 2))
    remainder = max(0., horizon - covered) * (error[-1] ** 2 if survived else 16.)
    yaw = np.array([math.atan2(np.asarray(r["rotation"])[1, 0], np.asarray(r["rotation"])[0, 0]) for r in rows])
    heading = np.arctan2(np.sin(yaw_reference - yaw), np.cos(yaw_reference - yaw)) * 180. / np.pi
    recovery, confirmed, inside, previous = None, None, None, None
    if mission == "hover" and force_off is not None:
        for stamp, distance in zip(t, error):
            if stamp < force_off - 1e-9:
                continue
            if previous is not None and stamp - previous > .030001:
                inside = None
            previous = stamp
            if distance > .25:
                inside = None
                continue
            if inside is None:
                inside = float(stamp)
            if stamp - inside >= 1. - 1e-9:
                recovery, confirmed = max(0., inside - force_off), float(stamp)
                break
    return {"rmse_position_m": math.sqrt(integral / covered) if covered > 0 else None,
            "penalized_rmse_position_m": math.sqrt((integral + remainder) / horizon),
            "altitude_drop_m": max(0., float(np.max(targets[:, 2] - pos[:, 2]))),
            "heading_rmse_deg": math.sqrt(float(np.dot(w, heading ** 2)) / covered) if covered > 0 else None,
            "heading_peak_deg": float(np.abs(heading).max()), "survived": bool(survived), "failure": failure,
            "recovery_applicable": mission == "hover" and force_off is not None,
            "recovery_time_s": recovery, "recovery_confirmed_at_s": confirmed,
            "recovery_censored": recovery is None if mission == "hover" and force_off is not None else None,
            "actual_force_off_s": force_off, "duration_s": float(t[-1]), "horizon_s": horizon,
            "failure_remainder_error_m": 4., "weighting": "preceding control interval; preserved 4m failed remainder"}


def forward(weights, observation):
    x = np.asarray(observation, np.float32)
    for i in range(int(weights["layers"])):
        x = x @ weights["w" + str(i)].T + weights["b" + str(i)]
        if i + 1 < int(weights["layers"]):
            x = np.tanh(x)
    return np.clip(x, -1., 1.)


def physical_model_signature(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.text = element.text.strip() if element.text else None
        element.tail = None
        if element.tag == "model":
            element.set("name", "same_physical_model")
        for child in list(element):
            if child.tag == "visual":
                element.remove(child)
        if element.tag == "include":
            if element.find("uri") is not None:
                element.find("uri").text = "same_physical_body"
            if element.find("name") is not None:
                element.find("name").text = "same_physical_body"
        if element.tag == "plugin" and element.get("name") == "ArduPilotPlugin":
            element.find("fdm_port_in").text = "INDEPENDENT_PORT"
    return ET.tostring(root)


def verify_start_gates(out, ready, start, audit):
    rows = audit.read(out / "start_gate.jsonl", lines=True)
    result = {}
    for label, gate in (("READY", ready["qualified_start_gate"]), ("MEASUREMENT_START", start["qualified_start_gate"])):
        audit.check(gate["position_tolerance_m"] == .03 and gate["speed_tolerance_mps"] == .05 and gate["required_stable_s"] == 1., label + ":fixed_fair_start_limits")
        since, end = gate["stable_since_sim_time"], gate["sim_time"]
        audit.check(since is not None and end - since >= 1. - 1e-9, label + ":qualified_one_second_span")
        evidence = [r for r in rows if since - 1e-9 <= r["sim_time"] <= end + 1e-9]
        audit.check(bool(evidence) and abs(evidence[0]["sim_time"] - since) < 1e-8 and abs(evidence[-1]["sim_time"] - end) < 1e-8,
                    label + ":full_gate_interval_archived")
        errors, speeds, spread = [], [], []
        for row in evidence:
            positions = []
            all_passed = True
            for role in ROLES:
                state = row["states"][role]
                p, v = np.asarray(state["position"], float), np.asarray(state["velocity"], float)
                if not np.isfinite(p).all() or not np.isfinite(v).all():
                    raise ValueError("Nonfinite fair-start actor state")
                error, speed = float(np.linalg.norm(p - POINTS[0])), float(np.linalg.norm(v))
                within = error <= .03 and speed <= .05
                audit.check(abs(error - state["goal0_error_m"]) < 1e-8 and abs(speed - state["speed_mps"]) < 1e-8
                            and bool(state["within_position_tolerance"]) == (error <= .03)
                            and bool(state["within_speed_tolerance"]) == (speed <= .05), label + ":independent_gate_state:" + role)
                all_passed = all_passed and within
                errors.append(error); speeds.append(speed); positions.append(p)
            spread.append(max(float(np.linalg.norm(a - b)) for a in positions for b in positions))
            audit.check(all_passed and row["passed"] is True, label + ":all_three_passed_every_gate_sample")
        audit.check(any(r == gate for r in rows), label + ":qualified_snapshot_equals_actual_gate_row")
        result[label] = {"stable_since_sim_time": since, "qualified_at_sim_time": end, "span_s": end - since,
                         "gate_samples": len(evidence), "maximum_goal0_error_m": max(errors), "maximum_speed_mps": max(speeds),
                         "maximum_pairwise_task_position_spread_m": max(spread)}
    audit.check(0. <= start["world_sim_time"] - start["qualified_start_gate"]["sim_time"] <= .160001,
                "measurement_follows_actual_qualified_common_state")
    return rows, result


def formal_demo_case(plan):
    return plan["seed"] == 83301 and (plan["mission"], plan["scenario"], float(plan["duration_s"])) in (
        ("route", "force_strong", 40.), ("hover", "gust", 30.))


def verify_cadence(out, plan, summary, commands, t0, audit):
    required = formal_demo_case(plan)
    expected = round(float(plan["duration_s"]) * 50.)
    present = "control_cadence_gate" in summary
    coverage = {role: len(commands.get(role, [])) / expected for role in ROLES}
    passed = all(.95 <= fraction <= 1.05 for fraction in coverage.values())
    if required:
        audit.check(present, "formal_demo_requires_actual_shared_50Hz_schedule")
        audit.check(passed, "formal_demo_requires_at_least_95_percent_expected_slots", coverage)
    if not present:
        return {"required_for_formal_demo": required, "schedule_archived": False, "target_hz": 50.,
                "expected_slots_each": expected, "actual_coverage_by_role": coverage, "passed": passed,
                "scope": "legacy smoke counts reported; no formal 50Hz acceptance"}
    gate = summary["control_cadence_gate"]
    audit.check(gate["target_hz"] == 50. and gate["minimum_expected_slot_fraction"] == .95
                and gate["common_fresh_position_and_attitude_barrier"] is True and gate["no_duplicate_observation_forwards"] is True,
                "fixed_cadence_contract")
    schedule = audit.read(out / "control_schedule.jsonl", lines=True)
    executed = [r for r in schedule if r["event"] == "shared_fresh_control_slot"]
    skipped = [r for r in schedule if r["event"] == "missed_control_slots"]
    audit.check(all(r["event"] in ("shared_fresh_control_slot", "missed_control_slots") for r in schedule), "known_schedule_record_types")
    slots = {int(r["slot"]): r for r in executed}
    audit.check(len(slots) == len(executed), "one_execution_per_shared_slot")
    missed = [slot for row in skipped for slot in range(int(row["first_slot"]), int(row["first_slot"]) + int(row["count"]))]
    audit.check(len(set(missed)) == len(missed) and not (set(missed) & set(slots)), "missed_slots_are_explicit_and_never_forwarded")
    audit.check(sum(int(row["count"]) for row in skipped) == gate["missed_slots"], "actual_missed_slot_count")
    all_slots = set(missed) | set(slots)
    audit.check(all(0 <= slot < expected for slot in all_slots) and all_slots == set(range(max(all_slots) + 1)), "schedule_is_a_fixed_50Hz_prefix")
    for slot, row in slots.items():
        audit.check(abs(row["scheduled_sim_time"] - (t0 + .02 * slot)) < 1e-7
                    and abs(row["lateness_s"] - (row["observed_sim_time"] - row["scheduled_sim_time"])) < 1e-8
                    and -.00000001 <= row["lateness_s"] < .020001, "actual_fixed_slot_time_not_reset_to_latest_observation")
    for role in ROLES:
        reported = gate["roles"][role]
        rows = commands.get(role, [])
        audit.check(reported["samples"] == len(rows) and reported["expected_slots"] == expected
                    and abs(reported["coverage"] - coverage[role]) < 1e-10
                    and bool(reported["passed"]) == (.95 <= coverage[role] <= 1.05), role + ":independent_expected_slot_coverage")
        used, previous_pair = set(), None
        for row in rows:
            slot = row["shared_control_slot"]
            audit.check(slot in slots and slot not in used, role + ":one_forward_for_actual_common_slot")
            used.add(slot)
            reference = slots[slot]
            audit.check(abs(row["scheduled_control_sim_time"] - reference["scheduled_sim_time"]) < 1e-8
                        and abs(row["world_sim_time"] - reference["observed_sim_time"]) < 1e-8, role + ":shared_slot_and_world_stamp")
            pair = row["measured_input_pair_boot_ms"]
            audit.check(previous_pair is None or all(current > prior for current, prior in zip(pair, previous_pair)), role + ":fresh_position_AND_attitude_each_forward")
            audit.check(pair == reference["input_pairs_boot_ms"][role], role + ":exact_same_received_input_pair_as_common_barrier")
            prefix = reference["input_receive_prefixes"][role]
            audit.check(all(row[key] == prefix[key] for key in ("receive_cutoff", "position_record_index", "attitude_record_index")),
                        role + ":no_hidden_redrain_after_common_barrier")
            previous_pair = pair
    audit.check((gate["status"] == "PASS") == passed, "producer_cadence_result_matches_actual_counts")
    return {"required_for_formal_demo": required, "schedule_archived": True, "target_hz": 50., "minimum_slot_fraction": .95,
            "expected_slots_each": expected, "actual_coverage_by_role": coverage, "executed_common_slots": len(slots),
            "explicit_missed_slots": len(missed), "passed": passed}


def force_audit(out, manifest, events, t0, duration, audit):
    bindings = {r["name"]: r for r in events if r.get("event") == "unique_model_link_binding"}
    expected = set(manifest["vehicles"])
    audit.check(set(bindings) == expected, "three_explicit_scene_hierarchy_bindings")
    audit.check(len({v["base_link_id"] for v in bindings.values()}) == 3, "three_distinct_base_links")
    force_rows = {}
    for name, spec in manifest["vehicles"].items():
        path = out / Path(spec["force_log"]).name
        rows = audit.read(path, lines=True)
        rows = [r for r in rows if r.get("event") == "force_physics_step"]
        audit.check(bool(rows), name + ":actual_force_rows")
        audit.check(all(r["link_valid"] and r.get("group_atomic") is True and r["entity_id"] == bindings[name]["base_link_id"] for r in rows), name + ":atomic_actual_force_entity")
        audit.check(len({int(r["seq"]) for r in rows}) == len(rows), name + ":unique_force_sequence")
        force_rows[name] = {int(r["seq"]): r for r in rows}
    all_sequences = set.union(*(set(v) for v in force_rows.values()))
    for name, values in force_rows.items():
        audit.check(set(values) == all_sequences, name + ":all_shared_force_sequences")
    ledger = []
    for seq in sorted(all_sequences):
        group = [values[seq] for values in force_rows.values() if seq in values]
        if len(group) != 3:
            continue
        audit.check(len({int(v["iteration"]) for v in group}) == 1 and max(v["sim_time"] for v in group) - min(v["sim_time"] for v in group) < 1e-9,
                    "shared_force_exact_physics_tick:" + str(seq))
        audit.check(all(max_error(v["force_enu_N"], group[0]["force_enu_N"]) < 1e-10 for v in group), "shared_force_vector:" + str(seq))
        ledger.append({"seq": seq, "sim_time": group[0]["sim_time"], "measurement_t": group[0]["sim_time"] - t0,
                       "iteration": group[0]["iteration"], "force_enu_N": group[0]["force_enu_N"]})
    nonzero = [r for r in ledger if np.linalg.norm(r["force_enu_N"]) > 1e-10 and -.1 <= r["measurement_t"] < duration]
    last_off = None
    if nonzero:
        last_on = max(r["sim_time"] for r in nonzero)
        zero_after = [r["measurement_t"] for r in ledger if r["sim_time"] > last_on and np.linalg.norm(r["force_enu_N"]) < 1e-10]
        if zero_after:
            last_off = min(zero_after)
    return ledger, last_off


def audit_run(out, audit):
    plan = audit.read(out / "plan.json")
    summary = audit.read(out / "complete.json")
    manifest = audit.read(out / "manifest.json")
    scene = audit.read(out / "gazebo_manifest.json")
    start = audit.read(out / "MEASUREMENT_START.json")
    ready = audit.read(out / "READY.json")
    start_gate_rows, qualified_start = verify_start_gates(out, ready, start, audit)
    events = audit.read(out / "scene_events.jsonl", lines=True)
    t0, duration = float(summary["measurement_sim_start_s"]), float(plan["duration_s"])
    epoch_ms = start.get("mission_epoch_boot_ms")
    if formal_demo_case(plan):
        audit.check(epoch_ms is not None, "formal_demo_requires_explicit_actual_sensor_mission_epoch")
    if epoch_ms is not None:
        audit.check(isinstance(epoch_ms, int) and not isinstance(epoch_ms, bool)
                    and abs(start["mission_epoch_boot_s"] - epoch_ms * .001) < 1e-10
                    and start["mission_epoch_source"] == "first_common_actual_sensor_pair_before_any_controlled_forward",
                    "one_explicit_frozen_actual_sensor_epoch")
        audit.check(set(start["mission_epoch_raw_evidence"]) == set(ROLES), "mission_epoch_has_all_three_raw_received_prefixes")
        epoch_world = float(start["mission_epoch_observer_world_sim_time"])
        audit.check(abs(epoch_world - t0 - start["mission_epoch_observer_measurement_t"]) < 1e-8 and epoch_world >= t0 - 1e-8,
                    "explicit_sensor_epoch_to_observer_world_anchor")
    else:
        epoch_world = t0
    audit.check(summary["one_physical_world"] is True and plan["physical_world_count"] == 1, "one_real_physical_world")
    audit.check(start["world_sim_time"] == t0 and summary["world"] == scene["world"] == plan["world"], "common_world_and_t0")
    audit.check(plan["new_training_steps"] == summary["new_training_steps"] == 0 and summary["scientific_matrix_updated"] is False, "demo_not_training_or_research_acceptance")
    audit.check(set(summary["roles"]) == set(ROLES), "exact_three_roles")
    audit.check(len(set(manifest["SITL_pids"])) == 3, "three_distinct_SITL_processes")
    world_path = archived_asset(out, scene["world_sdf"])
    audit.bind(world_path, scene["world_sha256"])
    world = ET.parse(world_path).getroot().find("world")
    audit.check(world is not None and world.get("name") == plan["world"], "actual_world_identity")
    audit.check(abs(float(world.find("physics/max_step_size").text) - .001) < 1e-12, "physics_step_1ms")
    audit.check(len([x for x in world.findall("include") if x.findtext("name") in scene["vehicles"]]) == 3, "three_physical_model_includes")
    signatures = {"body_model_sdf": [], "control_model_sdf": []}
    physical_masses = {}
    for spec in scene["vehicles"].values():
        for kind in signatures:
            path = archived_asset(out, spec[kind])
            audit.bind(path)
            signatures[kind].append(physical_model_signature(path))
        physical_masses[spec["name"]] = sum(float(n.text) for n in ET.parse(archived_asset(out, spec["body_model_sdf"])).getroot().findall(".//link/inertial/mass"))
    audit.check(all(len(set(values)) == 1 for values in signatures.values()), "three_identical_body_motor_physics_except_visuals_names_ports")
    audit.check(all(abs(mass - 1.75) < 1e-12 for mass in physical_masses.values()) and abs(scene["total_mass_kg_each"] - 1.75) < 1e-12,
                "actual_XML_mass_1p75_kg_each")
    assets = world_path.parent
    original_body = assets / "iris_with_standoffs/model.sdf"
    body_source_hash = next(value for key, value in scene["sources"].items() if key.endswith("/iris_with_standoffs/model.sdf"))
    audit.bind(original_body, body_source_hash)
    audit.check(all(value == physical_model_signature(original_body) for value in signatures["body_model_sdf"]), "physical_body_XML_equals_archived_upstream_except_visuals_name")
    for filename in ("GroupForceSystem.cc", "libThreeDroneForceSystem.so"):
        source_hash = next(value for key, value in scene["sources"].items() if key.endswith("/" + filename))
        audit.bind(assets / filename, source_hash)
    source_snapshot = out / "source_snapshot"
    for relative, expected in manifest["source_sha256"].items():
        original = Path(str(relative).replace("\\", "/"))
        snapshot = source_snapshot / (original.parent.name + "_" + original.name)
        audit.bind(snapshot, expected)
    force_ledger, force_off = force_audit(out, scene, events, t0, duration, audit)
    force_off_mission = None if force_off is None else force_off + t0 - epoch_world
    expected_events = start["common_force_schedule"]["force_events"]
    physical_changes = [r for r in force_ledger if -.1 <= r["measurement_t"] <= duration + .1]
    for event in expected_events:
        for stamp, vector in ((event["on_s"], event["force_enu_N"]), (event["off_s"], [0., 0., 0.])):
            matches = [r for r in physical_changes if max_error(r["force_enu_N"], vector) < 1e-9 and abs(r["measurement_t"] - stamp) <= .100001]
            audit.check(bool(matches), "scheduled_force_boundary_physically_observed", {"time": stamp, "force": vector})
    truth = audit.read(out / "groundtruth.jsonl", lines=True)
    truth_by_name = {name: sorted([r for r in truth if r["name"] == name], key=lambda r: r["sim_time"]) for name in scene["vehicles"]}
    public = plan.get("public_root")
    # Public scripts live at repository/demo_three, policies at repository/v2.
    # The original machine's absolute public_root is lineage, not a dependency
    # on a surviving tmpfs or private checkout when replaying a copied archive.
    code_root = DATA_ROOT
    if not (code_root / "quadrl/policy.py").is_file():
        raise FileNotFoundError("Local NumPy controller source package is absent")
    sys.path.insert(0, str(code_root))
    # Imports are pure NumPy inference modules; no env/Gazebo/optimizer exists here.
    old = importlib.import_module("quadrl.policy")
    memory = importlib.import_module("quadrl_route_memory_v20.policy")
    model_inputs = audit.read(out / "source_snapshot/model_inputs.json")
    for name, item in model_inputs.items():
        digest = audit.bind(out / "source_snapshot/models" / name, item["snapshot_sha256"])
        audit.check(digest == item["sha256"], "actual_model_snapshot_matches_input:" + name)
    calibration = audit.read(out / "source_snapshot/models/calibration.json")
    expected_cal = {"hover_pwm_gazebo_order": PWM_HOVER.tolist(), "n_samples": 1000, "window": "measurement time >5 seconds", "reference_fraction": .5, "nominal_only": True}
    audit.check(all(calibration.get(key) == value for key, value in expected_cal.items()),
                "unchanged_fixed_nominal_direct_calibration")
    calibration_sha = sha(out / "source_snapshot/models/calibration.json")
    original_cal_sha = "2e20df83b9dadf5231c2f93e8070d89c03ffbdbe17c14e277935d7c87743a649"
    calibration_projected = calibration_sha != original_cal_sha
    if calibration_projected:
        meta = calibration.get("_meta", {})
        audit.check(meta.get("original_sha256") == original_cal_sha and meta.get("changed_fields") == ["source_run"]
                    and set(calibration) - {"_meta"} == set(expected_cal) | {"source_run"}, "explicit_public_calibration_projection_only_source_run_metadata")
    profile = audit.read(out / "source_snapshot/models/pid.json")
    audit.check(profile["parameters"] == PID_PARAMETERS, "actual_frozen_PID_profile_snapshot")
    original_pid_sha = "086eeea17bf39436246f4638aa0d8bbb81cabfb2e873413488ce58b741fb29bd"
    pid_projected = sha(out / "source_snapshot/models/pid.json") != original_pid_sha
    if pid_projected:
        audit.check(profile.get("_public_projection", {}).get("original_sha256") == original_pid_sha, "explicit_public_PID_projection_original_lineage")
    phase_bundle = audit.read(out / "source_snapshot/models/phase_bundle.json")
    audit.check(sha(out / "source_snapshot/models/phase_bundle.json") == manifest["hybrid"]["bundle_sha256"], "actual_phase_bundle_snapshot_binding")
    public_contract = manifest["hybrid"]["kind"] == "distinct_public_inference_contract"
    if public_contract:
        audit.check(manifest["hybrid"]["kind"] == "distinct_public_inference_contract" and manifest["hybrid"]["original_ZIP_verified_locally"] is False
                    and phase_bundle["observation_contract"] == "public-shared-phase-expert-inference-v1.0", "honest_distinct_public_inference_contract")
        local_inference = code_root / "quadrl_phase_expert_inference_v1/policy.py"
        inference_source = local_inference if local_inference.is_file() else Path(manifest["hybrid"]["inference_source_path"])
        audit.bind(inference_source, "55afc699bda8835460509636760f6a90d1532050b457e566ed11677237a5e1c5")
    else:
        audit.check(manifest["hybrid"]["kind"] == "original_strict_phase29" and phase_bundle["observation_contract"] == "measured-shared-phase-expert-v29.0",
                    "original_strict_phase_contract")
    audit.bind(out / "source_snapshot/models/source68.npz", "68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5")
    weights = {}
    snapshot_names = {"direct": "direct111.npz", "F0": "F0.npz", "route": "route_final.npz"}
    for role, rel in PATHS.items():
        path = out / "source_snapshot/models" / snapshot_names[role]
        audit.bind(path, MODEL_SHA[role])
        with np.load(path, allow_pickle=False) as data:
            weights[role] = {k: data[k].copy() for k in data.files}
    actors = {
        "direct": old.ObservationEncoder(old.EncoderConfig(**json.loads(str(weights["direct"]["encoder_config"])))),
        "hybrid": memory.ObservationEncoderRouteMemory(memory.raw_config(weights["F0"]), "zero_route_integral"),
    }
    source_files = {Path(module.__file__).resolve() for module in tuple(sys.modules.values()) if getattr(module, "__file__", None)
                    and any(name in str(module.__file__).replace("\\", "/") for name in ("quadrl/policy", "quadrl_path_v7", "quadrl_route_basis_v14", "quadrl_route_memory_v20"))}
    for path in source_files:
        if path.is_file():
            audit.bind(path)
    result, commands_by_role = {}, {}
    for spec in plan["vehicles"]:
        role, name = spec["role"], spec["model"]
        folder = out / role
        rows = audit.read(folder / "commands.jsonl", lines=True)
        audit.check(bool(rows), role + ":nonempty_actual_commands")
        if not rows:
            continue
        commands_by_role[role] = rows
        readback = audit.read(folder / "parameter_readback.json")
        audit.check(all(abs(readback[k] - v) < 1e-5 for k, v in PID_PARAMETERS.items()), role + ":competent_frozen_PID_parameters")
        audit.check(abs(readback["AHRS_EKF_TYPE"] - 10.) < 1e-5, role + ":same_AHRS10_interface")
        audit.check(abs(readback["MAV_SYSID"] - (spec["instance"] + 1)) < 1e-5, role + ":distinct_readback_MAV_system")
        connection = audit.read(folder / "connection_identity.json")
        audit.check(connection["parameter_name"] == "MAV_SYSID" and connection["heartbeat_source_system"] == connection["connection_target_system"]
                    == connection["expected_system"] == spec["instance"] + 1, role + ":actual_heartbeat_connection_identity")
        if role == "hybrid":
            audit.check(abs(readback["ATC_ANGLE_BOOST"] - 1.) < 1e-5, "hybrid_existing_angle_boost_retained")
        launch = audit.read(folder / "sitl_launch.json")
        audit.check(launch["vehicle"]["model"] == name, role + ":SITL_model_binding")
        audit.bind(folder / "experiment.parm", launch["params_sha256"])
        raw = audit.read(folder / "mavlink.jsonl", lines=True, mavlink=True)
        indexed = {r["record_index"]: r for r in raw}
        if epoch_ms is not None:
            evidence = start["mission_epoch_raw_evidence"][role]
            ep_position, ep_attitude = indexed[evidence["position_record_index"]], indexed[evidence["attitude_record_index"]]
            pair = [int(ep_position["message"]["time_boot_ms"]), int(ep_attitude["message"]["time_boot_ms"])]
            audit.check(ep_position["message"]["mavpackettype"] == "LOCAL_POSITION_NED" and ep_attitude["message"]["mavpackettype"] == "ATTITUDE"
                        and ep_position["source_system"] == ep_attitude["source_system"] == spec["instance"] + 1,
                        role + ":mission_epoch_uses_own_actual_sensor_messages")
            audit.check(pair == evidence["pair_boot_ms"] and max(pair) == epoch_ms
                        and max(evidence["position_record_index"], evidence["attitude_record_index"]) <= evidence["receive_cutoff"],
                        role + ":same_common_epoch_independently_from_raw_sensor_pair")
            audit.check(all(rows[0][key] == evidence[key] for key in ("receive_cutoff", "position_record_index", "attitude_record_index"))
                        and abs(rows[0]["world_sim_time"] - epoch_world) < 1e-8 and abs(sample_time(rows[0])) < 1e-10,
                        role + ":epoch_is_first_controlled_frame_group_not_later_per_role_reset")
        role_events = audit.read(folder / "events.jsonl", lines=True)
        start_events = [r for r in role_events if r["event"] == "common_measurement_start"]
        audit.check(len(start_events) == 1 and start_events[0]["world_sim_time"] == t0, role + ":same_measurement_start")
        yaw = start_events[0]["yaw_reference"]
        origin = np.array(ready["coordinate_origins"][role], float)
        offset = np.array(spec["offset_enu"], float)
        audit.check(max_error(offset, scene["vehicles"][name]["origin_enu"]) < 1e-12, role + ":same_lane_translation")
        p_rows = [r for r in raw if r["message"]["mavpackettype"] == "LOCAL_POSITION_NED"]
        p_indices = [r["record_index"] for r in p_rows]
        for gate in start_gate_rows:
            state = gate["states"][role]
            raw_index = bisect_right(p_indices, state["receive_cutoff"]) - 1
            if raw_index < 0:
                raise ValueError("Gate precedes actual measured local state: " + role)
            measured = p_rows[raw_index]["message"]
            gate_p = ENU @ np.array([measured[k] for k in ("x", "y", "z")]) - origin
            gate_v = ENU @ np.array([measured[k] for k in ("vx", "vy", "vz")])
            audit.check(max(max_error(gate_p, state["position"]), max_error(gate_v, state["velocity"])) < 1e-7,
                        role + ":all_start_gate_positions_and_velocities_from_actual_MAV")
        if role in actors:
            actors[role].reset(yaw)
        previous = np.zeros(4, np.float32)
        obs_error, nn_error, coordinate_errors, truth_gaps = 0., 0., [], []
        previous_t, previous_boot = None, None
        tr = truth_by_name[name]
        tt = [r["sim_time"] for r in tr]
        for row in rows:
            t = row["measurement_t"]
            audit.check(row["role"] == role and abs(row["world_sim_time"] - t0 - t) < 1e-8 and t >= 0., role + ":world_clock_association")
            audit.check(previous_t is None or t > previous_t, role + ":strict_sim_time")
            observation_boot = row.get("observation_boot_s", row["mav_boot_s"])
            if "observation_boot_s" in row:
                audit.check(row["dt_source"] == "max_actual_position_attitude_boot_ms", role + ":explicit_measured_observation_clock")
            expected_dt = .02 if previous_boot is None else observation_boot - previous_boot
            audit.check(abs(expected_dt - row["control_dt_s"]) < 1e-9 and expected_dt > 0, role + ":actual_boot_dt")
            audit.check(max_error(row["actual_previous_action"], previous) < 1e-6, role + ":own_actual_previous_action")
            pmsg = indexed[row["position_record_index"]]["message"]
            amsg = indexed[row["attitude_record_index"]]["message"]
            if "measured_input_pair_boot_ms" in row:
                audit.check(row["measured_input_pair_boot_ms"] == [int(pmsg["time_boot_ms"]), int(amsg["time_boot_ms"])],
                            role + ":slot_input_pair_equals_actual_raw_message_timestamps")
            if "observation_boot_s" in row:
                audit.check(abs(observation_boot - max(int(pmsg["time_boot_ms"]), int(amsg["time_boot_ms"])) * .001) < 1e-10,
                            role + ":observation_clock_independently_from_actual_sensor_pair")
            verify_mission_sample_clock(row, epoch_ms, pmsg, amsg, audit, role)
            audit.check(indexed[row["position_record_index"]]["source_system"] == indexed[row["attitude_record_index"]]["source_system"] == spec["instance"] + 1,
                        role + ":actual_source_MAV_headers")
            audit.check(pmsg["mavpackettype"] == "LOCAL_POSITION_NED" and amsg["mavpackettype"] == "ATTITUDE", role + ":raw_message_types")
            audit.check(max(row["position_record_index"], row["attitude_record_index"]) <= row["receive_cutoff"], role + ":no_future_telemetry")
            p = ENU @ np.array([pmsg[k] for k in ("x", "y", "z")]) - origin
            v = ENU @ np.array([pmsg[k] for k in ("vx", "vy", "vz")])
            R = ENU @ rotation(amsg["roll"], amsg["pitch"], amsg["yaw"]) @ np.diag([1., -1., -1.])
            rates = np.array([amsg["rollspeed"], -amsg["pitchspeed"], -amsg["yawspeed"]])
            numeric = [p, v, R, rates, row["task_position"], row["target_before_update"], row["target_next"], row["actual_previous_action"], row["control_dt_s"], row["world_sim_time"]]
            if role != "pid":
                numeric.extend([row["observation"], row["action"]])
            if not all(np.isfinite(np.asarray(value, float)).all() for value in numeric):
                raise ValueError("Nonfinite actual actor input/action/metric state: " + role)
            audit.check(max(max_error(p, row["task_position"]), max_error(v, row["velocity"]), max_error(R, row["rotation"]), max_error(rates, row["body_rates"])) < 1e-7, role + ":raw_measured_actor_state")
            audit.check(max_error(row["raw_mav_position"], p + origin) < 1e-7 and max_error(row["task_origin_mav_enu"], origin) < 1e-7
                        and max_error(row["world_position_from_fixed_translation"], p + offset) < 1e-7, role + ":explicit_coordinate_ledger")
            target = np.asarray(row["target_before_update"], float)
            audit.check(max_error(row["target_world_enu"], target + offset) < 1e-7 and max_error(row["target_mav_enu"], target + origin) < 1e-7,
                        role + ":world_and_PID_inverse_targets")
            send = row["actual_send_arguments"]
            mask = 128 if role == "hybrid" else 2552
            audit.check(send["type_mask"] == row["mavlink_type_mask"] == mask and send["target_system"] == spec["instance"] + 1
                        and send["time_boot_ms"] == (int(observation_boot * 1000) & 0xffffffff), role + ":actual_send_identity_and_mask")
            if role != "hybrid":
                audit.check(send["message"] == "SET_POSITION_TARGET_LOCAL_NED" and max_error(send["position_ned"], ENU @ (target + origin)) < 1e-7
                            and abs(send["yaw_ned"] - (math.pi / 2 - yaw)) < 1e-8, role + ":native_position_command_transform")
            actual_truth, time_gap = nearest(tr, tt, row["world_sim_time"])
            truth_gaps.append(time_gap)
            coordinate_errors.append(float(np.linalg.norm(np.asarray(actual_truth["position"]) - (p + offset))))
            if role != "pid":
                encoder = actors[role]
                observation = encoder.observe(p, v, R, rates, row["target_before_update"], previous, row["control_dt_s"])
                obs_error = max(obs_error, max_error(observation, row["observation"]))
                expert = "direct" if role == "direct" else ("route" if encoder.path_context.valid_leg else "F0")
                action = forward(weights[expert], observation)
                nn_error = max(nn_error, max_error(action, row["action"]))
                previous = np.asarray(row["action"], np.float32)
                if role == "hybrid":
                    diag = row["diagnostics"]
                    audit.check(bool(diag["valid_leg"]) == bool(encoder.path_context.valid_leg) and diag["selected_source_sha256"] == MODEL_SHA[expert], role + ":current_observed_expert_not_oracle")
                    mapping = row["adapter"]
                    collective = float(np.clip((action[0] + 1.) * .302651215314865, 0., 1.))
                    denominator = max(.1, min(1., float(R[2, 2])))
                    inverse_factor = max(0., min(1., 10. * float(R[2, 2])))
                    expected = float(np.clip(collective * denominator / inverse_factor, 0., 1.)) if inverse_factor > 0 else 1e30
                    audit.check(abs(mapping["desired_body_thrust"] - collective) < NN_TOL and abs(mapping["preboost_thrust"] - expected) < NN_TOL, role + ":measured_body_collective_inverse")
                    audit.check(send["message"] == "SET_ATTITUDE_TARGET" and max_error(send["quaternion"], [0., 0., 0., 0.]) == 0
                                and max_error(send["rates_frd"], action[1:] * [2., -2., -1.]) <= NN_TOL and abs(send["thrust"] - expected) <= NN_TOL,
                                "hybrid_actual_body_rates_and_preboost_send_arguments")
                else:
                    pwm = np.rint(np.clip(1100. + (PWM_HOVER - 1100.) * np.sqrt(np.clip(action, -1., 1.) + 1.), 1100., 1900.)).astype(int)
                    audit.check(max_error(pwm, row["diagnostics"]["motor_pwm"]) == 0, role + ":FR_RL_FL_RR_PWM_mapping")
            previous_t, previous_boot = t, observation_boot
        audit.check(max(truth_gaps) <= TRUTH_MAX_TIME_GAP_S, role + ":physical_truth_time_coverage", max(truth_gaps))
        audit.check(max(coordinate_errors) <= COORDINATE_TOL_M, role + ":task_to_world_translation", max(coordinate_errors))
        audit.check(obs_error <= NN_TOL and nn_error <= NN_TOL, role + ":independent_numpy_encoder_and_NN", {"observation": obs_error, "NN": nn_error})
        audit.check(len(rows) == summary["roles"][role]["samples"], role + ":actual_count")
        geometry = recompute_route(rows, plan["mission"], audit, role)
        reported_time = summary["roles"][role]["route_completion_time_s"]
        audit.check((reported_time is None) == (geometry["route_completion_time_s"] is None) and
                    (reported_time is None or abs(reported_time - geometry["route_completion_time_s"]) < 1e-8), role + ":independent_completion_time")
        metrics = thesis_metrics(rows, duration, summary["roles"][role]["failure"], plan["mission"], yaw, force_off_mission)
        guard_violations = [r for r in rows if np.linalg.norm(np.asarray(r["task_position"]) - r["target_before_update"]) > 4.
                            or r["task_position"][2] < .25 or r["task_position"][2] > 10. or np.asarray(r["rotation"])[2, 2] < .2
                            or r["mode"] != 4 or not r["armed"]]
        audit.check(not guard_violations or metrics["failure"] is not None, role + ":all_actual_guard_failures_preserved")
        result[role] = {**metrics, **geometry, "actual_commands": len(rows), "encoder_max_error": obs_error, "NN_max_error": nn_error,
                        "task_world_error_max_m": max(coordinate_errors), "task_world_error_RMS_m": float(np.sqrt(np.mean(np.square(coordinate_errors)))),
                        "truth_association_max_gap_s": max(truth_gaps), "actual_commands_per_requested_second": len(rows) / duration,
                        "actual_interval_frequency_hz": (len(rows) - 1) / (sample_time(rows[-1]) - sample_time(rows[0])) if len(rows) > 1 else None,
                        "metric_clock": "explicit_actual_sensor_pair_minus_epoch" if "mission_time_s" in rows[0] else "archived_legacy_observer_clock",
                        "fifty_hz_within_five_percent": len(rows) / duration >= 47.5}
    launches = [audit.read(out / role / "sitl_launch.json") for role in ROLES]
    audit.check(len({l["binary_sha256"] for l in launches}) == 1, "same_actual_ArduCopter_binary")
    actuators = audit.read(out / "direct/actuators.jsonl", lines=True)
    active = [r for r in actuators if r.get("override")]
    audit.check(bool(active), "actual_direct_motor_overrides_not_PID_only")
    audit.check(all(len(r["output_pwm"]) == 4 and all(1100 <= p <= 1900 for p in r["output_pwm"]) for r in active), "physical_motor_PWM_bounds")
    audit.check(all(r["action_age_s"] <= .500001 for r in active), "existing_direct_wall_watchdog_retained")
    command_pwm = {tuple(r["diagnostics"]["motor_pwm"]) for r in commands_by_role.get("direct", [])}
    audit.check(all(tuple(r["output_pwm"]) in command_pwm for r in active), "all_actual_overridden_PWM_are_archived_NN_commands")
    audit.bind(out / "direct/physics.jsonl")
    capture = audit.read(out / "network_capture_finished.json")
    audit.bind(out / "network.pcap.gz", capture["sha256"])
    audit.check(capture["bytes"] > 0, "actual_passive_network_capture")
    pair_minimum = {}
    for a, b in (("iris_pid", "iris_hybrid"), ("iris_pid", "iris_direct"), ("iris_hybrid", "iris_direct")):
        rb = truth_by_name[b]
        bt = [r["sim_time"] for r in rb]
        values = []
        for r in truth_by_name[a]:
            if t0 <= r["sim_time"] <= t0 + duration:
                other, gap = nearest(rb, bt, r["sim_time"])
                if gap <= TRUTH_MAX_TIME_GAP_S:
                    values.append(float(np.linalg.norm(np.asarray(r["position"]) - other["position"])))
        pair_minimum[a + ":" + b] = min(values) if values else None
    failed = {role: value["failure"] for role, value in result.items() if value["failure"] is not None}
    cadence = verify_cadence(out, plan, summary, commands_by_role, t0, audit)
    return {"schema": "independent-shared-world-three-demo-audit-v1.0", "run": str(out),
            "scope": "fresh shared-world demonstration only; no old metrics or applicability acceptance inherited",
            "scenario": plan["scenario"], "mission": plan["mission"], "seed": plan["seed"],
            "prospective_two_case_protocol": formal_demo_case(plan), "control_cadence_verification": cadence,
            "actual_new_world_runs": 1, "actual_new_aircraft_attempts": 3, "new_optimizer_steps": 0,
            "qualified_common_start": qualified_start, "physical_mass_kg_by_model": physical_masses,
            "replay_code_root": str(code_root), "PID_document_is_explicit_public_projection": pid_projected,
            "calibration_document_is_explicit_public_projection": calibration_projected,
            "unused_standard_MAV_missing_values": dict(audit.unused_mavlink_missing_fields),
            "roles": result, "controller_failures_preserved": failed, "all_three_survived": all(r["survived"] for r in result.values()) and len(result) == 3,
            "minimum_pair_distance_m": pair_minimum, "collision_avoidance_guarantee": False,
            "actual_common_force_ledger": force_ledger, "actual_final_force_off_s": force_off,
            "actual_final_force_off_mission_clock_s": force_off_mission,
            "mission_clock_epoch": {"boot_ms": epoch_ms, "observer_world_anchor_s": epoch_world,
                                    "force_time_mapping": "actual world force tick minus recorded observer world epoch anchor; sensor reception latency is not claimed zero"},
            "coordinate_evidence_tolerance_m": COORDINATE_TOL_M, "NN_evidence_tolerance": NN_TOL,
            "wire_scope": "actual send arguments and proxy output replayed; PCAP preserved, not decoded by this auditor",
            "scientific_table_updated": False, "own_new_environment_steps": 0, "own_new_simulator_steps": 0,
            "own_backward_calls": 0, "own_optimizer_steps": 0, "own_NN_scope": "archived measured states only; one selected mean replay per archived NN command"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing receipts; choose a new output")
    audit = Audit()
    try:
        result = audit_run(args.run.resolve(), audit)
    except Exception as exc:
        audit.check(False, "archive_schema_or_evidence_error", repr(exc))
        result = {"schema": "independent-shared-world-three-demo-audit-v1.0", "run": str(args.run.resolve()),
                  "own_new_environment_steps": 0, "own_new_simulator_steps": 0, "own_backward_calls": 0, "own_optimizer_steps": 0}
    changed = [path for path, digest in audit.inputs.items() if not Path(path).is_file() or sha(path) != digest]
    audit.check(not changed, "all_archived_inputs_unchanged_after_read", changed)
    result.update(status="PASS" if not audit.failures else "FAIL", integrity_passed=not audit.failures,
                  checks=audit.checks, failed_checks=audit.failures, input_sha256=audit.inputs,
                  auditor_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks", "failed_checks", "roles", "controller_failures_preserved", "minimum_pair_distance_m")}, ensure_ascii=False))
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

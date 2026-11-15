"""Independent V2 judge. No imports from the environment or evaluator.

This module recomputes metrics; it does not create simulation outcomes. Adapters
must provide actual trace arrays and declare a failed/truncated flight. Safety
flags, source integrity, policy replay and simulator provenance are separate checks.
The primary RMSE weights each sample by its preceding measured interval. The
equally weighted sample RMSE is diagnostic. Both backends use the same rule.
"""
from __future__ import annotations
import numpy as np


def _array(value, shape, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"invalid {name}: expected finite array {shape}, got {arr.shape}")
    return arr


def wrapped_angle(angle):
    return (np.asarray(angle, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def first_dwell(time, inside, seconds, max_gap, earliest=0.):
    """Return START of first fully evidenced interval, never a sample count guess."""
    time, inside = np.asarray(time, float), np.asarray(inside, bool)
    start = None
    last = None
    for stamp, valid in zip(time, inside):
        if stamp < earliest - 1e-9:
            continue
        if last is not None and stamp - last > max_gap + 1e-9:
            start = None
        last = stamp
        if not valid:
            start = None
            continue
        if start is None:
            start = float(stamp)
        if stamp - start >= seconds - 1e-9:
            return start
    return None


def judge_route(time, position, points, band=.25, dwell=.5, expected_dt=.02):
    """Independently reconstruct sequential goal confirmations from sampled pose."""
    time, position, points = np.asarray(time, float), np.asarray(position, float), np.asarray(points, float)
    index, inside_since, previous_time = 0, None, None
    events = []
    for stamp, pose in zip(time, position):
        if previous_time is not None and stamp - previous_time > 1.5 * expected_dt + 1e-9:
            inside_since = None
        previous_time = stamp
        if np.linalg.norm(pose - points[index]) > band + 1e-12:
            inside_since = None
            continue
        if inside_since is None:
            inside_since = float(stamp)
        if stamp - inside_since < dwell - 1e-9:
            continue
        events.append({"goal_index": index, "time_s": float(stamp), "dwell_started_s": inside_since})
        if index == len(points) - 1:
            return {"completed": True, "completion_time_s": float(stamp), "censored": False, "events": events}
        index += 1
        inside_since = None
    return {"completed": False, "completion_time_s": None, "censored": True, "events": events}


def compute_metrics(*, time, position, target, yaw, yaw_target, planned_seconds,
                    failed, mission="hover", expected_dt=.02, last_force_off_s=None,
                    first_force_on_s=None, route_points=None, band=.25,
                    recovery_dwell=1., route_dwell=.5, failure_tail_error=4., initial_position=None):
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("nonempty finite one-dimensional time required")
    n = len(time)
    if np.any(np.diff(time) <= 0):
        raise ValueError("trace time must strictly increase; duplicates are not silently removed")
    if expected_dt <= 0 or planned_seconds <= 0:
        raise ValueError("positive horizon and sampling interval required")
    if time[0] < -1e-9 or time[-1] > planned_seconds + 2 * expected_dt:
        raise ValueError("trace outside declared measurement horizon")
    position = _array(position, (n, 3), "position")
    target = _array(target, (n, 3), "target")
    yaw = _array(yaw, (n,), "yaw")
    yaw_target = np.broadcast_to(np.asarray(yaw_target, float), (n,))
    if not np.isfinite(yaw_target).all():
        raise ValueError("finite yaw reference required")
    error = np.linalg.norm(position - target, axis=1)
    yaw_error_deg = np.degrees(wrapped_angle(yaw - yaw_target))
    planned_steps = int(round(planned_seconds / expected_dt))
    # A producer may include a t=0 sample in addition to N integration endpoints.
    endpoint_samples = n - int(abs(time[0]) < 1e-9)
    missing_steps = max(0, planned_steps - endpoint_samples)
    full_horizon = bool(time[-1] >= planned_seconds - 2 * expected_dt - 1e-9)
    survived = bool(not failed and full_horizon)
    weights = np.diff(np.r_[0., np.minimum(time, planned_seconds)])
    covered = float(weights.sum())
    if covered <= 0 and not failed:
        raise ValueError("trace proves no positive measurement duration")
    error_integral = float(np.dot(weights, error ** 2))
    missing_seconds = max(0., planned_seconds - covered)
    tail_error = float(error[-1]) if survived else failure_tail_error
    penalty = np.sqrt((error_integral + missing_seconds * tail_error ** 2) / planned_seconds)
    gaps = np.diff(time)
    timing = {
        "first_time_s": float(time[0]), "last_time_s": float(time[-1]),
        "samples": n, "integration_endpoint_samples": endpoint_samples,
        "planned_steps": planned_steps, "missing_steps": missing_steps,
        "max_gap_s": float(gaps.max()) if len(gaps) else None,
        "gaps_above_1_5_dt": int(np.count_nonzero(gaps > 1.5 * expected_dt + 1e-9)),
        "full_horizon": full_horizon,
        "covered_seconds": covered, "tail_seconds": missing_seconds,
        "unexplained_short_trace": bool(not failed and not full_horizon),
    }
    if mission != "hover":
        recovery = {"applicable": False, "reason": "route_target_changes", "time_s": None, "censored": None}
    elif last_force_off_s is None:
        recovery = {"applicable": False, "reason": "no_recorded_force_off", "time_s": None, "censored": None}
    else:
        start = first_dwell(time, error <= band + 1e-12, recovery_dwell,
                            1.5 * expected_dt, earliest=last_force_off_s)
        recovery = {"applicable": True, "force_off_s": float(last_force_off_s),
                    "dwell_start_s": start,
                    "time_s": None if start is None else max(0., start - last_force_off_s),
                    "censored": start is None}
    route = None
    if mission == "route":
        if route_points is None:
            raise ValueError("explicit frozen route points required")
        route_time, route_position = time, position
        if initial_position is not None and time[0] > 1e-9:
            initial = _array(initial_position, (3,), "initial position")
            route_time = np.r_[0., time]
            route_position = np.vstack([initial, position])
        route = judge_route(route_time, route_position, route_points, band, route_dwell, expected_dt)
    elif mission != "hover":
        raise ValueError("unknown mission")
    altitude_drop = float(max(0., np.max(target[:, 2] - position[:, 2])))
    disturbance_drop = None
    if first_force_on_s is not None:
        before = np.flatnonzero(time < first_force_on_s - 1e-9)
        after = np.flatnonzero(time >= first_force_on_s - 1e-9)
        if len(before) and len(after):
            initial_z = position[before[-1], 2]
            disturbance_drop = {"reference_time_s": float(time[before[-1]]),
                                "reference_z_m": float(initial_z),
                                "drop_m": float(max(0., initial_z - position[after, 2].min())),
                                "interpretation": "relative_to_pre_force_altitude; includes subsequent commanded altitude changes"}
    heading_rmse = float(np.sqrt(np.dot(weights, yaw_error_deg ** 2) / covered)) if covered > 1e-12 else None
    heading_peak = float(np.max(np.abs(yaw_error_deg)))
    completion = bool(route["completed"]) if route else None
    return {
        "rmse_m": float(np.sqrt(error_integral / covered)) if covered > 1e-12 else None,
        "observed_duration_positive": covered > 1e-12,
        "sample_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "penalized_rmse_m": float(penalty), "max_error_m": float(error.max()),
        "max_altitude_drop_m": altitude_drop, "altitude_drop_since_force": disturbance_drop,
        "recovery": recovery, "survived": survived,
        "declared_failed": bool(failed), "route": route, "completion": completion,
        "heading_rmse_deg": heading_rmse, "heading_peak_deg": heading_peak,
        "heading_guard_pass": bool(heading_rmse is not None and heading_rmse <= 5 + 1e-9 and heading_peak <= 15 + 1e-9),
        "heading_full_flight_eligible": survived,
        "yaw_unwrapped_change_deg": float(np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])),
        "timing": timing,
        "metric_time_weighting": "preceding sample interval; failure/incomplete tail4m; surviving final<=2controlsteps filled last error",
    }

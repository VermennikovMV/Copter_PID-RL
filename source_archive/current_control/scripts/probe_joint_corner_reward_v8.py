"""Bounded OFFLINE corner reward prototype audit; never imports a simulator.

Replay uses confirmed-event metadata as AUDIT_YARDSTICK, never as actor inputs.
Only the tracker itself is proposed for a future own-mission-judge integration.
The prospective plan permits exactly fixed K20 or separately root-authorized
fixed K40. This script never searches weights. No training is run.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit.turns_corners import segment_distance as frozen_segment_distance
from audit.verify_joint_metrics_v2 import geometry_metrics
from quadrl_corner_v8.reward import (
    CORNER_WEIGHT, ConfirmedTransition, CornerPeakReward, finite_segment_distance)
from quadrl.reward import reward_value

RUN = ROOT / "artifacts/refinement/joint_v2/path_pilot111"
OUT = ROOT / "artifacts/refinement/joint_v2/corner_reward_probe_v8"
CHECKS, FAILURES = 0, []
MAXIMA = dict(peak_frozen_error_m=0., peak_numpy_error_m=0.,
              peak_saved_geometry_error_m=0., peak_existing_geometry_error_m=0.,
              telescoping_error_reward=0., row_geometry_error_m=0.)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def check(ok, context):
    global CHECKS
    CHECKS += 1
    if not bool(ok):
        FAILURES.append(context)


def close(a, b, context, tol=1e-10):
    check(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a - b) <= tol,
          context + ": " + repr(a) + " vs " + repr(b))


def raises(call, context):
    try:
        call()
    except (ValueError, TypeError, OverflowError):
        check(True, context)
    else:
        check(False, context)


def independent_distance(p, a, b):
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    leg = b - a
    length_sq = np.dot(leg, leg)
    q = np.clip(np.sum((p - a) * leg, axis=-1) / length_sq, 0, 1) if length_sq else 0
    return np.linalg.norm(p - a - np.asarray(q)[..., None] * leg, axis=-1)


def prime(tracker):
    tracker.observe(0., (0, 0, 0), (0, 0, 0))
    tracker.observe(.02, (0, 0, 0), (1, 0, 0),
                    ConfirmedTransition(0, .02, (0, 0, 0)), dt=.02)


def open_corner(tracker, stamp=.1, p=(1, 0, 0), target=(1, 1, 0)):
    return tracker.observe(stamp, p, target, ConfirmedTransition(1, stamp, (1, 0, 0)))


def fixture_checks():
    start_checks = CHECKS
    records = {}
    # Projection cases distinguish finite vs line, altitude, and no error cap.
    for p, a, b, expected, name in [
        ((2, 0, 0), (0, 0, 0), (1, 0, 0), 1., "overshoot_after_b"),
        ((-2, 0, 0), (0, 0, 0), (1, 0, 0), 2., "before_a"),
        ((.5, 0, 3), (0, 0, 0), (1, 0, 0), 3., "true_3d"),
        ((0, 0, 4), (0, 0, 0), (0, 0, 0), 4., "degenerate_point"),
        ((.5, 100, 0), (0, 0, 0), (1, 0, 0), 100., "unclipped_error")]:
        close(finite_segment_distance(p, a, b), expected, name)
    # Same continuous linear/step positions and closed endpoint at 20..100Hz.
    for kind in ("linear", "step"):
        totals = []
        for hz in (20, 25, 50, 100):
            tracker = CornerPeakReward()
            prime(tracker)
            reward = open_corner(tracker)
            for i in range(1, hz + 1):
                local = i / hz
                deviation = .4 * local if kind == "linear" else (.4 if local >= .4 else .1)
                reward += tracker.observe(.1 + local, (1 + deviation, local, 0), (1, 1, 0), dt=1 / hz)
            report = tracker.finish("early_termination")
            close(reward, -8., kind + "/rate/" + str(hz))
            close(reward, report["telescoping_expected_reward"], kind + "/telescope")
            check(report["windows"][0]["full_post_window"], kind + "/closed complete")
            check(report["episode_partial"] and report["unreached_corners"] == 2,
                  kind + "/unreached retained")
            totals.append(reward)
        records[kind + "_reward_by_hz20_25_50_100"] = totals
    # Both transition sample and exact endpoint can establish the peak.
    for transition_error, endpoint_error, expected in ((.6, .1, -12.), (.1, .8, -16.)):
        tracker = CornerPeakReward()
        prime(tracker)
        reward = open_corner(tracker, p=(1 + transition_error, 0, 0))
        for i in range(1, 51):
            reward += tracker.observe(.1 + i * .02, (1 + (endpoint_error if i == 50 else .01), .5, 0), (1, 1, 0))
        # An enormous excursion just AFTER end must have no influence.
        close(tracker.observe(1.12, (100, 100, 100), (1, 1, 0)), 0., "outside closed window")
        close(reward, expected, "closed transition/endpoint maximum")
    # Pre-transition excursion earns zero; outgoing segment starts SAME row.
    tracker = CornerPeakReward()
    prime(tracker)
    close(tracker.observe(.08, (1, 100, 0), (1, 0, 0)), 0., "pre transition zero")
    close(open_corner(tracker, p=(1, .4, 0)), 0., "new current vertical segment immediately")
    # A different newly observed target changes the segment; there is no route lookup.
    other = CornerPeakReward()
    prime(other)
    close(open_corner(other, p=(1, .4, 0), target=(2, 0, 0)), -8., "observed current target controls segment")
    raises(lambda: tracker.observe(.12, (1, .4, 0), (9, 9, 9)), "unconfirmed future preview rejected")
    # Goal0/start and hover cannot create windows even at huge deviations.
    close(CornerPeakReward().observe(0., (100, 100, 100), (0, 0, 0)), 0., "start zero")
    hover = CornerPeakReward("hover")
    hover_rewards = [hover.observe(i * .02, (i, -i, i * 2), (0, 0, 3)) for i in range(1501)]
    check(all(r == 0. for r in hover_rewards) and not hover.windows, "verifiable hover zero1501")
    raises(lambda: hover.observe(30.02, (0, 0, 0), (1, 1, 1),
                                ConfirmedTransition(0, 30.02, (0, 0, 3))), "hover event rejected")
    # Three independent overlapping windows survive subsequent target changes.
    overlap = CornerPeakReward()
    prime(overlap)
    target = (1, 1, 0)
    overlap.observe(.1, (1.2, 0, 0), target, ConfirmedTransition(1, .1, (1, 0, 0)))
    for i in range(1, 71):
        stamp = round(.1 + i * .02, 10)
        event = None
        if i == 10:
            event, target = ConfirmedTransition(2, stamp, target), (0, 1, 0)
        elif i == 20:
            event, target = ConfirmedTransition(3, stamp, target), (0, 0, 0)
        overlap.observe(stamp, (1 + .1 * i, .1 * i, .01 * i), target, event)
        if i == 20:
            check(len([w for w in overlap.windows if not w.closed]) == 3, "three simultaneous windows")
    overlap_report = overlap.finish("early_termination")
    check(overlap.windows[0].a == (1., 0., 0.) and overlap.windows[0].b == (1., 1., 0.), "segment endpoints remain frozen")
    close(overlap.total_reward, -20 * sum(w.peak_m for w in overlap.windows), "overlap telescoping")
    check(overlap_report["complete_corner_windows"] == 3, "overlap all endpoints included")
    records["overlap"] = overlap_report
    # Missing endpoint/interior coverage and early termination stay partial.
    censored = CornerPeakReward()
    prime(censored)
    open_corner(censored)
    censored.observe(.2, (1.2, .1, 0), (1, 1, 0))
    partial = censored.finish("failure")
    check(partial["episode_partial"] and partial["complete_corner_windows"] == 0 and
          partial["windows"][0]["full_window_peak_m"] is None, "failure partial not full/zero")
    close(partial["reward"], -4., "failure observed peak penalty retained")
    skip = CornerPeakReward()
    prime(skip)
    open_corner(skip)
    skip.observe(1.12, (9, 9, 9), (1, 1, 0))
    check(not skip.report()["windows"][0]["full_post_window"] and
          "missing_end_coverage" in skip.report()["windows"][0]["flags"], "skip end missing flag")
    # Jittered sampling does not need an exact endpoint. The actual first
    # outside sample proves coverage but cannot establish a new corner peak.
    jitter = CornerPeakReward()
    prime(jitter)
    open_corner(jitter)
    for stamp in np.arange(.119, 1.1, .02):
        jitter.observe(float(stamp), (1.4, .5, 0), (1, 1, 0))
    pending = jitter.report()["windows"][0]
    check(not pending["full_post_window"] and not pending["end_coverage_confirmed"],
          "jitter no outside evidence remains censored")
    close(jitter.observe(1.119, (1000, 1000, 1000), (1, 1, 0)), 0.,
          "jitter first outside sample excluded from peak")
    proved = jitter.report()["windows"][0]
    check(proved["full_post_window"] and proved["end_coverage_confirmed"] and
          not proved["closed_endpoint_sample_included"], "jitter bracketing proves coverage")
    close(proved["full_window_peak_m"], .4, "jitter no interpolated/outside peak")
    records["jitter_bracketed_window"] = proved
    gap = CornerPeakReward()
    prime(gap)
    open_corner(gap)
    gap.observe(1.1, (1, .5, 0), (1, 1, 0))
    check("missing_interior_coverage" in gap.report()["windows"][0]["flags"] and
          not gap.report()["windows"][0]["full_post_window"], "interior gap flagged even endpoint present")
    # Guards leave state untouched, so a caller can recover from rejected data.
    guard = CornerPeakReward()
    prime(guard)
    before = guard.report()
    cases = [lambda: guard.observe(.02, (0, 0, 0), (1, 0, 0)),
             lambda: guard.observe(.01, (0, 0, 0), (1, 0, 0)),
             lambda: guard.observe(float("nan"), (0, 0, 0), (1, 0, 0)),
             lambda: guard.observe(float("inf"), (0, 0, 0), (1, 0, 0)),
             lambda: guard.observe(.04, (float("nan"), 0, 0), (1, 0, 0)),
             lambda: guard.observe(.04, (0, 0, 0), (1, float("inf"), 0)),
             lambda: guard.observe(.04, (0, 0), (1, 0, 0)),
             lambda: guard.observe(.04, (0, 0, 0), (1, 0, 0), dt=0),
             lambda: guard.observe(.04, (0, 0, 0), (1, 0, 0), dt=-.02),
             lambda: guard.observe(.04, (0, 0, 0), (1, 0, 0), dt=float("nan")),
             lambda: guard.observe(.04, (0, 0, 0), (1, 0, 0), dt=.01),
             lambda: guard.observe(.04, (0, 0, 0), (2, 0, 0), ConfirmedTransition(1, .03, (1, 0, 0))),
             lambda: guard.observe(.04, (0, 0, 0), (2, 0, 0), ConfirmedTransition(2, .04, (1, 0, 0))),
             lambda: guard.observe(.04, (0, 0, 0), (2, 0, 0), ConfirmedTransition(1, .04, (9, 0, 0)))]
    for index, call in enumerate(cases):
        raises(call, "guard/" + str(index))
        check(guard.report() == before, "guard atomic state/" + str(index))
    guard.observe(.04, (0, 0, 0), (1, 0, 0), dt=.02)
    guard.finish("early_termination")
    raises(lambda: guard.observe(.06, (0, 0, 0), (1, 0, 0)), "must reset after finish")
    guard.reset("hover")
    check(guard.last_time_s is None and guard.total_reward == 0 and not guard.windows and
          guard.current_target is None and guard.last_confirmed_goal == -1, "reset clears all state")
    close(guard.observe(0., (0, 0, 0), (0, 0, 0)), 0., "reset permits time restart")
    forty = CornerPeakReward(weight=40.)
    prime(forty)
    close(open_corner(forty, p=(1.2, 0, 0)), -8., "explicit fixedK40 linear increment")
    close(forty.report()["telescoping_expected_reward"], -8., "explicit fixedK40 telescope")
    forty.reset()
    check(forty.weight == 40. and forty.total_reward == 0, "reset preserves prospective instance weight")
    for invalid in (0, -1, float("nan"), float("inf")):
        raises(lambda: CornerPeakReward(weight=invalid), "invalid reward weight")
    # W8 shortcut/overshoot costs and base arrival10/completion20 are unchanged.
    for label, p in (("shortcut", (.5, .5, 0)), ("overshoot", (1.3, 0, 0)), ("altitude", (.5, 0, .4))):
        distance = finite_segment_distance(p, (0, 0, 0), (1, 0, 0))
        check(-.02 * 8 * distance ** 2 < 0, label + "/existing W8 penalty")
    args = dict(error=.1, velocity_sq=0., tilt_cos=1., yaw_error=0., rate_sq=0.,
                action_delta_sq=0., distance_progress=0., arrival=False,
                completion=False, failed=False, dt=.02)
    ordinary = reward_value(**args)
    close(reward_value(**dict(args, arrival=True)) - ordinary, 10., "original arrival10")
    close(reward_value(**dict(args, completion=True)) - ordinary, 20., "original completion20")
    check(reward_value(**dict(args, velocity_sq=9., arrival=True)) <
          reward_value(**dict(args, velocity_sq=0., arrival=True)), "fast arrival velocity cost retained")
    records["checks"] = CHECKS - start_checks
    records["semantic_scope"] = "No fast-arrival credit or new arrival bonus: the caller still uses original arrival10/completion20; direct corner peak cannot replace finite/heading/full40 guards"
    return records


def frozen_selector_checks():
    """Execute only the frozen pure selector AST, without driver imports."""
    source_path = ROOT / "scripts/train_joint_path_context_v7.py"
    parsed = ast.parse(source_path.read_text(encoding="utf8"))
    names = {"SCENARIOS", "SELECTION_VERSION", "ROUTE_TIME_LIMIT_S",
             "ROUTE_GOAL_RMSE_LIMIT_M", "ABSOLUTE_SEGMENT_PEAK_LIMIT_M",
             "FINITE_TARGETS_M", "CORNER_TARGETS_M"}
    nodes = [node for node in parsed.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                   for t in node.targets)) or
             (isinstance(node, ast.FunctionDef) and node.name == "selection_score")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    selector = namespace["selection_score"]
    scenarios = namespace["SCENARIOS"]
    metrics = dict(flight_failures=0, route_incomplete=0, heading_violations=0, geometry_ineligible=0,
                   active_segment_peak_failures=0,
                   active_segment_peak_checks=[dict(scenario=s, seed=84001, peak_m=.1) for s in scenarios],
                   cells={})
    for scenario in scenarios:
        metrics["cells"][scenario + "/hover"] = dict(goal_rmse_m=.03)
        metrics["cells"][scenario + "/route"] = dict(goal_rmse_m=.4, restricted_time_s=9.,
                                                    segment_rmse_m=.1, mean_corner_peak_m=.19)
    check(selector(metrics, metrics)["fixed_development_guards_pass"], "frozen selector otherwise passing fixture")
    rejected = []
    for scenario in scenarios:
        for field, limit in (("segment_rmse_m", namespace["FINITE_TARGETS_M"][scenario]),
                             ("mean_corner_peak_m", namespace["CORNER_TARGETS_M"][scenario])):
            bad = copy.deepcopy(metrics)
            bad["cells"][scenario + "/route"][field] = limit + 1e-8
            check(not selector(bad, metrics)["fixed_development_guards_pass"], "frozen selector rejects/" + scenario + "/" + field)
            rejected.append(scenario + "/" + field)
    for field in ("flight_failures", "route_incomplete", "heading_violations", "geometry_ineligible"):
        bad = copy.deepcopy(metrics)
        bad[field] = 1
        check(not selector(bad, metrics)["fixed_development_guards_pass"], "frozen selector rejects/" + field)
        rejected.append(field)
    bad = copy.deepcopy(metrics)
    bad["active_segment_peak_checks"][0]["peak_m"] = .50000001
    check(not selector(bad, metrics)["fixed_development_guards_pass"], "frozen selector rejects individual peak")
    rejected.append("individual_active_segment_peak_m")
    return dict(source=str(source_path.relative_to(ROOT)), source_sha256=sha(source_path),
                execution="Only original pure selection_score AST and its original fixed constants; no driver/simulator import",
                rejected_violations=rejected, reward_cannot_override_guard=True)


def source_purity_checks():
    source = (ROOT / "quadrl_corner_v8/reward.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module)
    check(set(imported) <= {"__future__", "dataclasses", "math", "typing"}, "tracker stdlib only")
    check("ROUTE_POINTS" not in source and "POINTS[" not in source, "tracker has no future route table")
    check(CORNER_WEIGHT == 20., "fixed prospective K20")
    base = (ROOT / "quadrl/reward.py").read_text(encoding="utf8")
    check("10*arrival+20*completion" in base, "unchanged arrival10/completion20 source")
    return dict(imports=imported, no_route_table=True, no_actor_or_environment_calls=True,
                public_inputs=["measured_position", "measurement_time", "current_observed_target",
                               "same_time_own_confirmed_old_goal_event"],
                actor_observation_unchanged=True, event_metadata_role="AUDIT_YARDSTICK in replay only")


def scores(base, w8, corner, completion, t, weight):
    result = {}
    for scope, mask in (("training_prefix", t <= completion + 1e-9), ("full40", np.ones(len(t), bool))):
        discount = .995 ** np.arange(np.sum(mask))
        b, w, c = base[mask], w8[mask], corner[mask]
        result[scope] = dict(steps=int(mask.sum()),
            base_reward_sum=float(b.sum()), W8_reward_sum=float(w.sum()),
            old_W8_reward_sum=float((b + w).sum()), corner_K_penalty=float(c.sum()),
            corner_weight_reward_per_m=weight,
            new_W8_plus_corner_reward_sum=float((b + w + c).sum()),
            gamma=.995, old_W8_discounted_return=float(discount @ (b + w)),
            corner_K_discounted_penalty=float(discount @ c),
            new_W8_plus_corner_discounted_return=float(discount @ (b + w + c)))
    return result


def replay(summary_path, weight):
    folder = summary_path.parent
    relative = str(folder.relative_to(RUN))
    arm, _, label, case = folder.relative_to(RUN).parts
    summary, saved = read(summary_path), read(folder / "geometry.json")
    with (folder / "trace.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    def cols(keys):
        return np.array([[float(row[k]) for k in keys] for row in rows])
    t, p = cols(["time"])[:, 0], cols(["x", "y", "z"])
    target = cols(["target_x", "target_y", "target_z"])
    current = cols(["next_target_x", "next_target_y", "next_target_z"])
    base = cols(["reward"])[:, 0]
    events = summary["mission_events"]
    event_at = {round(float(e["time_s"]), 8): e for e in events}
    check(len(event_at) == len(events), relative + "/unique event times")
    tracker = CornerPeakReward(weight=weight)
    tracker.observe(0., summary["initial_state"]["position"], target[0])
    corner = np.zeros(len(t))
    w8 = np.zeros(len(t))
    old_previous, old_current = None, target[0].copy()
    observed_events = []
    for index, (stamp, position, judged, now) in enumerate(zip(t, p, target, current)):
        event = event_at.get(round(float(stamp), 8))
        check(np.allclose(judged, old_current, rtol=0, atol=1e-9), relative + "/pretarget/" + str(index))
        if old_previous is not None:
            distance = float(independent_distance(position, old_previous, old_current))
            w8[index] = -(stamp - (t[index - 1] if index else 0.)) * 8 * distance ** 2
        confirmation = ConfirmedTransition(int(event["goal_index"]), float(event["time_s"]), event["target"]) if event else None
        corner[index] = tracker.observe(float(stamp), position, now, confirmation,
                                       dt=float(stamp - (t[index - 1] if index else 0.)))
        if event:
            observed_events.append(event)
            if int(event["goal_index"]) < 4:
                old_previous, old_current = old_current.copy(), now.copy()
        check(np.allclose(now, old_current, rtol=0, atol=1e-9), relative + "/posttarget/" + str(index))
    completion = float(summary["metrics"]["route_completion_time_s"])
    check(observed_events == events, relative + "/same-time all confirmed events consumed")
    check(summary["metrics"]["survived"] and t[-1] >= 40 - 1e-9 and not summary["metrics"]["failure"], relative + "/actual full40 guard")
    report = tracker.finish("full_horizon")
    check(report["corner_coverage_complete"] and len(report["windows"]) == 3,
          relative + "/three complete windows")
    existing = geometry_metrics(t, p, target, events, completion)
    peak_records = []
    for window, old, independent in zip(report["windows"], saved["corners"], existing["corners"]):
        a, b, stamp, end = window["a"], window["b"], window["transition_s"], window["window_end_s"]
        mask = (t >= stamp - 1e-9) & (t <= end + 1e-9)
        frozen = [frozen_segment_distance(position, a, b) for position in p[mask]]
        vector = independent_distance(p[mask], a, b)
        peak = window["full_window_peak_m"]
        references = dict(peak_frozen_error_m=max(frozen), peak_numpy_error_m=float(vector.max()),
                          peak_saved_geometry_error_m=old["post_outgoing_max_deviation_m"],
                          peak_existing_geometry_error_m=independent["post_outgoing_max_deviation_m"])
        for key, reference in references.items():
            MAXIMA[key] = max(MAXIMA[key], abs(peak - reference))
            close(peak, reference, relative + "/" + key)
        row_error = float(np.max(np.abs(vector - frozen)))
        MAXIMA["row_geometry_error_m"] = max(MAXIMA["row_geometry_error_m"], row_error)
        check(row_error <= 1e-10, relative + "/all closed-window samples frozen vs numpy")
        peak_records.append(dict(corner_index=window["corner_index"], peak_m=peak,
                                 frozen_peak_m=max(frozen), numpy_peak_m=float(vector.max()),
                                 saved_peak_m=old["post_outgoing_max_deviation_m"],
                                 closed_sample_count=int(mask.sum())))
    telescope = abs(corner.sum() + weight * report["observed_sum_peak_m"])
    MAXIMA["telescoping_error_reward"] = max(MAXIMA["telescoping_error_reward"], telescope)
    close(corner.sum(), -weight * report["observed_sum_peak_m"], relative + "/prefix telescope")
    return dict(arm=arm, label=label, case=case, scenario=summary["scenario"], seed=summary["eval_seed"],
                trace=str((folder / "trace.csv").relative_to(ROOT)), steps=len(t),
                finite_segment_rmse_m=saved["segment_rmse_m"],
                mean_corner_peak_m=report["observed_sum_peak_m"] / 3,
                windows=report["windows"], peak_agreement=peak_records,
                reward_scores=scores(base, w8, corner, completion, t, weight),
                phase_exposure=[dict(corner_index=w["corner_index"], transition_s=w["transition_s"],
                    force_N_at_transition=cols(["force_x", "force_y", "force_z"])[np.argmin(abs(t-w["transition_s"]))].tolist())
                    for w in report["windows"]],
                event_source="AUDIT_YARDSTICK: archived own mission confirmations at their actual observed timestamps",
                full40_guard_pass=True)


def score_delta(blind_score, path_score):
    b, p = blind_score, path_score
    weight = b["corner_weight_reward_per_m"]
    old_delta = p["old_W8_discounted_return"] - b["old_W8_discounted_return"]
    new_delta = p["new_W8_plus_corner_discounted_return"] - b["new_W8_plus_corner_discounted_return"]
    penalty_delta = p["corner_K_penalty"] - b["corner_K_penalty"]
    discounted_penalty_delta = p["corner_K_discounted_penalty"] - b["corner_K_discounted_penalty"]
    slope = discounted_penalty_delta / weight
    old_undiscounted = p["old_W8_reward_sum"] - b["old_W8_reward_sum"]
    undiscounted_slope = penalty_delta / weight
    crossing = -old_delta / slope if slope != 0 else None
    minimum_K = 0. if old_delta < 0 else (crossing if slope < 0 else None)
    return dict(fixed_K_reward_per_m=weight, old_discounted_path_minus_blind=old_delta,
        corner_K_undiscounted_path_minus_blind=penalty_delta,
        corner_K_discounted_path_minus_blind=discounted_penalty_delta,
        new_discounted_path_minus_blind=new_delta,
        old_undiscounted_path_minus_blind=old_undiscounted,
        new_undiscounted_path_minus_blind=old_undiscounted + penalty_delta,
        discounted_corner_delta_per_unit_K=slope,
        analytic_critical_K_discounted_zero_crossing=crossing,
        analytic_critical_K_undiscounted_zero_crossing=(-old_undiscounted / undiscounted_slope if undiscounted_slope != 0 else None),
        minimum_nonnegative_K_boundary_for_path_rejection=minimum_K,
        analytic_formula="D(K)=D_old+K*slope; K_zero=-D_old/slope; slope<0: rejection strictly above K_zero, equality tie. Negative roots are not positive-weight thresholds.",
        positive_weight_can_reject_path=old_delta < 0 or slope < 0,
        corner_penalty_disfavors_path=penalty_delta < 0,
        full_counterfactual_discounted_return_rejects_path=new_delta < 0)


def comparison(blind, path):
    result = dict(blind_label=blind["label"], path_label=path["label"],
                  scenario=blind["scenario"], seed=blind["seed"],
                  blind_mean_corner_peak_m=blind["mean_corner_peak_m"],
                  path_mean_corner_peak_m=path["mean_corner_peak_m"],
                  corner_regression=path["mean_corner_peak_m"] > blind["mean_corner_peak_m"] + 1e-10,
                  score_scopes={})
    for scope in ("training_prefix", "full40"):
        result["score_scopes"][scope] = score_delta(blind["reward_scores"][scope], path["reward_scores"][scope])
    return result


def aggregate_comparisons(flights, labels):
    """Equal-weight cells; candidates from the same trajectory are not replicates."""
    output = {}
    for scope_name, scenarios in (("all3_scenarios", ("nominal", "force_strong", "gust")),
                                  ("nominal", ("nominal",)), ("force_strong", ("force_strong",)),
                                  ("gust", ("gust",))):
        rows = {arm: [f for f in flights if f["arm"] == arm and f["label"] == labels[arm] and
                      f["scenario"] in scenarios] for arm in ("blind", "path")}
        aggregate_scores = {}
        for reward_scope in ("training_prefix", "full40"):
            arm_scores = {}
            for arm in rows:
                records = [f["reward_scores"][reward_scope] for f in rows[arm]]
                arm_scores[arm] = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
            aggregate_scores[reward_scope] = score_delta(arm_scores["blind"], arm_scores["path"])
        output[scope_name] = dict(cells_per_arm=len(rows["blind"]), labels=labels,
            blind_mean_corner_peak_m=float(np.mean([f["mean_corner_peak_m"] for f in rows["blind"]])),
            path_mean_corner_peak_m=float(np.mean([f["mean_corner_peak_m"] for f in rows["path"]])),
            blind_mean_finite_segment_rmse_m=float(np.mean([f["finite_segment_rmse_m"] for f in rows["blind"]])),
            path_mean_finite_segment_rmse_m=float(np.mean([f["finite_segment_rmse_m"] for f in rows["path"]])),
            score_scopes=aggregate_scores)
    return output


def corroborate_K40_against_preserved_K20(flights, plan):
    evidence = plan["prior_K20_evidence"]
    prior_path = ROOT / evidence["path"].replace("\\", "/")
    check(sha(prior_path) == evidence["sha256"], "preserved correctedK20 receipt hash")
    prior = read(prior_path)
    prior_flights = {(f["arm"], f["label"], f["case"]): f for f in prior["flights"]}
    max_linearity = 0.
    for flight in flights:
        old = prior_flights[flight["arm"], flight["label"], flight["case"]]
        close(flight["mean_corner_peak_m"], old["mean_corner_peak_m"], "K40 same measured geometry")
        for scope in ("training_prefix", "full40"):
            new_score, old_score = flight["reward_scores"][scope], old["reward_scores"][scope]
            for field in ("old_W8_reward_sum", "old_W8_discounted_return"):
                close(new_score[field], old_score[field], "K40 preserved base+W8")
            for new_key, old_key in (("corner_K_penalty", "corner_K20_penalty"),
                                     ("corner_K_discounted_penalty", "corner_K20_discounted_penalty")):
                error = abs(new_score[new_key] - 2 * old_score[old_key])
                max_linearity = max(max_linearity, error)
                check(error <= 1e-10, "K40 exact twofold fixedK20 penalty")
    snapshot_root = ROOT / evidence["preserved_coverage_final_sources"].replace("\\", "/")
    for name, digest in plan["coverage_final_v8_source_sha256"].items():
        check(sha(snapshot_root / name) == digest, "preserved finalK20 source/" + name)
    return dict(prior_receipt=str(prior_path.relative_to(ROOT)), prior_receipt_sha256=sha(prior_path),
                same72_trajectories=True, maximum_penalty_linearity_error=max_linearity,
                root_authorized_fixed_K40_not_grid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    if not plan_path.is_relative_to(OUT) or plan_path.name != "prospective_plan.json":
        raise ValueError("Require a prospectively frozen plan under the isolated v8 artifact directory")
    plan = read(plan_path)
    weight = float(plan["K_reward_per_m"])
    if len(plan["input_sha256"]) != 219 or weight not in (20., 40.):
        raise ValueError("Incomplete or changed prospective plan")
    if weight == 40. and "root_authorization" not in plan:
        raise ValueError("K40 requires separate explicit root authorization and frozen plan")
    source_seal = plan_path.parent / "implementation_source_seal.json"
    if weight == 40.:
        seal = read(source_seal)
        for name, digest in seal["source_sha256"].items():
            check(sha(ROOT / name) == digest, "sealed implementation/" + name)
        check(seal["K_reward_per_m"] == weight and seal["plan_sha256"] == sha(plan_path),
              "implementation sealed for this exact prospective plan/weight")
    for group in ("old_source_sha256", "input_sha256"):
        for name, expected in plan[group].items():
            check(sha(ROOT / name) == expected, "before/" + name)
    if FAILURES:
        raise ValueError("Frozen sources/inputs changed before replay: " + repr(FAILURES))
    receipt_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = plan_path.parent / receipt_id
    output.mkdir(exist_ok=False)
    purity = source_purity_checks()
    fixtures = fixture_checks()
    selector_fixtures = frozen_selector_checks()
    paths = sorted(RUN.glob("*/validation/*/route_*/summary.json"))
    check(len(paths) == 72, "all72 available v7 candidate routes")
    flights = []
    for path in paths:
        flights.append(replay(path, weight))
        if len(flights) % 12 == 0:
            print(json.dumps(dict(event="REPLAYED_ARCHIVED_ROUTES", routes=len(flights), checks=CHECKS,
                                  failures=len(FAILURES))), flush=True)
    lookup = {(f["arm"], f["label"], f["scenario"], f["seed"]): f for f in flights}
    selected = read(RUN / "pair_comparison.json")["selected_labels"]
    selected_rows = []
    matched_rows = []
    for scenario in ("nominal", "force_strong", "gust"):
        for seed in (84001, 84002, 84003, 84004):
            selected_rows.append(comparison(lookup["blind", selected["blind"], scenario, seed],
                                            lookup["path", selected["path"], scenario, seed]))
            for label in ("checkpoint_131072", "checkpoint_262144", "final"):
                matched_rows.append(comparison(lookup["blind", label, scenario, seed], lookup["path", label, scenario, seed]))
    tail = next(r for r in selected_rows if r["scenario"] == "force_strong" and r["seed"] == 84001)
    check(tail["corner_regression"], "selected actual tail corner regression retained")
    check(tail["score_scopes"]["training_prefix"]["corner_K_undiscounted_path_minus_blind"] < 0,
          "metric-aligned penalty alone disfavors actual tail corner regression")
    aggregates = dict(selected=aggregate_comparisons(flights, selected),
        matched={label: aggregate_comparisons(flights, dict(blind=label, path=label)) for label in
                 ("checkpoint_131072", "checkpoint_262144", "final")})
    prior_corroboration = corroborate_K40_against_preserved_K20(flights, plan) if weight == 40 else None
    before_after = {}
    changed = []
    for group in ("old_source_sha256", "input_sha256"):
        before_after[group] = {}
        for name, before in plan[group].items():
            after = sha(ROOT / name)
            before_after[group][name] = dict(before=before, after=after, unchanged=before == after)
            if before != after:
                changed.append(name)
            check(before == after, "after/" + name)
    new_sources = {str(p.relative_to(ROOT)): sha(p) for p in
                   (ROOT / "quadrl_corner_v8/__init__.py", ROOT / "quadrl_corner_v8/reward.py", Path(__file__))}
    if weight == 40:
        check(new_sources == seal["source_sha256"], "new implementation unchanged after K40 replay")
    tail_ranking_pass = all(r["full_counterfactual_discounted_return_rejects_path"] for r in tail["score_scopes"].values())
    composite_status = "FAIL" if FAILURES else ("FORMULA_PASS_TAIL_RANKING_PASS" if tail_ranking_pass else "FORMULA_PASS_TAIL_RANKING_FAIL")
    report = dict(schema="corner-reward-offline-prototype-v8.0", status=composite_status,
        formula_integrity_status="PASS" if not FAILURES else "FAIL",
        tail_total_discounted_return_ranking_status="PASS" if tail_ranking_pass else "FAIL_FIXED_K_STILL_PREFERS_CORNER_REGRESSION",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), checks=CHECKS, failures=FAILURES,
        scope="OFFLINE_ONLY_NO_TRAINING_NO_SIMULATOR_NO_NEW_FLIGHTS", new_flights=0, training_steps=0,
        prospective_plan=str(plan_path.relative_to(ROOT)), prospective_plan_sha256=sha(plan_path),
        fixed_K_reward_per_m=weight, gamma_for_counterfactual=.995,
        reward_formula="-K * sum_window(max(previous_peak, current_finite_distance) - previous_peak); no dt factor",
        closed_window="[same-time confirmed pre-goal1/2/3 transition, transition+1s]; outgoing endpoints fixed from confirmed old target and newly observed current target",
        base_and_W8="Archived trace.reward is original unwrapped precision-arrival-v3; independently add -dt*8*finite_preupdate_active_leg_distance^2; arrival10/completion20 retained",
        score_interpretation="Training-prefix is counterfactual original successful-route termination; full40 separately retains all flight reward. Discount gamma0.995 applies once per recorded50Hz control step; only undiscounted peak telescoping is controls-rate invariant",
        actor_information=purity, fixtures=fixtures, frozen_selector_fixtures=selector_fixtures, maximum_errors=MAXIMA,
        coverage_clarification="Frozen judge: actual endpoint if present or <=0.060001s actual inside/outside bracket; no interpolated/outside peak",
        coverage=dict(candidate_routes=72, candidate_control_rows=sum(f["steps"] for f in flights),
                      complete_corner_windows=sum(len(f["windows"]) for f in flights),
                      selected_route_pair_cells=len(selected_rows), matched_label_route_pair_cells=len(matched_rows),
                      training_seed_ids=[111], backend="archived fast_native_shaper_v3 development only",
                      candidate_states="blind/path x checkpoint131072/checkpoint262144/final x3 scenarios x4 development seeds; snapshots from same learning trajectory"),
        selected_labels=selected, selected_tail=tail,
        selected_pair_comparisons=selected_rows, matched_checkpoint_comparisons=matched_rows,
        equal_cell_aggregates=aggregates,
        linearity_scope="Analytic criticalK is for exactly these reused recorded trajectories and fixed gamma50Hz; it is not a model-training result, a universal optimum or an independent validation",
        tail_interpretation=dict(
            undiscounted_corner_penalty_rejects_regression=tail["score_scopes"]["training_prefix"]["corner_K_undiscounted_path_minus_blind"] < 0,
            full_counterfactual_discounted_return_rejects_regression_training_prefix=tail["score_scopes"]["training_prefix"]["full_counterfactual_discounted_return_rejects_path"],
            full_counterfactual_discounted_return_rejects_regression_full40=tail["score_scopes"]["full40"]["full_counterfactual_discounted_return_rejects_path"],
            failure_rule="If total counterfactual return still prefers the regressed path, report that ranking failure; do not tune K or claim PPO convergence"),
        source_and_input_integrity=dict(unchanged=not changed, changed=changed, before_after=before_after),
        new_source_sha256=new_sources, flights=flights,
        preserved_K20_linearity_corroboration=prior_corroboration,
        limitations=["Event metadata is AUDIT_YARDSTICK, not actor information or a new actor feature",
                     "No claim that K20 improves nominal performance or PPO convergence; no optimization was performed",
                     "No acceptance/Gazebo inheritance from replay; unchanged selector conjunction and full40 evaluation remain mandatory",
                     "Counterfactual rewards use recorded trajectories, not trajectories from a policy trained with this reward",
                     "Fixed absolute force times give different corner exposure for different route speeds; paired initial seeds do not align event-relative wind",
                     "Endpoint coverage uses actual endpoint or a <=0.060001s inside/outside bracket; outside samples never contribute peaks and no interpolation is used"],
        next_authorization="Root review only; no training, long-run/2M, simulator or Gazebo authorization")
    write_new(output / "report.json", report)
    # Preserve exact source bytes for each immutable receipt even when a later
    # separately authorized prototype extends the current workspace source.
    for name, digest in new_sources.items():
        destination = output / "frozen_sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        if sha(destination) != digest:
            raise RuntimeError("Could not preserve exact source snapshot: " + name)
    audit_path = ROOT / "audit" / ("joint_corner_reward_probe_v8_" + receipt_id + ".json")
    write_new(audit_path, dict(status=report["status"], checks=CHECKS, failures=FAILURES,
        report=str((output / "report.json").relative_to(ROOT)), report_sha256=sha(output / "report.json"),
        plan_sha256=sha(plan_path), new_source_sha256=new_sources, maximum_errors=MAXIMA,
        fixed_K_reward_per_m=weight, formula_integrity_status=report["formula_integrity_status"],
        source_and_input_integrity_unchanged=not changed, selected_tail=tail,
        new_flights=0, new_training_steps=0, root_review_only=True))
    print(json.dumps(dict(status=report["status"], checks=CHECKS, failures=FAILURES,
                          report=str(output / "report.json"), audit=str(audit_path),
                          selected_tail=tail, maximum_errors=MAXIMA), indent=2), flush=True)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())

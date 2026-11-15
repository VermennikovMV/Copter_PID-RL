"""Causal, stateful closed-window outgoing-corner peak increments.

This module has no route table, simulator, encoder, or policy dependency. The
new current target is public only AFTER the own mission judge confirms a
transition. Never pass a future-target preview as ``current_target``.

W8 and precision-arrival-v3 remain the caller's existing reward. Add the return
of observe() without multiplying it by dt. For any observed prefix the sum is
exactly -weight times the sum of its observed window peaks, up to roundoff.
The prospective original default is20; an explicit offline weight is recorded.
Missing
samples and early termination are reported, never zero-filled or extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

CORNER_WEIGHT = 20.0
WINDOW_S = 1.0
MAX_COVERAGE_GAP_S = 0.060001
TIME_TOL = 1e-9


def _vector(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be a finite 3-vector") from exc
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError(name + " must be a finite 3-vector")
    return result


def _close(a, b):
    return all(abs(x - y) <= 1e-9 for x, y in zip(a, b))


def finite_segment_distance(position, a, b) -> float:
    """Finite 3D projection. Only q is clipped; distance is never clipped."""
    p, a, b = (_vector(value, name) for value, name in
               ((position, "position"), (a, "a"), (b, "b")))
    ab = tuple(y - x for x, y in zip(a, b))
    pa = tuple(x - y for x, y in zip(p, a))
    length_sq = sum(x * x for x in ab)
    if not math.isfinite(length_sq):
        raise ValueError("segment arithmetic overflow")
    numerator = sum(x * y for x, y in zip(pa, ab))
    if not math.isfinite(numerator):
        raise ValueError("projection arithmetic overflow")
    q = numerator / length_sq if length_sq else 0.0
    if not math.isfinite(q):
        raise ValueError("projection arithmetic overflow")
    q = max(0.0, min(1.0, q))
    result = math.hypot(*(x - q * y for x, y in zip(pa, ab)))
    if not math.isfinite(result):
        raise ValueError("distance arithmetic overflow")
    return result


@dataclass(frozen=True)
class ConfirmedTransition:
    """Own mission judge's already-confirmed PRE-transition goal and time.

    The event deliberately contains no new/future waypoint. The latter must
    come from the current public target observed at the same timestamp.
    """
    goal_index: int
    time_s: float
    target: Sequence[float]


@dataclass
class _Window:
    corner_index: int
    transition_s: float
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    peak_m: float = 0.0
    samples: int = 0
    last_sample_s: float | None = None
    max_gap_s: float = 0.0
    end_observed: bool = False
    end_covered: bool = False
    first_sample_after_end_s: float | None = None
    closed: bool = False
    flags: set[str] = field(default_factory=set)

    @property
    def end_s(self):
        return self.transition_s + WINDOW_S

    def record(self):
        flags = set(self.flags)
        if not self.end_covered:
            flags.add("window_end_coverage_unconfirmed")
        complete = self.closed and self.end_covered and not flags
        return dict(corner_index=self.corner_index, transition_s=self.transition_s,
                    window_end_s=self.end_s, a=list(self.a), b=list(self.b),
                    observed_peak_m=self.peak_m,
                    full_window_peak_m=self.peak_m if complete else None,
                    full_post_window=complete, partial=not complete,
                    samples=self.samples, last_sample_s=self.last_sample_s,
                    max_sample_gap_s=self.max_gap_s,
                    transition_sample_included=self.samples > 0,
                    closed_endpoint_sample_included=self.end_observed,
                    end_coverage_confirmed=self.end_covered,
                    first_sample_after_end_s=self.first_sample_after_end_s,
                    flags=sorted(flags))


class CornerPeakReward:
    """Additive reward tracker; reset once per episode, observe once per sample.

    Start by observing the initial public current target (normally at t=0).
    Each later target change must have its own confirmed event at that same t.
    Goal0 opens no corner window; goals1/2/3 open independent fixed segments;
    goal4 completion opens none. Overlapping windows remain independent.

    An exact endpoint sample is included if present. Otherwise the first sample
    after the endpoint can prove coverage, with a bracketing gap <=0.060001s.
    That outside sample never contributes to the peak; no peak is interpolated.
    Until endpoint coverage is proved, the observed window remains partial.
    """

    def __init__(self, mission: str = "route", *, weight: float = CORNER_WEIGHT):
        self.weight = float(weight)
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")
        self.reset(mission)

    def reset(self, mission: str = "route") -> None:
        if mission not in ("route", "hover"):
            raise ValueError("mission must be route or hover")
        self.mission = mission
        self.last_time_s = None
        self.current_target = None
        self.last_confirmed_goal = -1
        self.windows: list[_Window] = []
        self.total_reward = 0.0
        self.finished = False
        self.finish_reason = None

    def observe(self, time_s: float, position: Sequence[float],
                current_target: Sequence[float],
                confirmed_transition: ConfirmedTransition | None = None,
                *, dt: float | None = None) -> float:
        """Consume only present/past data; invalid input cannot advance state."""
        stamp = float(time_s)
        p, target = _vector(position, "position"), _vector(current_target, "current_target")
        if self.finished:
            raise ValueError("observe after finish requires reset")
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("time must be finite and nonnegative")
        gap = stamp - self.last_time_s if self.last_time_s is not None else None
        if gap is not None and gap <= 0:
            raise ValueError("observation time must strictly increase")
        if dt is not None:
            step = float(dt)
            if not math.isfinite(step) or step <= 0:
                raise ValueError("dt must be finite and positive")
            if gap is not None and abs(step - gap) > TIME_TOL:
                raise ValueError("dt must match the measured timestamp increment")
        changed = self.current_target is not None and not _close(target, self.current_target)
        event = confirmed_transition
        opening = None
        if event is None:
            if changed:
                raise ValueError("target change needs same-time confirmed transition")
        else:
            if not isinstance(event, ConfirmedTransition) or self.mission != "route":
                raise ValueError("confirmed transitions require a route ConfirmedTransition")
            if isinstance(event.goal_index, bool) or not isinstance(event.goal_index, int):
                raise ValueError("confirmed goal index must be an integer")
            index = event.goal_index
            event_stamp = float(event.time_s)
            old_target = _vector(event.target, "confirmed old target")
            if not math.isfinite(event_stamp) or abs(event_stamp - stamp) > TIME_TOL:
                raise ValueError("transition must be observed at the same timestamp")
            if index != self.last_confirmed_goal + 1 or not 0 <= index <= 4:
                raise ValueError("confirmed route goals must be unique and ordered0..4")
            if self.current_target is None or not _close(old_target, self.current_target):
                raise ValueError("event old target must equal previously observed current target")
            if (index < 4 and not changed) or (index == 4 and changed):
                raise ValueError("new observed target must match transition/completion semantics")
            if index in (1, 2, 3):
                opening = _Window(index, stamp, old_target, target)
        # Validate all arithmetic before modifying existing windows or state.
        candidates = [w for w in self.windows if not w.closed]
        if opening is not None:
            candidates.append(opening)
        distances = {id(w): finite_segment_distance(p, w.a, w.b) for w in candidates
                     if stamp <= w.end_s + TIME_TOL}
        increments = []
        for window in candidates:
            if stamp > window.end_s + TIME_TOL:
                window.closed = True
                window.first_sample_after_end_s = stamp
                end_gap = stamp - window.last_sample_s
                window.max_gap_s = max(window.max_gap_s, end_gap)
                window.end_covered = end_gap <= MAX_COVERAGE_GAP_S
                if not window.end_covered:
                    window.flags.add("missing_end_coverage")
                continue
            if window.last_sample_s is not None:
                window_gap = stamp - window.last_sample_s
                window.max_gap_s = max(window.max_gap_s, window_gap)
                if window_gap > MAX_COVERAGE_GAP_S:
                    window.flags.add("missing_interior_coverage")
            new_peak = max(window.peak_m, distances[id(window)])
            increments.append(new_peak - window.peak_m)
            window.peak_m = new_peak
            window.samples += 1
            window.last_sample_s = stamp
            if abs(stamp - window.end_s) <= TIME_TOL:
                window.end_observed = True
                window.end_covered = True
                window.closed = True
        if opening is not None:
            self.windows.append(opening)
        if event is not None:
            self.last_confirmed_goal = event.goal_index
        self.last_time_s, self.current_target = stamp, target
        reward = -self.weight * math.fsum(increments)
        self.total_reward += reward
        return reward

    def finish(self, reason: str = "full_horizon") -> dict:
        """Finalize at the last real observation; no invented terminal sample."""
        if reason not in ("full_horizon", "completed_training_route", "failure", "early_termination"):
            raise ValueError("unknown finish reason")
        if self.finished:
            raise ValueError("finish may be called only once")
        for window in self.windows:
            if not window.closed:
                window.flags.add("terminated_before_window_end")
                window.closed = True
        self.finished, self.finish_reason = True, reason
        return self.report()

    def report(self) -> dict:
        records = [w.record() for w in self.windows]
        expected = 3 if self.mission == "route" else 0
        complete = sum(w["full_post_window"] for w in records)
        observed_peaks = math.fsum(w.peak_m for w in self.windows)
        return dict(mission=self.mission, expected_corners=expected,
                    observed_corners=len(records), unreached_corners=expected - len(records),
                    complete_corner_windows=complete,
                    corner_coverage_complete=complete == expected,
                    episode_partial=self.finish_reason in ("failure", "early_termination"),
                    full40_evaluation_guard_still_required=True,
                    finish_reason=self.finish_reason, last_time_s=self.last_time_s,
                    observed_sum_peak_m=observed_peaks, reward=self.total_reward,
                    weight_reward_per_m=self.weight,
                    telescoping_expected_reward=-self.weight * observed_peaks,
                    windows=records)

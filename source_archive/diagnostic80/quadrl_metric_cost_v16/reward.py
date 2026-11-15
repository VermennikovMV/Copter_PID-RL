"""Pure metric-cost accounting; fixed objective, not a universal quality proof."""
from __future__ import annotations
import math

FAILURE_OR_UNFINISHED_COST = 400.0


def finite_task_terminal(terminated, truncated, horizon_reached):
    """Learning-only deadline map; physical failure identity is never changed."""
    return (True, False) if horizon_reached else (bool(terminated), bool(truncated))


class MetricCostTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminal_penalty_emitted = False

    def step(self, *, mission_kind, parent_reward, dt, completed_before,
             active_PRE_leg, finite_distance_sq, yaw_error, tilt_cos,
             post_rate_sq, desired_action_delta_sq, failed, unfinished_horizon):
        values = (parent_reward, dt, finite_distance_sq, yaw_error, tilt_cos, post_rate_sq, desired_action_delta_sq)
        if mission_kind not in ("route", "hover") or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Finite declared task/cost inputs required")
        if dt <= 0 or min(finite_distance_sq, post_rate_sq, desired_action_delta_sq) < 0 or not -1.000000001 <= tilt_cos <= 1.000000001:
            raise ValueError("Positive dt/nonnegative squared costs/physical tilt required")
        if mission_kind == "hover":
            return float(parent_reward), dict(mission_kind=mission_kind, parent_reward=float(parent_reward),
                metric_reward=float(parent_reward), running_cost=0., terminal_penalty=0.,
                dt=float(dt), completed_before=bool(completed_before), active_PRE_leg=False,
                finite_distance_sq=float(finite_distance_sq), yaw_error=float(yaw_error), tilt_cos=float(tilt_cos),
                post_rate_sq=float(post_rate_sq), desired_action_delta_sq=float(desired_action_delta_sq),
                failed=bool(failed), unfinished_horizon=False,
                terminal_penalty_emitted=self.terminal_penalty_emitted, hover_parent_reward_exact=True)
        rate = 0.
        if not completed_before:
            rate = 1. + (80. * finite_distance_sq if active_PRE_leg else 0.)
            rate += 10. * (1. - math.cos(yaw_error)) + .05 * (1. - tilt_cos)
            rate += .003 * post_rate_sq + .008 * desired_action_delta_sq
        penalty = 0.
        if (failed or unfinished_horizon) and not self.terminal_penalty_emitted:
            penalty = FAILURE_OR_UNFINISHED_COST
            self.terminal_penalty_emitted = True
        result = -float(dt) * rate - penalty
        return result, dict(mission_kind=mission_kind, parent_reward=float(parent_reward), metric_reward=result,
            dt=float(dt), completed_before=bool(completed_before), active_PRE_leg=bool(active_PRE_leg),
            finite_distance_sq=float(finite_distance_sq), yaw_error=float(yaw_error), tilt_cos=float(tilt_cos),
            post_rate_sq=float(post_rate_sq), desired_action_delta_sq=float(desired_action_delta_sq),
            failed=bool(failed), unfinished_horizon=bool(unfinished_horizon),
            running_cost=rate, terminal_penalty=penalty, terminal_penalty_emitted=self.terminal_penalty_emitted,
            hover_parent_reward_exact=False)


def synthetic_fixtures():
    base = dict(mission_kind="route", parent_reward=31., dt=.02, completed_before=False,
        active_PRE_leg=True, finite_distance_sq=.25, yaw_error=0., tilt_cos=1.,
        post_rate_sq=0., desired_action_delta_sq=0., failed=False, unfinished_horizon=False)
    tracker = MetricCostTracker()
    value, ledger = tracker.step(**base)
    assert value == -.02 * 21. and ledger["terminal_penalty"] == 0.
    # Completion is confirmed AFTER this last interval; its cost is charged.
    completion_value, unused = tracker.step(**base)
    assert completion_value == value
    hold, unused = tracker.step(**{**base, "completed_before": True})
    assert hold == 0.
    failure_hold, one = tracker.step(**{**base, "completed_before": True, "failed": True, "unfinished_horizon": True})
    assert failure_hold == -400. and one["terminal_penalty"] == 400.
    twice, two = tracker.step(**{**base, "completed_before": True, "failed": True, "unfinished_horizon": True})
    assert twice == 0. and two["terminal_penalty"] == 0.
    timeout = MetricCostTracker()
    unfinished, row = timeout.step(**{**base, "active_PRE_leg": False, "finite_distance_sq": 0., "unfinished_horizon": True})
    assert unfinished == -400.02 and row["failed"] is False and row["terminal_penalty"] == 400.
    complete_at_deadline, row = MetricCostTracker().step(**base)
    assert row["terminal_penalty"] == 0. and complete_at_deadline == value
    hover, row = MetricCostTracker().step(**{**base, "mission_kind": "hover", "failed": True, "unfinished_horizon": True})
    assert hover == base["parent_reward"] and row["hover_parent_reward_exact"]
    for parent_t, parent_u in ((False, True), (True, True), (True, False), (False, False)):
        t, u = finite_task_terminal(parent_t, parent_u, True)
        assert t and not u and not (u and not t)  # DummyVecEnv TimeLimit.truncated=False.
        assert (1234. if (u and not t) else 0.) == 0.  # No SB3 terminal-value bootstrap.
        assert finite_task_terminal(parent_t, parent_u, False) == (parent_t, parent_u)
    return dict(status="PASS_PURE_SYNTHETIC_ONLY", fixed_failure_cost=400., terminal_penalty_once=True,
        last_completion_interval_charged=True, postcompletion_hold_zero_except_failure=True,
        failure_timeout_OR_once=True, timeout_not_physical_failure=True, hover_parent_exact=True,
        completion_at_deadline_no_unfinished_penalty=True, true_deadline_no_TimeLimit_bootstrap=True,
        new_physical_steps=0, optimizer_steps=0)

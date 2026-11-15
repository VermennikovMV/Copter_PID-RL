"""Common measured Path77/W8 environment; only effective K40 reward differs.

The tracker uses physical same-sample position, POST current next_target, and
the own mission's confirmed PRE-target event. No future-route table is read.
"""
from __future__ import annotations

from quadrl_path_v7.env import JointTaskEnv as FrozenPathTaskEnv
from .reward import CornerPeakReward, ConfirmedTransition

CORNER_WEIGHT = 40.0
VARIANTS = ("w8_control", "corner")


def coverage_termination(*, training, mission, successful, failed, horizon,
                         pending_windows, parent_terminated, parent_truncated):
    """Wait only after success; failure and the actual time limit override."""
    hold = bool(training and mission == "route" and successful and not failed
                and not horizon and not parent_truncated and pending_windows)
    if failed:
        return True, bool(horizon or parent_truncated), False
    if horizon or parent_truncated:
        return bool(parent_terminated and not pending_windows), True, False
    if hold:
        return False, False, True
    return bool(parent_terminated), bool(parent_truncated), False


class JointTaskEnv(FrozenPathTaskEnv):
    """Both reward arms execute identical tracking and common coverage hold."""

    def __init__(self, variant="corner", **kwargs):
        if variant not in VARIANTS:
            raise ValueError("Expected w8_control/corner reward arm")
        self.corner_variant = variant
        self.corner_reward_enabled = variant == "corner"
        self.corner_tracker = None
        self.coverage_hold_steps = 0
        self.completion_event_count = 0
        # Both actor AND critic always receive the unmasked causal Path77.
        super().__init__(variant="path", **kwargs)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.corner_tracker = CornerPeakReward(self.mission.kind, weight=CORNER_WEIGHT)
        self.corner_tracker.observe(0., self.state()[0], self.target)
        self.coverage_hold_steps = 0
        self.completion_event_count = 0
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = super().step(action)
        event = info.get("mission_event")
        confirmed = None
        if event is not None:
            if event.get("type") != "goal_reached":
                raise AssertionError("Unexpected own mission event")
            confirmed = ConfirmedTransition(int(event["goal_index"]),
                                            float(event["time_s"]), event["target"])
            if confirmed.goal_index == 4:
                self.completion_event_count += 1
                if self.completion_event_count != 1:
                    raise AssertionError("Base completion reward must emit once")
        raw = self.corner_tracker.observe(float(info["time"]), info["position"],
                  info["next_target"], confirmed, dt=self.control_dt)
        adjustment = raw if self.corner_reward_enabled else 0.
        pending = sum(not window.closed for window in self.corner_tracker.windows)
        successful = bool(self.training and self.mission.kind == "route" and
                          self.mission.completion_time is not None and not info["failed"])
        horizon = self.steps >= self.max_steps
        terminated, truncated, hold = coverage_termination(training=self.training,
            mission=self.mission.kind, successful=successful, failed=bool(info["failed"]),
            horizon=horizon, pending_windows=pending,
            parent_terminated=terminated, parent_truncated=truncated)
        self.coverage_hold_steps += int(hold)
        info.update(corner_reward_enabled=self.corner_reward_enabled,
            corner_configured_weight_reward_per_m=CORNER_WEIGHT,
            corner_tracker_reward_adjustment=raw, corner_reward_adjustment=adjustment,
            base_w8_reward=base_reward, corner_coverage_hold=hold,
            corner_coverage_hold_steps=self.coverage_hold_steps,
            corner_pending_windows=pending, completion_event_count=self.completion_event_count,
            reward_variant=self.corner_variant)
        if self.training and self.mission.kind == "route":
            # A completed route with a censored open window has not completed
            # this common training episode's coverage requirement.
            info["is_success"] = bool(successful and not pending and not hold)
        # The inherited summary is generated at its success edge. Emit ours
        # only at the actual common end, including any coverage hold steps.
        info.pop("curriculum_episode_summary", None)
        if terminated or truncated:
            reason = ("failure" if info["failed"] else "full_horizon" if horizon or truncated else
                      "completed_training_route" if successful else "early_termination")
            report = self.corner_tracker.finish(reason)
            report.update(enabled=self.corner_reward_enabled, effective_reward=(
                report["reward"] if self.corner_reward_enabled else 0.),
                censored_corner_windows=sum(window["partial"] for window in report["windows"]),
                coverage_hold_steps=self.coverage_hold_steps,
                completion_event_count=self.completion_event_count)
            info["corner_episode_summary"] = report
            if self.training:
                info["curriculum_episode_summary"] = self._curriculum_summary(info)
        return observation, base_reward + adjustment, terminated, truncated, info

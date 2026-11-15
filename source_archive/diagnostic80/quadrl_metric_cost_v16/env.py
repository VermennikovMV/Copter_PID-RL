"""One frozen v15 velocity-basis training step, then common learning deadline.

The actual plant/mission/arrival/stopping time/curriculum is inherited. Only
route training reward differs between arms. BOTH use finite-task termination
at existing30/40s training deadlines so gamma1 has no TimeLimit value bootstrap.
Frozen v15 NativeEvalEnv/evaluate are never changed or wrapped.
"""
from __future__ import annotations
import numpy as np
from quadrl_route_basis_v15.env import JointTaskEnv as FrozenTrainEnv
from .reward import MetricCostTracker, finite_task_terminal

ARMS = ("w8_control", "metric_cost")
BASIS_KIND = "velocity_residual"


class JointTaskEnv(FrozenTrainEnv):
    def __init__(self, variant="metric_cost", **kwargs):
        if variant not in ARMS: raise ValueError("Only fixed v16 reward arms supported")
        if kwargs.get("training", True) is not True:
            raise ValueError("v16 wrapper is training-only; use the frozen v15 native evaluator")
        self.reward_arm = variant
        self.metric_cost_tracker = MetricCostTracker()
        super().__init__(basis_kind=BASIS_KIND, **kwargs)

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.metric_cost_tracker.reset()
        return result

    def step(self, action):
        completed_before = self.mission.completion_time is not None
        previous_action = self.previous_action.copy()
        observation, parent_reward, terminated, truncated, info = super().step(action)
        horizon = self.steps >= self.max_steps
        completed_after = self.mission.completion_time is not None
        kind, judged = self.mission.kind, int(info["judged_goal_index"])
        active = kind == "route" and 1 <= judged <= 4
        distance_sq = -float(info["geometry_reward_adjustment"]) / (8. * self.control_dt) if active else 0.
        delta = self.previous_action - previous_action
        rates = np.asarray(info["body_rates"], dtype=float)
        cost_reward, ledger = self.metric_cost_tracker.step(mission_kind=kind, parent_reward=parent_reward,
            dt=self.control_dt, completed_before=completed_before, active_PRE_leg=active,
            finite_distance_sq=distance_sq, yaw_error=float(info["yaw_error"]), tilt_cos=float(info["tilt_cos"]),
            post_rate_sq=float(rates @ rates), desired_action_delta_sq=float(delta @ delta),
            failed=bool(info["failed"]), unfinished_horizon=bool(kind == "route" and horizon and not completed_after))
        old_t, old_u = terminated, truncated
        terminated, truncated = finite_task_terminal(terminated, truncated, horizon)
        ledger.update(completed_after=completed_after, judged_goal_index=judged,
            old_W8_geometry_adjustment=float(info["geometry_reward_adjustment"]),
            horizon_reached=horizon, parent_terminated=bool(old_t), parent_truncated=bool(old_u),
            learning_terminated=terminated, learning_truncated=truncated)
        info["metric_cost_ledger"] = ledger
        info["reward_arm"] = self.reward_arm
        info["finite_task_deadline"] = horizon
        if horizon: info["TimeLimit.truncated"] = False
        reward = cost_reward if self.reward_arm == "metric_cost" and kind == "route" else parent_reward
        return observation, reward, terminated, truncated, info

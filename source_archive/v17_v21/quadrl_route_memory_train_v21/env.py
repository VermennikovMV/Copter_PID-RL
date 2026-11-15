"""Unchanged v16 metric-cost plant/reward; exact causal v20 effective inputs."""
from __future__ import annotations
import numpy as np
from quadrl_metric_cost_v16.env import JointTaskEnv as FrozenMetricEnv
from quadrl_route_memory_v20.policy import ObservationEncoderRouteMemory,raw_config,source_values
from .policy import MODES


class JointTaskEnv(FrozenMetricEnv):
    def __init__(self,conditioning_mode,**kwargs):
        if conditioning_mode not in MODES:raise ValueError("Preregistered v21 mode required")
        self.conditioning_mode=conditioning_mode
        self.conditioner=ObservationEncoderRouteMemory(raw_config(source_values()),conditioning_mode)
        super().__init__(variant="metric_cost",**kwargs)

    def reset(self,seed=None,options=None):
        self.conditioner.reset()
        return super().reset(seed=seed,options=options)

    def observation(self,state=None):
        raw=super().observation(state)
        if self.conditioner.observation_calls==0:self.conditioner.reset(self.yaw_reference)
        measured=self.measured_state
        effective=self.conditioner.observe(measured["position"],measured["velocity"],measured["rotation"],
            measured["body_rates"],self.target,self.previous_action,self.control_dt)
        diagnostic=self.conditioner.diagnostics()
        assert np.array_equal(raw,diagnostic["raw_observation"]),"Training original80 encoder mismatch"
        sample=self.current_actor_sample
        sample.update(observation=effective.copy(),raw_observation=raw.copy(),effective_observation=effective.copy(),
            raw_integral_world=diagnostic["raw_integral_world"].copy(),conditioned_integral_world=diagnostic["conditioned_integral_world"].copy(),
            memory_valid_leg=diagnostic["valid_leg"],memory_target_changes=diagnostic["target_changes"],
            memory_target_changed=diagnostic["target_changed"],memory_raw_encoder_calls=diagnostic["raw_encoder_calls"],
            yaw_reference=float(self.yaw_reference))
        return effective

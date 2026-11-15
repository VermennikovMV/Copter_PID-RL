"""Observation-only CPU evaluator, with no training/Torch dependency.

This packaging adaptation mirrors PathObservationMixin/PathNativeEnv exactly for
the full-horizon evaluator. Original source hashes remain in the manifest.
"""
import numpy as np
from gymnasium import spaces
from quadrl_joint.env_v3 import QuadEnvJointV3
from quadrl_path_v7.policy import FiniteLegContext


class Path77CpuEnv(QuadEnvJointV3):
    def __init__(self, **kwargs):
        self.path_context = FiniteLegContext()
        super().__init__(architecture="hybrid", **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)

    def reset(self, seed=None, options=None):
        self.path_context.reset()
        return super().reset(seed=seed, options=options)

    def observation(self, state=None):
        base = super().observation(state)
        measured = self.measured_state
        extra = self.path_context.observe(measured["position"], measured["rotation"], self.target)
        self.path_feature = extra.copy()
        return np.concatenate((base, extra)).astype(np.float32)

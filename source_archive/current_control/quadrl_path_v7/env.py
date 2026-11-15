"""Observation-only Path77 integration over the frozen native/v5 environments.

The base74 encoder is called ONCE by the original observation method. Its noisy
measured sample and POST-mission-update current target supply the causal3. The
blind arm masks those3 at the common actor AND critic input. W8/reward judging,
mission transitions, plant, timing, sampler and curriculum remain inherited.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from quadrl_joint.env_v3 import QuadEnvJointV3
from scripts.train_joint_tailwind_v5 import JointTaskEnv as FrozenW8TaskEnv
from .policy import FiniteLegContext, CONTRACT_VERSION


class PathObservationMixin:
    def __init__(self, variant="path", **kwargs):
        if variant not in ("blind", "path"):
            raise ValueError("Only blind/path information ablation is supported")
        self.context_variant = variant
        self.path_context = FiniteLegContext()
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)

    def reset(self, seed=None, options=None):
        self.path_context.reset()
        return super().reset(seed=seed, options=options)

    def observation(self, state=None):
        base = super().observation(state)
        measured = self.measured_state
        raw = self.path_context.observe(measured["position"], measured["rotation"], self.target)
        self.path_feature = raw.copy()
        self.path_observation_flags = dict(valid_leg=self.path_context.valid_leg,
            target_changes=self.path_context.target_changes,
            feature_enabled=self.context_variant == "path")
        extra = raw if self.context_variant == "path" else np.zeros(3, np.float32)
        result = np.r_[base, extra].astype(np.float32)
        if base.shape != (74,) or result.shape != (77,):
            raise AssertionError("Original74 plus causal3 observation contract required")
        return result


class PathNativeEnv(PathObservationMixin, QuadEnvJointV3):
    """Full-horizon native evaluator with the new77 input contract."""


class JointTaskEnv(PathObservationMixin, FrozenW8TaskEnv):
    """Both arms inherit exactly the frozen v5 W8 reward and curriculum."""

    def __init__(self, variant="path", **kwargs):
        # The common W8 parent's reward variant is always geometry. The sole
        # between-arm choice is consumed by the observation mixin above it.
        PathObservationMixin.__init__(self, variant=variant, **kwargs)


OBSERVATION_CONTRACT = CONTRACT_VERSION

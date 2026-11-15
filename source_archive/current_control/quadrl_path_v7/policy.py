"""Measured finite-leg context appended to the unchanged74D encoder.

An observed target change makes the OLD observed target the new segment start
before computing this sample's features. Only measured current position/current
heading and previously observed targets are used. There is no route array, event
oracle, mission kind/completion, force, future waypoint or independent clock.

At mission start/hover, with no observed target change, the additional3 values
are zero. A known segment persists through final hold until reset or another
target change. A missed intermediate target cannot be recovered: the handler
uses the last ACTUALLY observed target, never an inferred missing waypoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np

from quadrl.policy import (
    CONTRACT_VERSION as BASE_CONTRACT_VERSION,
    EncoderConfig,
    NumpyPolicy,
    ObservationEncoder,
    heading_rotation,
)

CONTRACT_VERSION = "measured-memory-finite-leg-v3.0"
FINITE_LEG_SCALE_M = .25
DEGENERATE_LENGTH_SQUARED_M2 = 1e-15


@dataclass(frozen=True)
class EncoderConfigPath77(EncoderConfig):
    finite_leg_scale_m: float = FINITE_LEG_SCALE_M

    @property
    def size(self):
        return super().size + 3

    def base_config(self):
        return EncoderConfig(**{field.name: getattr(self, field.name) for field in fields(EncoderConfig)})


class FiniteLegContext:
    """Causal target memory and unclipped3D finite correction, in current heading."""

    def __init__(self, scale_m=FINITE_LEG_SCALE_M):
        if not np.isfinite(scale_m) or scale_m <= 0:
            raise ValueError("Positive finite path feature scale required")
        self.scale_m = float(scale_m)
        self.reset()

    def reset(self):
        self.current_target = None
        self.previous_target = None
        self.valid_leg = False
        self.target_changes = 0

    @staticmethod
    def validate(position, rotation, target):
        p, rot, goal = (np.asarray(value, dtype=float) for value in (position, rotation, target))
        if p.shape != (3,) or rot.shape != (3, 3) or goal.shape != (3,):
            raise ValueError("Measured position/rotation/current target shapes must be3/3x3/3")
        if not all(np.isfinite(value).all() for value in (p, rot, goal)):
            raise ValueError("Finite measured position/rotation/current target required")
        return p, rot, goal

    def observe(self, position, rotation, target):
        p, rot, goal = self.validate(position, rotation, target)
        if self.current_target is None:
            self.current_target = goal.copy()
        elif not np.array_equal(goal, self.current_target):
            self.previous_target = self.current_target.copy()
            self.current_target = goal.copy()
            self.target_changes += 1
        self.valid_leg = False
        if self.previous_target is None:
            return np.zeros(3, dtype=np.float32)
        leg = self.current_target - self.previous_target
        length2 = float(leg @ leg)
        if not np.isfinite(length2):
            raise ValueError("Finite segment length overflow")
        if length2 <= DEGENERATE_LENGTH_SQUARED_M2:
            return np.zeros(3, dtype=np.float32)
        fraction = float((p - self.previous_target) @ leg / length2)
        closest = self.previous_target + np.clip(fraction, 0., 1.) * leg
        _, heading = heading_rotation(rot)
        result = heading @ (closest - p) / self.scale_m
        if not np.isfinite(result).all():
            raise ValueError("Finite correction overflow")
        self.valid_leg = True
        # Scale is an engineering normalization, not geometric error clipping.
        return result.astype(np.float32)


class ObservationEncoderPath77:
    """The original74 are computed once, from the exact same sample and dt."""

    def __init__(self, config=None):
        self.config = config or EncoderConfigPath77()
        if not isinstance(self.config, EncoderConfigPath77) or self.config.base_config().size != 74:
            raise ValueError("Path77 contract requires the original74D history configuration")
        self.base_encoder = ObservationEncoder(self.config.base_config())
        self.path_context = FiniteLegContext(self.config.finite_leg_scale_m)

    def reset(self, yaw_reference=None):
        self.base_encoder.reset(yaw_reference)
        self.path_context.reset()

    def observe(self, position, velocity, rotation, body_rates, target, previous_action=None, dt=.02):
        # Validate path-required fields before either stateful encoder advances.
        FiniteLegContext.validate(position, rotation, target)
        base = self.base_encoder.observe(position, velocity, rotation, body_rates, target, previous_action, dt)
        extra = self.path_context.observe(position, rotation, target)
        result = np.r_[base, extra].astype(np.float32)
        if result.shape != (77,):
            raise AssertionError("Path77 observation size mismatch")
        return result


class NumpyPolicyPath77(NumpyPolicy):
    """Separate loader; the frozen74 loader and its metadata remain untouched."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.weights = [z[f"w{i}"].copy() for i in range(int(z["layers"]))]
            self.biases = [z[f"b{i}"].copy() for i in range(int(z["layers"]))]
            self.architecture, self.contract = str(z["architecture"]), str(z["contract"])
            base = json.loads(str(z["encoder_config"]))
            base["history_times_s"] = tuple(base["history_times_s"])
            path_config = json.loads(str(z["path_context_config"]))
            if str(z["base_contract"]) != BASE_CONTRACT_VERSION:
                raise ValueError("Unexpected base74 observation contract")
        if self.contract != CONTRACT_VERSION:
            raise ValueError("Path77 policy observation contract mismatch")
        if self.architecture not in ("hybrid", "direct"):
            raise ValueError("Unknown control architecture")
        self.encoder = ObservationEncoderPath77(EncoderConfigPath77(**base, **path_config))
        if self.weights[0].shape[1] != 77 or self.weights[-1].shape[0] != 4:
            raise ValueError("Invalid Path77 model dimensions")
        self.reset()


def lift_npz_path77(source, destination, scale_m=FINITE_LEG_SCALE_M):
    """Zero-column lift; retain every old actor array and base metadata exactly."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Fresh Path77 NPZ destination required")
    with np.load(source, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if str(values["contract"]) != BASE_CONTRACT_VERSION or values["w0"].shape[1] != 74:
        raise ValueError("Expected a frozen74D source policy")
    context = FiniteLegContext(scale_m)
    values["w0"] = np.c_[values["w0"], np.zeros((values["w0"].shape[0], 3), dtype=values["w0"].dtype)]
    values["base_contract"] = values["contract"].copy()
    values["contract"] = np.array(CONTRACT_VERSION)
    values["path_context_config"] = np.array(json.dumps(dict(finite_leg_scale_m=context.scale_m)))
    np.savez(destination, **values)
    NumpyPolicyPath77(destination)


def export_policy_path77(model, path, architecture="hybrid", config=None, base_encoder_metadata=None):
    """Export a loaded77D SB3 actor; this function never learns or steps a plant."""
    import torch
    cfg = config or EncoderConfigPath77()
    if cfg.base_config().size != 74:
        raise ValueError("Original74D base encoder required")
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    values = dict(layers=np.array(len(layers)), architecture=np.array(architecture),
                  contract=np.array(CONTRACT_VERSION), base_contract=np.array(BASE_CONTRACT_VERSION),
                  encoder_config=np.array(base_encoder_metadata if base_encoder_metadata is not None else json.dumps(asdict(cfg.base_config()))),
                  path_context_config=np.array(json.dumps(dict(finite_leg_scale_m=cfg.finite_leg_scale_m))))
    for index, layer in enumerate(layers):
        values[f"w{index}"] = layer.weight.detach().cpu().numpy()
        values[f"b{index}"] = layer.bias.detach().cpu().numpy()
    if values["w0"].shape[1] != 77:
        raise ValueError("A77D actor must be supplied")
    np.savez(path, **values)

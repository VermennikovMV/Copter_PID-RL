"""Frozen77 prefix plus one of two gated coordinate bases, never new sensors.

Both bases use only the exact measured/current-target sample already supplied
to Path77. A valid leg must have been formed by ACTUALLY observed target changes.
No mission kind/event/time/force/future waypoint or PID command enters here.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import numpy as np

from quadrl.policy import CONTRACT_VERSION as BASE74_CONTRACT, NumpyPolicy, heading_rotation
from quadrl_path_v7.policy import (CONTRACT_VERSION as PATH77_CONTRACT,
    EncoderConfigPath77, FiniteLegContext, ObservationEncoderPath77)

CONTRACT_VERSION = "measured-memory-route-basis-v4.0"
BASIS_KINDS = ("duplicate_path", "velocity_residual")
POSITION_GAIN_S_INV = 1.5
REFERENCE_SPEED_CAP_MPS = 2.0
VELOCITY_RESIDUAL_SCALE_MPS = 1.0


@dataclass(frozen=True)
class EncoderConfigRoute80(EncoderConfigPath77):
    basis_kind: str = "duplicate_path"
    position_gain_s_inv: float = POSITION_GAIN_S_INV
    reference_speed_cap_mps: float = REFERENCE_SPEED_CAP_MPS
    velocity_residual_scale_mps: float = VELOCITY_RESIDUAL_SCALE_MPS

    def __post_init__(self):
        if self.basis_kind not in BASIS_KINDS:
            raise ValueError("Only the two declared Route80 bases are supported")
        if (self.position_gain_s_inv, self.reference_speed_cap_mps, self.velocity_residual_scale_mps) != (1.5, 2., 1.):
            raise ValueError("Route80 prototype uses fixed1.5 gain/2mps reference normcap/1mps scale")
        if self.path77_config().size != 77:
            raise ValueError("Frozen77 history prefix required")

    @property
    def size(self):
        return self.path77_config().size + 3

    def path77_config(self):
        return EncoderConfigPath77(**{field.name: getattr(self, field.name) for field in fields(EncoderConfigPath77)})

    def basis_config(self):
        return {name: getattr(self, name) for name in
            ("basis_kind", "position_gain_s_inv", "reference_speed_cap_mps", "velocity_residual_scale_mps")}


def velocity_residual(position, velocity, rotation, current_target):
    """Fixed engineering reference feature; not a PID output or control blend."""
    p, rot, target = FiniteLegContext.validate(position, rotation, current_target)
    vel = np.asarray(velocity, dtype=float)
    if vel.shape != (3,) or not np.isfinite(vel).all():
        raise ValueError("Finite measured3D velocity required")
    with np.errstate(over="raise", invalid="raise"):
        reference = POSITION_GAIN_S_INV * (target - p)
        norm = math.hypot(*reference)
        if norm > REFERENCE_SPEED_CAP_MPS:
            reference = reference * (REFERENCE_SPEED_CAP_MPS / norm)
        unused, heading = heading_rotation(rot)
        result = heading @ (reference - vel) / VELOCITY_RESIDUAL_SCALE_MPS
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Velocity residual arithmetic overflow")
    return result


class ObservationEncoderRoute80:
    """Call frozen base77 once; append features from that SAME measured sample."""
    def __init__(self, config=None):
        self.config = config or EncoderConfigRoute80()
        if not isinstance(self.config, EncoderConfigRoute80):
            raise ValueError("Explicit Route80 encoder configuration required")
        self.base77 = ObservationEncoderPath77(self.config.path77_config())
        self.reset()

    @property
    def path_context(self):
        return self.base77.path_context

    def reset(self, yaw_reference=None):
        self.base77.reset(yaw_reference)
        self.base77_observe_calls = 0
        self.last_basis = np.zeros(3, dtype=np.float32)

    def observe(self, position, velocity, rotation, body_rates, target, previous_action=None, dt=.02):
        FiniteLegContext.validate(position, rotation, target)
        vel, rates = np.asarray(velocity, dtype=float), np.asarray(body_rates, dtype=float)
        if vel.shape != (3,) or rates.shape != (3,) or not np.isfinite(vel).all() or not np.isfinite(rates).all():
            raise ValueError("Finite measured velocity/body rates3-vectors required")
        # Validate arithmetic before advancing the stateful frozen encoder.
        residual = velocity_residual(position, vel, rotation, target) if self.config.basis_kind == "velocity_residual" else None
        prefix = self.base77.observe(position, velocity, rotation, body_rates, target, previous_action, dt)
        self.base77_observe_calls += 1
        extra = np.zeros(3, dtype=np.float32)
        if self.path_context.valid_leg:
            extra = prefix[74:77].copy() if self.config.basis_kind == "duplicate_path" else residual
        self.last_basis = extra.copy()
        observation = np.r_[prefix, extra].astype(np.float32)
        if observation.shape != (80,):
            raise AssertionError("Frozen77 plus3 Route80 inputs required")
        return observation


class NumpyPolicyRoute80(NumpyPolicy):
    """Separate strict80 loader; frozen74/77 loaders are never modified."""
    def __init__(self, path, expected_kind=None):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["contract"]) != CONTRACT_VERSION:
                raise ValueError("Route80 observation contract required; old77 is rejected")
            if str(archive["path_contract"]) != PATH77_CONTRACT or str(archive["base_contract"]) != BASE74_CONTRACT:
                raise ValueError("Unexpected frozen77/74 prefix contracts")
            self.weights = [archive[f"w{index}"].copy() for index in range(int(archive["layers"]))]
            self.biases = [archive[f"b{index}"].copy() for index in range(int(archive["layers"]))]
            self.architecture, self.contract = str(archive["architecture"]), str(archive["contract"])
            base = json.loads(str(archive["encoder_config"]))
            base["history_times_s"] = tuple(base["history_times_s"])
            context = json.loads(str(archive["path_context_config"]))
            basis = json.loads(str(archive["route_basis_config"]))
        config = EncoderConfigRoute80(**base, **context, **basis)
        if expected_kind is not None and config.basis_kind != expected_kind:
            raise ValueError("Route80 policy basis kind differs from expected metadata")
        if self.architecture != "hybrid":
            raise ValueError("This tested-source Route80 prototype is hybrid only")
        self.encoder = ObservationEncoderRoute80(config)
        if len(self.weights) != 3 or [weight.shape for weight in self.weights] != [(128, 80), (128, 128), (4, 128)]:
            raise ValueError("Route80 requires the frozen128x128 hybrid topology plus3 inputs")
        if any(bias.shape != (weight.shape[0],) for weight, bias in zip(self.weights, self.biases)):
            raise ValueError("Invalid Route80 bias dimensions")
        if not all(np.isfinite(array).all() for array in self.weights + self.biases):
            raise ValueError("Finite Route80 policy weights required")
        self.reset()


def export_policy_route80(model, path, config, base_encoder_metadata=None, path_context_metadata=None):
    """Readonly80 actor export with explicit immutable basis configuration."""
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError("Fresh Route80 export path required")
    if not isinstance(config, EncoderConfigRoute80):
        raise ValueError("Explicit Route80 basis configuration required")
    layers = [layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    if [tuple(layer.weight.shape) for layer in layers] != [(128, 80), (128, 128), (4, 128)]:
        raise ValueError("Loaded80 hybrid actor required")
    values = dict(layers=np.array(3), architecture=np.array("hybrid"), contract=np.array(CONTRACT_VERSION),
        base_contract=np.array(BASE74_CONTRACT), path_contract=np.array(PATH77_CONTRACT),
        encoder_config=np.array(base_encoder_metadata if base_encoder_metadata is not None else json.dumps(asdict(config.base_config()))),
        path_context_config=np.array(path_context_metadata if path_context_metadata is not None else json.dumps(dict(finite_leg_scale_m=config.finite_leg_scale_m))),
        route_basis_config=np.array(json.dumps(config.basis_config())))
    for index, layer in enumerate(layers):
        values[f"w{index}"] = layer.weight.detach().cpu().numpy()
        values[f"b{index}"] = layer.bias.detach().cpu().numpy()
    np.savez(path, **values)
    NumpyPolicyRoute80(path, expected_kind=config.basis_kind)

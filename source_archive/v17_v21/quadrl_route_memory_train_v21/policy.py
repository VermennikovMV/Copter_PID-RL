"""Strict trainable contract; exact v20 conditioning, finite learned actor arrays."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from quadrl.policy import NumpyPolicy
from quadrl_route_memory_v20.policy import (SOURCE_PATH,SOURCE_SHA,RAW_CONTRACT,source_values,
    raw_config,conditioning_config,ObservationEncoderRouteMemory,sha)

CONTRACT_VERSION="measured-memory-route-integral-train-v6.0"
MODES=("raw","zero_route_integral")
INITIAL_ZIP=Path("artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/checkpoint_131072.zip")
INITIAL_ZIP_SHA="0df1b6cd2811f5196395e8b707a3eee0387c8b0050375479a912511df779bf40"


class NumpyPolicyRouteMemoryTrain(NumpyPolicy):
    def __init__(self,path,expected_mode=None):
        with np.load(path,allow_pickle=False) as archive:values={name:archive[name].copy() for name in archive.files}
        if str(values.get("contract"))!=CONTRACT_VERSION:raise ValueError("Explicit trainable v21 contract required")
        conditioning=json.loads(str(values["conditioning_config"]))
        self.mode=conditioning["mode"]
        if self.mode not in MODES or (expected_mode is not None and self.mode!=expected_mode):raise ValueError("Expected v21 conditioning mode")
        if conditioning!=conditioning_config(self.mode):raise ValueError("Modified v20 conditioning semantics")
        original=source_values()
        extra={"conditioning_config","raw_old_contract","initial_actor_npz_sha256","initial_policy_zip_path","initial_policy_zip_sha256","weights_differ_from_initial"}
        if set(values)!=set(original)|extra:raise ValueError("Unexpected trainable NPZ schema")
        if str(values["raw_old_contract"])!=RAW_CONTRACT or str(values["initial_actor_npz_sha256"])!=SOURCE_SHA:
            raise ValueError("Source68 actor provenance required")
        if str(values["initial_policy_zip_path"])!=INITIAL_ZIP.as_posix() or str(values["initial_policy_zip_sha256"])!=INITIAL_ZIP_SHA:
            raise ValueError("Source68 continuation ZIP provenance required")
        actor={f"{prefix}{i}" for prefix in ("w","b") for i in range(3)}
        for name,value in original.items():
            if name not in actor|{"contract"} and not np.array_equal(values[name],value):raise ValueError("Changed original encoder metadata: "+name)
        self.weights=[values[f"w{i}"].copy() for i in range(3)]
        self.biases=[values[f"b{i}"].copy() for i in range(3)]
        if [w.shape for w in self.weights]!=[(128,80),(128,128),(4,128)] or [b.shape for b in self.biases]!=[(128,),(128,),(4,)]:raise ValueError("Frozen topology required")
        if not all(np.isfinite(a).all() for a in self.weights+self.biases):raise ValueError("Finite trained actor required")
        changed=any(not np.array_equal(values[name],original[name]) for name in actor)
        if bool(values["weights_differ_from_initial"])!=changed:raise ValueError("Actor-change metadata mismatch")
        self.architecture,self.contract=str(values["architecture"]),CONTRACT_VERSION
        self.encoder=ObservationEncoderRouteMemory(raw_config(values),self.mode)
        self.reset()


def export_policy(model,path,mode):
    import torch
    if mode not in MODES:raise ValueError("Preregistered mode required")
    path=Path(path)
    if path.exists():raise FileExistsError("Fresh trainable policy path required")
    values=source_values();original={n:v.copy() for n,v in values.items()}
    layers=[layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer,torch.nn.Linear)]+[model.policy.action_net]
    assert len(layers)==3
    for i,layer in enumerate(layers):
        values[f"w{i}"]=layer.weight.detach().cpu().numpy().copy();values[f"b{i}"]=layer.bias.detach().cpu().numpy().copy()
    values.update(contract=np.array(CONTRACT_VERSION),raw_old_contract=np.array(RAW_CONTRACT),
        conditioning_config=np.array(json.dumps(conditioning_config(mode),sort_keys=True)),initial_actor_npz_sha256=np.array(SOURCE_SHA),
        initial_policy_zip_path=np.array(INITIAL_ZIP.as_posix()),initial_policy_zip_sha256=np.array(INITIAL_ZIP_SHA),
        weights_differ_from_initial=np.array(any(not np.array_equal(values[f"{prefix}{i}"],original[f"{prefix}{i}"]) for prefix in ("w","b") for i in range(3))))
    np.savez(path,**values)
    NumpyPolicyRouteMemoryTrain(path,expected_mode=mode)

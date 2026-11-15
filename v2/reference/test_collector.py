"""Guard the data split and ensure offline collection never reads hidden truth."""
import json
import numpy as np
import pytest
from reference.collect_demonstrations import collect_dataset, ensure_development_seeds
from reference.dob import DOBReference
from quadrl.policy import ObservationEncoder, EncoderConfig


class MeasuredOnlyEnvironment:
    def __init__(self):
        self.encoder = ObservationEncoder(EncoderConfig(history=False))
        self.training_progress = 0.0

    @property
    def state(self):
        raise AssertionError("collector tried reading simulator truth")

    @property
    def parameters(self):
        raise AssertionError("collector tried reading disturbance parameters")

    def observation(self, previous_action):
        return self.encoder.observe([0, 0, 2], [0, 0, 0], np.eye(3), [0, 0, 0],
                                    [0, 0, 2], previous_action)

    def reset(self, seed):
        self.encoder.reset()
        self.steps = 0
        return self.observation(np.zeros(4)), {}

    def step(self, action):
        self.steps += 1
        return self.observation(action), 1.0, False, self.steps == 3, {"failed": False, "time": self.steps * 0.02}


def test_training_collector_rejects_validation_and_test_seeds():
    for seeds in ([82001], [202601], [100], [101, 101], []):
        with pytest.raises(ValueError):
            ensure_development_seeds(seeds)
    assert ensure_development_seeds([101, 102]) == [101, 102]


def test_collector_only_uses_measured_observations_and_preserves_episode_labels(tmp_path):
    destination = tmp_path / "data"
    manifest = collect_dataset(MeasuredOnlyEnvironment(), DOBReference(), [101, 102], destination)
    assert manifest["valid"] and manifest["samples"] == 6
    assert manifest["split"] == "development_training"
    assert manifest["runtime_teacher_required"] is False
    assert manifest["privileged_policy_inputs"] is False
    with np.load(destination / "observations_actions.npz") as data:
        assert set(data.files) == {"observations", "actions", "episode_ids"}
        assert data["observations"].shape == (6, 30)
        assert data["actions"].shape == (6, 4)
        np.testing.assert_array_equal(data["episode_ids"], [0, 0, 0, 1, 1, 1])
    assert json.loads((destination / "manifest.json").read_text())["dataset_sha256"]
    with pytest.raises(FileExistsError):
        collect_dataset(MeasuredOnlyEnvironment(), DOBReference(), [101], destination)

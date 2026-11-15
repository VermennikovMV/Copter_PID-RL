import numpy as np
from reference.collect_mixed_demonstrations import episode_design, MixedDemonstrationEnv
from quadrl.env import QuadEnv


def test_prospective_mixture_counts_and_seed_separation():
    specs = episode_design(range(2001, 2097))
    assert sum(not item["training_domain"] for item in specs) == 20
    assert sum(item["mission"] == "route" for item in specs) == 40
    assert sum(not item["training_domain"] and item["mission"] == "route" for item in specs) == 8
    assert len({item["seed"] for item in specs}) == 96
    assert all(101 <= item["seed"] <= 60000 for item in specs)


def test_mission_override_preserves_initial_measurements_and_early_physics():
    specs = episode_design(range(2001, 2097))
    chosen = [next(item for item in specs if item["training_domain"] == training and item["mission"] == mission)
              for training, mission in ((False, "route"), (True, "route"), (True, "hover"))]
    for item in chosen:
        wrapped = MixedDemonstrationEnv(specs)
        plain = QuadEnv(architecture="hybrid", training=item["training_domain"], scenario="nominal",
                        mission=item["mission"], seconds=30., history=True)
        plain.training_progress = 1.0
        try:
            actual, _ = wrapped.reset(item["seed"])
            expected, _ = plain.reset(seed=item["seed"])
            np.testing.assert_array_equal(actual, expected)
            assert wrapped._env.mission.kind == item["mission"]
            for _ in range(10):
                actual = wrapped.step(np.zeros(4))[0]
                expected = plain.step(np.zeros(4))[0]
                np.testing.assert_array_equal(actual, expected)
        finally:
            wrapped.close()
            plain.close()

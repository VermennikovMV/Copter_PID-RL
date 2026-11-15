"""Check standalone77 inference, causal geometry and actual CPU physical steps."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from quadrl_path_v7.policy import FiniteLegContext, NumpyPolicyPath77
from quadrl_path_v7.cpu_eval import Path77CpuEnv


def main():
    for name in ("quadrl.policy", "quadrl_path_v7.policy", "quadrl_path_v7.cpu_eval",
                 "quadrl_joint.env_v3", "quadrl_joint.native_rate_model_v3", "quadrl.env", "reference.direct_physics"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT), name
    assert "torch" not in sys.modules, "CPU inference must not import training/Torch"
    context = FiniteLegContext()
    R = np.eye(3)
    assert np.array_equal(context.observe(np.array([0., .2, 3.]), R, [0.,0.,3.]), np.zeros(3))
    feature = context.observe(np.array([1., .2, 3.]), R, [2.,0.,3.])
    assert np.allclose(feature, [0., -.8, 0.])
    assert np.allclose(context.observe(np.array([3.,0.,3.]), R, [2.,0.,3.]), [-4.,0.,0.])
    policy_path = ROOT / "v2/artifacts/refinement/joint_v2/corner_pilot111/w8_control/best_policy.npz"
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == "e0aa635603d5ec299cd526347891ef1b090e2f8ebd3879d5aa34f207a41fa781"
    policy = NumpyPolicyPath77(policy_path)
    env = Path77CpuEnv(scenario="nominal", mission="route", seconds=40.)
    observed = 0
    try:
        obs, _ = env.reset(seed=84001)
        assert obs.shape == (77,)
        for _ in range(env.max_steps):
            action = policy.predict(obs)
            assert action.shape == (4,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (77,) and np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated, info
            observed += 1
            if truncated:
                break
        assert observed == 2000 and len(env.mission.events) == 5 and env.mission.completion_time is not None
        completion = env.mission.completion_time
    finally:
        env.close()
    print(json.dumps(dict(status="PASS", scope="SELF_CONTAINED_CPU_SOFTWARE_QA_NOT_GAZEBO",
        observations=77, actual_control_steps=observed, actual_physics_substeps=observed*8,
        goals_reached=5, route_completion_time_s=completion, training_or_Gazebo_started=False)))


if __name__ == "__main__":
    main()

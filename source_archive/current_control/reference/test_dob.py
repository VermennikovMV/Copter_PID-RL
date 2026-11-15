"""Analytical checks for estimator sign, memory transport and saturation."""
import numpy as np
import pytest
from collections import deque
from dataclasses import replace
from reference.dob import DOBConfig, DOBReference, G
from quadrl.policy import ObservationEncoder, EncoderConfig


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_known_constant_acceleration_estimation_rotating_heading():
    controller = DOBReference(config=DOBConfig(nominal_drag_per_mass=0.0))
    encoder = ObservationEncoder(EncoderConfig(history=False))
    encoder.reset(yaw_reference=0.0)
    acceleration = np.array([1.4, -0.8, -0.3])
    for step in range(251):
        time = step * 0.02
        rotation = yaw_matrix(0.7 * time)
        observation = encoder.observe(0.5 * acceleration * time**2, acceleration * time,
                                      rotation, [0.0, 0.0, 0.7], np.zeros(3), np.zeros(4))
        controller.predict(observation)
    estimate_world = rotation @ controller.disturbance
    np.testing.assert_allclose(estimate_world, acceleration, atol=2e-5)


@pytest.mark.parametrize("delay_steps", [0, 1, 3])
@pytest.mark.parametrize("drag_z", [0.0, 0.9])
def test_first_order_motor_model_does_not_mistake_command_transient_for_wind(delay_steps, drag_z):
    cfg = DOBConfig(nominal_drag_per_mass=(0.01, 0.01, drag_z), nominal_command_delay_steps=delay_steps)
    controller = DOBReference(config=cfg)
    encoder = ObservationEncoder(EncoderConfig(history=False))
    velocity, position = np.zeros(3), np.zeros(3)
    specific_thrust = G
    previous_command = np.zeros(4)
    command_queue = deque([0.0] * delay_steps)
    for step in range(301):
        observation = encoder.observe(position, velocity, np.eye(3), np.zeros(3),
                                      np.zeros(3), previous_command)
        controller.predict(observation)
        command = 0.2 * np.sin(step * 0.02 * 2.0)
        command_queue.append(command)
        actual_command = command_queue.popleft()
        demand = (actual_command + 1) * G
        decay = np.exp(-0.02 / cfg.nominal_motor_tau_s)
        average = demand + (specific_thrust - demand) * cfg.nominal_motor_tau_s / 0.02 * (1 - decay)
        specific_thrust = demand + (specific_thrust - demand) * decay
        velocity[2] = (velocity[2] + (average - G) * 0.02 - 0.01 * drag_z * velocity[2]) / (1 + 0.01 * drag_z)
        position += velocity * 0.02
        previous_command = np.array([command, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(controller.disturbance, 0.0, atol=3e-5)


def test_saturation_prevents_integral_windup_and_all_outputs_remain_bounded():
    controller = DOBReference()
    encoder = ObservationEncoder(EncoderConfig(history=False))
    for step in range(300):
        observation = encoder.observe(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3),
                                      np.array([100.0, -100.0, 100.0]), np.zeros(4))
        action = controller.predict(observation)
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0
    np.testing.assert_allclose(controller.integral, 0.0, atol=1e-12)


def test_disabled_observer_changes_no_input_or_gains():
    enabled = DOBReference()
    disabled = DOBReference(config=replace(enabled.config, enabled=False))
    encoder = ObservationEncoder(EncoderConfig(history=False))
    observation = encoder.observe([0, 0, 2], [0, 0, 0], np.eye(3), [0, 0, 0], [0, 0, 2], np.zeros(4))
    np.testing.assert_array_equal(enabled.predict(observation), disabled.predict(observation))
    enabled_metadata, disabled_metadata = enabled.metadata(), disabled.metadata()
    enabled_metadata["config"]["enabled"] = False
    assert enabled_metadata == disabled_metadata


def test_reset_removes_previous_episode_adaptation_and_rejects_bad_observations():
    controller = DOBReference()
    controller.disturbance[:] = [2, 1, -1]
    controller.integral[:] = [1, 2, 3]
    controller.reset()
    assert controller.previous_velocity is None
    np.testing.assert_array_equal(controller.disturbance, np.zeros(3))
    np.testing.assert_array_equal(controller.integral, np.zeros(3))
    for invalid in (np.zeros(22), np.full(30, np.nan)):
        try:
            controller.predict(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid observation accepted")

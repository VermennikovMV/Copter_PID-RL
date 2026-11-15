"""Independent implementation of the documented measured-memory-yaw-v2 encoder.

No environment/policy code is imported. Used to replay actual measured-input logs.
The model producer's config is data; the feature calculations below are separate.
"""
import numpy as np


class IndependentEncoder:
    def __init__(self, config, yaw_reference=None):
        self.config = dict(config)
        self.reference = yaw_reference
        self.integral = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.previous_velocity = None
        self.elapsed = 0.
        self.frames = []

    def observe(self, position, velocity, rotation, body_rates, target, previous_action, dt):
        pos, vel, rot, rate, goal, action = [np.asarray(x, float) for x in
                                           (position, velocity, rotation, body_rates, target, previous_action)]
        rot = rot.reshape(3, 3)
        if not np.isfinite(np.r_[pos, vel, rot.ravel(), rate, goal, action, dt]).all() or dt <= 0:
            raise ValueError("invalid measured input")
        cfg = self.config
        heading = float(np.arctan2(rot[1, 0], rot[0, 0]))
        frame = np.array([[np.cos(heading), np.sin(heading), 0],
                          [-np.sin(heading), np.cos(heading), 0], [0, 0, 1.]])
        if self.reference is None:
            self.reference = heading
        error = goal - pos
        if self.previous_velocity is not None:
            self.integral += dt * error
            horizontal_length = np.linalg.norm(self.integral[:2])
            limit = cfg["integral_limit"]
            if horizontal_length > limit:
                self.integral[:2] *= limit / horizontal_length
            self.integral[2] = min(limit, max(-limit, self.integral[2]))
            acceleration_sample = (vel - self.previous_velocity) / dt
            weight = -np.expm1(-dt / cfg["acceleration_tau_s"])
            self.acceleration += weight * (acceleration_sample - self.acceleration)
            self.elapsed += dt
        self.previous_velocity = vel.copy()
        base = np.concatenate([frame @ error / cfg["position_scale"], frame @ vel / cfg["velocity_scale"],
                               (frame @ rot).ravel(), rate / cfg["rate_scale"], np.clip(action, -1, 1)])
        difference = self.reference - heading
        current = np.concatenate([base, frame @ self.integral / cfg["integral_limit"],
                                  frame @ self.acceleration / cfg["acceleration_scale"],
                                  [np.sin(difference), np.cos(difference)]])
        self.frames.append((self.elapsed, base.copy(), frame.copy()))
        # Match the documented finite history buffer; max requested lag is .2s.
        self.frames = self.frames[-128:]
        if cfg["history"]:
            memory = []
            for lag in cfg["history_times_s"]:
                candidates = [entry for entry in self.frames if entry[0] <= self.elapsed - lag + 1e-9]
                old_time, old_base, old_frame = candidates[-1] if candidates else self.frames[0]
                change = frame @ old_frame.T
                rebuilt = old_base.copy()
                rebuilt[:3] = change @ old_base[:3]
                rebuilt[3:6] = change @ old_base[3:6]
                rebuilt[6:15] = (change @ old_base[6:15].reshape(3, 3)).ravel()
                memory.append(rebuilt)
            current = np.concatenate([current, *memory])
        return current.astype(np.float32)


def independent_mlp(npz, observation):
    """Recompute exported deterministic actor without importing its wrapper."""
    state = np.asarray(observation, dtype=np.float32)
    count = int(npz["layers"])
    for index in range(count):
        state = np.matmul(state, np.asarray(npz[f"w{index}"]).T) + np.asarray(npz[f"b{index}"])
        if index + 1 < count:
            state = np.tanh(state)
    return np.clip(state, -1, 1)

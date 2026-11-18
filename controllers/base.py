"""Observation encoder shared by the networks in PyBullet, in Gazebo and in the firmware (C++ copy).

Everything is computed from what the autopilot measures (position, velocity, attitude, body rates)
and the target; no forces, mass or simulator parameters. Vectors are expressed in the heading frame
(yaw-only rotation), so the inputs do not depend on the global heading.

Inputs (74 with the default config):
  30  goal error / position_scale, velocity / velocity_scale, rotation in the heading frame (9),
      body rates / rate_scale, previous action (4), integral of the goal error, low-pass acceleration,
      sin/cos of the heading error
  22  the first 22 of these 0.1 s ago, and 22 more 0.2 s ago (world vectors in the current heading frame)
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

OBSERVATION_SCHEMA_ID = 'quadrotor-heading-observation-v2-clock'
COMMAND_SCHEMA_IDS = {
    'direct': 'motor-thrust-gazebo-order-v1',          # four motor commands
    'residual': 'ardupilot-cascade-residual-v1',       # ArduPilot cascade + bounded residual input
}


@dataclass(frozen=True)
class EncoderConfig:
    history: bool = True
    integral_limit: float = 6.
    acceleration_tau_s: float = .08
    position_scale: float = 3.
    velocity_scale: float = 3.
    rate_scale: float = 3.
    acceleration_scale: float = 5.
    history_times_s: tuple = (.1, .2)
    goal_clip_m: float = 0.     # > 0: the goal error vector is clipped to this length ("carrot")

    @property
    def size(self):
        return 30 + 22 * len(self.history_times_s) if self.history else 30


def heading_rotation(rotation):
    """Yaw of a body->world rotation and the world->heading-frame rotation."""
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    return yaw, np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])


class ObservationEncoder:
    """Stateful (integral, filtered acceleration, history): one instance per flight, reset at the start."""
    schema_id = OBSERVATION_SCHEMA_ID

    def __init__(self, config=None):
        self.config = config or EncoderConfig()
        self.reset()

    def reset(self, yaw_reference=None):
        self.yaw_reference = yaw_reference
        self.integral_world = np.zeros(3)
        self.acceleration_world = np.zeros(3)
        self.previous_velocity = None
        self.time = 0.
        self.history = deque()

    def observe(self, position, velocity, rotation, body_rates, target, previous_action=None, dt=.02):
        cfg = self.config
        pos, vel, rot, rates, goal = map(np.asarray, (position, velocity, rotation, body_rates, target))
        action = np.zeros(4) if previous_action is None else np.asarray(previous_action, dtype=float)
        yaw, heading = heading_rotation(rot)
        if self.yaw_reference is None:
            self.yaw_reference = yaw
        error = goal - pos
        if cfg.goal_clip_m > 0:
            length = float(np.linalg.norm(error))
            if length > cfg.goal_clip_m:
                error = error * (cfg.goal_clip_m / length)
        if self.previous_velocity is not None:
            self.integral_world += error * dt
            # the horizontal limit is on the norm (independent of the global heading), vertical separately
            length = np.linalg.norm(self.integral_world[:2])
            if length > cfg.integral_limit:
                self.integral_world[:2] *= cfg.integral_limit / length
            self.integral_world[2] = np.clip(self.integral_world[2], -cfg.integral_limit, cfg.integral_limit)
            observed_acceleration = (vel - self.previous_velocity) / dt
            alpha = 1 - np.exp(-dt / cfg.acceleration_tau_s)
            self.acceleration_world += alpha * (observed_acceleration - self.acceleration_world)
            self.time += dt
        self.previous_velocity = vel.copy()

        base = np.r_[heading @ error / cfg.position_scale, heading @ vel / cfg.velocity_scale,
                     (heading @ rot).reshape(-1), rates / cfg.rate_scale, np.clip(action, -1, 1)]
        yaw_error = self.yaw_reference - yaw
        current = np.r_[base, heading @ self.integral_world / cfg.integral_limit,
                        heading @ self.acceleration_world / cfg.acceleration_scale,
                        np.sin(yaw_error), np.cos(yaw_error)]
        if not cfg.history:
            return current.astype(np.float32)

        self.history.append((self.time, base.copy(), heading.copy()))
        oldest = self.time - max(cfg.history_times_s)
        while len(self.history) > 1 and self.history[1][0] <= oldest:
            self.history.popleft()
        old = []
        for lag in cfg.history_times_s:
            selected, old_heading = self.history[0][1:]
            for stamp, frame, frame_heading in self.history:
                if stamp <= self.time - lag + 1e-9:
                    selected, old_heading = frame, frame_heading
                else:
                    break
            # past world vectors are re-expressed in the CURRENT heading frame;
            # past body rates and actions keep their body axes
            change = heading @ old_heading.T
            moved = selected.copy()
            moved[:3] = change @ selected[:3]
            moved[3:6] = change @ selected[3:6]
            moved[6:15] = (change @ selected[6:15].reshape(3, 3)).reshape(-1)
            old.append(moved)
        return np.r_[current, *old].astype(np.float32)

"""Explicit waypoint tolerance for demos; frozen research protocol stays intact."""
import math
import numpy as np
from quadrl.protocol import Mission, CONTROL_DT, ROUTE_DWELL_S, ROUTE_POINTS


def acceptance_spec(radius):
    radius = float(radius)
    if not math.isfinite(radius) or not 0 < radius <= 1.:
        raise ValueError("Waypoint radius must be finite and in (0, 1] metres")
    return dict(waypoint_radius_m=radius, arrival_dwell_s=ROUTE_DWELL_S,
                measurement_gap_limit_s=1.5 * CONTROL_DT,
                hover_recovery_radius_m=.25,
                rule="shared across PID, hybrid and direct; measured 3D distance")


class DemoMission(Mission):
    """Same ordered goals and dwell; only the shared arrival radius is configurable."""
    def __init__(self, kind="hover", waypoint_radius_m=.25):
        self.acceptance = acceptance_spec(waypoint_radius_m)
        self.waypoint_radius_m = self.acceptance["waypoint_radius_m"]
        super().__init__(kind)

    def update(self, t, position):
        t = float(t)
        position = np.asarray(position, dtype=float)
        if not np.isfinite(t) or not np.all(np.isfinite(position)):
            raise ValueError("invalid mission observation")
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("mission time must increase")
        if self.last_time is not None and t - self.last_time > 1.5 * CONTROL_DT:
            self.inside_since = None
        self.last_time = t
        if self.kind == "hover" or self.completion_time is not None:
            return None
        if np.linalg.norm(position - self.target) > self.waypoint_radius_m:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = t
        if t - self.inside_since + 1e-9 < ROUTE_DWELL_S:
            return None
        event = {"type": "goal_reached", "goal_index": self.index, "time_s": t,
                 "dwell_started_s": self.inside_since, "target": self.target.tolist()}
        self.events.append(event)
        self.inside_since = None
        if self.index == len(ROUTE_POINTS) - 1:
            self.completion_time = t
        else:
            self.index += 1
        return event

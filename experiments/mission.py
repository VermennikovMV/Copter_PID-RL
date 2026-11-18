"""Controller-independent waypoint acceptance for experiments and training."""
import math
import numpy as np
from experiments.scenarios import CONTROL_DT, ROUTE_DWELL_S, ROUTE_POINTS


def acceptance_spec(radius):
    radius = float(radius)
    if not math.isfinite(radius) or not 0 < radius <= 1.:
        raise ValueError("Waypoint radius must be finite and in (0, 1] metres")
    return dict(waypoint_radius_m=radius, arrival_dwell_s=ROUTE_DWELL_S,
                measurement_gap_limit_s=1.5 * CONTROL_DT,
                hover_recovery_radius_m=.25,
                rule="shared across controllers; measured 3D distance")


class Mission:
    """One ordered route and one continuous-dwell rule for all controllers.

    Optional track description (all three per point, same length):
      points  - waypoints, metres (ENU); index 0 is the start point;
      dwell_s - 0 for a fly-through point, >0 for a stop point (hold this long inside);
      radii_m - acceptance radius per point.
    Without them the original fixed route (ROUTE_POINTS, ROUTE_DWELL_S) is used."""
    def __init__(self, kind="hover", waypoint_radius_m=.25, points=None, dwell_s=None, radii_m=None):
        if kind not in ("hover", "route"):
            raise ValueError("Mission must be hover or route")
        self.kind = kind
        self.acceptance = acceptance_spec(waypoint_radius_m)
        self.waypoint_radius_m = self.acceptance["waypoint_radius_m"]
        self.points = np.asarray(ROUTE_POINTS if points is None else points, dtype=float)
        n = len(self.points)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or n < 2 or not np.all(np.isfinite(self.points)):
            raise ValueError("Track needs at least two finite 3D points")
        self.dwell_s = np.full(n, ROUTE_DWELL_S) if dwell_s is None else np.asarray(dwell_s, dtype=float)
        self.radii_m = np.full(n, self.waypoint_radius_m) if radii_m is None else np.asarray(radii_m, dtype=float)
        if self.dwell_s.shape != (n,) or self.radii_m.shape != (n,) or np.any(self.dwell_s < 0) or np.any(self.radii_m <= 0):
            raise ValueError("Per-point dwell (>=0) and radius (>0) required")
        self.horizon_s = 30. if kind == "hover" else 40.
        self.reset()

    def reset(self):
        self.index = 0
        self.inside_since = None
        self.last_time = None
        self.completion_time = None
        self.events = []

    @property
    def target(self):
        return self.points[self.index].copy()

    @property
    def last_index(self):
        return len(self.points) - 1

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
        if np.linalg.norm(position - self.target) > self.radii_m[self.index]:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = t
        if t - self.inside_since + 1e-9 < self.dwell_s[self.index]:
            return None
        event = {"type": "goal_reached", "goal_index": self.index, "time_s": t,
                 "dwell_started_s": self.inside_since, "target": self.target.tolist()}
        self.events.append(event)
        self.inside_since = None
        if self.index == self.last_index:
            self.completion_time = t
        else:
            self.index += 1
        return event

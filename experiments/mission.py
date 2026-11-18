"""Waypoint acceptance shared by all controllers, in PyBullet and in Gazebo."""
import numpy as np

from experiments.scenarios import CONTROL_DT, ROUTE_DWELL_S, ROUTE_POINTS


class Mission:
    """Ordered waypoints; a point is reached after staying inside its radius for its dwell time.

      points  - waypoints, m (ENU); index 0 is the start point
      dwell_s - 0 for a fly-through point, > 0 for a stop point
      radii_m - acceptance radius per point
    Without a track the default route (ROUTE_POINTS, ROUTE_DWELL_S) is used. A hover mission
    holds point 0."""

    def __init__(self, kind='hover', waypoint_radius_m=.25, points=None, dwell_s=None, radii_m=None):
        if kind not in ('hover', 'route'):
            raise ValueError('mission must be hover or route')
        self.kind = kind
        self.points = np.asarray(ROUTE_POINTS if points is None else points, dtype=float)
        n = len(self.points)
        self.dwell_s = np.full(n, ROUTE_DWELL_S) if dwell_s is None else np.asarray(dwell_s, dtype=float)
        self.radii_m = np.full(n, float(waypoint_radius_m)) if radii_m is None else np.asarray(radii_m, dtype=float)
        if self.points.shape != (n, 3) or n < 2 or self.dwell_s.shape != (n,) or self.radii_m.shape != (n,):
            raise ValueError('track needs >= 2 points (x, y, z) with one dwell and one radius each')
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
        """Advance with the measured position at time t; returns the event when a point is reached."""
        t, position = float(t), np.asarray(position, dtype=float)
        if self.last_time is not None and t - self.last_time > 1.5 * CONTROL_DT:
            self.inside_since = None                       # a gap in measurements restarts the dwell
        self.last_time = t
        if self.kind == 'hover' or self.completion_time is not None:
            return None
        if np.linalg.norm(position - self.target) > self.radii_m[self.index]:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = t
        if t - self.inside_since + 1e-9 < self.dwell_s[self.index]:
            return None
        event = {'type': 'goal_reached', 'goal_index': self.index, 'time_s': t,
                 'dwell_started_s': self.inside_since, 'target': self.target.tolist()}
        self.events.append(event)
        self.inside_since = None
        if self.index == self.last_index:
            self.completion_time = t
        else:
            self.index += 1
        return event

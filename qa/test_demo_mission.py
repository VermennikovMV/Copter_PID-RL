"""Waypoint semantics: no simulator, neural forward or training required."""
import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "v2")]
from quadrl.protocol import Mission
from demo_three.mission import DemoMission, acceptance_spec


class WaypointTests(unittest.TestCase):
    def test_default_preserves_frozen_route(self):
        old, new = Mission("route"), DemoMission("route")
        for i in range(180):
            position = old.target.copy()
            self.assertEqual(old.target.tolist(), new.target.tolist())
            self.assertEqual(old.update(i * .02, position), new.update(i * .02, position))
        self.assertEqual(len(new.events), 5)
        self.assertEqual(old.completion_time, new.completion_time)

    def test_radius_changes_arrival_but_still_requires_dwell(self):
        old, wider = DemoMission("route", .25), DemoMission("route", .4)
        for i in range(25):
            self.assertIsNone(wider.update(i * .02, [.35, 0, 3]))
            self.assertIsNone(old.update(i * .02, [.35, 0, 3]))
        self.assertEqual(wider.update(.5, [.35, 0, 3])["goal_index"], 0)
        self.assertIsNone(old.update(.5, [.35, 0, 3]))

    def test_real_missing_measurement_breaks_dwell(self):
        mission = DemoMission("route", .4)
        for i in range(25):
            self.assertIsNone(mission.update(i * .02, [0, 0, 3]))
        self.assertIsNone(mission.update(.52, [0, 0, 3]))
        for i in range(1, 25):
            self.assertIsNone(mission.update(.52 + i * .02, [0, 0, 3]))
        self.assertEqual(mission.update(1.02, [0, 0, 3])["time_s"], 1.02)

    def test_radius_does_not_relax_hover_recovery_or_dwell(self):
        spec = acceptance_spec(.4)
        self.assertEqual(spec["arrival_dwell_s"], .5)
        self.assertEqual(spec["hover_recovery_radius_m"], .25)
        self.assertEqual(spec["measurement_gap_limit_s"], .03)
        hover = DemoMission("hover", .4)
        for i in range(100):
            self.assertIsNone(hover.update(i * .02, [0, 0, 3]))

    def test_invalid_radius_is_rejected(self):
        for radius in (0, -.1, 1.1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                DemoMission("route", radius)


if __name__ == "__main__":
    unittest.main()

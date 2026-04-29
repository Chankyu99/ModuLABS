from __future__ import annotations

import unittest

from pipeline.config import MAX_OPERATIONAL_OFF_NADIR_DEG
from pipeline.Level2a.pass_predictor import (
    _ground_range_for_off_nadir_km,
    _required_off_nadir_deg,
)


class PassPredictorOffNadirTest(unittest.TestCase):
    def test_required_off_nadir_round_trips_ground_range(self) -> None:
        altitude_km = 500.0
        ground_range_km = _ground_range_for_off_nadir_km(
            MAX_OPERATIONAL_OFF_NADIR_DEG,
            altitude_km,
        )

        required = _required_off_nadir_deg(ground_range_km, altitude_km)

        self.assertAlmostEqual(required, MAX_OPERATIONAL_OFF_NADIR_DEG, places=6)

    def test_thirty_degree_operational_limit_is_not_tangent_overestimate(self) -> None:
        altitude_km = 500.0

        ground_range_km = _ground_range_for_off_nadir_km(
            MAX_OPERATIONAL_OFF_NADIR_DEG,
            altitude_km,
        )

        self.assertLess(ground_range_km, altitude_km)
        self.assertAlmostEqual(ground_range_km, 292.7, places=1)


if __name__ == "__main__":
    unittest.main()

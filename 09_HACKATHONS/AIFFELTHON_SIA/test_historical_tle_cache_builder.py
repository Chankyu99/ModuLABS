from __future__ import annotations

import unittest

from build_historical_tle_cache import (
    build_cache_payload,
    parse_historical_tle_text,
    parse_tle_epoch,
)
from pipeline.Level2a.satellite_catalog import load_satellite_catalog


SAMPLE_TEXT = """
1 40697U 15028A   26059.50000000  .00000100  00000-0  10000-4 0  9991
2 40697  98.5658 146.7894 0001206 103.4362 256.6955 14.30814221559762
1 40697U 15028A   26064.50000000  .00000100  00000-0  10000-4 0  9992
2 40697  98.5658 147.7894 0001206 103.4362 256.6955 14.30814221559763
1 43797U 18099AR  26059.60000000  .00000100  00000-0  10000-4 0  9993
2 43797  96.9401  87.3139 0004547 161.9651 198.1761 15.45212732405050
1 43797U 18099AR  26076.60000000  .00000100  00000-0  10000-4 0  9994
2 43797  96.9401  88.3139 0004547 161.9651 198.1761 15.45212732405051
1 43800U 18099AU  26059.70000000  .00000100  00000-0  10000-4 0  9995
2 43800  97.4519 114.9730 0009443 328.1507  31.9153 15.16267814396295
1 63229U 25052V   26059.80000000  .00000100  00000-0  10000-4 0  9996
2 63229  97.4089 313.5503 0005312 152.7995 207.3520 15.21361388 53274
1 66820U 25279A   26059.90000000  .00000100  00000-0  10000-4 0  9997
2 66820  97.6691   8.9744 0010684  14.0373 346.1146 15.01339272 13283
"""


class HistoricalTleCacheBuilderTest(unittest.TestCase):
    def test_planetscope_spaceeye_catalog_excludes_missing_norads(self) -> None:
        excluded_norads = {60518, 60519, 60558, 60559, 60560, 60561, 60563}
        catalog = load_satellite_catalog("planetscope-spaceeye")
        norads = {int(sat["norad_id"]) for sat in catalog}

        self.assertTrue(excluded_norads.isdisjoint(norads))
        self.assertIn(63229, norads)
        self.assertEqual(len(catalog), 54)

    def test_parse_tle_epoch(self) -> None:
        epoch = parse_tle_epoch("1 40697U 15028A   26059.50000000  .00000100  00000-0  10000-4 0  9991")
        self.assertEqual(epoch.strftime("%Y%m%d"), "20260228")

    def test_parse_historical_tle_text(self) -> None:
        records = parse_historical_tle_text(SAMPLE_TEXT)
        self.assertEqual(len(records), 7)
        self.assertEqual(records[0].norad_id, 40697)
        self.assertEqual(records[-1].norad_id, 66820)

    def test_build_cache_payload_picks_latest_before_target_date(self) -> None:
        records = parse_historical_tle_text(SAMPLE_TEXT)
        payload = build_cache_payload(records, "20260305", scenario="default", allow_partial=True)
        self.assertEqual(payload["40697"]["line1"].split()[3], "26064.50000000")
        self.assertEqual(payload["43797"]["line1"].split()[3], "26059.60000000")

        payload_late = build_cache_payload(records, "20260317", scenario="default", allow_partial=True)
        self.assertEqual(payload_late["43797"]["line1"].split()[3], "26076.60000000")


if __name__ == "__main__":
    unittest.main()

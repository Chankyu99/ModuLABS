from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from pipeline.Level2a import weather_checker


class WeatherCheckerTest(unittest.TestCase):
    def test_selects_archive_for_past_date(self) -> None:
        now_utc = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
        kind = weather_checker.resolve_weather_source_kind(
            "2026-03-16T12:15:00Z",
            now_utc=now_utc,
        )
        self.assertEqual(kind, "archive")

    def test_selects_forecast_for_same_day_or_future(self) -> None:
        now_utc = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
        same_day = weather_checker.resolve_weather_source_kind(
            "2026-04-23T23:15:00Z",
            now_utc=now_utc,
        )
        future_day = weather_checker.resolve_weather_source_kind(
            "2026-04-24T01:15:00Z",
            now_utc=now_utc,
        )
        self.assertEqual(same_day, "forecast")
        self.assertEqual(future_day, "forecast")

    @patch("pipeline.Level2a.weather_checker._fetch_cloud_archive")
    @patch("pipeline.Level2a.weather_checker._fetch_cloud_forecast")
    def test_get_cloud_cover_uses_archive_for_past_dates(
        self,
        forecast_mock,
        archive_mock,
    ) -> None:
        archive_mock.return_value = {
            "cloud_map": {
                "2026-03-16T12:00": 37,
            }
        }
        forecast_mock.return_value = {
            "cloud_map": {
                "2026-03-16T12:00": 99,
            }
        }

        result = weather_checker.get_cloud_cover(
            35.6892,
            51.3890,
            "2026-03-16T12:15:00Z",
            now_utc=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result["cloud_cover_pct"], 37)
        self.assertEqual(result["cloud_status"], "partial")
        archive_mock.assert_called_once()
        forecast_mock.assert_not_called()

    @patch("pipeline.Level2a.weather_checker._fetch_cloud_archive")
    @patch("pipeline.Level2a.weather_checker._fetch_cloud_forecast")
    def test_get_cloud_cover_uses_forecast_for_future_dates(
        self,
        forecast_mock,
        archive_mock,
    ) -> None:
        forecast_mock.return_value = {
            "cloud_map": {
                "2026-04-24T01:00": 10,
            }
        }
        archive_mock.return_value = {
            "cloud_map": {
                "2026-04-24T01:00": 90,
            }
        }

        result = weather_checker.get_cloud_cover(
            35.6892,
            51.3890,
            "2026-04-24T00:45:00Z",
            now_utc=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result["cloud_cover_pct"], 10)
        self.assertEqual(result["cloud_status"], "clear")
        forecast_mock.assert_called_once()
        archive_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import unittest

from pipeline.Level2a.schedule_builder import (
    build_city_execution_plan,
    compute_policy_preference,
    print_schedule,
)


class ScheduleBuilderOutputTest(unittest.TestCase):
    def test_policy_preference_prioritizes_sia_business_assets(self) -> None:
        base_event = {
            "sensor_type": "optical",
            "daylight": True,
            "cloud_cover_pct": 10,
            "max_elevation_deg": 80,
            "required_off_nadir_abs_deg": 5,
            "off_nadir_limit_deg": 30,
        }

        spaceeye = compute_policy_preference({**base_event, "satellite": "SpaceEye-T"})
        planetscope = compute_policy_preference({**base_event, "satellite": "PlanetScope-58270"})
        other_optical = compute_policy_preference({**base_event, "satellite": "Sentinel-2-40697"})

        self.assertGreater(spaceeye, planetscope)
        self.assertGreater(planetscope, other_optical)

    def test_build_city_execution_plan_orders_by_city_earliest_time(self) -> None:
        events = [
            {
                "city": "Tehran",
                "satellite": "ICEYE-X",
                "sensor_type": "sar",
                "pass_time_utc": "2026-03-03T03:00:00Z",
                "action_priority_label": "즉시 촬영",
                "llm_event_summary": "Tehran summary",
                "source_urls": ["https://a", "https://b"],
            },
            {
                "city": "Beirut",
                "satellite": "SpaceEye-T",
                "sensor_type": "optical",
                "pass_time_utc": "2026-03-02T01:00:00Z",
                "action_priority_label": "우선 촬영",
                "llm_event_summary": "Beirut summary",
                "source_urls": ["https://c", "https://d"],
            },
            {
                "city": "Tehran",
                "satellite": "SpaceEye-T",
                "sensor_type": "optical",
                "pass_time_utc": "2026-03-02T05:00:00Z",
                "action_priority_label": "우선 촬영",
                "llm_event_summary": "Tehran summary",
                "source_urls": ["https://a", "https://b"],
            },
        ]

        plan = build_city_execution_plan(events)

        self.assertEqual([item["city"] for item in plan], ["Beirut", "Tehran"])
        self.assertEqual(plan[1]["timeline"][0]["satellite"], "SpaceEye-T")
        self.assertEqual(plan[1]["timeline"][0]["source_urls"], ["https://a", "https://b"])

    def test_print_schedule_shows_city_plan_llm_and_urls(self) -> None:
        schedule = {
            "generated_utc": "2026-03-01T00:00:00Z",
            "mode": "backtest",
            "prediction_start_utc": "2026-03-01T00:00:00Z",
            "prediction_end_utc": "2026-03-08T00:00:00Z",
            "tle_reference_date": "20260301",
            "tle_source": "historical-cache",
            "satellites_tracked": 2,
            "cities_monitored": 1,
            "total_passes": 10,
            "swath_passes": 3,
            "shootable_passes": 2,
            "scheduled_cities": 1,
            "sensor_condition_summary": {
                "optical_total": 1,
                "optical_shootable": 1,
                "sar_total": 1,
                "sar_shootable": 1,
            },
            "satellite_allocation": [
                {"satellite": "SpaceEye-T", "sensor_type": "optical", "scheduled_count": 1, "cities": ["Tehran"]},
            ],
            "city_execution_plan": [
                {
                    "city": "Tehran",
                    "scheduled_count": 2,
                    "timeline": [
                        {
                            "execution_order": 1,
                            "city": "Tehran",
                            "satellite": "SpaceEye-T",
                            "sensor_type": "optical",
                            "action_priority_label": "우선 촬영",
                            "pass_time_utc": "2026-03-02T06:00:00Z",
                            "recommendation_reason": "EO 촬영 좋음",
                            "llm_event_summary": "시설 타격 확인",
                            "source_urls": ["https://news/1", "https://news/2"],
                        }
                    ],
                }
            ],
            "satellite_execution_plan": [],
        }

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_schedule(schedule)

        output = buffer.getvalue()
        self.assertIn("도시별 실행 계획", output)
        self.assertIn("위성 할당", output)
        self.assertIn("시설 타격 확인", output)
        self.assertIn("https://news/1", output)
        self.assertIn("https://news/2", output)
        self.assertNotIn("위성별 실행 계획", output)


if __name__ == "__main__":
    unittest.main()

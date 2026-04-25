from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.integrated_pipeline import (
    build_level1_output,
    run_integrated_pipeline,
    resolve_level2a_mode,
    resolve_main_data_path,
    resolve_url_data_path,
)


class IntegratedPipelineTest(unittest.TestCase):
    def test_resolve_data_paths_prefers_existing_repo_files(self) -> None:
        self.assertTrue(resolve_main_data_path().exists())
        self.assertTrue(resolve_url_data_path().exists())

    def test_build_level1_output_filters_nonpassing_llm_rows(self) -> None:
        anomalies = pd.DataFrame(
            [
                {
                    "date": "20260317",
                    "city": "Tehran",
                    "risk_level": 3,
                    "risk_label": "위기",
                    "conflict_index": 10.0,
                    "innov_z": 5.2,
                    "events": 8,
                    "lat": 35.6892,
                    "lng": 51.3890,
                    "country_code": "IR",
                    "risk_guide": "즉시 대응",
                    "llm_status": "SUCCESS",
                    "llm_report": '{"Summary": "도시 내 직접 타격"}',
                    "source_urls": [
                        "https://example.com/1",
                        "https://example.com/2",
                        "https://example.com/3",
                    ],
                    "is_anomaly": True,
                },
                {
                    "date": "20260317",
                    "city": "Dubai",
                    "risk_level": 2,
                    "risk_label": "위험",
                    "conflict_index": 7.0,
                    "innov_z": 3.4,
                    "events": 4,
                    "lat": 25.2048,
                    "lng": 55.2708,
                    "country_code": "AE",
                    "risk_guide": "정밀 분석",
                    "llm_status": "DROPPED",
                    "llm_report": '{"Summary": "데이터라인 오탐"}',
                    "is_anomaly": True,
                },
            ]
        )

        output = build_level1_output(anomalies, "20260317", use_llm=True, top_k=20)

        self.assertEqual(output["alert_count"], 1)
        alert = output["alerts"][0]
        self.assertEqual(alert["city"], "Tehran")
        self.assertEqual(alert["risk_label"], "RED")
        self.assertEqual(alert["lon"], 51.389)
        self.assertEqual(alert["llm_status"], "SUCCESS")
        self.assertEqual(alert["source_urls"], ["https://example.com/1", "https://example.com/2"])

    def test_build_level1_output_without_llm_keeps_top_anomalies(self) -> None:
        anomalies = pd.DataFrame(
            [
                {
                    "date": "20260317",
                    "city": "Baghdad",
                    "risk_level": 2,
                    "risk_label": "위험",
                    "conflict_index": 8.0,
                    "innov_z": 4.1,
                    "events": 5,
                    "lat": 33.3152,
                    "lng": 44.3661,
                    "is_anomaly": True,
                },
                {
                    "date": "20260317",
                    "city": "Beirut",
                    "risk_level": 1,
                    "risk_label": "주의",
                    "conflict_index": 6.0,
                    "innov_z": 2.2,
                    "events": 3,
                    "lat": 33.8938,
                    "lng": 35.5018,
                    "is_anomaly": True,
                },
            ]
        )

        output = build_level1_output(anomalies, "20260317", use_llm=False, top_k=1)

        self.assertEqual(output["alert_count"], 1)
        self.assertEqual(output["alerts"][0]["city"], "Baghdad")
        self.assertEqual(output["alerts"][0]["risk_label"], "ORANGE")

    def test_resolve_level2a_mode_auto_uses_backtest_for_past_date(self) -> None:
        self.assertEqual(resolve_level2a_mode("20200101", mode="auto"), "backtest")

    @patch("pipeline.integrated_pipeline._load_level2a_modules")
    @patch("pipeline.integrated_pipeline._load_level1_modules")
    @patch("pipeline.integrated_pipeline.load_gdelt_datasets")
    def test_run_integrated_pipeline_suppresses_internal_logs_by_default(
        self,
        load_gdelt_datasets_mock,
        load_level1_modules_mock,
        load_level2a_modules_mock,
    ) -> None:
        raw_df = pd.DataFrame({"SQLDATE": ["20260317"]})
        load_gdelt_datasets_mock.return_value = (raw_df, None, resolve_main_data_path(), None)

        def noisy_compute_conflict_index(*args, **kwargs):
            print("NOISY_L1_INTERNAL")
            anomalies = pd.DataFrame(
                [
                    {
                        "date": "20260317",
                        "city": "Tehran",
                        "risk_level": 3,
                        "risk_label": "위기",
                        "conflict_index": 10.0,
                        "innov_z": 5.2,
                        "events": 8,
                        "lat": 35.6892,
                        "lng": 51.3890,
                        "is_anomaly": True,
                    }
                ]
            )
            return anomalies, pd.DataFrame()

        def fake_detect_anomalies(city_daily, target_date):
            return city_daily.copy()

        def noisy_build_schedule_from_level1_result(**kwargs):
            print("NOISY_L2_INTERNAL")
            return {
                "generated_utc": "2026-03-17T00:00:00Z",
                "mode": "backtest",
                "prediction_start_utc": "2026-03-17T00:00:00Z",
                "prediction_end_utc": "2026-03-24T00:00:00Z",
                "tle_reference_date": "20260317",
                "tle_source": "historical-cache",
                "satellites_tracked": 1,
                "cities_monitored": 1,
                "total_passes": 1,
                "swath_passes": 1,
                "shootable_passes": 1,
                "scheduled_cities": 1,
                "city_best_recommendations": [],
                "sensor_condition_summary": {
                    "optical_total": 1,
                    "optical_shootable": 1,
                    "sar_total": 0,
                    "sar_shootable": 0,
                },
                "city_execution_plan": [
                    {
                        "city": "Tehran",
                        "scheduled_count": 1,
                        "timeline": [
                            {
                                "city": "Tehran",
                                "satellite": "SpaceEye-T",
                                "sensor_type": "optical",
                                "action_priority_label": "우선 촬영",
                                "pass_time_utc": "2026-03-18T06:00:00Z",
                            }
                        ],
                    }
                ],
                "satellite_allocation": [
                    {"satellite": "SpaceEye-T", "sensor_type": "optical", "scheduled_count": 1, "cities": ["Tehran"]}
                ],
                "satellite_execution_plan": [],
            }

        preprocess_stub = type("PreprocessStub", (), {"VERBOSE_LOGS": False})()
        load_level1_modules_mock.return_value = (noisy_compute_conflict_index, fake_detect_anomalies, preprocess_stub)
        load_level2a_modules_mock.return_value = (
            noisy_build_schedule_from_level1_result,
            lambda schedule, verbose=False: print("CLEAN_REPORT"),
            lambda schedule, filename=None: "schedule.json",
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_integrated_pipeline(
                target_date="20260317",
                mode="backtest",
                save_level1=False,
                save_schedule_output=False,
            )

        output = buffer.getvalue()
        self.assertNotIn("NOISY_L1_INTERNAL", output)
        self.assertNotIn("NOISY_L2_INTERNAL", output)
        self.assertIn("CLEAN_REPORT", output)


if __name__ == "__main__":
    unittest.main()

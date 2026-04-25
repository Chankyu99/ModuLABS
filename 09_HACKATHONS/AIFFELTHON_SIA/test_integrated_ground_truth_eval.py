from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.archive.city_utils import normalize_city_key

from run_integrated_ground_truth_eval import (
    annotate_ground_truth_detectability,
    extract_predicted_cities,
    extract_level1_alert_cities,
    evaluate_single_date,
    evaluate_windowed_metrics,
    match_predictions_to_ground_truth,
)


class IntegratedGroundTruthEvalTest(unittest.TestCase):
    def test_extract_level1_alert_cities_deduplicates_by_normalized_city(self) -> None:
        level1_output = {
            "alerts": [
                {"city": "Esfahan, Iran", "lat": 32.6546, "lon": 51.6680, "llm_status": "SUCCESS"},
                {"city": "Isfahan", "lat": 32.6546, "lon": 51.6680, "llm_status": "SUCCESS"},
                {"city": "Tel Aviv", "lat": 32.0853, "lon": 34.7818, "llm_status": "AMBIGUOUS"},
            ]
        }

        extracted = extract_level1_alert_cities(level1_output)

        self.assertEqual([item["city"] for item in extracted], ["Esfahan", "Tel Aviv"])
        self.assertEqual(extracted[0]["normalized_city"], normalize_city_key("Isfahan"))

    def test_annotate_ground_truth_detectability_uses_mentions_threshold(self) -> None:
        ground_truth_rows = [
            {
                "date": "20260301",
                "city": "Esfahan",
                "normalized_city": normalize_city_key("Esfahan"),
                "lat": 32.6546,
                "lon": 51.6680,
            },
            {
                "date": "20260301",
                "city": "Dubai",
                "normalized_city": normalize_city_key("Dubai"),
                "lat": 25.2048,
                "lon": 55.2708,
            },
        ]
        gdelt_df = pd.DataFrame(
            [
                {
                    "SQLDATE": "20260301",
                    "ActionGeo_FullName": "Isfahan, Iran",
                    "ActionGeo_Lat": 32.6546,
                    "ActionGeo_Long": 51.6680,
                    "NumMentions": 60,
                },
                {
                    "SQLDATE": "20260301",
                    "ActionGeo_FullName": "Isfahan",
                    "ActionGeo_Lat": 32.6546,
                    "ActionGeo_Long": 51.6680,
                    "NumMentions": 50,
                },
                {
                    "SQLDATE": "20260301",
                    "ActionGeo_FullName": "Dubai",
                    "ActionGeo_Lat": 25.2048,
                    "ActionGeo_Long": 55.2708,
                    "NumMentions": 99,
                },
            ]
        )

        annotated = annotate_ground_truth_detectability(ground_truth_rows, gdelt_df, mentions_threshold=100)
        by_city = {row["city"]: row for row in annotated}

        self.assertEqual(by_city["Esfahan"]["mentions_on_date"], 110)
        self.assertEqual(by_city["Esfahan"]["tier"], "A")
        self.assertEqual(by_city["Dubai"]["mentions_on_date"], 99)
        self.assertEqual(by_city["Dubai"]["tier"], "C")

    def test_evaluate_windowed_metrics_respects_plus_seven_day_window_and_tiers(self) -> None:
        daily_results = [
            {
                "date": "20260301",
                "predictions": [
                    {
                        "city": "Gamma",
                        "normalized_city": normalize_city_key("Gamma"),
                        "lat": 20.0,
                        "lon": 20.0,
                    }
                ],
            },
            {
                "date": "20260308",
                "predictions": [
                    {
                        "city": "Alpha",
                        "normalized_city": normalize_city_key("Alpha"),
                        "lat": 0.0,
                        "lon": 0.0,
                    }
                ],
            },
        ]
        ground_truth_rows = [
            {
                "date": "20260301",
                "city": "Alpha",
                "normalized_city": normalize_city_key("Alpha"),
                "lat": 0.0,
                "lon": 0.0,
                "tier": "A",
                "mentions_on_date": 120,
            },
            {
                "date": "20260301",
                "city": "Beta",
                "normalized_city": normalize_city_key("Beta"),
                "lat": 10.0,
                "lon": 10.0,
                "tier": "C",
                "mentions_on_date": 40,
            },
        ]

        metrics = evaluate_windowed_metrics(daily_results, ground_truth_rows, window_days=7, radius_km=50.0)

        self.assertEqual(metrics["gt_count_total"], 2)
        self.assertEqual(metrics["gt_count_a"], 1)
        self.assertEqual(metrics["gt_count_c"], 1)
        self.assertEqual(metrics["prediction_count_total"], 2)
        self.assertEqual(metrics["tp_count"], 1)
        self.assertEqual(metrics["fp_count"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall_total"], 0.5)
        self.assertAlmostEqual(metrics["recall_a"], 1.0)
        self.assertAlmostEqual(metrics["recall_c"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    @patch("run_integrated_ground_truth_eval.load_ground_truth_rows")
    @patch("run_integrated_ground_truth_eval.run_integrated_pipeline")
    def test_evaluate_single_date_passes_use_llm_flag(
        self,
        run_integrated_pipeline_mock,
        load_ground_truth_rows_mock,
    ) -> None:
        run_integrated_pipeline_mock.return_value = {
            "mode": "backtest",
            "level1_output": {
                "alerts": [{"city": "Tehran", "lat": 35.6892, "lon": 51.3890, "llm_status": "SUCCESS"}]
            },
            "schedule": {
                "city_best_recommendations": [
                    {"city": "Tehran", "lat": 35.6892, "lon": 51.3890, "satellite": "ICEYE"}
                ],
                "sensor_condition_summary": {
                    "optical_total": 1,
                    "optical_shootable": 0,
                    "optical_blocked_unknown": 1,
                    "sar_total": 1,
                    "sar_shootable": 1,
                },
                "total_passes": 10,
                "swath_passes": 3,
                "shootable_passes": 1,
                "scheduled_cities": 1,
                "satellites_used": 1,
            },
        }
        load_ground_truth_rows_mock.return_value = [
            {
                "city": "Tehran",
                "normalized_city": normalize_city_key("Tehran"),
                "lat": 35.6892,
                "lon": 51.3890,
                "source": "GT",
                "event_description": "hit",
            }
        ]

        result = evaluate_single_date(date="20260317", scenario="tri-mix", use_llm=True)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["llm_candidate_count"], 1)
        self.assertEqual(result["llm_confirmed_count"], 1)
        self.assertEqual(result["llm_rejected_count"], 0)
        self.assertEqual(result["level1_predictions"][0]["city"], "Tehran")
        self.assertTrue(result["used_llm"])
        self.assertEqual(run_integrated_pipeline_mock.call_args.kwargs["use_llm"], True)

    def test_extract_predicted_cities_deduplicates_by_normalized_city(self) -> None:
        schedule = {
            "city_best_recommendations": [
                {"city": "Esfahan", "lat": 32.6546, "lon": 51.6680, "satellite": "A"},
                {"city": "Isfahan", "lat": 32.6546, "lon": 51.6680, "satellite": "B"},
                {"city": "Tel Aviv", "lat": 32.0853, "lon": 34.7818, "satellite": "C"},
            ]
        }

        extracted = extract_predicted_cities(schedule)

        self.assertEqual([item["city"] for item in extracted], ["Esfahan", "Tel Aviv"])
        self.assertEqual(extracted[0]["normalized_city"], normalize_city_key("Isfahan"))

    def test_match_predictions_to_ground_truth_uses_name_then_geo(self) -> None:
        predictions = [
            {"city": "Esfahan", "normalized_city": normalize_city_key("Esfahan"), "lat": 32.6546, "lon": 51.6680},
            {"city": "Ramat Gan", "normalized_city": normalize_city_key("Ramat Gan"), "lat": 32.0823, "lon": 34.8107},
            {"city": "Dubai", "normalized_city": normalize_city_key("Dubai"), "lat": 25.2048, "lon": 55.2708},
        ]
        ground_truth = [
            {"city": "Isfahan", "normalized_city": normalize_city_key("Isfahan"), "lat": 32.6546, "lon": 51.6680},
            {"city": "Tel Aviv", "normalized_city": normalize_city_key("Tel Aviv"), "lat": 32.0853, "lon": 34.7818},
        ]

        matched, unmatched_predictions, unmatched_ground_truth = match_predictions_to_ground_truth(
            predictions,
            ground_truth,
            radius_km=50.0,
        )

        self.assertEqual(len(matched), 2)
        self.assertEqual(matched[0]["match_type"], "name")
        self.assertEqual(matched[1]["match_type"], "geo")
        self.assertEqual(unmatched_predictions[0]["city"], "Dubai")
        self.assertEqual(unmatched_ground_truth, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.live_data_preparer import PreparedLiveInputs, compute_fetch_dates


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_INTEGRATED_PIPELINE_PATH = PROJECT_ROOT / "run_integrated_pipeline.py"


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveDataPreparerTest(unittest.TestCase):
    def test_compute_fetch_dates_returns_missing_dates_after_existing_max(self) -> None:
        self.assertEqual(
            compute_fetch_dates("20260331", "20260403"),
            ["20260401", "20260402", "20260403"],
        )

    def test_compute_fetch_dates_returns_empty_when_target_not_newer(self) -> None:
        self.assertEqual(compute_fetch_dates("20260421", "20260421"), [])
        self.assertEqual(compute_fetch_dates("20260421", "20260420"), [])

    def test_integrated_cli_fetch_flag_prepares_live_inputs_and_forces_operational(self) -> None:
        captured: dict[str, object] = {}

        def fake_prepare_live_prediction_inputs(**kwargs):
            return PreparedLiveInputs(
                main_path=Path("/tmp/main.parquet"),
                url_path=Path("/tmp/url.parquet"),
                fetched_dates=["20260401", "20260402"],
                base_max_date="20260331",
            )

        def fake_run_integrated_pipeline(**kwargs):
            captured.update(kwargs)
            return {"schedule": {}}

        module = load_module_from_path("test_run_integrated_pipeline_live", RUN_INTEGRATED_PIPELINE_PATH)

        with patch.object(module, "prepare_live_prediction_inputs", side_effect=fake_prepare_live_prediction_inputs):
            with patch.object(module, "run_integrated_pipeline", side_effect=fake_run_integrated_pipeline):
                with patch.object(
                    sys,
                    "argv",
                    ["run_integrated_pipeline.py", "--date", "20260421", "--fetch-gdelt"],
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        module.main()

        self.assertEqual(captured["target_date"], "20260421")
        self.assertEqual(captured["main_path"], "/tmp/main.parquet")
        self.assertEqual(captured["url_path"], "/tmp/url.parquet")
        self.assertEqual(captured["mode"], "operational")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.config import PREDICTION_HOURS


PROJECT_ROOT = Path(__file__).resolve().parent
LEVEL2A_PATH = PROJECT_ROOT / "pipeline" / "Level2a" / "level2a.py"
RUN_LEVEL2A_REAL_PATH = PROJECT_ROOT / "run_level2a_real.py"


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Level2aPredictionWindowTest(unittest.TestCase):
    def test_prediction_hours_defaults_to_seven_days(self) -> None:
        self.assertEqual(PREDICTION_HOURS, 24 * 7)

    def test_level2a_entry_defaults_follow_prediction_hours(self) -> None:
        schedule_builder_stub = types.ModuleType("pipeline.schedule_builder")
        schedule_builder_stub.build_schedule = lambda **kwargs: kwargs
        schedule_builder_stub.print_schedule = lambda schedule: None
        schedule_builder_stub.save_schedule = lambda schedule, filename=None: filename

        with patch.dict(sys.modules, {"pipeline.schedule_builder": schedule_builder_stub}):
            module = load_module_from_path("test_level2a_module", LEVEL2A_PATH)

        self.assertEqual(module.build_schedule_from_level1_result.__defaults__[0], PREDICTION_HOURS)
        self.assertEqual(module.run_level2a_for_date.__defaults__[0], PREDICTION_HOURS)

    def test_run_level2a_real_cli_uses_prediction_hours_default(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_level2a_for_date(**kwargs):
            captured.update(kwargs)
            return {"error": "missing", "missing_path": "/tmp/fake.json"}

        level2a_stub = types.ModuleType("pipeline.level2a")
        level2a_stub.run_level2a_for_date = fake_run_level2a_for_date

        with patch.dict(sys.modules, {"pipeline.level2a": level2a_stub}):
            module = load_module_from_path("test_run_level2a_real", RUN_LEVEL2A_REAL_PATH)

        with patch.object(sys, "argv", ["run_level2a_real.py"]):
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()

        self.assertEqual(captured["hours"], PREDICTION_HOURS)


if __name__ == "__main__":
    unittest.main()

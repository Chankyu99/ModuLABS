from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / ".vendor"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "Code" / ".mplconfig"))

from presentation_refactor.config import OUTPUT_DIR  # noqa: E402
from presentation_refactor.data import (  # noqa: E402
    build_delivery_frame,
    build_state_coordinates,
    build_state_scoring_frame,
    load_tables,
)
from presentation_refactor.folium_appendix import (  # noqa: E402
    build_appendix_rj_market_entry_map,
)
from presentation_refactor.metrics import build_state_delay_risk_summary  # noqa: E402


def main() -> None:
    appendix_dir = OUTPUT_DIR / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(PROJECT_ROOT / "Data")
    state_scores = build_state_scoring_frame(tables)
    delay_summary = build_state_delay_risk_summary(build_delivery_frame(tables))
    coordinates = build_state_coordinates(tables)

    output_path = appendix_dir / "appendix_rj_market_entry_map.html"
    build_appendix_rj_market_entry_map(
        state_scores=state_scores,
        delay_summary=delay_summary,
        coordinates=coordinates,
        output_path=output_path,
    )

    print(f"Appendix folium map generated: {output_path}")


if __name__ == "__main__":
    main()

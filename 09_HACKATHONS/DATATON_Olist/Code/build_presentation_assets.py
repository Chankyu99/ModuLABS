from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "Code" / ".mplconfig"))

from presentation_refactor.charts import (  # noqa: E402
    save_category_portfolio_chart,
    save_delivery_impact_chart,
    save_recommendation_card,
    save_state_delay_risk_chart,
    save_state_scoring_chart,
    save_state_strategy_map,
)
from presentation_refactor.config import FIGURE_DIR, OUTPUT_DIR, TABLE_DIR  # noqa: E402
from presentation_refactor.data import (  # noqa: E402
    build_category_frame,
    build_delivery_frame,
    build_state_coordinates,
    build_state_scoring_frame,
    load_tables,
)
from presentation_refactor.metrics import (  # noqa: E402
    build_delivery_impact_summary,
    build_recommendation_summary,
    build_state_delay_risk_summary,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Code" / ".mplconfig").mkdir(exist_ok=True)

    tables = load_tables(PROJECT_ROOT / "Data")

    delivery = build_delivery_frame(tables)
    state_scores = build_state_scoring_frame(tables)
    category_summary = build_category_frame(tables)
    rj_category_summary = build_category_frame(tables, state="RJ")
    coordinates = build_state_coordinates(tables)

    delivery_impact = build_delivery_impact_summary(delivery)
    state_delay_summary = build_state_delay_risk_summary(delivery)
    recommendation = build_recommendation_summary(
        state_scores=state_scores,
        category_summary=category_summary,
        rj_category_summary=rj_category_summary,
        state_delay_summary=state_delay_summary,
    )

    delivery_impact.to_csv(TABLE_DIR / "01_delivery_impact_summary.csv", index=False)
    state_delay_summary.to_csv(TABLE_DIR / "02_state_delay_risk_summary.csv", index=False)
    state_scores.to_csv(TABLE_DIR / "03_state_opportunity_scores.csv", index=False)
    category_summary.to_csv(TABLE_DIR / "04_category_portfolio_summary.csv", index=False)
    recommendation.to_csv(TABLE_DIR / "05_final_recommendation_summary.csv", index=False)

    save_delivery_impact_chart(
        delivery_impact, FIGURE_DIR / "01_problem_and_cause_validation.png"
    )
    save_state_delay_risk_chart(
        state_delay_summary, FIGURE_DIR / "02_state_delay_risk.png"
    )
    save_state_scoring_chart(
        state_scores, FIGURE_DIR / "03_market_entry_state_scoring.png"
    )
    save_category_portfolio_chart(
        category_summary, FIGURE_DIR / "04_launch_category_portfolio.png"
    )
    save_recommendation_card(
        recommendation, FIGURE_DIR / "05_final_recommendation.png"
    )
    save_state_strategy_map(
        state_scores=state_scores,
        delay_summary=state_delay_summary,
        coordinates=coordinates,
        output_path=FIGURE_DIR / "03_state_strategy_map.png",
    )

    print("Presentation assets generated successfully.")
    for path in sorted(FIGURE_DIR.iterdir()):
        print(f"FIGURE: {path}")
    for path in sorted(TABLE_DIR.iterdir()):
        print(f"TABLE: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "Code" / ".mplconfig"))

from src_code.charts import (  # noqa: E402
    save_preprocessing_filter_flow,
    save_zipcode_coordinate_concept,
)
from src_code.data import load_tables  # noqa: E402


FIGURE_DIR = PROJECT_ROOT / "Outputs" / "figures"


def build_filter_summary(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    orders = tables["orders"][["order_id", "order_delivered_customer_date", "order_estimated_delivery_date"]]
    reviews = tables["reviews"][["order_id", "review_score"]]

    merged = orders.merge(reviews, on="order_id", how="left")
    total_orders = int(merged["order_id"].nunique())

    has_delivered = int(merged[merged["order_delivered_customer_date"].notna()]["order_id"].nunique())
    has_estimated = int(
        merged[
            merged["order_delivered_customer_date"].notna()
            & merged["order_estimated_delivery_date"].notna()
        ]["order_id"].nunique()
    )
    has_review = int(
        merged[
            merged["order_delivered_customer_date"].notna()
            & merged["order_estimated_delivery_date"].notna()
            & merged["review_score"].notna()
        ]["order_id"].nunique()
    )

    return {
        "total_orders": total_orders,
        "has_delivered_date": has_delivered,
        "has_estimated_date": has_estimated,
        "has_review_score": has_review,
        "usable_orders": has_review,
    }


def build_zipcode_example(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    geo = tables["geolocation"][
        ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
    ].copy()

    counts = (
        geo.groupby("geolocation_zip_code_prefix")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    zipcode = str(counts.iloc[0]["geolocation_zip_code_prefix"])
    raw_points = geo[geo["geolocation_zip_code_prefix"] == counts.iloc[0]["geolocation_zip_code_prefix"]].copy()

    if len(raw_points) > 40:
        raw_points = raw_points.sample(40, random_state=42)

    return raw_points, zipcode


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Code" / ".mplconfig").mkdir(parents=True, exist_ok=True)

    tables = load_tables(PROJECT_ROOT / "Data")

    filter_summary = build_filter_summary(tables)
    save_preprocessing_filter_flow(
        filter_summary,
        FIGURE_DIR / "00_preprocessing_filter_flow.png",
    )

    raw_points, zipcode = build_zipcode_example(tables)
    save_zipcode_coordinate_concept(
        raw_points,
        zipcode,
        FIGURE_DIR / "00_preprocessing_zipcode_coordinate_concept.png",
    )

    print("Preprocessing assets generated successfully.")
    print(FIGURE_DIR / "00_preprocessing_filter_flow.png")
    print(FIGURE_DIR / "00_preprocessing_zipcode_coordinate_concept.png")


if __name__ == "__main__":
    main()

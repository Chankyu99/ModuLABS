from __future__ import annotations

from typing import Dict

import pandas as pd


def load_tables(data_dir) -> Dict[str, pd.DataFrame]:
    return {
        "orders": pd.read_csv(data_dir / "olist_orders_dataset.csv"),
        "items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv"),
        "reviews": pd.read_csv(data_dir / "olist_order_reviews_dataset.csv"),
        "products": pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "sellers": pd.read_csv(data_dir / "olist_sellers_dataset.csv"),
        "geolocation": pd.read_csv(data_dir / "olist_geolocation_dataset.csv"),
        "category_translation": pd.read_csv(
            data_dir / "product_category_name_translation.csv"
        ),
    }


def build_delivery_frame(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"].copy()
    customers = tables["customers"][["customer_id", "customer_unique_id", "customer_state"]]
    reviews = tables["reviews"][["order_id", "review_score"]]

    delivery = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .copy()
    )

    delivery["order_purchase_timestamp"] = pd.to_datetime(
        delivery["order_purchase_timestamp"]
    )
    delivery["order_delivered_customer_date"] = pd.to_datetime(
        delivery["order_delivered_customer_date"]
    )
    delivery["order_estimated_delivery_date"] = pd.to_datetime(
        delivery["order_estimated_delivery_date"]
    )

    delivery = delivery[
        delivery["order_delivered_customer_date"].notna()
        & delivery["order_estimated_delivery_date"].notna()
        & delivery["review_score"].notna()
    ].copy()

    delivery["is_delayed"] = (
        delivery["order_delivered_customer_date"]
        > delivery["order_estimated_delivery_date"]
    )
    delivery["is_low_review"] = delivery["review_score"] <= 2
    delivery["delay_days"] = (
        delivery["order_delivered_customer_date"]
        - delivery["order_estimated_delivery_date"]
    ).dt.days

    return delivery


def build_state_scoring_frame(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"].copy()
    items = tables["items"].copy()
    customers = tables["customers"][["customer_id", "customer_state"]]
    reviews = tables["reviews"][["order_id", "review_score"]]

    scored = (
        orders.merge(items, on="order_id", how="inner")
        .merge(customers, on="customer_id", how="left")
        .merge(reviews, on="order_id", how="inner")
        .copy()
    )

    scored["order_delivered_customer_date"] = pd.to_datetime(
        scored["order_delivered_customer_date"]
    )
    scored["order_estimated_delivery_date"] = pd.to_datetime(
        scored["order_estimated_delivery_date"]
    )

    scored = scored[
        scored["order_delivered_customer_date"].notna()
        & scored["order_estimated_delivery_date"].notna()
        & (scored["price"] > 0)
    ].copy()

    scored["freight_ratio"] = scored["freight_value"] / scored["price"]
    scored["arrival_diff"] = (
        scored["order_estimated_delivery_date"] - scored["order_delivered_customer_date"]
    ).dt.days

    state_scores = (
        scored.groupby("customer_state")
        .agg(
            order_count=("order_id", "count"),
            freight_ratio=("freight_ratio", "mean"),
            arrival_diff=("arrival_diff", "mean"),
            review_score=("review_score", "mean"),
        )
        .reset_index()
        .rename(columns={"customer_state": "state"})
    )

    state_scores["norm_demand"] = _min_max_scale(state_scores["order_count"])
    state_scores["norm_satisfaction"] = _min_max_scale(state_scores["review_score"])
    state_scores["opportunity_score"] = state_scores["norm_demand"] * (
        1 - state_scores["norm_satisfaction"]
    )
    return state_scores.sort_values("opportunity_score", ascending=False).reset_index(
        drop=True
    )


def build_category_frame(tables: Dict[str, pd.DataFrame], state: str | None = None) -> pd.DataFrame:
    orders = tables["orders"][["order_id", "customer_id"]]
    customers = tables["customers"][["customer_id", "customer_state"]]
    items = tables["items"][["order_id", "product_id", "price", "freight_value"]]
    products = tables["products"][["product_id", "product_category_name"]]
    translation = tables["category_translation"]
    reviews = tables["reviews"][["order_id", "review_score"]]

    category_df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
        .merge(reviews, on="order_id", how="left")
        .copy()
    )

    if state is not None:
        category_df = category_df[category_df["customer_state"] == state].copy()

    summary = (
        category_df.groupby("product_category_name_english")
        .agg(
            order_count=("order_id", "nunique"),
            revenue=("price", "sum"),
            avg_review=("review_score", "mean"),
            avg_freight=("freight_value", "mean"),
        )
        .reset_index()
        .dropna(subset=["product_category_name_english", "avg_review"])
        .rename(columns={"product_category_name_english": "category"})
    )

    total_orders = summary["order_count"].sum()
    total_revenue = summary["revenue"].sum()
    summary["order_share"] = summary["order_count"] / total_orders
    summary["revenue_share"] = summary["revenue"] / total_revenue

    return summary.sort_values("revenue", ascending=False).reset_index(drop=True)


def build_state_coordinates(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = tables["customers"][["customer_zip_code_prefix", "customer_state"]]
    geolocation = tables["geolocation"][
        ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
    ]

    geo_avg = (
        geolocation.groupby("geolocation_zip_code_prefix")[
            ["geolocation_lat", "geolocation_lng"]
        ]
        .mean()
        .reset_index()
    )

    coords = (
        customers.merge(
            geo_avg,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )
        .groupby("customer_state")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"customer_state": "state", "geolocation_lng": "geolocation_lon"})
    )

    return coords


def _min_max_scale(series: pd.Series) -> pd.Series:
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.min()) / denominator


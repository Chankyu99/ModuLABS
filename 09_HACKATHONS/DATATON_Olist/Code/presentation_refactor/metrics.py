from __future__ import annotations

import pandas as pd

from .config import MIN_DELAYED_ORDERS, MIN_STATE_ORDER_COUNT


def build_delivery_impact_summary(delivery: pd.DataFrame) -> pd.DataFrame:
    summary = (
        delivery.groupby("is_delayed")
        .agg(
            orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
            low_review_rate=("is_low_review", "mean"),
        )
        .reset_index()
    )
    summary["delivery_status"] = summary["is_delayed"].map(
        {False: "정시 배송", True: "지연 배송"}
    )
    summary["low_review_rate_pct"] = summary["low_review_rate"] * 100
    return summary[["delivery_status", "orders", "avg_review", "low_review_rate_pct"]]


def build_state_delay_risk_summary(delivery: pd.DataFrame) -> pd.DataFrame:
    order_counts = (
        delivery.groupby("customer_state")
        .agg(total_orders=("order_id", "nunique"))
        .reset_index()
    )

    delayed_only = delivery[delivery["is_delayed"]].copy()
    delayed_summary = (
        delayed_only.groupby("customer_state")
        .agg(
            delayed_orders=("order_id", "nunique"),
            delayed_avg_review=("review_score", "mean"),
            delayed_low_review_rate=("is_low_review", "mean"),
            avg_delay_days=("delay_days", "mean"),
        )
        .reset_index()
    )

    summary = order_counts.merge(delayed_summary, on="customer_state", how="left").fillna(0)
    summary = summary[
        (summary["total_orders"] >= MIN_STATE_ORDER_COUNT)
        & (summary["delayed_orders"] >= MIN_DELAYED_ORDERS)
    ].copy()

    summary["delayed_low_review_rate_pct"] = summary["delayed_low_review_rate"] * 100
    summary = summary.rename(columns={"customer_state": "state"})
    return summary.sort_values(
        ["delayed_low_review_rate_pct", "delayed_orders"], ascending=[False, False]
    ).reset_index(drop=True)


def build_recommendation_summary(
    state_scores: pd.DataFrame,
    category_summary: pd.DataFrame,
    rj_category_summary: pd.DataFrame,
    state_delay_summary: pd.DataFrame,
) -> pd.DataFrame:
    top_state = state_scores.iloc[0]
    top_category = category_summary[category_summary["category"] == "health_beauty"].iloc[0]
    rj_health = rj_category_summary[rj_category_summary["category"] == "health_beauty"].iloc[0]
    rj_delay = state_delay_summary[state_delay_summary["state"] == "RJ"].iloc[0]

    return pd.DataFrame(
        [
            {
                "recommendation": "1차 진입 지역",
                "value": "RJ",
                "evidence": (
                    f"기회점수 {top_state['opportunity_score']:.3f}, "
                    f"아이템 단위 수요 {int(top_state['order_count']):,}, "
                    f"평균 리뷰 {top_state['review_score']:.2f}"
                ),
            },
            {
                "recommendation": "런칭 카테고리",
                "value": "Health & Beauty",
                "evidence": (
                    f"브라질 전체 매출 순위 1위 ({top_category['revenue']:,.0f}), "
                    f"주문 순위 2위 ({int(top_category['order_count']):,})"
                ),
            },
            {
                "recommendation": "RJ 내 적합 카테고리",
                "value": "Health & Beauty",
                "evidence": (
                    f"RJ 매출 {rj_health['revenue']:,.0f}, "
                    f"평균 리뷰 {rj_health['avg_review']:.2f}"
                ),
            },
            {
                "recommendation": "운영 가드레일",
                "value": "배송 약속 관리",
                "evidence": (
                    f"RJ 지연 주문 저평점 비율 {rj_delay['delayed_low_review_rate_pct']:.1f}%, "
                    f"지연 주문 평균 리뷰 {rj_delay['delayed_avg_review']:.2f}"
                ),
            },
        ]
    )

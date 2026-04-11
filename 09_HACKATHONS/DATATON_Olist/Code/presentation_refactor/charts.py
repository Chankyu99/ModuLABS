from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import math
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import ACCENT, BACKGROUND, DANGER, LIGHT, MUTED, PRIMARY, SECONDARY, SUCCESS


def set_chart_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["AppleGothic", "Arial Unicode MS", "sans-serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": "white",
            "axes.edgecolor": LIGHT,
            "axes.labelcolor": SECONDARY,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "grid.color": LIGHT,
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "font.size": 11,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
        }
    )


def save_delivery_impact_chart(summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    status_order = ["정시 배송", "지연 배송"]
    ordered = summary.set_index("delivery_status").loc[status_order].reset_index()
    colors = [SUCCESS, DANGER]

    axes[0].bar(ordered["delivery_status"], ordered["avg_review"], color=colors)
    axes[0].set_title("배송이 지연되면 리뷰 점수에 영향을 미침")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("평균 리뷰 점수")
    axes[0].set_ylim(0, 5)
    for idx, row in ordered.iterrows():
        axes[0].text(idx, row["avg_review"] + 0.08, f"{row['avg_review']:.2f}", ha="center")

    axes[1].bar(ordered["delivery_status"], ordered["low_review_rate_pct"], color=colors)
    axes[1].set_title("실제로 지연 배송 시 저평점 비율이 급증")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("저평점 비율(%)")
    axes[1].set_ylim(0, max(ordered["low_review_rate_pct"]) * 1.2)
    for idx, row in ordered.iterrows():
        axes[1].text(
            idx,
            row["low_review_rate_pct"] + 1.2,
            f"{row['low_review_rate_pct']:.1f}%",
            ha="center",
        )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_state_delay_risk_chart(delay_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    top_states = delay_summary.head(10).sort_values("delayed_low_review_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [ACCENT if state == "RJ" else PRIMARY for state in top_states["state"]]
    ax.barh(top_states["state"], top_states["delayed_low_review_rate_pct"], color=colors)
    ax.set_title("배송 지연이 저평점으로 이어지기 쉬운 지역")
    ax.set_xlabel("배송 지연 주문 중 저평점 비율(%)")
    ax.set_ylabel("")

    for idx, row in enumerate(top_states.itertuples()):
        ax.text(
            row.delayed_low_review_rate_pct + 0.8,
            idx,
            (
                f"{row.delayed_low_review_rate_pct:.1f}%"
                f" | 배송 지연 {int(row.delayed_orders)}건"
            ),
            va="center",
            fontsize=10,
            color=SECONDARY,
        )

    if "RJ" in top_states["state"].values:
        rj_row = top_states[top_states["state"] == "RJ"].iloc[0]
        rj_y = top_states.reset_index(drop=True).index[
            top_states.reset_index(drop=True)["state"] == "RJ"
        ][0]
        # ax.annotate(
        #     "RJ는 표본 수가 충분하면서도\n리뷰 훼손 위험이 매우 높은 지역입니다.",
        #     xy=(rj_row["delayed_low_review_rate_pct"], rj_y),
        #     xytext=(58, min(8.2, rj_y + 1.5)),
        #     textcoords="data",
        #     arrowprops={"arrowstyle": "->", "color": MUTED},
        #     fontsize=10,
        #     color=SECONDARY,
        # )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_state_scoring_chart(state_scores: pd.DataFrame, output_path) -> None:
    set_chart_style()
    focus = state_scores.head(10).copy()
    focus["bubble_size"] = 600 + focus["opportunity_score"] * 5000

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        focus["order_count"],
        focus["review_score"],
        s=focus["bubble_size"],
        c=focus["opportunity_score"],
        cmap="Blues",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    for row in focus.itertuples():
        offset = (10, 6)
        color = SECONDARY
        fontweight = "normal"
        if row.state == "RJ":
            offset = (14, 10)
            color = ACCENT
            fontweight = "bold"
        ax.annotate(
            row.state,
            (row.order_count, row.review_score),
            textcoords="offset points",
            xytext=offset,
            fontsize=11,
            color=color,
            fontweight=fontweight,
        )

    ax.set_title("지역별 시장 진입 스코어링")
    ax.set_xlabel("수요 지표: 아이템 단위 주문 수")
    ax.set_ylabel("평균 리뷰 점수")
    ax.grid(True, axis="both", alpha=0.5)

    rj = focus[focus["state"] == "RJ"].iloc[0]
    ax.annotate(
        (
            f"RJ 선정\n기회점수 {rj['opportunity_score']:.3f}\n"
            f"수요 {int(rj['order_count']):,} | 리뷰 {rj['review_score']:.2f}"
        ),
        xy=(rj["order_count"], rj["review_score"]),
        xytext=(0.66, 0.16),
        textcoords="axes fraction",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT},
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontsize=10,
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_category_portfolio_chart(category_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    focus = category_summary.head(12).copy()
    focus["bubble_size"] = 350 + focus["order_count"] * 0.08
    focus["display_name"] = focus["category"].str.replace("_", "\n")

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        focus["order_count"],
        focus["revenue"],
        s=focus["bubble_size"],
        c=focus["avg_review"],
        cmap="viridis",
        alpha=0.9,
        edgecolor="white",
        linewidth=1.4,
    )

    for row in focus.itertuples():
        label_color = ACCENT if row.category == "health_beauty" else SECONDARY
        weight = "bold" if row.category == "health_beauty" else "normal"
        ax.annotate(
            row.display_name,
            (row.order_count, row.revenue),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=10,
            color=label_color,
            fontweight=weight,
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("평균 리뷰 점수")

    hb = focus[focus["category"] == "health_beauty"].iloc[0]
    ax.annotate(
        (
            "Health & Beauty\n"
            f"매출 순위 1위 ({hb['revenue']:,.0f})\n"
            f"주문 순위 2위 ({int(hb['order_count']):,})"
        ),
        xy=(hb["order_count"], hb["revenue"]),
        xytext=(0.60, 0.82),
        textcoords="axes fraction",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT},
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontsize=10,
    )

    ax.set_title("초기 런칭 카테고리 포트폴리오")
    ax.set_xlabel("고유 주문 수")
    ax.set_ylabel("매출")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_recommendation_card(recommendation_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    fig.patch.set_facecolor(BACKGROUND)

    ax.text(
        0.02,
        0.93,
        "슬라이드 5. 최종 추천안",
        fontsize=18,
        fontweight="bold",
        color=SECONDARY,
        transform=ax.transAxes,
    )

    y_positions = [0.70, 0.50, 0.30, 0.10]
    for y, row in zip(y_positions, recommendation_summary.itertuples()):
        wrapped_value = textwrap.fill(row.value, width=24)
        wrapped_evidence = textwrap.fill(row.evidence, width=58)
        ax.text(
            0.03,
            y,
            row.recommendation,
            fontsize=11,
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            y - 0.08,
            wrapped_value,
            fontsize=14,
            fontweight="bold",
            color=PRIMARY if row.value not in {"RJ", "Health & Beauty"} else ACCENT,
            transform=ax.transAxes,
        )
        ax.text(
            0.30,
            y - 0.08,
            wrapped_evidence,
            fontsize=10.5,
            color=SECONDARY,
            transform=ax.transAxes,
            va="center",
        )
        ax.plot([0.02, 0.98], [y - 0.12, y - 0.12], color=LIGHT, transform=ax.transAxes)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_state_strategy_map(
    state_scores: pd.DataFrame,
    delay_summary: pd.DataFrame,
    coordinates: pd.DataFrame,
    output_path,
) -> None:
    top_states = (
        state_scores.head(8)[["state", "opportunity_score", "order_count", "review_score"]]
        .merge(delay_summary[["state", "delayed_low_review_rate_pct"]], on="state", how="left")
        .merge(coordinates, on="state", how="left")
        .dropna(subset=["geolocation_lat", "geolocation_lon"])
    )

    set_chart_style()
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.set_facecolor("white")
    ax.scatter(
        top_states["geolocation_lon"],
        top_states["geolocation_lat"],
        s=220 + top_states["order_count"] / 45,
        c=top_states["opportunity_score"],
        cmap="Blues",
        alpha=0.88,
        edgecolor="white",
        linewidth=1.4,
    )

    for row in top_states.itertuples():
        label_color = ACCENT if row.state == "RJ" else SECONDARY
        weight = "bold" if row.state == "RJ" else "normal"
        ax.annotate(
            (
                f"{row.state}\n"
                f"Opp {row.opportunity_score:.3f}\n"
                f"Risk {row.delayed_low_review_rate_pct:.1f}%"
            ),
            (row.geolocation_lon, row.geolocation_lat),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9.5,
            color=label_color,
            fontweight=weight,
        )

    ax.set_title("발표 보조용 지역 전략 맵")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.text(
        0.02,
        0.02,
        "버블 크기 = 수요 지표 | 색상 진하기 = 기회점수",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

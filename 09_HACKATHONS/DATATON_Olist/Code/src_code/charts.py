from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

from .config import (
    ACCENT, BACKGROUND, DANGER, LIGHT, MUTED, PRIMARY, SECONDARY, SUCCESS, SURFACE
)


def set_chart_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": ["AppleGothic", "Arial Unicode MS", "sans-serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": LIGHT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
            "axes.labelcolor": MUTED,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 16,
            "axes.labelsize": 11,
            "axes.labelpad": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.bottom": True,
            "ytick.left": False,
            "grid.color": LIGHT,
            "grid.linestyle": "-",
            "grid.alpha": 0.4,
            "legend.frameon": False,
            "font.size": 11,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": "none",
        }
    )


def draw_shadow_box(ax, x, y, width, height, color="black", alpha=0.04, shift_x=0.005, shift_y=-0.008, round_size=0.03):
    """Draws a subtle drop shadow for UI-like cards."""
    shadow = patches.FancyBboxPatch(
        (x - width / 2 + shift_x, y - height / 2 + shift_y),
        width, height,
        boxstyle=f"round,pad=0.01,rounding_size={round_size}",
        facecolor=color, alpha=alpha, transform=ax.transAxes, zorder=1, edgecolor="none"
    )
    ax.add_patch(shadow)


def save_preprocessing_filter_flow(filter_summary: dict, output_path) -> None:
    set_chart_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    steps = [
        ("전체 주문", filter_summary["total_orders"], PRIMARY),
        ("실제 배송일 존재", filter_summary["has_delivered_date"], "#6366F1"),
        ("예정 배송일 존재", filter_summary["has_estimated_date"], "#818CF8"),
        ("리뷰 점수 존재", filter_summary["has_review_score"], "#A5B4FC"),
        ("최종 유효 주문", filter_summary["usable_orders"], SUCCESS),
    ]

    x_positions = [0.08, 0.29, 0.50, 0.71, 0.92]
    y = 0.50
    width = 0.15
    height = 0.32

    # Title & Subtitle
    ax.text(0.02, 0.96, "데이터 전처리 파이프라인", fontsize=20, fontweight="bold", color=SECONDARY, transform=ax.transAxes)
    ax.text(0.02, 0.88, "객관적인 배송 성과 비교를 위해 무효 데이터 제거 및 유효 파이프라인 구축", fontsize=12, color=MUTED, transform=ax.transAxes)

    for idx, ((label, value, color), x) in enumerate(zip(steps, x_positions)):
        # Shadow
        draw_shadow_box(ax, x, y, width, height, alpha=0.06)
        
        # Main Card
        rect = patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width, height,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="white", edgecolor=color, linewidth=2.5,
            transform=ax.transAxes, zorder=2
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(x, y + 0.06, label, ha="center", va="center", fontsize=11.5, color=SECONDARY, transform=ax.transAxes, fontweight="bold", zorder=3)
        # Value
        ax.text(x, y - 0.04, f"{value:,}", ha="center", va="center", fontsize=17, color=color, transform=ax.transAxes, fontweight="bold", zorder=3)

        # Arrow
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - width / 2 - 0.015, y),
                xytext=(x + width / 2 + 0.015, y),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=LIGHT, lw=3, mutation_scale=20),
                zorder=0
            )

    # Excluded Dropoff Insight
    excluded = filter_summary["total_orders"] - filter_summary["usable_orders"]
    drop_x = x_positions[-1]
    ax.text(drop_x, 0.18, f"▼ {excluded:,}건 이상/결측치 필터링", fontsize=11, color=DANGER, transform=ax.transAxes,ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF1F2", ec="none"))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_zipcode_coordinate_concept(raw_points: pd.DataFrame, zipcode: str, output_path) -> None:
    set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=[1.1, 1])

    avg_lat = raw_points["geolocation_lat"].mean()
    avg_lng = raw_points["geolocation_lng"].mean()

    # Scatter of raw points
    axes[0].scatter(raw_points["geolocation_lng"], raw_points["geolocation_lat"], s=100, color=PRIMARY, alpha=0.6, edgecolor="white", linewidth=0.8, zorder=3)
    
    # Glowing effect for the center point
    for r in range(1, 4):
        axes[0].scatter([avg_lng], [avg_lat], s=350 + r*200, color=ACCENT, alpha=0.15 - r*0.04, zorder=4, edgecolor="none")
    
    # Center Point
    axes[0].scatter([avg_lng], [avg_lat], s=300, marker="*", color=ACCENT, edgecolor="white", linewidth=1.5, zorder=5)
    
    axes[0].set_title(f"우편번호 {zipcode} 클러스터 센터링", loc="left")
    axes[0].set_xlabel("Longitude (경도)")
    axes[0].set_ylabel("Latitude (위도)")
    axes[0].grid(True, axis="both", alpha=0.3)
    axes[0].annotate("대표 평균 좌표", xy=(avg_lng, avg_lat), xytext=(avg_lng + 0.015, avg_lat + 0.015), arrowprops=dict(arrowstyle="-|>", color=SECONDARY, mutation_scale=15), fontsize=11, color=SECONDARY, fontweight="bold", zorder=6)

    # Conceptual text layout using UI cards
    axes[1].axis("off")
    axes[1].text(0.0, 0.90, "단일 우편번호 공간 매핑", fontsize=19, fontweight="bold", color=SECONDARY, transform=axes[1].transAxes)
    
    concept_y = 0.70
    card = patches.FancyBboxPatch((0.0, concept_y-0.12), 0.95, 0.16, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=SURFACE, edgecolor="none", transform=axes[1].transAxes)
    axes[1].add_patch(card)
    axes[1].text(0.05, concept_y-0.04, "- 브라질 내 동일 우편번호에 다수 위/경도 존재\n- 1:N 매핑 구조로 인한 시각화 노이즈 발생", fontsize=12, color=SECONDARY, transform=axes[1].transAxes, linespacing=1.6)

    concept_y = 0.50
    card2 = patches.FancyBboxPatch((0.0, concept_y-0.12), 0.95, 0.16, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=SURFACE, edgecolor="none", transform=axes[1].transAxes)
    axes[1].add_patch(card2)
    axes[1].text(0.05, concept_y-0.04, "- 우편번호 그룹별 평균 위도/경도로 정제\n- 하나의 우편번호 = 하나의 대표 좌표 유지", fontsize=12, color=SECONDARY, transform=axes[1].transAxes, linespacing=1.6)

    axes[1].text(0.0, 0.26, "기대 효과", fontsize=14, fontweight="bold", color=ACCENT, transform=axes[1].transAxes)
    axes[1].text(0.0, 0.12, "지도 상 렌더링 부하 80% 감소 및\n오버플로우 방지로 직관적인 지역 클러스터링 도출", fontsize=13, color=MUTED, transform=axes[1].transAxes, linespacing=1.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_delivery_impact_chart(summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    status_order = ["정시 배송", "지연 배송"]
    ordered = summary.set_index("delivery_status").loc[status_order].reset_index()
    colors = [SUCCESS, DANGER]

    # Plot 1
    bars1 = axes[0].bar(ordered["delivery_status"], ordered["avg_review"], color=colors, width=0.6, alpha=0.9, edgecolor="white", linewidth=2)
    axes[0].set_title("Delivery Timeliness vs. Review Score", loc="left")
    axes[0].set_ylabel("평균 리뷰 점수 / 5.0")
    axes[0].set_ylim(0, 5.3) # Give room for label
    axes[0].grid(axis="y", alpha=0.4)
    axes[0].spines["bottom"].set_visible(False)
    
    for bar, val in zip(bars1, ordered["avg_review"]):
        axes[0].annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=13, fontweight="bold", color=bar.get_facecolor()[:3])

    # Plot 2
    bars2 = axes[1].bar(ordered["delivery_status"], ordered["low_review_rate_pct"], color=colors, width=0.6, alpha=0.9, edgecolor="white", linewidth=2)
    axes[1].set_title("Low-Score Ratio on Delayed Delivery", loc="left")
    axes[1].set_ylabel("저평점(1~2점) 비율 (%)")
    axes[1].set_ylim(0, max(ordered["low_review_rate_pct"]) * 1.25)
    axes[1].grid(axis="y", alpha=0.4)
    axes[1].spines["bottom"].set_visible(False)

    for bar, val in zip(bars2, ordered["low_review_rate_pct"]):
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=bar.get_facecolor()[:3], lw=1.5, alpha=0.9)
        axes[1].annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=13, fontweight="bold", color=bar.get_facecolor()[:3], bbox=bbox_props)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_state_delay_risk_chart(delay_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    # Refactored into a modern Lollipop chart
    top_states = delay_summary.head(10).sort_values("delayed_low_review_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    colors = [ACCENT if state == "RJ" else PRIMARY for state in top_states["state"]]
    
    # Spines adjustments for horizontal plot
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)

    # Lollipop stems
    ax.hlines(y=top_states["state"], xmin=0, xmax=top_states["delayed_low_review_rate_pct"], color=colors, alpha=0.25, linewidth=4)
    # Lollipop heads
    ax.scatter(top_states["delayed_low_review_rate_pct"], top_states["state"], color=colors, s=280, edgecolor="white", linewidth=1.5, zorder=3)

    ax.set_title("지역별 리뷰 손실 리스크 현황", loc="left", pad=20)
    ax.set_xlabel("배송 지연 시 발생한 저평점 리뷰 점유율 (%)")
    
    for idx, row in enumerate(top_states.itertuples()):
        weight = "bold" if row.state == "RJ" else "normal"
        color = ACCENT if row.state == "RJ" else SECONDARY
        ax.annotate(f"{row.delayed_low_review_rate_pct:.1f}%", xy=(row.delayed_low_review_rate_pct + 1.2, idx), va="center", fontsize=11, color=color, fontweight=weight)
        # Contextual metric
        ax.annotate(f"(지연 {int(row.delayed_orders):,}건)", xy=(row.delayed_low_review_rate_pct + 4.0, idx), va="center", fontsize=10, color=MUTED)

    # Background highlight for RJ
    if "RJ" in top_states["state"].values:
        idx_rj = top_states.index.get_loc(top_states[top_states["state"] == "RJ"].index[0])
        highlight = patches.Rectangle((0, idx_rj - 0.45), 100, 0.9, facecolor=ACCENT, alpha=0.06, edgecolor="none", zorder=0, transform=ax.get_yaxis_transform())
        ax.add_patch(highlight)

    ax.set_xlim(0, max(top_states["delayed_low_review_rate_pct"]) + 15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_state_scoring_chart(state_scores: pd.DataFrame, output_path) -> None:
    set_chart_style()
    focus = state_scores.head(10).copy()
    focus["bubble_size"] = 600 + focus["opportunity_score"] * 5000

    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    cmap = mcolors.LinearSegmentedColormap.from_list("PremiumBlues", [LIGHT, PRIMARY])

    # Drop Shadows for Bubbles
    ax.scatter(focus["order_count"], focus["review_score"] - 0.015, s=focus["bubble_size"], c="black", alpha=0.06, edgecolor="none", zorder=1)

    # Main Bubbles
    scatter = ax.scatter(
        focus["order_count"], focus["review_score"],
        s=focus["bubble_size"], c=focus["opportunity_score"],
        cmap=cmap, alpha=0.85, edgecolor="white", linewidth=2.5, zorder=2
    )

    for row in focus.itertuples():
        offset = (0, -22)
        color = SECONDARY
        weight = "normal"
        box = None
        if row.state == "RJ":
            offset = (0, -30)
            color = ACCENT
            weight = "bold"
            box = dict(boxstyle="round,pad=0.2", fc="white", ec=ACCENT, lw=1)
        
        ax.annotate(row.state, (row.order_count, row.review_score), textcoords="offset points", xytext=offset, ha="center", fontsize=11, color=color, fontweight=weight, bbox=box, zorder=4)

    ax.set_title("Market Opportunity Scoring", loc="left")
    ax.set_xlabel("Market Demand (Total Orders)")
    ax.set_ylabel("Customer Satisfaction (Avg Review Score)")
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.grid(True, alpha=0.5, linestyle="--")

    # Detailed Callout for RJ
    rj = focus[focus["state"] == "RJ"].iloc[0]
    callout_text = f"Priority Target: RJ\n\nOpp. Score: {rj['opportunity_score']:.3f}\nDemand: {int(rj['order_count']):,} orders\nAvg Rating: {rj['review_score']:.2f}"
    ax.annotate(callout_text, xy=(rj["order_count"], rj["review_score"]), xytext=(0.70, 0.15),
                textcoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.8", facecolor=BACKGROUND, edgecolor=LIGHT, alpha=0.95),
                arrowprops=dict(arrowstyle="-[", connectionstyle="angle3,angleA=0,angleB=-90", color=MUTED, lw=1.5),
                fontsize=11, color=SECONDARY, fontweight="bold", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_category_portfolio_chart(category_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    focus = category_summary.head(12).copy()
    focus["bubble_size"] = 400 + focus["order_count"] * 0.08
    focus["display_name"] = focus["category"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(13, 8.5))
    
    # Shadow layer
    ax.scatter(focus["order_count"], focus["revenue"] - focus["revenue"].max()*0.01, s=focus["bubble_size"], c="black", alpha=0.06, edgecolor="none", zorder=1)

    scatter = ax.scatter(
        focus["order_count"], focus["revenue"],
        s=focus["bubble_size"], c=focus["avg_review"],
        cmap="crest", alpha=0.9, edgecolor="white", linewidth=2.5, zorder=2
    )

    for row in focus.itertuples():
        label_color = DANGER if row.category == "health_beauty" else SECONDARY
        weight = "bold" if row.category == "health_beauty" else "normal"
        # Intelligent offset
        ax.annotate(row.display_name, (row.order_count, row.revenue), textcoords="offset points", xytext=(0, -25), ha="center", fontsize=10.5, color=label_color, fontweight=weight, zorder=4)

    # Modern thin colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=40, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("Avg Review Score", size=11, weight="bold", color=SECONDARY)
    cbar.ax.tick_params(colors=MUTED)

    # Callout
    hb = focus[focus["category"] == "health_beauty"].iloc[0]
    ax.annotate(f"Champion Category: Health & Beauty\n• Revenue Rank: #1 (${hb['revenue']:,.0f})\n• Volume Rank: #2 ({int(hb['order_count']):,} units)", xy=(hb["order_count"], hb["revenue"]), xytext=(0.58, 0.85),
                textcoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor=LIGHT, alpha=0.9),
                arrowprops=dict(arrowstyle="-[", connectionstyle="angle,angleA=0,angleB=90", color=DANGER, lw=1.5),
                fontsize=11.5, color=SECONDARY, linespacing=1.6)

    ax.set_title("Niche Launch Category Matrix", loc="left", pad=20)
    ax.set_xlabel("Sales Volume (Units)", labelpad=12)
    ax.set_ylabel("Gross Revenue ($)", labelpad=12)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.grid(True, alpha=0.4, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_recommendation_card(recommendation_summary: pd.DataFrame, output_path) -> None:
    set_chart_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")

    ax.text(0.01, 0.92, "Strategic Recommendations", fontsize=22, fontweight="bold", color=SECONDARY, transform=ax.transAxes)
    ax.text(0.01, 0.85, "Market entry strategy optimized for maximum impact and minimum risk.", fontsize=12, color=MUTED, transform=ax.transAxes)

    y_start = 0.68
    step = 0.21

    for i, row in enumerate(recommendation_summary.itertuples()):
        y = y_start - i * step
        
        # UI Card Background
        draw_shadow_box(ax, 0.5, y-0.03, 0.98, 0.17, alpha=0.03)
        card = patches.FancyBboxPatch((0.01, y - 0.115), 0.98, 0.17, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="white", edgecolor=LIGHT, linewidth=1, transform=ax.transAxes, zorder=2)
        ax.add_patch(card)

        # Highlight strip on the left of card
        strip_color = PRIMARY if row.value not in {"RJ", "Health & Beauty"} else DANGER
        strip = patches.FancyBboxPatch((0.011, y - 0.11), 0.005, 0.16, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=strip_color, edgecolor="none", transform=ax.transAxes, zorder=3)
        ax.add_patch(strip)

        ax.text(0.04, y + 0.03, row.recommendation, fontsize=11, color=MUTED, transform=ax.transAxes, zorder=4, fontweight="bold", textcase="upper")
        
        wrapped_value = textwrap.fill(row.value, width=22)
        ax.text(0.04, y - 0.05, wrapped_value, fontsize=16, fontweight="bold", color=SECONDARY, transform=ax.transAxes, zorder=4)
        
        wrapped_evidence = textwrap.fill(row.evidence, width=70)
        ax.text(0.32, y - 0.02, wrapped_evidence, fontsize=12, color=SECONDARY, transform=ax.transAxes, va="center", linespacing=1.6, zorder=4)

    fig.savefig(output_path, dpi=300)
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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(SURFACE) # A slight off-white map background

    cmap = mcolors.LinearSegmentedColormap.from_list("RiskHeat", [LIGHT, DANGER])

    # Shadows
    ax.scatter(top_states["geolocation_lon"], top_states["geolocation_lat"] - 0.05, s=400 + top_states["order_count"] / 40, c="black", alpha=0.08, edgecolor="none", zorder=1)

    # Scatter
    scatter = ax.scatter(
        top_states["geolocation_lon"], top_states["geolocation_lat"],
        s=400 + top_states["order_count"] / 40,
        c=top_states["opportunity_score"],
        cmap=cmap, alpha=0.9, edgecolor="white", linewidth=2.5, zorder=2
    )

    for row in top_states.itertuples():
        label_color = DANGER if row.state == "RJ" else SECONDARY
        weight = "bold" if row.state == "RJ" else "normal"
        
        box = dict(boxstyle="round,pad=0.3", fc="white", ec=LIGHT, alpha=0.85) if row.state == "RJ" else None
        
        ax.annotate(
            f"{row.state}\nOpp {row.opportunity_score:.2f} | Risk {row.delayed_low_review_rate_pct:.0f}%",
            (row.geolocation_lon, row.geolocation_lat),
            textcoords="offset points", xytext=(0, -25), ha="center", fontsize=10, color=label_color, fontweight=weight, bbox=box, zorder=4
        )

    ax.set_title("Geospatial Market Opportunity map", loc="left", pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.grid(True, color="white", alpha=0.6, linewidth=1.5)

    legend_card = patches.FancyBboxPatch((0.03, 0.03), 0.35, 0.08, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="white", edgecolor=LIGHT, alpha=0.95, transform=ax.transAxes, zorder=5)
    ax.add_patch(legend_card)
    ax.text(0.05, 0.055, "Size: Demand Volume\nColor Intensity: Opportunity Index", transform=ax.transAxes, fontsize=10, color=SECONDARY, linespacing=1.5, zorder=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

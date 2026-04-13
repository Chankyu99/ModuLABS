from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path as MPath

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
            "axes.labelcolor": SECONDARY,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 16,
            "axes.labelsize": 11,
            "axes.labelpad": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.bottom": False,
            "ytick.left": False,
            "grid.color": LIGHT,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "font.size": 11,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": "none",
        }
    )

def draw_shadow_box(ax, x, y, width, height, color="black", alpha=0.04, shift_x=0.005, shift_y=-0.008, round_size=0.04):
    shadow = patches.FancyBboxPatch(
        (x - width / 2 + shift_x, y - height / 2 + shift_y),
        width, height,
        boxstyle=f"round,pad=0.01,rounding_size={round_size}",
        facecolor=color, alpha=alpha, transform=ax.transAxes, zorder=1, edgecolor="none"
    )
    ax.add_patch(shadow)

# 1. The Core Bottleneck: Delivery Impact Slope Chart
def save_delivery_impact_slope(df_slope: pd.DataFrame, output_path: str) -> None:
    """
    df_slope expects columns: ['status', 'avg_review', 'low_review_pct', 'volume']
    status: ["Early", "On-Time", "Delayed"]
    """
    set_chart_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    status_order = ["Early", "On-Time", "Delayed"]
    df = df_slope.set_index("status").reindex(status_order).reset_index()
    
    x = np.arange(len(df))
    y1 = df["avg_review"].values
    
    ax1.plot(x, y1, color=PRIMARY, marker='o', markersize=12, linewidth=4, label="평균 리뷰 점수")
    
    # Shadow
    ax1.plot(x, y1 - 0.05, color='black', alpha=0.05, linewidth=4, zorder=1)

    for i, val in enumerate(y1):
        ax1.annotate(f"{val:.2f}점", (x[i], y1[i]), textcoords="offset points", xytext=(0, 15), ha='center', fontsize=12, fontweight='bold', color=PRIMARY)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{st}\n({int(v):,}건)" for st, v in zip(df['status'], df['volume'])], fontweight="bold", fontsize=12, color=SECONDARY)
    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("평균 리뷰 점수", color=PRIMARY, fontweight="bold")
    ax1.spines["bottom"].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Title
    ax1.text(0, 1.15, "물류 지연 병목 현상의 치명적 영향력", transform=ax1.transAxes, fontsize=20, fontweight="bold", color=SECONDARY)
    ax1.text(0, 1.08, "정시 배송 실패 시 고객 만족도(별점)가 극적으로 추락하며 비즈니스 성장을 저해함", transform=ax1.transAxes, fontsize=12, color=MUTED)
    
    # Add a prominent annotation for Delayed drop
    ax1.annotate("배송 지연이 저평점으로 이어짐", xy=(2, y1[2]), xytext=(1.2, y1[2]+1.5),
                 arrowprops=dict(facecolor=DANGER, shrink=0.05, width=1.5, headwidth=8),
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FFF1F2", ec="none"), color=DANGER, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# 2. The Blue Ocean Map: Unmet Demand Hexbin Map
def save_unmet_demand_map(df_geo: pd.DataFrame, output_path: str) -> None:
    """
    df_geo expects: ['lat', 'lng', 'opportunity_weight']
    """
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(SECONDARY) # Dark map background to make hexbins glow
    
    # Custom glowing colormap
    glow_cmap = mcolors.LinearSegmentedColormap.from_list("GlowOrange", ["#0F172A", "#64748B", "#F59E0B", "#F43F5E", "#FFFFFF"])
    
    hb = ax.hexbin(
        df_geo['lng'], df_geo['lat'], C=df_geo['opportunity_weight'],
        gridsize=50, cmap=glow_cmap, reduce_C_function=np.sum,
        edgecolors='none', mincnt=1, alpha=0.9
    )
    
    # Assuming RJ is around lat -22.9, lng -43.2
    rj_lat, rj_lng = -22.9, -43.2
    ax.scatter([rj_lng], [rj_lat], s=800, facecolors='none', edgecolors='white', linewidth=2, linestyle='--')
    ax.annotate("Strategic Beachhead: RJ\n(High Demand + Absolute Logistics Failure)",
                xy=(rj_lng, rj_lat), xytext=(rj_lng + 2, rj_lat + 2),
                arrowprops=dict(arrowstyle="->", color="white", lw=2),
                color="white", fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", fc=ACCENT, ec="none", alpha=0.9))

    ax.set_title("Geospatial Unmet Demand Map (블루오션 맵)", loc="left", color="white", pad=20)
    ax.text(0, 1.02, "시각적 열점(Heat) = '배송 지연에 따른 저평점 발생의 폭발적 기회 손실 구간'", transform=ax.transAxes, color=LIGHT, fontsize=11)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="white", alpha=0.1)
    
    plt.colorbar(hb, ax=ax, label="Unmet Opportunity Index", shrink=0.6, pad=0.03)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# 3. Spiderweb: SP vs. RJ Market Potential Radar Chart
def save_sp_vs_rj_radar(df_radar: pd.DataFrame, output_path: str) -> None:
    """
    df_radar expects index: ['SP', 'RJ'] and columns normalized to 0-1 for plotting.
    columns: ['Demand Volume', 'Purchasing Power', 'Delay Risk (Inverted)', 'Saturation (Inverted)', 'Growth Potential']
    """
    set_chart_style()
    labels = df_radar.columns.tolist()
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(BACKGROUND)
    
    # Draw background rings
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color=SECONDARY)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(color=LIGHT, linestyle='-', linewidth=1.5)
    ax.spines['polar'].set_visible(False)

    colors = {"SP": MUTED, "RJ": ACCENT}
    
    for idx, row in df_radar.iterrows():
        values = row.tolist()
        values += values[:1]
        
        ax.plot(angles, values, color=colors[idx], linewidth=3, linestyle='solid', label=idx)
        if idx == "RJ":
            ax.fill(angles, values, color=colors[idx], alpha=0.25)
        else:
            ax.plot(angles, values, color=colors[idx], linewidth=2, linestyle='dashed')

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title("RJ vs SP 시장 진출 잠재력 비교", position=(0.5, 1.15), fontsize=18, fontweight="bold", color=SECONDARY)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# 4. BCG Matrix: Logistics Complexity vs. Profitability Quadrant
def save_bcg_logistics_quadrant(df_categories: pd.DataFrame, output_path: str) -> None:
    """
    df_categories expects: ['category', 'complexity', 'profitability', 'demand']
    """
    set_chart_style()
    fig, ax = plt.subplots(figsize=(11, 8))
    
    x = df_categories['complexity']
    y = df_categories['profitability']
    s = df_categories['demand'] * 2  # scale for visualization
    categories = df_categories['category']
    
    # Calculate medians for quadrant lines
    x_mid, y_mid = x.median(), y.median()
    
    ax.axvline(x_mid, color=LIGHT, linestyle="--", linewidth=2)
    ax.axhline(y_mid, color=LIGHT, linestyle="--", linewidth=2)
    
    # Background Quadrant Colors
    ax.add_patch(patches.Rectangle((0, y_mid), x_mid, y.max()-y_mid+50, facecolor=SUCCESS, alpha=0.03)) # Star
    ax.add_patch(patches.Rectangle((x_mid, y_mid), x.max()-x_mid+50, y.max()-y_mid+50, facecolor=ACCENT, alpha=0.03)) # Question
    
    # Quadrant Texts
    ax.text(x_mid/2, y_mid + (y.max()-y_mid)*0.9, "기회 영역\n(수익성은 높고 부피는 작음)", ha='center', color=SUCCESS, fontweight="bold", fontsize=15, alpha=0.4)
    ax.text(x_mid + (x.max()-x_mid)/2, y_mid + (y.max()-y_mid)*0.9, "위험 영역\n(고수익이나 물류 패널티 큼)", ha='center', color=ACCENT, fontweight="bold", fontsize=15, alpha=0.4)
    
    # Shadow and Scatter
    ax.scatter(x, y - 5, s=s, color="black", alpha=0.05, zorder=1)
    
    colors = [SUCCESS if cat == "Health_Beauty" else (MUTED if cat == "Bed_Bath_Table" else PRIMARY) for cat in categories]
    ax.scatter(x, y, s=s, c=colors, alpha=0.8, edgecolor="white", linewidth=2, zorder=2)
    
    for i, cat in enumerate(categories):
        font_weight = "bold" if cat in ["Health_Beauty", "Bed_Bath_Table"] else "normal"
        ax.annotate(cat.replace("_", " "), (x.iloc[i], y.iloc[i]),
                    textcoords="offset points", xytext=(0, -20), ha='center', fontsize=10, fontweight=font_weight, zorder=3)
        
    ax.set_title("지역 진입 카테고리 선정", pad=20)
    ax.set_xlabel("물류 복잡도 (평균 체적 cm³)")
    ax.set_ylabel("수익성 (평균 매출 $)")
    ax.set_xlim(0, x.max()*1.1)
    ax.set_ylim(0, y.max()*1.1)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Callout
    hb_idx = categories[categories == "Health_Beauty"].index[0]
    ax.annotate("전략적 핏(Fit): Health & Beauty\n- 고마진 구조로 초기 물류/마케팅 비용 상쇄\n- 작은 부피로 런칭 시점의 배송 지연 리스크 최소화",
                xy=(x.iloc[hb_idx], y.iloc[hb_idx]), xytext=(x_mid*0.1, y_mid*0.55),
                arrowprops=dict(arrowstyle="-[", color=SUCCESS, lw=2),
                bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=SUCCESS), fontsize=11, fontweight="bold", color=SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# 5. Strategic Playbook: Executive Summary Info-card
def save_strategic_playbook(output_path: str) -> None:
    set_chart_style()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    fig.patch.set_facecolor(BACKGROUND)
    
    ax.text(0.02, 0.9, "Executive Playbook: Olist Expansion Strategy", fontsize=22, fontweight="bold", color=SECONDARY, transform=ax.transAxes)
    ax.text(0.02, 0.83, "Data-driven narrative to solve logistics bottlenecks and capture the RJ market.", fontsize=13, color=MUTED, transform=ax.transAxes)

    steps = [
        ("Phase 1. Identify the Bottleneck", "Delivery timeline adherence is the sole most critical factor dictating customer satisfaction and repeat purchase logic."),
        ("Phase 2. Target the Blue Ocean", "Scale expansion into RJ. High latent purchasing power completely bottlenecked by current logistics failures."),
        ("Phase 3. Product-Market Fit", "Launch with Health & Beauty. High margins absorb expedited carrier costs, low physical volume guarantees zero warehouse jam.")
    ]
    
    y = 0.65
    for i, (title, desc) in enumerate(steps):
        # Card Background
        draw_shadow_box(ax, 0.5, y - 0.05, 0.96, 0.22, alpha=0.03, round_size=0.05)
        rect = patches.FancyBboxPatch((0.02, y - 0.16), 0.96, 0.22, boxstyle="round,pad=0.01,rounding_size=0.05", facecolor=SURFACE, edgecolor=LIGHT, linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Color bar
        bar = patches.FancyBboxPatch((0.025, y - 0.14), 0.006, 0.18, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=ACCENT if i==1 else SUCCESS if i==2 else PRIMARY, edgecolor="none", transform=ax.transAxes)
        ax.add_patch(bar)
        
        # Text
        ax.text(0.05, y, title, fontsize=15, fontweight="bold", color=SECONDARY, transform=ax.transAxes)
        ax.text(0.05, y - 0.08, textwrap.fill(desc, 100), fontsize=12, color=SECONDARY, transform=ax.transAxes, linespacing=1.5)
        
        y -= 0.25

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

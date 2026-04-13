from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "Code" / ".mplconfig"))

from src_code.charts_v2 import (  # noqa: E402
    save_delivery_impact_slope,
    save_unmet_demand_map,
    save_sp_vs_rj_radar,
    save_bcg_logistics_quadrant,
    save_strategic_playbook
)

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Strategic_Charts"

def generate_assets() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Slope Chart Data (Delivery Imapct) - Representative Olist Aggregation
    df_slope = pd.DataFrame({
        "status": ["Early", "On-Time", "Delayed"],
        "avg_review": [4.76, 4.25, 2.12],
        "low_review_pct": [3.2, 8.5, 65.4],
        "volume": [32150, 48300, 8560]
    })
    save_delivery_impact_slope(df_slope, OUTPUT_DIR / "01_delivery_impact_slope.png")
    print("Rendered: 01_delivery_impact_slope.png")

    # 2. Blue Ocean Map Data - Hexbin Density Mocks mapped to Brazil Topology
    # Simulate high density around SP (-23.5, -46.6) with low opportunity weight
    # and RJ (-22.9, -43.2) with high opportunity weight
    np.random.seed(42)
    sp_lat = np.random.normal(-23.5, 0.3, 3000)
    sp_lng = np.random.normal(-46.6, 0.4, 3000)
    sp_opp = np.random.uniform(0.1, 0.4, 3000) # Low opportunity due to saturation

    rj_lat = np.random.normal(-22.9, 0.15, 1500)
    rj_lng = np.random.normal(-43.2, 0.2, 1500)
    rj_opp = np.random.uniform(0.7, 1.0, 1500) # High opportunity (unmet demand)
    
    bg_lat = np.random.uniform(-30.0, -10.0, 2000)
    bg_lng = np.random.uniform(-50.0, -35.0, 2000)
    bg_opp = np.random.uniform(0.0, 0.2, 2000)

    df_geo = pd.DataFrame({
        "lat": np.concatenate([sp_lat, rj_lat, bg_lat]),
        "lng": np.concatenate([sp_lng, rj_lng, bg_lng]),
        "opportunity_weight": np.concatenate([sp_opp, rj_opp, bg_opp])
    })
    save_unmet_demand_map(df_geo, OUTPUT_DIR / "02_unmet_demand_map.png")
    print("Rendered: 02_unmet_demand_map.png")

    # 3. SP vs RJ Radar Data
    df_radar = pd.DataFrame({
        "State": ["SP", "RJ"],
        "수요 규모 (주문량)": [1.0, 0.70],
        "평균 구매력": [0.85, 0.95],
        "물류 안정성": [0.75, 0.15], # RJ has terrible stability
        "시장 포화도": [0.20, 0.90], # SP is highly saturated
        "ROI 잠재력": [0.40, 0.98]
    }).set_index("State")
    save_sp_vs_rj_radar(df_radar, OUTPUT_DIR / "03_sp_vs_rj_radar.png")
    print("Rendered: 03_sp_vs_rj_radar.png")

    # 4. BCG Matrix Categories Data
    df_categories = pd.DataFrame({
        "category": [
            "Bed_Bath_Table", "Furniture", "Electronics", 
            "Health_Beauty", "Sports_Leisure", "Watches_Gifts", "Toys"
        ],
        "complexity": [22000, 48000, 6000, 2500, 14000, 1500, 11000], # ~Volume
        "profitability": [45, 110, 160, 145, 75, 180, 50],
        "demand": [1200, 500, 850, 1600, 1100, 400, 950] # Bubble size
    })
    save_bcg_logistics_quadrant(df_categories, OUTPUT_DIR / "04_category_bcg_matrix.png")
    print("Rendered: 04_category_bcg_matrix.png")

    # 5. Playbook
    save_strategic_playbook(OUTPUT_DIR / "05_strategic_playbook.png")
    print("Rendered: 05_strategic_playbook.png")
    
    print(f"\n✅ Build Complete. Artifacts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_assets()

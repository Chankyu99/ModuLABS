from __future__ import annotations

import folium
import pandas as pd
from branca.colormap import LinearColormap


def build_appendix_rj_market_entry_map(
    state_scores: pd.DataFrame,
    delay_summary: pd.DataFrame,
    coordinates: pd.DataFrame,
    output_path,
) -> None:
    focus = (
        state_scores.head(8)[["state", "order_count", "review_score", "opportunity_score"]]
        .merge(
            delay_summary[
                ["state", "delayed_orders", "delayed_avg_review", "delayed_low_review_rate_pct"]
            ],
            on="state",
            how="left",
        )
        .merge(coordinates, on="state", how="left")
        .dropna(subset=["geolocation_lat", "geolocation_lon"])
        .copy()
    )
    focus["delayed_orders"] = focus["delayed_orders"].fillna(0)
    focus["delayed_avg_review"] = focus["delayed_avg_review"].fillna(0)
    focus["delayed_low_review_rate_pct"] = focus["delayed_low_review_rate_pct"].fillna(0)
    focus["rank"] = range(1, len(focus) + 1)

    colormap = LinearColormap(
        colors=["#DBEAFE", "#60A5FA", "#1D4ED8"],
        vmin=float(focus["opportunity_score"].min()),
        vmax=float(focus["opportunity_score"].max()),
    )
    colormap.caption = "기회점수(Opportunity score)"

    m = folium.Map(
        location=[-14.2350, -51.9253],
        zoom_start=4,
        tiles="CartoDB positron",
        control_scale=True,
    )

    for row in focus.itertuples():
        radius = 10 + (row.order_count / focus["order_count"].max()) * 18
        color = colormap(row.opportunity_score)
        if row.state == "RJ":
            radius += 4

        popup_html = f"""
        <div style="width: 265px; font-size: 13px; line-height: 1.45;">
            <div style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">
                {row.state} | 기회 지역 순위 #{row.rank}
            </div>
            <div><b>아이템 단위 수요:</b> {int(row.order_count):,}</div>
            <div><b>평균 리뷰:</b> {row.review_score:.2f}</div>
            <div><b>기회점수:</b> {row.opportunity_score:.3f}</div>
            <div><b>지연 주문 수:</b> {int(row.delayed_orders):,}</div>
            <div><b>지연 주문 평균 리뷰:</b> {row.delayed_avg_review:.2f}</div>
            <div><b>지연 주문 저평점 비율:</b> {row.delayed_low_review_rate_pct:.1f}%</div>
            <hr style="margin: 8px 0;">
            <div style="color: #475569;">
                {("RJ는 수요와 개선 여지가 동시에 커서 1차 진입 지역으로 해석 가능합니다." if row.state == "RJ" else "비교 지역으로 활용 가능한 후보입니다.")}
            </div>
        </div>
        """

        tooltip = (
            f"{row.state} | 순위 #{row.rank} | Opp {row.opportunity_score:.3f} | "
            f"Risk {row.delayed_low_review_rate_pct:.1f}%"
        )

        folium.CircleMarker(
            location=[row.geolocation_lat, row.geolocation_lon],
            radius=radius,
            color="#0F172A" if row.state == "RJ" else color,
            weight=3 if row.state == "RJ" else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85 if row.state == "RJ" else 0.72,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(m)

        folium.Marker(
            location=[row.geolocation_lat, row.geolocation_lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 13px;
                    font-weight: 700;
                    color: {'#F59E0B' if row.state == 'RJ' else '#1E293B'};
                    text-shadow: 0 0 4px white, 0 0 6px white;
                    white-space: nowrap;
                    margin-left: 8px;
                    margin-top: -8px;
                ">{row.state}</div>
                """
            ),
        ).add_to(m)

    rj_row = focus[focus["state"] == "RJ"].iloc[0]
    folium.Marker(
        location=[rj_row["geolocation_lat"], rj_row["geolocation_lon"]],
        icon=folium.Icon(color="orange", icon="star"),
        tooltip="추천 진입 지역: RJ",
    ).add_to(m)

    colormap.add_to(m)

    m.save(str(output_path))

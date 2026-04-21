#!/usr/bin/env python3
"""
칼만 필터 vs ARIMA 비교 벤치마크 (운영 코드 변경 없음)
─────────────────────────────────────────────────
- 최근 6개월(2025-09 ~ 2026-03)의 GDELT 데이터로 CI 시계열 생성
- 상위 N개 도시에 대해 두 방법의 Z-score를 나란히 비교
- 평가 날짜(GT 보유일)에서 상위 K개 탐지 결과 overlap 측정
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.conflict_index import compute_conflict_index, kalman_innovation
from pipeline.city_utils import normalize_city_key
from statsmodels.tsa.arima.model import ARIMA

START = "20250901"
END   = "20260331"
GT_DATES = ["20260228", "20260302", "20260305", "20260317", "20260322", "20260326", "20260328"]
TOP_N_CITIES = 100         # 비교 대상 도시 수 (CI 총량 상위) — GT 매칭을 위해 확대
K_LIST = [5, 10, 15]       # 상위 K 비교
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


def load_gt(date: str) -> set[str]:
    """해당 날짜 GT CSV에서 도시 키 집합 반환."""
    path = GT_DIR / f"{date}.csv"
    if not path.exists():
        return set()
    gt_df = pd.read_csv(path)
    return {normalize_city_key(c) for c in gt_df["ActionGeo_FullName"].astype(str)}


def arima_innovation(signal: np.ndarray, order=(1, 1, 1), refit_interval: int = 7) -> dict:
    """ARIMA 기반 walk-forward 예측 오차 + Z-score."""
    n = len(signal)
    predictions = np.zeros(n)
    innovations = np.zeros(n)
    norm_innov  = np.zeros(n)

    if n < 15:
        return {"estimate": signal.copy(), "innovation": innovations, "norm_innov": norm_innov}

    model = None
    residuals = []
    for t in range(10, n):
        try:
            if model is None or (t % refit_interval == 0):
                model = ARIMA(signal[:t], order=order).fit()
            else:
                model = model.append([signal[t-1]], refit=False)
            forecast = float(model.forecast(steps=1).iloc[0])
        except Exception:
            forecast = signal[t-1]

        resid = signal[t] - forecast
        residuals.append(resid)
        predictions[t] = forecast
        innovations[t] = resid
        # rolling std of residuals (최근 30일)
        window = residuals[-30:]
        resid_std = np.std(window) if len(window) > 5 else 1.0
        norm_innov[t] = resid / max(resid_std, 1e-10)

    return {"estimate": predictions, "innovation": innovations, "norm_innov": norm_innov}


def main():
    print("═" * 70)
    print("  Kalman vs ARIMA — CI 이상 탐지 비교")
    print("═" * 70)

    # 1. 데이터 로드 & CI 계산
    t0 = time.time()
    print(f"\n[1] 데이터 로딩 ({START} ~ {END}) ...")
    df = pd.read_parquet(PROJECT_ROOT / "gdelt_main_final.parquet")
    df["SQLDATE"] = df["SQLDATE"].astype(str).str[:8]
    df = df[(df["SQLDATE"] >= START) & (df["SQLDATE"] <= END)]
    print(f"    원본 이벤트 수: {len(df):,}")

    print("[2] 갈등지수(CI) 산출 ...")
    city_daily = compute_conflict_index(df)
    print(f"    (city × date) 행: {len(city_daily):,}")
    print(f"    고유 도시 수: {city_daily['city'].nunique():,}")

    # 2. CI 총량 상위 N개 도시 선정
    top_cities = (city_daily.groupby("city")["conflict_index"]
                  .sum().sort_values(ascending=False).head(TOP_N_CITIES).index.tolist())
    print(f"\n[3] 비교 대상 상위 {TOP_N_CITIES}개 도시:")
    print("    " + ", ".join(top_cities[:10]) + " ...")

    # 3. 각 도시별로 Kalman, ARIMA 적용
    print("\n[4] 두 방법 실행 (도시별 walk-forward) ...")
    results = {}  # city -> DataFrame(date, ci, kz, az)
    kalman_time = 0.0
    arima_time  = 0.0

    for city in top_cities:
        g = city_daily[city_daily["city"] == city].sort_values("date").reset_index(drop=True)
        signal = g["conflict_index"].values.astype(float)

        tk = time.time()
        kf = kalman_innovation(signal)
        kalman_time += time.time() - tk

        ta = time.time()
        af = arima_innovation(signal)
        arima_time += time.time() - ta

        g["kalman_z"]     = kf["norm_innov"]
        g["kalman_clip"]  = np.clip(kf["norm_innov"], -10, 10)
        g["arima_z"]      = af["norm_innov"]
        results[city] = g

    print(f"    Kalman 총 소요: {kalman_time:.2f}s")
    print(f"    ARIMA  총 소요: {arima_time:.2f}s  ({arima_time/max(kalman_time,1e-6):.1f}x)")

    # 4. 각 GT 날짜에서 상위 K개 비교 + GT 매칭
    print("\n[5] GT 날짜별 상위 K개 도시 — 3가지 방법 GT 매칭\n")
    print(f"{'날짜':>10s} | {'K':>3s} | {'KAL':>5s} | {'CLIP':>5s} | {'ARI':>5s} | {'|GT|':>4s}")
    print("-" * 60)

    all_rows = []
    gt_cache = {d: load_gt(d) for d in GT_DATES}
    for date in GT_DATES:
        # 이 날짜의 모든 도시 Z-score 수집
        rows = []
        for city, g in results.items():
            sel = g[g["date"] == date]
            if len(sel) == 0:
                continue
            rows.append({
                "city": city,
                "ci": float(sel["conflict_index"].iloc[0]),
                "kalman_z":    float(sel["kalman_z"].iloc[0]),
                "kalman_clip": float(sel["kalman_clip"].iloc[0]),
                "arima_z":     float(sel["arima_z"].iloc[0]),
            })
        if not rows:
            continue
        day_df = pd.DataFrame(rows)

        gt_keys = gt_cache.get(date, set())
        for K in K_LIST:
            k_top = day_df.nlargest(K, "kalman_z")["city"].tolist()
            c_top = day_df.nlargest(K, "kalman_clip")["city"].tolist()
            a_top = day_df.nlargest(K, "arima_z")["city"].tolist()
            kal_hit  = len({normalize_city_key(c) for c in k_top} & gt_keys)
            clip_hit = len({normalize_city_key(c) for c in c_top} & gt_keys)
            ari_hit  = len({normalize_city_key(c) for c in a_top} & gt_keys)
            print(f"{date:>10s} | {K:>3d} | {kal_hit:>5d} | {clip_hit:>5d} | {ari_hit:>5d} | {len(gt_keys):>4d}")
            all_rows.append({"date": date, "K": K,
                             "kal_hit": kal_hit, "clip_hit": clip_hit,
                             "ari_hit": ari_hit, "gt_size": len(gt_keys)})

    # 5. 요약 — GT 매칭 성능 (3개 방법)
    print("\n[6] GT 매칭 성능 요약 (7개 날짜 평균)")
    rows_df = pd.DataFrame(all_rows)
    print(f"    {'K':>3s} | {'Kalman P':>9s} | {'Kalman R':>9s} | {'Clip P':>7s} | {'Clip R':>7s} | {'ARIMA P':>8s} | {'ARIMA R':>8s}")
    for K in K_LIST:
        sub = rows_df[rows_df["K"] == K]
        kal_p  = sub["kal_hit"].sum()  / (len(sub) * K) * 100
        clip_p = sub["clip_hit"].sum() / (len(sub) * K) * 100
        ari_p  = sub["ari_hit"].sum()  / (len(sub) * K) * 100
        kal_r  = (sub["kal_hit"]  / sub["gt_size"].clip(lower=1)).mean() * 100
        clip_r = (sub["clip_hit"] / sub["gt_size"].clip(lower=1)).mean() * 100
        ari_r  = (sub["ari_hit"]  / sub["gt_size"].clip(lower=1)).mean() * 100
        print(f"    {K:>3d} | {kal_p:>8.1f}% | {kal_r:>8.1f}% | {clip_p:>6.1f}% | {clip_r:>6.1f}% | {ari_p:>7.1f}% | {ari_r:>7.1f}%")

    # 6. GT 날짜 Z-score 분포 비교
    print("\n[7] 전체 GT 날짜 Z-score 분포")
    all_kz, all_cz, all_az = [], [], []
    for date in GT_DATES:
        for city, g in results.items():
            sel = g[g["date"] == date]
            if len(sel):
                all_kz.append(float(sel["kalman_z"].iloc[0]))
                all_cz.append(float(sel["kalman_clip"].iloc[0]))
                all_az.append(float(sel["arima_z"].iloc[0]))
    kz = np.array(all_kz); cz = np.array(all_cz); az = np.array(all_az)
    print(f"    Kalman Z   : mean={kz.mean():+.3f}, std={kz.std():.3f}, max={kz.max():.2f}, >2: {(kz>2).sum()}개")
    print(f"    Kalman Clip: mean={cz.mean():+.3f}, std={cz.std():.3f}, max={cz.max():.2f}, >2: {(cz>2).sum()}개")
    print(f"    ARIMA  Z   : mean={az.mean():+.3f}, std={az.std():.3f}, max={az.max():.2f}, >2: {(az>2).sum()}개")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

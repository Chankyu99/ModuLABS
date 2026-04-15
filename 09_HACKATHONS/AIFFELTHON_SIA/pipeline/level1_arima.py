"""
Level 1 — ARIMA 기반 이상 징후 탐지 및 Kalman 비교 유틸리티
──────────────────────────────────────────────────────────────
- 도시별 갈등 지수 시계열에 대해 one-step-ahead ARIMA 예측 수행
- 예측 오차(residual)를 표준화하여 ARIMA Z-Score 산출
- 기존 Kalman 결과와 동일한 ground truth 기준으로 비교 가능
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from pipeline.config import MIN_HISTORY, get_risk_level
from pipeline.conflict_index import detect_anomalies as detect_anomalies_kalman


DEFAULT_ARIMA_ORDERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
)


def _prepare_signal(series: pd.Series, transform: str = "log1p") -> np.ndarray:
    """ARIMA 입력 시계열 전처리."""
    values = series.astype(float).fillna(0.0).values
    if transform == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    if transform == "none":
        return values
    raise ValueError(f"지원하지 않는 transform입니다: {transform}")


def fit_best_arima(
    train_series: pd.Series,
    candidate_orders: Iterable[tuple[int, int, int]] = DEFAULT_ARIMA_ORDERS,
) -> tuple[object | None, tuple[int, int, int] | None, float]:
    """후보 order 중 AIC가 가장 좋은 ARIMA 모델을 선택."""
    best_model = None
    best_order = None
    best_aic = np.inf

    for order in candidate_orders:
        try:
            fitted = ARIMA(
                train_series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        except Exception:
            continue

        if np.isfinite(fitted.aic) and fitted.aic < best_aic:
            best_model = fitted
            best_order = order
            best_aic = float(fitted.aic)

    return best_model, best_order, best_aic


def rolling_arima_anomaly(
    series: pd.Series,
    min_history: int = MIN_HISTORY,
    transform: str = "log1p",
    candidate_orders: Iterable[tuple[int, int, int]] = DEFAULT_ARIMA_ORDERS,
    min_residual_history: int = 10,
) -> pd.DataFrame:
    """도시별 시계열에 대해 rolling one-step ARIMA 이상 점수를 계산."""
    signal = _prepare_signal(series, transform=transform)
    n = len(signal)

    forecast = np.full(n, np.nan)
    residual = np.full(n, np.nan)
    residual_z = np.full(n, np.nan)
    order_used: list[str | None] = [None] * n

    if n < max(2, min_history):
        return pd.DataFrame(
            {
                "arima_forecast": forecast,
                "arima_residual": residual,
                "arima_z": residual_z,
                "arima_order": order_used,
            }
        )

    for t in range(min_history, n):
        train = pd.Series(signal[:t])
        model, order, _ = fit_best_arima(train, candidate_orders=candidate_orders)
        if model is None:
            continue

        try:
            yhat = float(model.forecast(steps=1)[0])
        except Exception:
            continue

        forecast[t] = yhat
        residual[t] = signal[t] - yhat
        order_used[t] = str(order)

        hist_residuals = residual[min_history:t]
        hist_residuals = hist_residuals[~np.isnan(hist_residuals)]
        if len(hist_residuals) < min_residual_history:
            continue

        sigma = float(np.std(hist_residuals))
        if sigma > 1e-8:
            residual_z[t] = residual[t] / sigma

    return pd.DataFrame(
        {
            "arima_forecast": forecast,
            "arima_residual": residual,
            "arima_z": residual_z,
            "arima_order": order_used,
        }
    )


def detect_anomalies_arima(
    city_daily: pd.DataFrame,
    target_date: str | None = None,
    min_history: int = MIN_HISTORY,
    transform: str = "log1p",
    candidate_orders: Iterable[tuple[int, int, int]] = DEFAULT_ARIMA_ORDERS,
) -> pd.DataFrame:
    """도시별 갈등 지수 시계열에 대해 ARIMA 기반 이상 징후를 계산."""
    results = []

    for city, group in city_daily.groupby("city"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < min_history:
            continue

        arima_result = rolling_arima_anomaly(
            group["conflict_index"],
            min_history=min_history,
            transform=transform,
            candidate_orders=candidate_orders,
        )

        group = group.copy()
        group["arima_forecast"] = arima_result["arima_forecast"]
        group["arima_residual"] = arima_result["arima_residual"]
        group["arima_z"] = arima_result["arima_z"].fillna(0.0)
        group["arima_order"] = arima_result["arima_order"]

        risk_info = group["arima_z"].apply(get_risk_level)
        group["risk_level"] = risk_info.apply(lambda x: x["level"])
        group["risk_label"] = risk_info.apply(lambda x: x["label"])
        group["risk_emoji"] = risk_info.apply(lambda x: x["emoji"])
        group["risk_guide"] = risk_info.apply(lambda x: x["guide"])
        group["is_anomaly"] = group["risk_level"] >= 1

        results.append(group)

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)
    if target_date:
        all_results = all_results[all_results["date"] == target_date]

    return all_results.sort_values(["date", "arima_z"], ascending=[True, False])


def _normalize_city(value: str) -> str:
    return str(value).strip().lower()


def extract_ground_truth_targets(
    ground_truth_df: pd.DataFrame,
    use_selection_decision: bool = True,
    allow_fallback_all_cities: bool = False,
) -> dict[str, set[str]]:
    """ground truth CSV에서 날짜별 positive 도시 셋을 만든다.

    우선순위:
    1. selection_decision == should_keep
    2. manual_label == TP
    3. allow_fallback_all_cities=True일 때만 해당 파일의 전체 city를 positive로 간주
    """
    gt = ground_truth_df.copy()
    gt["date"] = gt["date"].astype(str)
    truth_by_date: dict[str, set[str]] = defaultdict(set)

    for date, group in gt.groupby("date"):
        selected = pd.DataFrame()
        if use_selection_decision and "selection_decision" in group.columns:
            keep_mask = (
                group["selection_decision"]
                .fillna("")
                .astype(str)
                .str.lower()
                .eq("should_keep")
            )
            if keep_mask.any():
                selected = group.loc[keep_mask].copy()

        if selected.empty and "manual_label" in group.columns:
            tp_mask = group["manual_label"].fillna("").astype(str).str.upper().eq("TP")
            if tp_mask.any():
                selected = group.loc[tp_mask].copy()

        if selected.empty and allow_fallback_all_cities:
            selected = group.copy()

        for _, row in selected.iterrows():
            truth_by_date[date].add(_normalize_city(row["city"]))

    return dict(truth_by_date)


def evaluate_model_predictions(
    model_results: pd.DataFrame,
    score_col: str,
    ground_truth_df: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    allow_fallback_all_cities: bool = False,
) -> dict:
    """모델 결과를 ground truth와 비교해 Top-N 기준 랭킹 지표를 계산.

    평가 원칙:
    - Ground Truth는 명시적으로 positive 라벨된 도시만 사용한다.
    - 날짜별로 score 상위 N개 도시 안에 GT 도시가 몇 개 포함되는지 계산한다.
    """
    if model_results.empty:
        return {
            "dates_evaluated": 0,
            "avg_mrr": 0.0,
            "avg_rank_of_first_hit": None,
            "precision_at_k": {f"p@{k}": 0.0 for k in k_values},
            "recall_at_k": {f"r@{k}": 0.0 for k in k_values},
            "per_date": [],
        }

    truth_by_date = extract_ground_truth_targets(
        ground_truth_df,
        allow_fallback_all_cities=allow_fallback_all_cities,
    )
    eval_dates = sorted(set(model_results["date"].astype(str)) & set(truth_by_date))
    if not eval_dates:
        return {
            "dates_evaluated": 0,
            "avg_mrr": 0.0,
            "avg_rank_of_first_hit": None,
            "precision_at_k": {f"p@{k}": 0.0 for k in k_values},
            "recall_at_k": {f"r@{k}": 0.0 for k in k_values},
            "per_date": [],
        }

    precision_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    reciprocal_ranks = []
    first_ranks = []
    per_date = []
    max_k = max(k_values) if k_values else 10

    for date in eval_dates:
        day_truth = truth_by_date[date]
        day_pred = model_results.loc[model_results["date"].astype(str) == date].copy()
        day_pred = day_pred.sort_values(score_col, ascending=False)
        ranked_city_labels = day_pred["city"].astype(str).tolist()
        ranked_cities = [_normalize_city(city) for city in ranked_city_labels]

        first_hit_rank = None
        for idx, city in enumerate(ranked_cities, start=1):
            if city in day_truth:
                first_hit_rank = idx
                break

        if first_hit_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / first_hit_rank)
            first_ranks.append(first_hit_rank)

        day_metrics = {
            "date": date,
            "truth_count": len(day_truth),
            "truth_cities": sorted(day_truth),
            "prediction_count": len(ranked_cities),
            "first_hit_rank": first_hit_rank,
            f"top_{max_k}_predictions": ranked_city_labels[:max_k],
        }

        for k in k_values:
            topk = ranked_cities[:k]
            hits = len(set(topk) & day_truth)
            precision = hits / max(k, 1)
            recall = hits / max(len(day_truth), 1)
            precision_scores[k].append(precision)
            recall_scores[k].append(recall)
            topk_labels = ranked_city_labels[:k]
            matched_truth = sorted(set(topk) & day_truth)
            day_metrics[f"top_{k}"] = topk_labels
            day_metrics[f"hits@{k}"] = hits
            day_metrics[f"matched_truth@{k}"] = matched_truth
            day_metrics[f"p@{k}"] = precision
            day_metrics[f"r@{k}"] = recall

        per_date.append(day_metrics)

    return {
        "dates_evaluated": len(eval_dates),
        "avg_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "avg_rank_of_first_hit": float(np.mean(first_ranks)) if first_ranks else None,
        "precision_at_k": {
            f"p@{k}": float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
            for k in k_values
        },
        "recall_at_k": {
            f"r@{k}": float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
            for k in k_values
        },
        "per_date": per_date,
    }


def compare_kalman_vs_arima(
    city_daily: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    target_dates: list[str] | None = None,
    min_history: int = MIN_HISTORY,
    transform: str = "log1p",
    candidate_orders: Iterable[tuple[int, int, int]] = DEFAULT_ARIMA_ORDERS,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """동일한 갈등 지수 시계열에 대해 Kalman vs ARIMA 결과를 비교."""
    kalman_results = detect_anomalies_kalman(city_daily)
    arima_results = detect_anomalies_arima(
        city_daily,
        min_history=min_history,
        transform=transform,
        candidate_orders=candidate_orders,
    )

    if target_dates:
        target_dates = [str(d) for d in target_dates]
        kalman_results = kalman_results[kalman_results["date"].astype(str).isin(target_dates)]
        arima_results = arima_results[arima_results["date"].astype(str).isin(target_dates)]
        ground_truth_df = ground_truth_df[ground_truth_df["date"].astype(str).isin(target_dates)]

    kalman_eval = evaluate_model_predictions(
        kalman_results,
        score_col="innov_z",
        ground_truth_df=ground_truth_df,
        k_values=k_values,
    )
    arima_eval = evaluate_model_predictions(
        arima_results,
        score_col="arima_z",
        ground_truth_df=ground_truth_df,
        k_values=k_values,
    )

    return {
        "config": {
            "min_history": min_history,
            "transform": transform,
            "candidate_orders": [list(order) for order in candidate_orders],
            "k_values": list(k_values),
        },
        "kalman": {
            "score_col": "innov_z",
            "evaluation": kalman_eval,
        },
        "arima": {
            "score_col": "arima_z",
            "evaluation": arima_eval,
        },
    }

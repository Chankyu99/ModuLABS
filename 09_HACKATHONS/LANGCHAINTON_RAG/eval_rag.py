"""

eval_rag.py : RAG 파이프라인 중간 평가 모듈

- 프로젝트 의도대로 설계한 RAG 파이프라인이 실제로 잘 작동하는지 중간 점검
- 슬롯 추출, 누락 슬롯 판단, 검색 실행 등 핵심 로직이 의도대로 작동하는지 테스트
- 레이턴시 및 응답 품질도 간단히 체크

"""

# 필요한 라이브러리 임포트

import time
from pathlib import Path

import pandas as pd

from bot_logic import (
    extract_slots_and_map,
    check_missing_slots,
    retrieve_docs,
)

# 설정값 모음

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"
EVAL_DATASET_DIR = EVAL_DIR / "datasets"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

EVAL_DATASET_FILE = EVAL_DATASET_DIR / "eval_dataset.csv"
EVAL_RESULTS_FILE = EVAL_RESULTS_DIR / "eval_results.csv"
EVAL_SUMMARY_FILE = EVAL_RESULTS_DIR / "eval_summary.csv"
EVAL_NOISE_SUMMARY_FILE = EVAL_RESULTS_DIR / "eval_summary_by_noise_type.csv"

EVAL_LIMIT = 5              # 전체 평가 시 None으로 설정
RUN_RETRIEVAL_EVAL = True
RUN_ANSWER_EVAL = False
ENABLE_RAGAS = False        # 최종 답변 품질은 RAGAS 적용 

# 사용할 함수
def is_empty(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def normalize(value) -> str:
    if is_empty(value):
        return ""
    return str(value).strip().lower()


def bool_from_csv(value) -> bool:
    return str(value).strip().lower() == "true"


def empty_match(pred, expected) -> bool:
    pred_norm = normalize(pred)
    expected_norm = normalize(expected)

    if expected_norm == "":
        return pred_norm == ""

    return expected_norm in pred_norm or pred_norm in expected_norm


def list_contains(candidates, expected) -> bool | None:
    if is_empty(expected):
        return None

    expected_norm = normalize(expected)

    for candidate in candidates or []:
        candidate_norm = normalize(candidate)

        if expected_norm in candidate_norm or candidate_norm in expected_norm:
            return True

    return False


# 평가 데이터셋 로드
def load_eval_dataset(filepath: str | Path) -> pd.DataFrame:
   
    df = pd.read_csv(filepath)

    if EVAL_LIMIT is not None:
        return df.head(EVAL_LIMIT)

    return df

# 슬롯 추출 및 DB 매핑 평가
def run_slot_extraction(user_query: str) -> tuple[dict, dict, float]:
    
    start = time.time()

    updated_slots, mapped_items = extract_slots_and_map(
        user_message=user_query,
        chat_history=[],
        current_slots={},
    )

    latency = time.time() - start

    return updated_slots, mapped_items, latency

# 검색 평가
def run_retrieval_eval(slots: dict, mapped_items: dict, should_run: bool) -> tuple[list, bool | None, float]:
    """검색 평가를 실행할지 결정하고, 검색 결과와 latency를 반환한다."""
    if not should_run:
        return [], None, 0.0

    start = time.time()

    retrieved, all_mapping_failed = retrieve_docs(slots, mapped_items)

    latency = time.time() - start

    return retrieved, all_mapping_failed, latency

# 예상 슬롯과 예측 슬롯을 비교
def evaluate_slots(updated_slots: dict, row: pd.Series) -> dict:
    
    departure_correct = empty_match(
        updated_slots.get("departure"),
        row.get("expected_departure"),
    )
    arrival_correct = empty_match(
        updated_slots.get("arrival"),
        row.get("expected_arrival"),
    )
    item_correct = empty_match(
        updated_slots.get("item"),
        row.get("expected_item"),
    )

    return {
        "departure_correct": departure_correct,
        "arrival_correct": arrival_correct,
        "item_correct": item_correct,
        "slot_all_correct": departure_correct and arrival_correct and item_correct,
    }

# 예상 DB 항목과 예측 DB 항목이 일치하는지 평가
def evaluate_mapping(mapped_items: dict, row: pd.Series, pred_missing: bool) -> dict:
    """LLM이 고른 DB item이 기대 item과 맞는지 평가한다."""
    kr_mapping_hit = list_contains(
        mapped_items.get("KR", []),
        row.get("expected_kr_item"),
    )
    us_mapping_hit = list_contains(
        mapped_items.get("US", []),
        row.get("expected_us_item"),
    )

    valid_scores = [
        score for score in [kr_mapping_hit, us_mapping_hit]
        if score is not None
    ]

    if pred_missing:
        mapping_hit = None
    elif valid_scores:
        mapping_hit = any(valid_scores)
    else:
        mapping_hit = None

    return {
        "kr_mapping_hit": kr_mapping_hit,
        "us_mapping_hit": us_mapping_hit,
        "mapping_hit": mapping_hit,
    }

# 검색 결과에서 item 목록과 doc_id 목록을 추출
def extract_retrieved_metadata(retrieved: list[dict]) -> tuple[list[str], list[str]]:
   
    retrieved_items = []
    retrieved_doc_ids = []

    for result in retrieved:
        doc = result["doc"]
        metadata = doc.metadata

        retrieved_items.append(metadata.get("item", ""))
        retrieved_doc_ids.append(metadata.get("doc_id", ""))

    return retrieved_items, retrieved_doc_ids


# retrieval 결과가 예상 결과와 일치하는지 평가
def evaluate_retrieval(retrieved_items: list[str], row: pd.Series, pred_missing: bool) -> bool | None:
    
    expected_targets = [
        row.get("expected_kr_item"),
        row.get("expected_us_item"),
    ]

    valid_targets = [
        target for target in expected_targets
        if not is_empty(target)
    ]

    if pred_missing:
        return None

    if not valid_targets:
        return None

    return any(
        list_contains(retrieved_items, target)
        for target in valid_targets
    )

# evaluate_row를 짧게 다시 구성
def evaluate_row(row: pd.Series) -> dict:
    user_query = row["user_query"]
    total_start = time.time()

    updated_slots, mapped_items, slot_latency = run_slot_extraction(user_query)

    missing_question = check_missing_slots(updated_slots)
    pred_missing = missing_question is not None
    expected_missing = bool_from_csv(row["expected_missing"])

    should_run_retrieval = RUN_RETRIEVAL_EVAL and not pred_missing

    retrieved, all_mapping_failed, retrieval_latency = run_retrieval_eval(
        slots=updated_slots,
        mapped_items=mapped_items,
        should_run=should_run_retrieval,
    )

    total_latency = time.time() - total_start

    slot_scores = evaluate_slots(updated_slots, row)
    mapping_scores = evaluate_mapping(mapped_items, row, pred_missing)

    retrieved_items, retrieved_doc_ids = extract_retrieved_metadata(retrieved)
    retrieval_hit = evaluate_retrieval(retrieved_items, row, pred_missing)

    return {
        "id": row["id"],
        "case_type": row["case_type"],
        "user_query": user_query,

        "base_case_id": row.get("base_case_id"),
        "noise_type": row.get("noise_type"),
        "canonical_query": row.get("canonical_query"),

        "expected_departure": row.get("expected_departure"),
        "expected_arrival": row.get("expected_arrival"),
        "expected_item": row.get("expected_item"),
        "expected_kr_item": row.get("expected_kr_item"),
        "expected_us_item": row.get("expected_us_item"),
        "expected_missing": expected_missing,

        "pred_departure": updated_slots.get("departure"),
        "pred_arrival": updated_slots.get("arrival"),
        "pred_item": updated_slots.get("item"),
        "pred_quantity": updated_slots.get("quantity"),
        "pred_missing": pred_missing,
        "missing_question": missing_question,

        "mapped_kr": " | ".join(mapped_items.get("KR", [])),
        "mapped_us": " | ".join(mapped_items.get("US", [])),

        "retrieved_items": " | ".join(retrieved_items),
        "retrieved_doc_ids": " | ".join(retrieved_doc_ids),
        "num_retrieved": len(retrieved),
        "all_mapping_failed": all_mapping_failed,

        "missing_correct": pred_missing == expected_missing,
        "retrieval_hit": retrieval_hit,

        **slot_scores,
        **mapping_scores,

        "slot_latency_sec": round(slot_latency, 3),
        "retrieval_latency_sec": round(retrieval_latency, 3),
        "total_latency_sec": round(total_latency, 3),
    }

# 평가 결과 요약
def summarize(result_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []

    metric_cols = [
        "departure_correct",
        "arrival_correct",
        "item_correct",
        "slot_all_correct",
        "missing_correct",
        "mapping_hit",
        "retrieval_hit",
    ]

    for col in metric_cols:
        valid = result_df[col].dropna()

        if len(valid) > 0:
            metrics.append({
                "metric": col,
                "value": round(float(valid.mean()), 4),
                "count": len(valid),
            })

    metrics.extend([
        {
            "metric": "avg_slot_latency_sec",
            "value": round(float(result_df["slot_latency_sec"].mean()), 4),
            "count": len(result_df),
        },
        {
            "metric": "avg_retrieval_latency_sec",
            "value": round(float(result_df["retrieval_latency_sec"].mean()), 4),
            "count": len(result_df),
        },
        {
            "metric": "avg_total_latency_sec",
            "value": round(float(result_df["total_latency_sec"].mean()), 4),
            "count": len(result_df),
        },
    ])

    return pd.DataFrame(metrics)

# noise_type별로 평가 지표 추가
def summarize_by_noise_type(result_df: pd.DataFrame) -> pd.DataFrame:
    """noise_type별로 평가 지표를 요약한다."""
    if "noise_type" not in result_df.columns:
        return pd.DataFrame()

    return (
        result_df.groupby("noise_type")
        .agg(
            count=("id", "count"),
            item_accuracy=("item_correct", "mean"),
            slot_all_accuracy=("slot_all_correct", "mean"),
            missing_accuracy=("missing_correct", "mean"),
            mapping_hit=("mapping_hit", "mean"),
            retrieval_hit=("retrieval_hit", "mean"),
            avg_latency=("total_latency_sec", "mean"),
        )
        .reset_index()
    )

# 메인 함수: 전체 평가 실행
def main():
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_eval_dataset(EVAL_DATASET_FILE)

    print(f"평가 대상: {len(df)}개")
    print(f"RUN_RETRIEVAL_EVAL={RUN_RETRIEVAL_EVAL}")
    print(f"RUN_ANSWER_EVAL={RUN_ANSWER_EVAL}")
    print(f"ENABLE_RAGAS={ENABLE_RAGAS}")

    results = []

    for _, row in df.iterrows():
        print(f"[EVAL] {row['id']} | {row['case_type']} | {row['user_query']}")
        result = evaluate_row(row)
        results.append(result)

    result_df = pd.DataFrame(results)
    summary_df = summarize(result_df)
    noise_summary_df = summarize_by_noise_type(result_df)

    result_df.to_csv(EVAL_RESULTS_FILE, index=False, encoding="utf-8-sig")
    summary_df.to_csv(EVAL_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    if not noise_summary_df.empty:
        noise_summary_df.to_csv(
            EVAL_NOISE_SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n=== Evaluation Summary ===")
    print(summary_df)

    if not noise_summary_df.empty:
        print("\n=== Noise Type Summary ===")
        print(noise_summary_df)


if __name__ == "__main__":
    main()

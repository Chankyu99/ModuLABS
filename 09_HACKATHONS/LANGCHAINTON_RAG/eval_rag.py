import time
import pandas as pd

from bot_logic import (
    extract_slots_and_map,
    check_missing_slots,
    retrieve_docs,
)


def is_empty(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def normalize(value) -> str:
    if is_empty(value):
        return ""
    return str(value).strip().lower()


def bool_from_csv(value) -> bool:
    return str(value).strip().lower() == "true"


def empty_match(pred, expected) -> bool:
    """
    expected가 비어 있으면 pred도 비어 있어야 정답.
    expected가 있으면 문자열 포함 기준으로 비교.
    """
    pred_norm = normalize(pred)
    exp_norm = normalize(expected)

    if exp_norm == "":
        return pred_norm == ""

    return exp_norm in pred_norm or pred_norm in exp_norm


def list_contains(candidates, expected) -> bool | None:
    """
    expected가 비어 있으면 평가 제외(None).
    expected가 있으면 candidates 안에 포함되는지 확인.
    """
    if is_empty(expected):
        return None

    exp_norm = normalize(expected)

    for candidate in candidates or []:
        cand_norm = normalize(candidate)
        if exp_norm in cand_norm or cand_norm in exp_norm:
            return True

    return False


def evaluate_row(row: pd.Series) -> dict:
    user_query = row["user_query"]

    start = time.time()

    # 독립 평가: 이전 대화/슬롯 절대 넣지 않음
    updated_slots, mapped_items = extract_slots_and_map(
        user_message=user_query,
        chat_history=[],
        current_slots={},
    )

    slot_latency = time.time() - start

    missing_question = check_missing_slots(updated_slots)
    pred_missing = missing_question is not None
    expected_missing = bool_from_csv(row["expected_missing"])

    retrieved = []
    all_mapping_failed = None
    retrieval_latency = 0.0

    # 슬롯이 충분할 때만 검색 평가
    if not pred_missing:
        t_retrieve = time.time()
        retrieved, all_mapping_failed = retrieve_docs(updated_slots, mapped_items)
        retrieval_latency = time.time() - t_retrieve

    total_latency = time.time() - start

    # Slot 평가
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

    slot_all_correct = departure_correct and arrival_correct and item_correct
    missing_correct = pred_missing == expected_missing

    # Mapping 평가
    kr_mapping_hit = list_contains(
        mapped_items.get("KR", []),
        row.get("expected_kr_item"),
    )
    us_mapping_hit = list_contains(
        mapped_items.get("US", []),
        row.get("expected_us_item"),
    )

    valid_mapping_scores = [
        x for x in [kr_mapping_hit, us_mapping_hit]
        if x is not None
    ]

    if pred_missing:
        mapping_hit = None
    elif valid_mapping_scores:
        mapping_hit = any(valid_mapping_scores)
    else:
        mapping_hit = None

    # Retrieval 평가
    retrieved_items = []
    retrieved_doc_ids = []

    for r in retrieved:
        doc = r["doc"]
        meta = doc.metadata
        retrieved_items.append(meta.get("item", ""))
        retrieved_doc_ids.append(meta.get("doc_id", ""))

    expected_retrieval_targets = [
        row.get("expected_kr_item"),
        row.get("expected_us_item"),
    ]

    valid_targets = [
        target for target in expected_retrieval_targets
        if not is_empty(target)
    ]

    if pred_missing:
        retrieval_hit = None
    elif valid_targets:
        retrieval_hit = any(
            list_contains(retrieved_items, target)
            for target in valid_targets
        )
    else:
        retrieval_hit = None

    return {
        "id": row["id"],
        "case_type": row["case_type"],
        "user_query": user_query,

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

        "departure_correct": departure_correct,
        "arrival_correct": arrival_correct,
        "item_correct": item_correct,
        "slot_all_correct": slot_all_correct,
        "missing_correct": missing_correct,
        "kr_mapping_hit": kr_mapping_hit,
        "us_mapping_hit": us_mapping_hit,
        "mapping_hit": mapping_hit,
        "retrieval_hit": retrieval_hit,

        "slot_latency_sec": round(slot_latency, 3),
        "retrieval_latency_sec": round(retrieval_latency, 3),
        "total_latency_sec": round(total_latency, 3),
    }


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


def main():
    df = pd.read_csv("eval_dataset.csv")

    results = []

    for _, row in df.iterrows():
        print(f"[EVAL] {row['id']} | {row['case_type']} | {row['user_query']}")
        result = evaluate_row(row)
        results.append(result)

    result_df = pd.DataFrame(results)
    summary_df = summarize(result_df)

    result_df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv("eval_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Evaluation Summary ===")
    print(summary_df)

    print("\nSaved:")
    print("- eval_results.csv")
    print("- eval_summary.csv")


if __name__ == "__main__":
    main()
import pandas as pd

df = pd.read_csv("eval_results.csv")

summary_by_noise = (
    df.groupby("noise_type")
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

summary_by_noise.to_csv("eval_summary_by_noise_type.csv", index=False, encoding="utf-8-sig")
print(summary_by_noise)
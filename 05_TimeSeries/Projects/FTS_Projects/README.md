# ETH Minute Anomaly Monitoring Project

## Project Summary

This project reframes Ethereum 1-minute candle data from Upbit as a monitoring problem rather than a trading problem.
The goal is to detect abnormal states early, reduce false alerts, and explain how the final operating rule can be translated into a semiconductor process monitoring story.

The full workflow is organized as five notebooks:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_eda_feature_review.ipynb`
3. `notebooks/03_baseline_modeling.ipynb`
4. `notebooks/04_threshold_tuning_and_rules.ipynb`
5. `notebooks/05_monitoring_story_and_dashboard.ipynb`

## Why This Project

Instead of optimizing for trading profit, the project focuses on:

- anomaly detection
- change monitoring
- false alert reduction
- rule tuning for operations
- dashboard-oriented communication

This framing is much closer to semiconductor process monitoring, excursion detection, and manufacturing analytics than a pure finance prediction project.

## Data

- Source: `sub_upbit_eth_min_tick.csv`
- Market: Upbit ETH
- Frequency: 1-minute OHLCV
- Raw rows: `1,000,000`
- Raw period: `2017-09-25 03:00:00` to `2019-11-03 10:33:00`

Main raw columns:

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

Existing engineered feature set:

- `sub_upbit_eth_min_feature_labels.pkl`
- rows: `908,845`
- columns: `29`
- label candidate: `t_value`

## Workflow

### 1. Data Audit

The first notebook checks:

- timestamp continuity
- missing minute ratio
- gap distribution
- low-liquidity artifacts
- extreme event candidates

Key audit findings:

- missing minute ratio: about `9.73%`
- gap larger than 1 minute: about `5.59%`
- maximum gap: `4,924` minutes

Conclusion:
This dataset should not be treated as a perfectly continuous 1-minute series.

### 2. Feature Review

The second notebook reviews:

- monthly drift
- feature redundancy
- label separability

Key findings:

- price-level features drift the most over time
- some engineered features are highly redundant
- momentum and return features separate `t_value` better than volatility-only features

### 3. Baseline Modeling

The third notebook compares three baselines:

- rolling z-score
- EWMA
- Isolation Forest

Pseudo anomaly labels are defined from:

- extreme absolute 1-minute return
- extreme 30-minute realized volatility
- extreme 60-minute volume z-score

### 4. Threshold Tuning And Rules

The fourth notebook tunes:

- alert threshold
- cooldown rule

The result is two operating modes:

- `balanced_selected`
- `conservative_selected`

### 5. Monitoring Story And Dashboard

The fifth notebook converts the modeling result into:

- monitoring KPIs
- daily summary tables
- dashboard mockups
- semiconductor-friendly storytelling

## Final Result

Test-set comparison from the tuned rule notebook:

| config | point_f1 | point_fpr | event_f1 | false_alerts_per_day |
|---|---:|---:|---:|---:|
| baseline_zscore | 0.2323 | 0.00947 | 0.4493 | 8.5592 |
| balanced_selected | 0.2510 | 0.00531 | 0.4461 | 4.5137 |
| conservative_selected | 0.2143 | 0.00266 | 0.4100 | 2.3209 |

Interpretation:

- `balanced_selected` improves point-level F1 while cutting false alerts per day by about `47%` versus baseline.
- `conservative_selected` is the quietest operating rule and cuts false alerts per day by about `73%`, but misses more events.

Daily monitoring summary:

- test period days: `204`
- average daily pseudo events: `8.82`
- average daily baseline alerts: `14.18`
- average daily balanced alerts: `8.64`
- average daily conservative alerts: `4.68`
- balanced missed-event days: `0`
- conservative missed-event days: `2`

## Recommended Operating Rule

Recommended default:

- `balanced_selected`

Why:

- lower false alert burden than baseline
- no missed day among days with pseudo events in the test summary
- better point-level F1 than baseline
- easier to justify as a production monitoring default

Optional conservative mode:

- `conservative_selected`

Why:

- useful when alert fatigue cost is very high
- good backup option for night shift or low-touch operations

## Semiconductor Translation

This project can be explained in semiconductor language as follows:

- price shock -> process excursion
- volume burst -> sudden equipment or sensor signal burst
- volatility spike -> process instability increase
- false alert -> false alarm
- cooldown rule -> duplicate alarm suppression
- balanced rule -> detection-oriented operating mode
- conservative rule -> low-false-alarm operating mode

## Repository Structure

```text
FTS_Projects/
├── README.md
├── docs/
│   └── PRESENTATION_OUTLINE.md
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_feature_review.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_threshold_tuning_and_rules.ipynb
│   └── 05_monitoring_story_and_dashboard.ipynb
├── presentation/
│   ├── build/
│   ├── final/
│   │   ├── eth_monitoring_portfolio_draft.pptx
│   │   └── narrative_plan.md
│   └── reference-images/
└── outputs/
    ├── 01_data_audit/
    ├── 02_eda_feature_review/
    ├── 03_baseline_modeling/
    ├── 04_threshold_tuning_and_rules/
    └── 05_monitoring_story_and_dashboard/
```

## How To Review This Project

If you want to follow the project quickly:

1. read this `README.md`
2. open `notebooks/05_monitoring_story_and_dashboard.ipynb`
3. review `notebooks/04_threshold_tuning_and_rules.ipynb`
4. go back to `notebooks/01_data_audit.ipynb` if you want the full data logic

If you want to reproduce the full story:

1. run `notebooks/01_data_audit.ipynb`
2. run `notebooks/02_eda_feature_review.ipynb`
3. run `notebooks/03_baseline_modeling.ipynb`
4. run `notebooks/04_threshold_tuning_and_rules.ipynb`
5. run `notebooks/05_monitoring_story_and_dashboard.ipynb`

## Limitations

- pseudo anomaly labels are rule-based, not ground truth events
- the dataset has time gaps and low-liquidity periods
- this is a monitoring prototype, not a live trading system
- further gains are possible with event merging, adaptive thresholds, and richer features

## Portfolio Message

This project shows that I can:

- audit noisy time-series data carefully
- design anomaly labels when ground truth is limited
- compare statistical and machine-learning baselines
- tune rules for operational false-alert control
- translate a finance dataset into a manufacturing monitoring story

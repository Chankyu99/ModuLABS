# Presentation Outline

## Slide 1. Title

Title:
`ETH 1-Minute Time Series Based Anomaly Monitoring And Explainable Alert Rule Design`

Subtitle:
`From high-frequency market data to a semiconductor-style monitoring portfolio`

Speaker note:
This project started with Upbit Ethereum minute data, but I did not approach it as a trading problem.
I reframed it as an anomaly monitoring problem, focused on detecting abnormal states early and reducing false alerts.

## Slide 2. Why This Problem

Slide message:

- I wanted to build a project closer to monitoring and operations than price prediction.
- Semiconductor data roles value anomaly detection, drift awareness, and false alarm control.
- High-frequency crypto time series can be reinterpreted as a process monitoring problem.

Speaker note:
Instead of asking which model predicts the next price best, I asked which rule detects abnormal states with manageable alert volume.
That framing is much closer to process excursion monitoring.

## Slide 3. Data And Audit

Slide message:

- Raw data: `1,000,000` rows of 1-minute OHLCV
- Period: `2017-09-25` to `2019-11-03`
- Missing minute ratio: about `9.73%`
- Gap > 1 minute ratio: about `5.59%`
- Max gap: `4,924` minutes

Speaker note:
The first important finding was that the dataset is not a perfectly continuous 1-minute series.
That meant I needed to separate true abnormal behavior from artifacts caused by gaps or low-liquidity periods.

## Slide 4. Feature Review

Slide message:

- Reviewed monthly drift, feature redundancy, and label separability
- Price-level features drifted the most
- Momentum and return features separated labels better than volatility-only features
- Some features were highly redundant

Speaker note:
This step mattered because I wanted a model that is explainable and stable, not just accurate on one split.
The feature review showed that momentum and short-horizon return features were more informative than many redundant technical indicators.

## Slide 5. Baseline Modeling

Slide message:

- Baselines compared:
  - rolling z-score
  - EWMA
  - Isolation Forest
- Pseudo anomaly labels built from:
  - extreme absolute return
  - volatility spike
  - volume burst

Speaker note:
I intentionally started with simple baselines before rule tuning.
The goal was not to prove one complex model is best, but to understand the trade-off between sensitivity and false alert burden.

## Slide 6. Rule Tuning

Slide message:

- Tuned threshold and cooldown on validation data
- Selected two operating modes:
  - `balanced_selected`
  - `conservative_selected`
- Evaluated both point metrics and event metrics

Speaker note:
I introduced cooldown because repeated alerts from the same event are costly in operations.
I also evaluated event-level performance, not just point-level performance, because monitoring systems are judged on incidents, not isolated timestamps.

## Slide 7. Final Test Result

Slide table:

| config | point_f1 | event_f1 | false_alerts_per_day |
|---|---:|---:|---:|
| baseline_zscore | 0.232 | 0.449 | 8.56 |
| balanced_selected | 0.251 | 0.446 | 4.51 |
| conservative_selected | 0.214 | 0.410 | 2.32 |

Slide message:

- `balanced_selected` reduced false alerts/day by about `47%` vs baseline
- `balanced_selected` also improved point-level F1
- `conservative_selected` reduced false alerts/day by about `73%`

Speaker note:
The key result is that I did not just improve a score.
I reduced alert burden substantially while keeping event coverage at a useful level, which is exactly the kind of trade-off that matters in monitoring operations.

## Slide 8. Monitoring Dashboard View

Slide message:

- Average daily pseudo events: `8.82`
- Average daily alerts:
  - baseline: `14.18`
  - balanced: `8.64`
  - conservative: `4.68`
- Balanced missed-event days: `0`
- Conservative missed-event days: `2`

Speaker note:
This slide shows why I recommend the balanced rule as the default operating mode.
It brings daily alert volume much closer to the actual event level without missing event days in the test summary.

## Slide 9. Semiconductor Translation

Slide message:

- price shock -> process excursion
- volume burst -> sudden sensor or equipment signal burst
- volatility spike -> process instability increase
- false alert -> false alarm
- cooldown rule -> duplicate alarm suppression

Speaker note:
Although the raw data is financial, the workflow is directly relevant to semiconductor analytics:
data audit, drift-aware modeling, anomaly detection, false alarm control, and operational rule design.

## Slide 10. What I Learned

Slide message:

- Data quality rules matter before model complexity
- Monitoring systems should optimize for operational usefulness, not just model score
- It is valuable to present both a balanced mode and a conservative mode

Speaker note:
The biggest lesson was that a good monitoring project is not only about detecting more anomalies.
It is about designing a rule set that people can actually trust and use.

## Slide 11. Closing

Slide message:

- Built an end-to-end anomaly monitoring prototype
- Tuned rules for false alert reduction
- Translated results into an operations-ready story

Speaker note:
If I extend this project further, the next step would be adaptive thresholds or a live dashboard.
But as a portfolio piece, I believe the current project already demonstrates strong monitoring, analytics, and communication skills.

## Short Q&A Prep

### Q1. Why did you not focus on trading return?

Suggested answer:
Because my target role is closer to monitoring and anomaly detection than alpha generation.
I wanted the project to show data audit, event detection, false alarm control, and operational rule design.

### Q2. Why did you use pseudo labels?

Suggested answer:
There was no verified anomaly ground truth in the dataset.
So I defined rare events using extreme return, volatility, and volume behavior, and then kept the evaluation consistent across train, validation, and test.

### Q3. Why is balanced your recommended rule?

Suggested answer:
It reduced false alerts per day by about 47 percent versus baseline while also improving point-level F1.
It was the best compromise between detection coverage and operating burden.

### Q4. Why keep a conservative rule?

Suggested answer:
In real operations, some teams care more about reducing false alarms than maximizing sensitivity.
The conservative mode gives a lower-alert option for that environment.

# 이더리움 1분봉 기반 이상 상태 모니터링 프로젝트

## 프로젝트 개요

- Upbit 이더리움 1분봉 데이터를 사용해 `트레이딩 전략`이 아니라 `이상 상태 조기경보 시스템`을 설계  
- 핵심 목표는 이상 징후를 가능한 한 빨리 탐지하고, 동시에 오경보를 줄여 실제 운영에 가까운 규칙을 만드는 것

## 사용 데이터

원본 데이터:

- 파일: `sub_upbit_eth_min_tick.csv`
- 시장: Upbit ETH
- 주기: 1분봉 OHLCV
- 행 수: `1,000,000`
- 기간: `2017-09-25 03:00:00` ~ `2019-11-03 10:33:00`

원본 컬럼:

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

기존 피처 파일 (시계열 노드 학습에서 제작한 피처 파일)

- 파일: `sub_upbit_eth_min_feature_labels.pkl`
- 행 수: `908,845`
- 컬럼 수: `29`
- 라벨 후보: `t_value`

## 프로젝트 과정

### 1. 데이터 검증 

원본 1분봉 데이터(`sub_upbit_eth_min_tick.csv`)가 실제로 모델링에 적합한지부터 확인

주요 항목

- 시간축 연속성
- 결측 분 비율
- gap 분포
- 저유동성 구간
- 극단 이벤트 후보

핵심 결과

- 누락 분 비율: 약 `9.73%`
- 1분 초과 gap 비율: 약 `5.59%`
- 최대 gap: `4,924분`

> 따라서 프로젝트에 사용된 데이터는 “완전한 연속 1분 시계열”로 다루기보다, 공백과 저유동성 구간을 분리해서 보는 것이 더 적절했습니다.

### 2. 피처 EDA

기존 피처 파일(`sub_upbit_eth_min_feature_labels.pkl`)에서 미리 생성된 feature들 중 쓸모가 있을 만한 feature를 선별

주요 점검 항목

- 월별 drift
- 피처 간 상관과 중복
- `t_value`와의 분리력

핵심 결과

- 가격 레벨 feature의 drift가 가장 큼
- 일부 기술지표는 높은 중복 정보를 가짐
- `momentum`, `return` 계열 feature가 `t_value`와 더 잘 갈림

> 즉, 피처를 무조건 많이 쓰는 것보다 중복을 줄이고, 목표에 맞는 feature를 고르는 것이 더 중요하다는 점을 확인

### 3. 베이스라인 모델링

- 단순 baseline으로 rolling z-score, EWMA, Isolation Forest를 먼저 비교

- pseudo anomaly label은 절대 1분 수익률, 30분 실현 변동성, 60분 거래량 z-score를 기준으로 정의

> 최고 성능 모델을 바로 찾는 것도 좋지만, 베이스라인 모델링을 통해 `탐지형 모델`과 `보수형 모델`이 어떤 trade-off를 가지는지 확인

### 4. 임계값 튜닝과 룰 설계

baseline score를 실제 운영 가능한 경보 규칙으로 바꾸는 작업을 진행

alert threshold, cooldown rule 를 튜닝해 최종적으로 balanced, conservative 두 가지 rule 수립

point-level F1만 보는 것이 아니라 event_f1, false_alerts_per_day, point_fpr 를 함께 고려

> 즉, “얼마나 많이 잡는가”뿐 아니라 “얼마나 운영 가능한가”를 같이 보도록 설계

### 5. 모니터링 대시보드

최종 rule 결과를 대시보드로 정리

- monitoring KPI summary
- daily monitoring summary
- dashboard mockup

## 결론

### 튜닝 후 test 구간 비교 결과

| config | point_f1 | point_fpr | event_f1 | false_alerts_per_day |
|---|---:|---:|---:|---:|
| baseline_zscore | 0.2323 | 0.00947 | 0.4493 | 8.5592 |
| balanced_selected | 0.2510 | 0.00531 | 0.4461 | 4.5137 |
| conservative_selected | 0.2143 | 0.00266 | 0.4100 | 2.3209 |

- `balanced_selected`는 baseline 대비 point-level F1이 좋아졌습니다.
- 동시에 false alerts per day를 약 `47%` 줄였습니다.
- `conservative_selected`는 가장 조용한 rule이며 false alerts per day를 약 `73%` 줄였습니다.
- 대신 conservative는 더 많은 이벤트를 놓칠 수 있습니다.

### 튜닝 후 test 구간 기준 daily summary:

- 운영 일수: `204일`
- 일평균 pseudo event: `8.82건`
- 일평균 baseline alert: `14.18건`
- 일평균 balanced alert: `8.64건`
- 일평균 conservative alert: `4.68건`
- balanced missed day: `0일`
- conservative missed day: `2일`

### 주요 수치 요약

| 항목 | 값 |
|---|---:|
| raw 데이터 행 수 | 1,000,000 |
| feature 파일 행 수 | 908,845 |
| 누락 분 비율 | 9.73% |
| 최종 모델링 데이터 행 수 | 832,228 |
| 기본 운영안 | balanced_selected |
| 보수 운영안 | conservative_selected |

### 주요 시각화

#### 1. 최종 rule KPI 비교

`balanced_selected`는 baseline보다 point F1을 개선하면서 false alert/day를 크게 낮췄고,  
`conservative_selected`는 가장 낮은 경보량을 만드는 대신 일부 이벤트를 더 놓치는 구조입니다.

![Rule KPI 비교](docs/assets/readme_kpi_comparison.png)

#### 2. 일별 모니터링 추이

일별 집계로 보면 `balanced_selected`의 경보량이 실제 pseudo event 규모와 가장 비슷하게 맞춰집니다.  
즉, 너무 많이 울리지 않으면서도 운영 누락을 최소화하는 쪽에 가깝습니다.

![일별 모니터링 추이](docs/assets/readme_daily_monitoring.png)

#### 3. 이벤트 창 예시

같은 이벤트 창에서도 baseline, balanced, conservative의 반응 강도가 다릅니다.  
이 그림은 threshold와 cooldown 조정이 실제 경보 개수에 어떤 영향을 주는지 보여줍니다.

![이벤트 창 예시](docs/assets/readme_event_window.png)

> 최종적으로는 `balanced_selected` + `conservative_selected`를 함께 사용하는 것을 추천

- baseline보다 경보 부담이 낮음
- point-level F1이 baseline보다 좋음
- test summary에서 이벤트가 있었던 날을 놓치지 않음
- 운영 기본 모드로 설명하기 좋음

- 오경보 비용이 매우 큰 환경에 적합
- 야간 모드, 저터치 운영, 보수적 모니터링 모드로 설명 가능

## 배운 점

- 데이터 품질 문제를 먼저 정리하지 않으면 뒤의 모델 성능 해석이 쉽게 왜곡된다.
- 이상탐지에서는 모델 성능만큼이나 `threshold`, `cooldown`, `false alert/day` 같은 운영 규칙이 중요하다.
- point-level 성능과 event-level 성능은 다를 수 있어서, 두 지표를 분리해서 보는 습관이 필요하다.
- 좋은 운영안은 하나만 있는 것이 아니라, `balanced`와 `conservative`처럼 운영 목적에 따라 여러 모드로 제안하는 것이 더 실무적이다.
- 같은 분석이라도 어떤 언어로 설명하느냐에 따라 프로젝트의 직무 적합도가 크게 달라진다.

## 폴더 구조

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

## 한계

- pseudo anomaly label은 rule-based label이며 실제 ground truth 이벤트는 아닙니다.
- 데이터에는 시간 gap과 저유동성 구간이 존재합니다.
- 이 프로젝트는 live trading 시스템이 아니라 monitoring prototype입니다.
- adaptive threshold, richer feature, event merging 개선 여지가 남아 있습니다.

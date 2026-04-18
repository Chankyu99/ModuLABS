# 이더리움 1분봉 기반 이상 상태 모니터링 프로젝트

## 프로젝트 한 줄 소개

Upbit 이더리움 1분봉 데이터를 사용해 `트레이딩 전략`이 아니라 `이상 상태 조기경보 시스템`을 설계한 프로젝트입니다.  
핵심 목표는 이상 징후를 가능한 한 빨리 탐지하고, 동시에 오경보를 줄여 실제 운영에 가까운 규칙을 만드는 것입니다.

이 프로젝트는 금융 데이터를 사용했지만, 전체 흐름은 `반도체 공정 모니터링`, `이상 감지`, `운영 규칙 튜닝`, `오경보 제어` 문제로 자연스럽게 번역할 수 있도록 설계했습니다.

## 왜 이 프로젝트를 했는가

이 프로젝트는 단순한 수익률 예측보다 아래 역량을 보여주는 데 초점을 맞췄습니다.

- 시계열 데이터 품질 점검
- 이상 상태 탐지
- 분포 이동과 drift 해석
- false alert 감소를 위한 rule tuning
- 운영 관점의 KPI와 대시보드 설계

즉, “모델을 하나 잘 돌렸다”보다 “현업에서 실제로 쓸 수 있는 모니터링 규칙을 어떻게 만들 것인가”에 더 가까운 프로젝트입니다.

## 데이터

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

기존 피처 파일:

- 파일: `sub_upbit_eth_min_feature_labels.pkl`
- 행 수: `908,845`
- 컬럼 수: `29`
- 라벨 후보: `t_value`

## 프로젝트 흐름

전체 프로젝트는 아래 5개 노트북으로 구성했습니다.

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_eda_feature_review.ipynb`
3. `notebooks/03_baseline_modeling.ipynb`
4. `notebooks/04_threshold_tuning_and_rules.ipynb`
5. `notebooks/05_monitoring_story_and_dashboard.ipynb`

### 1. 데이터 검증

첫 번째 노트북에서는 원본 1분봉 데이터가 실제로 모델링에 적합한지부터 확인했습니다.

주요 점검 항목:

- 시간축 연속성
- 결측 분 비율
- gap 분포
- 저유동성 구간
- 극단 이벤트 후보

핵심 결과:

- 누락 분 비율: 약 `9.73%`
- 1분 초과 gap 비율: 약 `5.59%`
- 최대 gap: `4,924분`

즉, 이 데이터는 “완전한 연속 1분 시계열”로 다루기보다, 공백과 저유동성 구간을 분리해서 보는 것이 더 적절했습니다.

### 2. 피처 EDA 검토

두 번째 노트북에서는 기존 피처 파일을 그대로 믿기보다, 실제로 어떤 feature가 쓸 만한지 검토했습니다.

주요 점검 항목:

- 월별 drift
- 피처 간 상관과 중복
- `t_value`와의 분리력

핵심 결과:

- 가격 레벨 feature의 drift가 가장 큼
- 일부 기술지표는 높은 중복 정보를 가짐
- `momentum`, `return` 계열 feature가 `t_value`와 더 잘 갈림

즉, 피처를 무조건 많이 쓰는 것보다 `중복을 줄이고`, `목표에 맞는 feature를 고르는 것`이 더 중요하다는 점을 확인했습니다.

### 3. 베이스라인 모델링

세 번째 노트북에서는 복잡한 모델로 바로 가지 않고, 먼저 단순 baseline을 비교했습니다.

비교한 모델:

- `rolling z-score`
- `EWMA`
- `Isolation Forest`

pseudo anomaly label은 아래 3가지를 기준으로 정의했습니다.

- 극단적인 절대 1분 수익률
- 극단적인 30분 실현 변동성
- 극단적인 60분 거래량 z-score

이 단계의 목적은 “최고 성능 모델 찾기”보다 `탐지형 모델`과 `보수형 모델`이 어떤 trade-off를 가지는지 확인하는 것이었습니다.

### 4. 임계값 튜닝과 룰 설계

네 번째 노트북에서는 baseline score를 실제 운영 가능한 경보 규칙으로 바꾸는 작업을 진행했습니다.

튜닝한 요소:

- alert threshold
- cooldown rule

최종적으로 두 가지 운영안을 만들었습니다.

- `balanced_selected`
- `conservative_selected`

이 단계에서 핵심은 point-level F1만 보는 것이 아니라 아래 지표를 함께 보는 것이었습니다.

- `event_f1`
- `false_alerts_per_day`
- `point_fpr`

즉, “얼마나 많이 잡는가”뿐 아니라 “얼마나 운영 가능한가”를 같이 보도록 설계했습니다.

### 5. 모니터링 스토리와 대시보드

다섯 번째 노트북에서는 최종 rule 결과를 분석 결과로 끝내지 않고, 실제 운영 화면과 포트폴리오 스토리로 정리했습니다.

정리한 내용:

- monitoring KPI summary
- daily monitoring summary
- dashboard mockup
- 반도체 공정 모니터링 언어로의 번역

이 단계에서 프로젝트는 단순 모델링 실습이 아니라 `운영형 이상 감지 포트폴리오`로 완성됩니다.

## 최종 결과

튜닝 후 test 구간 비교 결과는 아래와 같습니다.

| config | point_f1 | point_fpr | event_f1 | false_alerts_per_day |
|---|---:|---:|---:|---:|
| baseline_zscore | 0.2323 | 0.00947 | 0.4493 | 8.5592 |
| balanced_selected | 0.2510 | 0.00531 | 0.4461 | 4.5137 |
| conservative_selected | 0.2143 | 0.00266 | 0.4100 | 2.3209 |

핵심 해석:

- `balanced_selected`는 baseline 대비 point-level F1이 좋아졌습니다.
- 동시에 false alerts per day를 약 `47%` 줄였습니다.
- `conservative_selected`는 가장 조용한 rule이며 false alerts per day를 약 `73%` 줄였습니다.
- 대신 conservative는 더 많은 이벤트를 놓칠 수 있습니다.

## 운영 관점 요약

test 구간 기준 daily summary:

- 운영 일수: `204일`
- 일평균 pseudo event: `8.82건`
- 일평균 baseline alert: `14.18건`
- 일평균 balanced alert: `8.64건`
- 일평균 conservative alert: `4.68건`
- balanced missed day: `0일`
- conservative missed day: `2일`

즉, `balanced_selected`는 실제 이벤트 규모와 가장 비슷한 경보량을 만들면서, 이벤트가 있었던 날을 놓치지 않는 기본 운영안으로 해석할 수 있습니다.

## 추천 운영안

기본 추천안:

- `balanced_selected`

이유:

- baseline보다 경보 부담이 낮음
- point-level F1이 baseline보다 좋음
- test summary에서 이벤트가 있었던 날을 놓치지 않음
- 운영 기본 모드로 설명하기 좋음

보수 운영안:

- `conservative_selected`

이유:

- 오경보 비용이 매우 큰 환경에 적합
- 야간 모드, 저터치 운영, 보수적 모니터링 모드로 설명 가능

## 반도체 공정 데이터 직무와의 연결

이 프로젝트는 아래처럼 제조/공정 언어로 번역할 수 있습니다.

- 가격 급등락 -> 공정 excursion 또는 이상 상태
- 거래량 burst -> 센서/설비 신호 급증
- 변동성 spike -> 공정 안정성 저하
- false alert -> 오경보
- cooldown rule -> 중복 알람 억제 규칙
- balanced rule -> 탐지형 운영안
- conservative rule -> 보수형 운영안

즉, 이 프로젝트는 금융 데이터셋을 사용했지만 실제로 보여주는 역량은 아래와 더 가깝습니다.

- 공정 데이터 품질 점검
- 이상 상태 탐지
- rule-based monitoring
- false alarm 제어
- 운영 KPI 설계

## 현재 폴더 구조

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

## 빠르게 보는 방법

프로젝트를 빠르게 이해하고 싶다면:

1. `README.md`를 읽습니다.
2. `notebooks/05_monitoring_story_and_dashboard.ipynb`를 먼저 봅니다.
3. `notebooks/04_threshold_tuning_and_rules.ipynb`에서 rule 선택 과정을 확인합니다.
4. 필요하면 `notebooks/01_data_audit.ipynb`로 돌아가 데이터 품질 문제를 확인합니다.

전체 흐름을 순서대로 재현하고 싶다면:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_eda_feature_review.ipynb`
3. `notebooks/03_baseline_modeling.ipynb`
4. `notebooks/04_threshold_tuning_and_rules.ipynb`
5. `notebooks/05_monitoring_story_and_dashboard.ipynb`

## 한계

- pseudo anomaly label은 rule-based label이며 실제 ground truth 이벤트는 아닙니다.
- 데이터에는 시간 gap과 저유동성 구간이 존재합니다.
- 이 프로젝트는 live trading 시스템이 아니라 monitoring prototype입니다.
- adaptive threshold, richer feature, event merging 개선 여지가 남아 있습니다.

## 이 프로젝트로 보여줄 수 있는 역량

이 프로젝트는 제가 아래 역량을 갖추고 있다는 점을 보여줍니다.

- noisy time-series 데이터 감사
- ground truth가 없는 상황에서의 anomaly label 설계
- 통계적 baseline과 ML baseline 비교
- 운영 관점의 false alert 제어
- 금융 데이터를 제조 모니터링 언어로 번역하는 커뮤니케이션 능력

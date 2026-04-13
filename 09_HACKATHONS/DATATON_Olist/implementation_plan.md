# Olist Market Entry Strategy: Visualization Architecture Redesign

본 계획은 기존의 단순 지표 시각화를 넘어서, **"왜 우리는 지금 당장 RJ 지역에 Health & Beauty 전용 물류망을 구축해야 하는가?"** 라는 강력한 비즈니스 스토리를 설득력 있게 전달하기 위해 시각화 구성을 바닥부터 새롭게 설계(Zero-base Redesign)합니다.

## User Review Required

> [!WARNING]
> 이 계획은 기존 `charts.py`의 함수 시그니처와 출력물을 완전히 새로운 차트들로 교체합니다. 만약 기존 데이터 정제나 산출물 파이프라인(예: `build_preprocessing_assets.py`)이 깨지는 것을 원치 않으신다면, `charts_v2.py`라는 새로운 파일로 독립 구축할지 의견 부탁드립니다.

---

## 전략적 내러티브 기반 시각화 파이프라인 (Proposed Changes)

전체 프레젠테이션의 기승전결(문제 제기 -> 타겟 지역 선정 -> 타겟 카테고리 선정 -> 실행 방안)을 담당하는 5개의 메인 시각화를 제안합니다.

### 1. The Core Bottleneck (문제 제기)
* **목적:** "물류 지연이 수익성(고객 이탈 및 악플)에 미치는 치명적 영향"을 강조합니다.
* **시각화 기법: Delivery Impact Slope Chart (슬로프 차트)**
  * 배송 형태(조기, 정시, 지연)에 따라 리뷰(1~5점) 분포가 어떻게 극적으로 무너지는지를 보여주는 역동적인 슬로프/밀도 차트 도입. 기존의 밋밋한 바 차트를 대체합니다.

### 2. The Blue Ocean Map (타겟 지역 탐색)
* **목적:** 브라질 전역에서 수요는 많으나 서비스 질이 낮아 '블루오션'이 될 수 있는 지역을 지도상에 직관적으로 매핑합니다.
* **시각화 기법: Unmet Demand Hexbin & Topology Map**
  * 리뷰 점수 대비 수요(주문수)를 맵핑한 지도로, 단순 스캐터가 아닌 육각 격자(Hexbin) 또는 밀도맵을 활용해 RJ(Rio de Janeiro) 지역이 시각적으로 선명하게 '핵심 기회 창출지'로 타오르는 듯한 디자인 구현.

### 3. Spiderweb: SP vs. RJ (타겟 지역 확정)
* **목적:** 이미 포화 상태인 상파울루(SP)와 비교하여 리우데자네이루(RJ)의 잠재력을 설득.
* **시각화 기법: Market Potential Radar Chart**
  * 두 지역의 지표 (총 수요, 객단가, 배송 지연율, 저평점 비율, 확장성)를 Radar 차트로 비교하여 RJ가 '물류만 해결되면 터지는 폭발적 시장'임을 입증합니다.

### 4. BCB Matrix (타겟 카테고리 선정)
* **목적:** RJ의 열악한 배송 환경을 극복하기 위해 물리적 제약은 적고 수익성은 높은 카테고리를 찾습니다.
* **시각화 기법: Logistics Complexity vs. Profitability Quadrant (버블 차트 기반의 매트릭스)**
  * X축: 물류 복잡도 (상품의 부피+무게 등)
  * Y축: 수익성 (건당 매출)
  * 4분면으로 나누어 우상단(고수익-저난이도)에 위치한 **Health & Beauty**가 가장 완벽한 런칭 아이템임을 증명하는 차트입니다.

### 5. Final Strategic Playbook (결론 및 제안)
* **목적:** 경영진을 위한 원페이지 서머리.
* **시각화 기법: Executive Summary Info-card**
  * 이전에 텍스트로만 구성되었던 `save_recommendation_card`를 발전시켜, 시각적 인포그래픽 요소와 아이콘 뱃지가 결합된 세련된 형태의 전략 로드맵 렌더링.

---

## Open Questions

> [!IMPORTANT]
> **데이터 구성원(Data Dependencies) 여부**
> 전략 시나리오(예: 물류 복잡성인 무게/부피, 지연시간 등)를 구현하기 위해 기존 테이블(`orders`, `products`, `reviews`, `geolocation`)의 새로운 컬럼(예: `product_weight_g`, `product_volume`)을 집계하는 데이터 파이프라인의 변경이 필요합니다. 
> 
> 1. 시각화 코드(그리는 로직)에 집중하여 가상의 Dummy 또는 Aggregation 로직을 작성해도 괜찮습니까? 
> 2. 아니면 제가 실제 데이터 전처리 코드(`data.py` 또는 `build_...py`)까지 파고들어서 완전한 End-to-End로 구축하기를 원하시나요?

---

## Verification Plan

### 시각화 에셋 생성 테스트
- 새롭게 작성할 `draw_strategic_charts.py` 런너 스크립트를 작성합니다.
- Olist 데이터셋을 로드하고, 제안한 5종 전략 차트를 `Outputs/Strategic_Charts/` 폴더에 렌더링.
- 최종 산출된 PNG 이미지들을 Artifact 형태로 Walkthrough에서 리포트하여, 단순한 '그래프'가 아니라 한 편의 '비즈니스 덱(Deck)'이 시각적으로 어떻게 완성되었는지 검증 및 시연합니다.

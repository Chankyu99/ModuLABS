# 고객 세그먼테이션 (RFM Segmentation)

## 프로젝트 개요
본 프로젝트는 e-커머스 거래 데이터를 활용하여 **고객 세그먼테이션 (Customer Segmentation)**을 수행하는 파이프라인입니다. 
데이터 정제부터 피쳐 엔지니어링(Feature Engineering)까지의 과정은 **SQL**로 처리하였으며, 추출된 데이터를 기반으로 **Python**을 활용해 머신러닝(군집화) 및 3D 데이터 시각화를 수행하여 고객군을 분류했습니다.

---

## 데이터 파이프라인 및 워크플로우

### 1단계: SQL을 통한 데이터 전처리 및 피쳐 추출 (`Query.sql`)
- **데이터 정제**: 결측치(CustomerID, Description) 및 중복 데이터 제거, 오류값(취소 건, 단가 0 이하 등) 처리.
- **RFM 추출**: Recency(최근성), Frequency(구매 빈도), Monetary(구매 금액) 계산.
- **추가 파생 변수 생성**: 구매하는 제품의 다양성(`unique_products`), 평균 구매 주기(`average_interval`), 구매 취소 비율(`cancel_rate`) 등.
- **데이터 추출**: 정제된 최종 유저별 특징 데이터를 `user_data.csv` 형태로 추출.

### 2단계: Python을 활용한 머신러닝 군집화 (`Code.ipynb`)
- **이상치 처리 및 스케일링**: Z-score 기반 이상치 제거 후 `StandardScaler`를 이용한 데이터 정규화.
- **차원 축소 (PCA)**: 주성분 분석(PCA)을 통해 군집화 연산 최적화.
- **K-Means 클러스터링**: K-Means 알고리즘(`k=3`)을 사용하여 고객을 3개의 세그먼트로 분류.
- **결과 시각화**: Plotly를 활용하여 각 군집을 3D Scatter Plot 형태로 시각화.

---

## 활용 기술
- **Database**: `MySQL`
- **Language**: `Python`, `SQL`
- **Data Analysis & ML**: `Pandas`, `Scipy`, `Scikit-learn (PCA, KMeans)`
- **Visualization**: `Matplotlib`, `Seaborn`, `Plotly (3D Scatter)`

---

# 2. 사이킷런으로 시작하는 머신러닝

## 2-1. 사이킷런 소개와 특징

사이킷런(scikit-learn) : 파이썬 기반의 머신러닝 라이브러리

특징
1. 가장 쉽고 파이썬스러운 API 제공
2. ML을 위한 다양한 알고리즘, 개발을 위한 편리한 프레임워크 및 API 제공
3. 대중적으로 가장 많이 쓰임

## 2-2. 첫 번째 머신러닝 만들어 보기 - 붓꽃 품종 예측



## 2-3. 사이킷런의 기반 프레임 워크 익히기

### Estimator 이해 및 fit(), predict() 메서드

### 사이킷런의 주요 모듈

| 분류 | 모듈명 | 설명 | 
| :---: | :---: | :---: | 
| 예제 데이터 | sklearn.datasets | 사이킷런에 내장되어 에제로 제공하는 데이터 세트 |
| 피처 처리 | sklearn.preprocessing | 데이터 전처리에 필요한 다양한 가공 기능 제공 (인코딩, 정규화, 스케일링 등) |
| 피처 처리 | sklearn.feature_selection | 알고리즘에 큰 영햐을 미치는 피처를 우선순위대로 셀렉션 작업을 수행 |
| 피처 처리 | sklearn.feature_extraction | 텍스트나 이미지 데이터에서 피처를 추출하는 기능 제공 |
| 피처 처리 & 차원 축소 | sklearn.decompostion | 차원 축소 관련 알고리즘 지원 (PCA, NMF, Truncated SVD 등) |
| 데이터 분리, 검증 & 파라미터 튜닝 | sklean.model_selection | 데이터 분할, 교차 검증, 파라미터 튜닝 관련 기능 제공 |
| 평가 | sklearn.metrics | 분류, 회귀, 클러스터링, 페어와이즈에 대한 다양한 성능 측정 방법 제공 |
| ML 알고리즘 | sklearn.ensemble | 앙상블 알고리즘 |
| ML 알고리즘 | sklearn.linear_model | 선형 회귀, 로지스틱 회귀 등 |
| ML 알고리즘 | sklearn.naive_bayes | 나이브 베이즈 알고리즘 |
| ML 알고리즘 | sklearn.neighbors | 최근접 이웃 알고리즘 |
| ML 알고리즘 | sklearn.svm | 서포트 벡터 머신 알고리즘 |
| ML 알고리즘 | sklearn.tree | 결정 트리 알고리즘 |
| ML 알고리즘 | sklearn.cluster | 비지도 학습인 군집화 알고리즘 |
| 유틸리티 | sklearn.pipeline | 피처 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 함께 묶어서 실행 |

### 내장된 예제 데이터 세트

사이킷런에는 내장된 예제 세트가 존재해 API를 통해 쉽게 불러와서 사용할 수 있다.

datasets.load_boston()
datasets.load_breast_cancer()
datasets.load_diabetes()
datasets.load_digits()
datasets.load_iris()

fetch 계열 명령은 다운로드 후 사용 -> scikit_learn_data라는 서브 디렉터리에 저장 후 불러오기

fetch_covtype()
fetch_20newsgroups()
fetch_olivetti_faces()
fetch_lfw_people()
fetch_lfw_pairs()
fetch_rcv1()
fetch_mldata()

분류와 클러스터링을 위한 표본 데이터 생성기


# Main Quest 03 : 통계학 활용 실습

> 온라인 리테일 데이터셋(Kaggle)을 활용하여 그동안 학습한 통계학 개념들을 실전에 적용하는 종합 실습 + [MQ3 실습 코드](https://github.com/Chankyu99/ModuLABS/blob/master/03_Statistics/MQ3.ipynb)

---

## 데이터셋 소개

Kaggle의 **Online Retail** 데이터셋을 활용하였다. 영국 기반 온라인 리테일 회사의 약 54만 건 거래 데이터로, 주요 컬럼은 다음과 같다.

| 컬럼 | 설명 |
| :--- | :--- |
| `InvoiceNo` | 거래 번호 (C로 시작하면 반품) |
| `StockCode` | 상품 코드 |
| `Quantity` | 판매 수량 (음수면 반품) |
| `UnitPrice` | 단가 |
| `InvoiceDate` | 거래 일시 |
| `Country` | 판매 국가 |

피처 엔지니어링을 통해 다음 변수를 추가 생성하였다.

```python
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]       # 주문 가격
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek           # 요일 (0:Mon ~ 6:Sun)
df["InvoiceMonth"] = df["InvoiceDate"].dt.month            # 판매 월
```

---

## 1. 모집단과 표본 집단의 TotalPrice 비교

> 중심극한정리를 실전 데이터에서 직접 확인한다.

전체 데이터셋에서 무작위로 1000개의 표본을 추출하고, 모집단과 표본의 평균·표준편차를 비교하였다.

```python
sample_df = df.sample(n=1000, random_state=42)
```

| 구분 | 평균 | 표준편차 |
| :---: | :---: | :---: |
| **전체 데이터셋(모집단)** | 17.99 | 378.81 |
| **표본(n=1000)** | 16.83 | 64.47 |

이어서 표본 크기 $n=30$으로 1000번 반복 추출하여 표본평균의 분포를 시각화하였다.

![MQ3-1](MQ3-1.png)

> **중심극한정리**에 따르면, 모집단의 분포 모양과는 관계없이 추출된 표본평균의 분포가 **정규분포**에 가까워진다. 왼쪽의 원본 TotalPrice 분포는 극도로 왜곡된(right-skewed) 형태이지만, 오른쪽의 표본평균 분포는 정규분포에 매우 가까운 종 모양을 보인다.

---

## 2. 상위 두 국가의 TotalPrice 평균 차이 검정

> 독립표본 t-검정을 통해 두 국가 간 평균 차이가 통계적으로 유의미한지 검증한다.

주문 수가 가장 많은 상위 두 국가는 **영국(United Kingdom)**과 **독일(Germany)**이다. 이 두 국가의 `TotalPrice` 평균에 차이가 있는지 검증하였다.

**검정 절차** :

1. 등분산 검정(Levene's Test) → $p = 0.6125$로 등분산 가정 만족
2. 독립표본 t-검정 수행

```python
levene_stat, levene_p = stats.levene(uk_data, germany_data)
t_stat, p_val = stats.ttest_ind(uk_data, germany_data, equal_var=True)
```

| 지표 | 값 |
| :---: | :---: |
| **t-statistic** | -1.6839 |
| **p-value** | 0.0922 |

$p = 0.0922 > 0.05$ 이므로 귀무가설을 기각할 수 없다. 즉, **"두 국가 간 TotalPrice 평균 차이는 통계적으로 유의미하지 않다"**는 결론이 나온다.

그런데 실제 평균을 확인하면:
- **영국 평균** : 16.53
- **독일 평균** : 23.35

실제로는 독일의 가격이 영국보다 높다. 귀무가설이 옳다고 판단했지만 실제로는 대립가설이 옳았기 때문에, 이 경우 **2종 오류**를 범한 셈이다. 이는 표본 수의 불균형(영국 >> 독일)이나 분산의 크기 등이 원인일 수 있다.

---

## 3. 주중/주말에 따른 UnitPrice 분포 변화 확인

> 시각적 탐색을 통해 주중과 주말의 단가 분포 차이를 확인한다.

```python
df['DayType'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
```

| DayType | count | mean | std | min | 25% | 50% | 75% | max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Weekday | 477,534 | 4.84 | 103.03 | -11,062 | 1.25 | 2.10 | 4.13 | 38,970 |
| Weekend | 64,375 | 2.89 | 7.96 | 0.00 | 1.25 | 1.85 | 3.75 | 1,237 |

![MQ3-2](MQ3-2.png)

> UnitPrice 데이터는 가격이 낮은 상품이 대다수이고, 소수의 매우 비싼 상품(이상치)이 존재한다. 주중과 주말의 UnitPrice는 크게 차이가 없으며, 주중이 주말보다 약간 더 비싸게 판매되었다.

---

## 4. 판매 수량(Quantity) 예측 : 선형 회귀 모델

> 회귀 모델을 사용하여 종속변수(Quantity)를 예측한다.

**독립변수** : `UnitPrice`, `DayOfWeek`, `Country` (원-핫 인코딩)
**종속변수** : `Quantity`

```python
df_clean = df[(df['Quantity'] > 0) & (df['Quantity'] < 10000)]
df_encoded = pd.get_dummies(df_clean, columns=['Country'], drop_first=True)

model = LinearRegression()
model.fit(X_train, y_train)
```

| 평가 지표 | 값 |
| :---: | :---: |
| **MSE** | 1534.26 |
| **$R^2$** | 0.03 |

> $R^2 = 0.03$으로 **매우 낮은 성능**을 보인다. 독립변수들이 종속변수(Quantity)와의 선형관계가 매우 약하다는 뜻이다. 즉, 단가·요일·국가만으로는 판매 수량을 설명하기 어렵다. 다른 변수의 추가나 비선형 모델을 고안해야 한다.

---

## 5. 반품 가능성 예측 : 로지스틱 회귀 모델

> 거래 정보를 바탕으로 해당 거래가 반품될 가능성을 예측하는 이진 분류 문제를 해결한다.

**피처 엔지니어링** : `Quantity < 0`이면 반품(`IsReturn = 1`)으로 라벨링

**독립변수** : `TotalPrice`, `DayOfWeek`, `InvoiceMonth`, `Country` (원-핫 인코딩)
**종속변수** : `IsReturn` (0 또는 1)

#### 모델 개선 과정

| 시도 | 방법 | F1 Score | 비고 |
| :---: | :--- | :---: | :--- |
| 1차 | 기본 로지스틱 회귀 | — | $R^2 = 0.61$, 평가지표 부적절 |
| 2차 | Country 컬럼 추가 | — | $R^2 = 0.63$으로 소폭 향상 |
| 3차 | F1 Score로 전환 | 0.7532 | precision=0.64, recall=1.00 |
| 4차 | `class_weight` 튜닝 | **0.9350** | precision=0.97, recall=0.92 |

#### 최종 모델

```python
model = LogisticRegression(class_weight={0: 1, 1: 10})
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

**최종 성능 보고서** :

```
교차 검증 평균 F1 점수: 0.9350

              precision    recall  f1-score   support
           0       1.00      1.00      1.00    106226
           1       0.97      0.92      0.94      2156
    accuracy                           1.00    108382

Confusion Matrix:
[[106158     68]
 [   181   1975]]
```

> **핵심**: 
> 1. 분류 문제에서의 평가지표는 결정계수($R^2$)가 아닌 **F1 Score**를 사용해야 한다.
> 2. 반품 데이터는 전체의 약 2%에 불과한 **클래스 불균형** 문제가 존재한다. `class_weight` 파라미터를 조정하여 소수 클래스(반품)에 더 큰 가중치를 부여함으로써 F1을 0.78 → 0.94로 대폭 향상시켰다.
> 3. 약간의 precision 감소(1.00 → 0.97)를 감수하되, recall을 크게 끌어올려(0.64 → 0.92) 실제 반품 건을 놓치지 않도록 하는 것이 비즈니스적으로 합리적인 선택이다.

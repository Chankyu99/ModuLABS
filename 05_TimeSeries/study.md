# 시계열 분석 정리

> 노드 학습 내용 + [실습 코드 노트북](https://github.com/Chankyu99/ModuLABS/blob/master/05_TimeSeries/%EB%85%B8%EB%93%9C%ED%95%99%EC%8A%B5.ipynb)

---

## 1. 시계열 데이터란?

> 일정 시간 간격으로 배치된 데이터들의 수열

시계열(Time Series) 데이터란 어떤 순서를 내포하고 있으며, **시간적으로 종속된 모든 데이터**를 일컫는다. 이 시계열 데이터를 분석해 의미있는 요약 혹은 통계정보를 추출하고 **미래를 예측**하거나 **과거의 행동을 진단**한다.

> 과거가 미래에 어떤 영향을 주는가? 과거와 미래에 어떻게 연관되어 있는가?

시계열 데이터의 역사를 보면, 1676년도에 초판이 발행된 최초의 의료 시계열 데이터를 다룬 사실이 있다. 사람이 죽을 때마다 교회의 종을 울려 인간의 죽음을 추적하고, 사망 인원 수를 시계열적으로 기록한 최초 사례이다.

![TS-1](Img/Pasted%20image%2020260413233916.png)

이처럼 시계열은 단순한 사례로부터 **금융, 의료, 리테일** 등 다양한 도메인에서 활용되며 예측에 도움을 주었다. 시계열 공부 전에 데이터 관련 라이브러리를 복기해두면 좋다. [여기](https://github.com/Chankyu99/ModuLABS/blob/master/02_DataPreprocessing/Study/README.md)에 Pandas 핵심 메서드를 정리해 두었다.

> **핵심**: 시계열 분석은 시간에 따른 데이터의 패턴을 파악하고, 이를 기반으로 미래를 예측하거나 과거를 진단하는 것이 목적이다.

---

## 2. 시계열의 기본 성질

시계열 데이터를 공부할 때 가장 유명한 `AirPassengers` 데이터셋을 이용해 시계열의 기본 성질에 대해 알아보았다.

![TS-2](Img/Pasted%20image%2020260413234845.png)

1949년 1월 ~ 1960년 12월 승객 수 데이터로, 여기서 시계열의 기본 성질 3가지로 분해해보자.

![TS-3](Img/Pasted%20image%2020260413235539.png)

### 2.1 추세 (Trend)

**장기적으로 증가하거나 감소하는 경향성**이 존재하는 것을 의미한다.

- 주로 기울기가 증가 혹은 감소할 때 관찰된다
- 일정 기간 동안 지속되는 변화이며, 반복적인 패턴이 아니더라도 장기적인 방향성을 보여준다
- 위 이미지의 경우 확정적 추세만을 뽑아 시각화한 것이다

### 2.2 계절성 (Seasonality)

**계절적 요인의 영향을 받아** 1년, 혹은 일정 기간 안에 **반복적으로 나타나는 패턴**을 의미한다.

- 빈도의 형태로 나타나며, 항상 일정한 경우가 많다
- 위 이미지를 보면 비슷한 모양이 구간별로 12번 반복되는 것을 알 수 있다

### 2.3 주기성 (Cycle)

정해지지 않은 빈도 및 기간으로 일어나는 **상승 혹은 하락**을 의미한다. 계절성과 달리 주기가 고정되어 있지 않다.

### 2.4 Observed & Random

- **Observed** : 원본 데이터를 의미
- **Random** : 추세에서 계절성을 뺀 **잔차(Residual)**

> **핵심**: 잔차에서는 더 이상 뽑아낼 수 있는 시계열의 기본 성질이 없어야 한다. 시계열 = 추세 + 계절성 + 잔차로 분해할 수 있다.

---

## 3. 정상성 (Stationarity)

### 3.1 정상성이란?

> 시간에 상관없이 시계열이 일정한 성질을 띠는 시계열의 특징

만약 시계열의 특징이 관측된 시간과 무관하지 않다면, 해당 시계열은 시간에 따라 특징이 변하고 있다고 볼 수 있다. 그렇게 되면 **과거 시점을 관측해 얻은 모델을 앞으로의 시점에 적용할 수 없다.** 따라서 시계열은 정상성이 필요하다.

크게 **강정상성**과 **약정상성** 2가지 정의로 분류된다.

- **강정상성(Strict Stationarity)** : 모든 적률(Moments, 확률 분포 성질이 일정한 정도)이 시간과 무관하게 일정한 특징. 기저를 이루는 확률분포가 시간에 무관하게 언제나 일정하다.
- **약정상성(Weak Stationarity)** : 현실의 데이터는 노이즈가 많고 강정상성을 만족하기 어려워 주로 활용된다.

약정상성은 임의의 $t, h$에 대하여 다음 3가지 조건을 만족해야 한다.

$$E(X_t) = \mu$$

$$\text{Var}(X_t) < \infty$$

$$\text{Cov}(X_{t+h}, X_t) = \gamma(h)$$

> 쉽게 표현하자면, 약정상성을 띠는 시계열 데이터는 **어느 시점에 관측해도 확률 과정의 성질이 변하지 않는다**는 뜻이다.

### 3.2 정상성을 확인하는 법

통계학에서 배운 [가설검정](https://github.com/Chankyu99/ModuLABS/blob/master/03_Statistics/Study/DAY2/DAY2_Statistics.md) 방법으로 2가지가 존재한다.

#### KPSS 검정

KPSS(Kwiatkowski-Phillips-Schmidt-Shin Test) 검정의 **귀무가설**은 '시계열 과정이 **정상적(Stationary)**이다'로 설정되어 있으며, 대립가설이 '시계열 과정이 비정상적(Non-stationary)이다'로 설정되어 있다.

> $p \geq \alpha$ → 귀무가설 채택 → **정상적** / $p < \alpha$ → 귀무가설 기각 → **비정상적**

#### ADF 검정

ADF(Augmented Dickey-Fuller) 검정의 **귀무가설**은 '시계열에 **단위근이 존재한다**(비정상적)'이며, 대립가설은 '시계열이 정상성을 만족한다'이다.

> $p < \alpha$ → 귀무가설 기각 → **정상적** / $p \geq \alpha$ → 귀무가설 채택 → **비정상적**

#### KPSS 검정과 ADF 검정의 차이

**확정적 추세(Deterministic Trend)**가 존재하는 경우에 둘의 검정 결과에는 차이가 존재한다. KPSS 검정은 기본적(default option)으로 'around the mean'에 대한 검정을 진행하기 때문에, 확정적 추세가 존재하는 경우 ADF와는 다른 결론을 내릴 수도 있다. **항상 시각적으로도 정상적인지 파악하는 것이 중요하다.**

### 코드 구현 : KPSS / ADF 검정

`statsmodels` 라이브러리의 `kpss`와 `adfuller` 함수를 이용하여 쉽게 사용 가능하다.

```python
# KPSS 검정
from statsmodels.tsa.stattools import kpss

time_series_data_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
kpss_outputs = kpss(time_series_data_test)

print('KPSS test 결과 : ')
print('KPSS Statistic:', kpss_outputs[0])
print('p-value:', kpss_outputs[1])
```

```
KPSS Statistic: 0.5941
p-value: 0.0232
```

```python
# ADF 검정
from statsmodels.tsa.stattools import adfuller

adf_outputs = adfuller(time_series_data_test)

print('ADF Test 결과 : ')
print('ADF Statistic:', adf_outputs[0])
print('p-value:', adf_outputs[1])
```

```
ADF Statistic: -2.0510
p-value: 0.2647
```

> **핵심**: KPSS는 p-value가 0.023으로 귀무가설(정상적)을 기각 → **비정상적**. ADF는 p-value가 0.265로 귀무가설(비정상적)을 채택할 수 없음 → 역시 **비정상적**으로 판단. 단순 선형 증가 데이터이므로 두 검정 모두 비정상적이라는 결론이 일치한다.

### 3.3 정상성을 부여하는 방법

비정상 시계열을 정상 시계열로 변환하려면 **분산을 일정하게** 하거나 **평균을 일정하게** 만드는 방법이 있다.

#### 분산을 일정하게 : 로그 변환 (Log Transformation)

```python
import numpy as np
import matplotlib.pyplot as plt

time_series_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
time_series_data_log = np.log(time_series_data)
```

![TS-4](Img/notebook_output_01.png)

![TS-5](Img/notebook_output_02.png)

단순 선형 데이터가 로그 변환을 통해 **비선형의 특징**을 가지는 것을 알 수 있다. 분포 범위가 줄어들어 분산이 안정화된다.

#### 평균을 일정하게 : 평활 (Smoothing)

시계열 데이터의 **잡음을 제거**하기 위해 주로 사용된다. 이동평균(rolling mean)을 적용하면 원본보다 훨씬 평활해진 것을 확인할 수 있다.

```python
import random
import pandas as pd

df0 = pd.DataFrame({'orig_value': [random.uniform(0, 100) for _ in range(100)]})
df0['smoothed_value'] = df0['orig_value'].rolling(5).mean()

df0.plot(legend=True, subplots=True, title='smoothed')
```

![TS-6](Img/Pasted%20image%2020260414002426.png)

#### 평균을 일정하게 : 차분 (Differencing)

시계열 데이터들의 **시간상의 차이**를 구하는 것이다. 차분 횟수에 따라 1차, 2차, ... 차수가 늘어나고, 데이터의 길이가 충분할 경우 여러 번 수행된다.

```python
df1 = pd.DataFrame({'orig_value': [random.uniform(0, 100) for _ in range(100)]})
df1['diff_value'] = df1['orig_value'].diff()
df1.plot(legend=True, subplots=True, title='diff')
```

![TS-7](Img/Pasted%20image%2020260414002603.png)

> **핵심**: 많은 비정상적 시계열은 **누적 과정(Integrated Process)**이고, 정상적 시계열이 누적되어 비정상적 시계열을 이루었기 때문에, 다시 누적된 것을 차분해줌으로써 그 이면의 **정상적 과정**을 볼 수 있게 되는 것이다.

---

## 4. 시계열 데이터의 시각화

`EuStockMarkets` 시계열 데이터셋(DAX, SMI, CAC, FTSE 등 유럽 주가지수)을 이용해 4가지 시각화 방법을 살펴보았다.

### 4.1 Line Plot

간단한 **선**을 그려 데이터의 시간과 순서에 따라 어떻게 변하는지 기본적으로 시계열 파악 시 활용된다.

![TS-8](Img/Pasted%20image%2020260414235617.png)

### 4.2 Histogram

데이터의 **분포**를 판단할 수 있으며, 일반적으로 시계열 데이터는 1차 차분에 대해 히스토그램을 그리면 특정한 분포를 볼 수 있다.

![TS-9](Img/Pasted%20image%2020260414235802.png)

### 4.3 Scatter Plot

두 시계열의 **관계**에 대한 유의미한 정보를 확인할 수 있다. 차분을 하면 새로운 의미를 찾을 수 있다.

![TS-10](Img/Pasted%20image%2020260414235844.png)

### 4.4 Circular Chart

시계열에 대한 유의미한 인사이트를 얻긴 어렵지만, **시각적 효과**를 위해 사용한다. 계절성을 시각화할 때 가끔 유용하다.

![TS-11](Img/Pasted%20image%2020260414235926.png)

### 코드 구현 : EuStockMarket 탐색

**Line Plot**으로 추세를 확인해보자.

![TS-12](Img/notebook_output_07.png)

**Histogram** — 차분을 적용하면 분포의 평균이 일정하고, 정상성을 가지게 되며 시각적으로 가운데로 밀집한 형태의 히스토그램으로 변경된다.

![TS-13](Img/notebook_output_08.png)

차분 적용 후 히스토그램:

![TS-14](Img/notebook_output_09.png)

**Scatter Plot** — 차분이 적용되고 안 되고의 차이가 산점도에서도 명확히 확인된다.

![TS-15](Img/notebook_output_10.png)

차분 적용 후 산점도:

![TS-16](Img/notebook_output_11.png)

> **핵심**: 차분을 적용하면 추세가 제거되어 정상성에 가까워지고, 히스토그램과 산점도에서도 그 변화가 명확히 드러난다.

---

## 5. 시계열 EDA : ACF & PACF

### 5.1 ACF (AutoCorrelation Function) Plot

**자기상관(AutoCorrelation)**은 시계열 데이터에서 일정 간격이 있는 값들 사이의 상관관계를 의미한다. 이것을 함수로 나타내어 시간에 따른 상관 정도를 나타내는 그래프가 **ACF Plot**이다.

![TS-17](Img/Pasted%20image%2020260415000654.png)

일반적인 사인함수와 그 ACF Plot인데, 임계값이 파란색으로 나타나 있다.

### 5.2 PACF (Partial AutoCorrelation Function) Plot

**편자기상관(PACF)**은 자신에 대한 그 시차의 편상관을 의미한다. 즉, 두 시점 사이의 전체 상관관계에서 **그 사이 다른 시점의 조건부 상관관계를 뺀 것**이다.

![TS-18](Img/Pasted%20image%2020260415000907.png)

> **ACF vs PACF**: ACF는 두 시점 간 상관관계 계산 시 두 시점 사이 모든 lag에 대한 정보가 들어간다. PACF는 오로지 두 시점만의 상관관계만을 계산해 다른 lag는 조건부 상관관계로 제거한다.

### 코드 구현 : ACF & PACF 시각화

`statsmodels` 라이브러리의 `plot_acf`, `plot_pacf` 모듈을 임포트해 간단히 사용할 수 있다. 일반적인 사인함수를 그려보고 ACF/PACF로 분석하였다.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 50, 300)
y = np.sin(x)

plt.plot(y)
plt.show()
```

![TS-19](Img/notebook_output_15.png)

```python
plot_acf(y)
```

![TS-20](Img/notebook_output_16.png)

```python
plot_pacf(y)
```

![TS-21](Img/notebook_output_17.png)

**ACF 해석**: lag가 진행됨에 따라 상관계수가 0으로 수렴하지 않고, 특정 시차마다 파동을 그리며 다시 커지는 패턴이 반복된다. 데이터에 **일정한 주기가 있음**을 시사한다. 정상적인 시계열이라면 ACF가 초기에 급감해야 하는데, 신뢰구간을 넘는 값이 많으며 전체적으로 값이 크다. 이는 **정상성을 가지고 있다고 할 수 없다**.

**PACF 해석**: 불규칙하게 튀는 지점이 많다. 데이터의 추세나 계절성이 제거되지 않았을 때 전형적으로 나타나는 모습이다.

> **핵심**: 시계열 EDA에서 체크해야 할 가장 중요한 위험은 **'허위 상관(spurious correlation)'**이며, 이것은 두 개 이상의 변수가 통계적 상관은 있지만 인과관계가 없는 관계를 말한다.

---

## 6. 시계열 데이터의 특징 추출

### 6.1 Tsfresh 라이브러리

각 시계열들은 고유의 특징을 지니고 있다. 주요한 시계열 요약 통계 특징은 다음과 같다.

- 평균과 분산
- 최댓값과 최솟값
- 시작과 마지막 값의 차이
- 국소적 최소와 최대의 개수
- 시계열의 평활 정도
- 시계열의 주기성과 자기상관

다양한 특징을 추출하는 데 편리한 라이브러리로 **Tsfresh**가 있다. 논문에 따르면 63개의 시계열 특징 추출 방법론을 활용해 794개의 특징을 포함할 수 있다고 하며, 현재는 약 **1200개 이상의 특징**을 지원 중이다.

### 코드 구현 : Robot Execution 데이터 특징 추출

`tsfresh`의 `load_robot_execution_failures` 데이터셋을 활용하여 특징을 추출하였다.

```python
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures
timeseries, y = load_robot_execution_failures()
```

![TS-22](Img/스크린샷%202026-04-15%2015.22.53.png)

![TS-23](Img/스크린샷%202026-04-15%2015.23.04.png)

`extract_features`를 통해 그룹화 및 정렬 기준에 맞게 추출할 수 있다. 코드를 통해 약 **4698개의 특징**이 추출되었다.

```python
from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
```

여기서 `impute`를 통해 동일한 열의 중앙/극단값으로 바꾸고, `select_features()`로 특징 행렬의 모든 특징의 중요성을 확인해 관련 특징만 포함하는 **축소된 버전**의 특징 행렬을 반환한다.

```python
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)
```

중요한 컬럼만 남게 된 값을 확인해보면:

```python
F_count = [col for col in features_filtered.columns if col.startswith("F_")]
T_count = [col for col in features_filtered.columns if col.startswith("T_")]

print(len(F_count))  # 376
print(len(T_count))  # 295
```

> F(Force)가 376개, T(Torque)가 295개로, **F가 더 중요한 역할을 한다**는 결론을 얻었다.

---

## 7. 시계열 데이터 분류

시계열 데이터를 이용해 지도학습 **분류**를 해보자.

### 코드 구현 : 로지스틱 회귀 분류

데이터셋을 커스텀 함수로 Train/Test로 분할하고, `MinimalFCParameters`로 최소 특징만 추출하여 빠르게 테스트하였다.

```python
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

settings = MinimalFCParameters()  # 계산 효율을 위해 minimal 설정
minimal_features_train = extract_features(
    X_train, column_id="id", column_sort="time",
    default_fc_parameters=settings
)
```

추출된 특징을 시각화한 결과:

![TS-24](Img/notebook_output_18.png)

![TS-25](Img/notebook_output_19.png)

분류에 **로지스틱 회귀**를 사용한 결과, **0.69 정도의 정확도**를 얻을 수 있었다.

> 모델 혹은 알고리즘은 score로만 결과를 신뢰할 수 없다. 모델을 거쳐 나온 **결과값을 분석**하여 이유가 있고 **설명 가능한 상태**가 되어야 한다.

`classification_report`로 상세 지표를 확인해보면:

| 지표 | Precision | Recall | F1-score |
| :---: | :---: | :---: | :---: |
| True | 1.0 | 0.6 | 0.75 |
| False | 0.43 | 1.0 | 0.6 |
| **accuracy** | | | **0.69** |
| macro avg | 0.71 | 0.8 | 0.67 |
| weighted avg | 0.87 | 0.69 | 0.72 |

특징 추출 후 나온 feature의 개수는 **60개**이다. 매크로 평균(macro avg)은 각 클래스별 평균 / 클래스 수로, 가중 산술 평균(weighted avg)은 자료의 중요도를 반영한 평균이다.

> **핵심**: 시계열 분류에서는 원시 데이터 자체를 모델에 넣는 것보다, tsfresh 등으로 **특징을 추출한 후** 분류기에 입력하는 접근이 효과적이다. 다만 score만이 아니라 Precision, Recall, F1-score를 종합적으로 판단해야 한다.

---

## 8. ARIMA

2017년까지 LSTM과 Transformer를 이용한 접근이 ARIMA를 비롯한 통계 모델에게 우세를 점하지 못하고 있다. 현재도 **복잡성이 낮은 예측 문제**에서 ARIMA는 높은 예측 성능을 보인다.

$$\text{ARIMA} = \text{AR} + \text{Integration(차분)} + \text{MA}$$

### AR(p) : 자기 회귀 모델

AR(p) 모델은 **과거가 미래를 예측한다**는 직관적인 사실에 의존하는 모델로서, 특정 시점 $t$의 값은 이전 시점을 구성하는 값들의 함수라는 시계열 과정을 상정한다.

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t$$

- $p$ : AR lagged values (과거 몇 개의 시점을 참조할 것인가)

### MA(q) : 이동평균 모델

MA(q) 모델은 각 시점의 데이터가 최근의 **과거 오차항**으로 구성된 함수로 표현된다. 각 오차항은 서로 **독립적**이다. 이는 과거의 오차(그 시점에 발생한 독립적인 사건)가 현재의 과정에 영향을 미친다고 보는 것이다.

$$y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

- $q$ : MA lagged errors (과거 몇 개의 오차항을 참조할 것인가)

### ARIMA(p, d, q) 모델

AR과 MA를 결합하고, 차분(Integration)을 추가한 것이 ARIMA이다.

- **p** : AR lagged values
- **d** : 차분의 개수
- **q** : MA lagged errors

### 파라미터 적합 : Box-Jenkins 법

ACF, PACF 함수를 보며 **수동으로 파라미터를 적합**시키는 방법을 **박스-젠킨스 법(Box-Jenkins)**이라 한다.

1. 시각화 및 도메인 지식을 바탕으로 초기 파라미터 $(p, d, q)$를 추정
2. 모델 적합 후 시각화 및 성능 평가를 수행
3. 부적합하다고 판단될 경우 파라미터를 조정
4. 이를 반복

### Auto-ARIMA와 과적합 가능성

Log likelihood를 이용한 Step-wise 방법인 **Auto-ARIMA**가 존재한다. 하지만 해당 방법을 이용할 경우, **과도하게 복잡한 모델**을 적합할 위험성이 존재한다.

> 일반적으로 차분($d$)이 **2를 넘어가면 과적합**으로 보며, $\text{AR}(p)$, $\text{MA}(q)$도 **3을 넘기지 않는 것**이 바람직하다.

### 코드 구현 : ARIMA 실습

Daily Demand Forecasting Orders 데이터를 활용해 ARIMA를 실습하였다.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# 데이터 불러오기
df = pd.read_csv('Daily_Demand_Forecasting_Orders.csv', delimiter=';')
data = df['Target (Total orders)']

plt.plot(data)
plt.show()
```

![TS-26](Img/notebook_output_20.png)

ACF 및 PACF로 데이터의 자기상관 구조를 확인하였다.

```python
plot_acf(data)
```

![TS-27](Img/notebook_output_21.png)

```python
plot_pacf(data)
```

![TS-28](Img/notebook_output_23.png)

ACF/PACF를 바탕으로 여러 ARIMA 모델을 비교 적용하였다.

```python
# 여러 모델 비교
model1 = ARIMA(data, order=(1,0,0))  # AR(1)
model2 = ARIMA(data, order=(0,0,1))  # MA(1)
model3 = ARIMA(data, order=(1,0,1))  # ARMA(1,1)

res1 = model1.fit()
res2 = model2.fit()
res3 = model3.fit()
```

`fittedvalues`(적합된 값)를 실제 데이터와 비교하여 시각화하였다.

```python
predictions = res2.fittedvalues
plt.figure()
plt.plot(data)
plt.plot(predictions)
plt.show()
```

![TS-29](Img/notebook_output_25.png)

> $\hat{y}_i = b_0 + b_1 x_i$ 에서 $\hat{y}$는 적합된 값(fitted value)이며, $e_t = y_t - \hat{y}_t$는 **잔차**이다. 잔차는 관측치와 적합된 값 사이의 차이와 같다.

**Auto-ARIMA 적용**:

```python
model = pm.AutoARIMA(seasonal=False, stepwise=True,
                     suppress_warnings=True, trace=True)
res = model.fit(data)
print(res.summary())
```

> **핵심**: Auto-ARIMA는 편리하지만, 도메인 지식 없이 자동으로 적합하면 과적합 위험이 있다. ACF/PACF를 통해 데이터의 자기상관 구조를 먼저 파악하고, 적절한 범위 내에서 파라미터를 설정하는 것이 바람직하다.

---

## 9. ARIMA의 확장

ARIMA에 추가 요소를 결합하여 더 복잡한 시계열을 모델링할 수 있다.

### SARIMA

ARIMA 모델에 **계절성(Seasonality)** 요소를 추가한 모델이다.

$$\text{SARIMA} = \text{ARIMA} + \text{Seasonality}$$

계절($m$)에 대한 $\text{AR}(P)$, 차분($D$), $\text{MA}(Q)$를 ARIMA 모델에 포함시킴으로써 **계절성이 있는 데이터**도 모델링할 수 있다.

### ARIMAX (Dynamic Regression)

일반적으로 **선형 회귀는 시계열 회귀에 적합하지 않다.** 선형회귀는 **독립항등분포(iid)**를 가정하기 때문에 시계열의 특성과 다르기 때문이다. 하지만 회귀의 오차항이 **ARIMA 시계열**이라고 가정할 경우, 예측 작업에 회귀분석을 보다 손쉽게 적용할 수 있다.

$$\text{ARIMAX} = \text{ARIMA} + \text{Time Series Regression}$$

### SARIMAX

SARIMA와 ARIMAX를 결합한 모델이다.

$$\text{SARIMAX} = \text{SARIMA} + \text{ARIMAX}$$

> **핵심**: 단순 ARIMA로 부족할 때, 계절성이 있으면 SARIMA, 외생변수(exogenous variable)가 있으면 ARIMAX, 둘 다 있으면 SARIMAX를 고려한다.

---

## 10. 금융 시계열과 변동성 모델

### 수익률 (Return)

금융에서는 자산의 가치 자체보다는 **수익에 대해서 더 높은 관심**을 지니고 있다. $P_t$는 각 시점의 가격을 의미한다.

**다기간 수익률 (Multiperiod Simple Return)**

$$\frac{P_t}{P_{t-k}} = \frac{P_t}{P_{t-1}} \times \frac{P_{t-1}}{P_{t-2}} \times \cdots \times \frac{P_{t-k+1}}{P_{t-k}}$$

**다기간 로그 수익률 (Multiperiod Log Return)**

$$\log \frac{P_t}{P_{t-k}} = r_t + r_{t-1} + \cdots + r_{t-k+1}$$

> 연속된 기간의 수익률을 계산할 때, 로그 수익률은 **Additive**하게 적용될 수 있으므로 연산이 간편하다. Additive란 다른 요인(factors)들과 결합될 때 알려진 정도보다 더 누적 효과나 강도가 좋아지는 것을 말한다.

### 자산 수익률의 Stylized Facts

**경험기반의 사실(Stylized Fact)**은 과거부터 축적되어 있는 정형화된 사실들을 기반으로 접근하는 방법론이다.

1. 일반적으로 평균을 중심으로 **좌우 대칭(symmetric)** 분포를 보인다
2. 굉장히 **두꺼운 꼬리 분포(Heavy-tailed distribution)**를 따른다 — 단기간에 대박나거나 쪽박차는 사람들이 있다
3. 각 시점의 수익률들 간에는 **별다른 관계가 없다** → **효율적 시장 가설(EMH)**
4. 수익률의 **제곱** 간에는 높은 관계가 존재한다 → **조건부 이분산성(Conditional Heteroskedasticity)**: 자산의 수익률의 분산(=제곱)은 상수가 아니며, 자기회귀적(Autoregressive) 특성을 지니고 있다

> 수익률의 변동성이 높아지는 시기는 하나의 **군집(cluster)**처럼 뭉쳐 있는 것을 볼 수 있다 → 변동성에 대한 AR 모델의 적용 가능성이 있다.

### ARCH 모델

**ARCH(AutoRegressive Conditional Heteroskedasticity Model)**는 $m$개의 파라미터를 가지고 있으며 자산의 **변동성을 자기회귀**를 통해 모델링한다.

$$r_t^2 = \alpha_0 + \alpha_1 r_{t-1}^2 + \cdots + \alpha_m r_{t-m}^2 + \eta_t$$

Python에서는 `arch` 라이브러리를 통해 손쉽게 활용 가능하다.

### GARCH 모델

**GARCH(Generalized ARCH)**는 자산 변동성에 AR뿐만 아니라 **MA도 함께 적용**한 모델이다. GARCH(1,1)의 일반식은 다음과 같다.

$$\sigma_t^2 = \alpha_0 + \alpha_1 r_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

- $\text{GARCH}(p, q)$는 ARMA와 마찬가지로 $p$, $q$ 두 개의 파라미터를 지니고 있다
- 자산 변동성에 ARMA를 적용함으로써 **높은 수준의 변동성 예측**이 가능하다

### 변동성 모델링은 무엇을 가능하게 해주는가?

- 자산의 변동성이 중요한 **파생상품 가치 측정**, **위험관리** 부문에서 GARCH/ARCH 모델의 중요성은 매우 높다
- 트레이딩 측면에서도 변동성을 이용한 유용한 **투자전략**들이 존재한다
- S-GARCH, E-GARCH 등 여러 GARCH의 **파생 모델**들이 변동성 예측 및 트레이딩에 활용되고 있다

> **핵심**: 금융 시계열에서는 수익률 자체의 예측보다 **변동성의 예측**이 더 실용적인 경우가 많다. ARCH/GARCH 모델은 변동성의 군집 현상(Volatility Clustering)을 포착하여 위험관리와 파생상품 가격 결정에 핵심적인 역할을 한다.

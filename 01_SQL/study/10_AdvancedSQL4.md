# 10. SQL 심화 - SQL로 데이터 분석하기

> 인터넷 쇼핑몰에 근무하는 데이터 사이언티스트가 할 수 있는일 연습하기

## 학습 내용

- 회사의 **일별 매출액** 구하기
- 유저가 조회한 페이지 정보를 활용하여 **PV**, **Unique PV**, **UV** 계산하기
- 유저들의 구매 정보를 활용하여 **ARPU**, **ARPPU** 구하기
- 유저들의 액션 로그 정보를 활용하여 **퍼널 분석**하기
- 유저들의 구매 정보를 활용하여 **리텐션 분석**하기
- 유저들의 구매 정보를 활용하여 **RFM 분석**하기

## 10-1. 회사의 일별 매출액 구하기

> 회사의 일별 매출액 합계를 구해야 한다
> 주문이 일어난 시점에 대한 정보를 기록한 `orders`와 결제 금액에 대한 정보를 기록한 `payments` 테이블을 활용해 일별 매출액 총합을 구한다

### 데이터: orders
| order_id | user_id | order_timestamp |
| :--- | :--- | :--- |
| order_1 | user_1 | 2018-01-01 10:00:00 |
| order_2 | user_2 | 2018-01-01 12:30:00 |
| order_3 | user_3 | 2018-01-02 09:20:00 |
| order_4 | user_4 | 2018-01-02 10:15:00 |
| order_5 | user_5 | 2018-01-03 11:00:00 |
| order_6 | user_6 | 2018-01-03 13:45:00 |

### 데이터: payments
| order_id | value |
| :--- | :--- |
| order_1 | 100 |
| order_2 | 150 |
| order_3 | 200 |
| order_4 | 110 |
| order_5 | 120 |
| order_6 | 130 |

### 쿼리 작성

```sql
SELECT DATE(o.order_timestamp) AS dt, SUM(p.value) AS daily_value
FROM orders AS o
JOIN payments AS p
ON o.order_id = p.order_id
GROUP BY dt
```

> DATE() 함수는 날짜만 추출하는 함수이다. 시/분/초를 제외하고 출력하기 위해 사용했어야 했는데 못했다.
> GROUP BY로 데이터를 그룹화했다면, SELECT 절에는 집계 함수를 사용해야 한다. 이것을 지키지 않아서 오류가 발생했다.

## 10-2. 유저가 조회한 페이지 정보를 활용하여 PV, Unique PV, UV 계산하기

### 개념 정리

- PV(Page View) : 사용자가 페이지를 열어본 횟수의 총합, 새로고침도 횟수로 포함한다 
    - 특정 페이지에 얼마나 많은 사람들이 몰렸는지 측정가능
- Unique PV : 한번의 세션 내 본 횟수. 새로고침을 여러번해도 1로 카운트 된다
- UV(Unique Visitor) : 기간 내 서비스에 방문한 순수 방문자 수. 실제 서비스 이용자 수를 파악할 수 있다
- 분석과 주의해야 할 점
    1. UV ↑ , PV ↑ : 순방문자 수도 늘고, 방문한 페이지 수도 늘었다는 의미
    2. UV ↑ , PV ↓ : 순방문자 수는 늘었지만, 페이지 체류 시간이 줄었다는 의미
    3. UV ↓ , PV ↑ : 순방문자 수는 줄었지만, 페이지 체류 시간이 늘었다는 의미
    4. UV ↓ , PV ↓ : 순방문자 수도 줄고, 페이지 체류 시간도 줄었다는 의미

> 사용할 데이터는 다음과 같다

### 데이터: visits
| log_id | user_id | page_url | timestamp |
| :--- | :--- | :--- | :--- |
| 1 | A | home | 2023-10-01 10:15:22 |
| 3 | A | home | 2023-10-01 10:18:12 |
| 2 | B | product_detail | 2023-10-01 10:17:45 |
| 5 | B | checkout | 2023-10-01 10:21:30 |
| 4 | C | home | 2023-10-01 10:20:10 |
| 6 | C | product_detail | 2023-10-01 10:22:00 |

1. 페이지 별 PV

```sql
SELECT page_url, COUNT(*) AS PV
FROM visits 
GROUP BY page_url
```

2. 페이지 별 Unique PV

```sql
SELECT page_url, COUNT(DISTINCT user_id) AS Unique_PV
FROM visits
GROUP BY page_url
```

3. ⭐️ Visits 수

```sql
WITH LAST_SESSION AS (
	SELECT *, LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS last_timestamp 
    FROM visits
    ),
	SESSION_DIFF AS (
    SELECT *, TIMESTAMPDIFF(MINUTE, last_timestamp, timestamp) AS time_diff 
    FROM LAST_SESSION
    )
SELECT page_url, COUNT(CASE WHEN time_diff IS NULL OR time_diff >= 30 THEN 1 END) AS Visits
FROM SESSION_DIFF
GROUP BY page_url;
```

- WITH 구문은 하나의 WITH로 시작해서 여러개를 만들 수 있다. (,)로 구분하여 사용한다.
- 시차를 구하는 함수는 MySQL에서 TIMESTAMPDIFF(단위, 시작일, 종료일)이다.
- time_diff IS NULL: 유저가 처음 들어왔을 때 (새 세션 시작)
- time_diff >= 30: 마지막 활동 후 30분이 지나서 다시 들어왔을 때 (새 세션 시작)
- 이 두 경우만 1을 반환하여 COUNT하므로, 단순 페이지 조회(PV)가 아닌 **'세션 기반의 방문 수'**를 정확히 구할 수 있다.

4. 페이지별 UV

```sql
SELECT page_url, COUNT(DISTINCT user_id) AS UV
FROM visits
GROUP BY page_url;
```

> 3과 4의 차이가 헷갈리는데, 차이점을 정리해보았다
> 우선 중복 제거 기준의 차이가 있다. 페이지별 Visit은 **세션 기준**으로 중복을 제거하지만, 페이지별 UV는 **유저 기준**으로 중복을 제거한다.
> 즉 페이지 방문 수 가 몇인가? VS 접속자가 몇 명인가? 느낌.
> UV는 그래서 얼마나 페이지가 신규 사용자에게 노출이 잘 되었는지, 페이지별 Visits는 서비스 활성도를 측정 시 사용하는 지표가 된다.

## 10-3. 유저들의 구매 정보를 활용하여 ARPU, ARPPU 구하기

### 개념 정리

- ARPU (Average Revenue Per User) : 전체 유저 수 대비 평균 매출액 -> 게임이 대중적으로 매출이 잘 나오는 편인지 확인

$$ \text{ARPU} = \frac{총 매출액}{총 유저 수} $$

- ARPPU (Average Revenue Per Paying User) : 결제 유저 수 대비 평균 매출액 -> 과금 유저들의 충성도 지표

$$ \text{ARPPU} = \frac{총 매출액}{총 과금 유저 수} $$

> 유저들의 구매 정보를 기록한 `arpu` 테이블로 ARPU, ARPPU를 계산해보자

### 데이터: arpu

| user_id | purchase_date | revenue |
| :--- | :--- | :--- |
| 1 | 2023-01-01 | 10.00 |
| 2 | 2023-01-01 | 20.00 |
| 3 | 2023-01-01 | 0.00 |
| 1 | 2023-01-02 | 15.00 |
| 2 | 2023-01-02 | 0.00 |
| 3 | 2023-01-02 | 5.00 |
| 4 | 2023-01-03 | 20.00 |
| 5 | 2023-01-03 | 20.00 |
| 6 | 2023-01-03 | 0.00 |
| 1 | 2023-01-04 | 10.00 |
| 2 | 2023-01-04 | 25.00 |
| 7 | 2023-01-04 | 15.00 |
| 8 | 2023-01-05 | 40.00 |
| 5 | 2023-01-05 | 10.00 |
| 9 | 2023-01-05 | 0.00 |
| 10 | 2023-01-06 | 50.00 |
| 11 | 2023-01-06 | 35.00 |
| 6 | 2023-01-06 | 20.00 |
| 12 | 2023-01-07 | 15.00 |
| 13 | 2023-01-07 | 10.00 |
| 1 | 2023-01-08 | 5.00 |
| 14 | 2023-01-08 | 25.00 |
| 15 | 2023-01-08 | 30.00 |
| 16 | 2023-01-09 | 45.00 |
| 17 | 2023-01-09 | 0.00 |
| 18 | 2023-01-10 | 20.00 |
| 19 | 2023-01-10 | 35.00 |
| 20 | 2023-01-11 | 20.00 |
| 21 | 2023-01-11 | 25.00 |
| 22 | 2023-01-12 | 15.00 |

### 쿼리 작성

```sql
SELECT SUM(revenue) / COUNT(DISTINCT user_id) AS ARPU, 
SUM(revenue) / COUNT(DISTINCT CASE WHEN revenue > 0 THEN user_id END) AS ARPPU
FROM arpu;
```

## 10-4. 유저들의 액션 로그 정보를 활용하여 퍼널 분석하기

- 퍼널 분석 : 사용자를 최초로 유입시킨 이후, 최종 목적까지의 단계를 분석하는 프레임워크 -> 고객 이탈 분석
    - AIDA 모델  : Attention(인지) → Interest(흥미) → Desire(욕구) → Action(행동)
    - AARRR 모델 : Acquisition(획득) → Activation(활성화) → Retention(유지) → Revenue(수익) → Referral(추천)

> 유저들의 액션 로그 정보를 기록한 `funnel` 테이블로 각 페이지 단위의 방문 수와 visits 수를 모수로 한 페이지 단위의 전환율 계산

### 데이터: funnel

| user_id | action | action_date |
| :--- | :--- | :--- |
| 4 | visit | 2023-01-02 |
| 5 | visit | 2023-01-02 |
| 6 | visit | 2023-01-03 |
| 4 | signup | 2023-01-03 |
| 5 | add_to_cart | 2023-01-04 |
| 6 | purchase | 2023-01-04 |
| 7 | visit | 2023-01-04 |
| 7 | signup | 2023-01-05 |
| 7 | add_to_cart | 2023-01-06 |
| 7 | purchase | 2023-01-07 |
| 8 | visit | 2023-01-07 |
| 8 | signup | 2023-01-08 |
| 8 | add_to_cart | 2023-01-09 |
| 9 | visit | 2023-01-09 |
| 9 | signup | 2023-01-10 |
| 10 | visit | 2023-01-10 |
| 10 | add_to_cart | 2023-01-11 |
| 10 | purchase | 2023-01-12 |
| 11 | visit | 2023-01-12 |
| 11 | signup | 2023-01-13 |
| 12 | visit | 2023-01-13 |
| 12 | add_to_cart | 2023-01-14 |
| 13 | visit | 2023-01-14 |
| 13 | signup | 2023-01-15 |
| 13 | add_to_cart | 2023-01-16 |
| 13 | purchase | 2023-01-17 |
| 14 | visit | 2023-01-17 |
| 14 | signup | 2023-01-18 |
| 15 | visit | 2023-01-18 |
| 15 | signup | 2023-01-19 |
| 15 | add_to_cart | 2023-01-20 |
| 15 | purchase | 2023-01-21 |
| 16 | visit | 2023-01-21 |
| 16 | signup | 2023-01-22 |
| 17 | visit | 2023-01-22 |
| 17 | add_to_cart | 2023-01-23 |
| 18 | visit | 2023-01-23 |
| 18 | signup | 2023-01-24 |
| 18 | add_to_cart | 2023-01-25 |
| 18 | purchase | 2023-01-26 |
| 19 | visit | 2023-01-26 |
| 19 | signup | 2023-01-27 |
| 20 | visit | 2023-01-27 |
| 20 | add_to_cart | 2023-01-28 |
| 21 | visit | 2023-01-28 |
| 21 | signup | 2023-01-29 |
| 21 | add_to_cart | 2023-01-30 |
| 21 | purchase | 2023-01-31 |

### 쿼리 작성

- ⭐️ 퍼널 분석 해보기


```sql
-- 임시테이블 생성 -> 각 단계별 유저 수를 중복없이 계산

WITH Funnel_Counts AS (
  SELECT
    COUNT(DISTINCT CASE WHEN action = 'visit' THEN user_id END) AS Visits,
    COUNT(DISTINCT CASE WHEN action = 'signup' THEN user_id END) AS Signups,
    COUNT(DISTINCT CASE WHEN action = 'add_to_cart' THEN user_id END) AS AddToCarts,
    COUNT(DISTINCT CASE WHEN action = 'purchase' THEN user_id END) AS Purchases
  FROM `project_name.dataset_name.funnel`
)

-- 임시테이블을 바탕으로 각 단계별 전환율 계산

SELECT
  Visits,
  Signups,
  AddToCarts,
  Purchases,
    
-- 전환율 = (각 단계별 유저 수 / 전체 유저 수) * 100 
-- 2번째 자리 반올림

  ROUND(Signups / Visits * 100, 2) AS SignupRate,       
  ROUND(AddToCarts / Visits * 100, 2) AS AddToCartRate,
  ROUND(Purchases / Visits * 100, 2) AS PurchaseRate
FROM Funnel_Counts;
```

## 10-5. 유저들의 구매 정보를 활용하여 **리텐션 분석**하기

- 리텐션 분석 : 사용자의 방문 이후 우리 서비스에 남아 있는 비율과 특징, 원인을 분석하는 프레임워크
    - 핵심 목표 : 한번이라도 우리의 제품/서비스를 이용했던 고객이 지속적으로 서비스를 찾아오게 하는 것

- 리텐션 분석 종류
    - Classic Retention : N일뒤에 제품을 이용한 사용자 / Day 0에 제품을 이용한 사용자
        - 가장 보편적인 기법이지만, 특정 일에 대한 유지율만 보여주기 때문에 실제 사용자의 재방문율을 확인할 수 없다
    - Rolling Retention : N일 뒤에 한번이라도 제품을 이용한 사용자 / Day 0에 제품을 이용한 사용자
        - 사용자가 실제로 제품을 재사용하는지 여부를 확인가능, 한번이라도 방문한 사용자를 리텐션으로 간주해 리텐션이 다소 높게 측정됨.
    - Range Retention : Range N에 제품을 이용한 사용자 / Range 0에 제품을 이용한 사용자
        - 분석 구간을 유연하게 설정 가능, 구간 설정 기간이 길수록 리텐션이 과대추정 될 수 있음

> 유저들의 구매 정보를 기록한 `retention` 테이블을 활용하여 리텐션 분석
> 제품의 사용 주기가 긴 영양제를 주력 상품으로 밀고 있기 때문에 월별 리텐션을 구해야 한다

### 데이터 : retention

| user_id | invoice_date | sales_amount |
| :--- | :--- | :--- |
| 1 | 2023-01-05 | 120.00 |
| 2 | 2023-01-15 | 200.00 |
| 1 | 2023-01-20 | 80.00 |
| 3 | 2023-01-25 | 150.00 |
| 4 | 2023-02-01 | 100.00 |
| 2 | 2023-02-05 | 220.00 |
| 1 | 2023-02-15 | 50.00 |
| 3 | 2023-02-18 | 130.00 |
| 5 | 2023-02-20 | 175.00 |
| 6 | 2023-02-25 | 90.00 |
| 4 | 2023-03-01 | 110.00 |
| 5 | 2023-03-05 | 160.00 |
| 7 | 2023-03-10 | 200.00 |
| 6 | 2023-03-15 | 85.00 |
| 8 | 2023-03-20 | 140.00 |
| 2 | 2023-03-25 | 210.00 |
| 1 | 2023-04-01 | 95.00 |
| 9 | 2023-04-05 | 180.00 |
| 5 | 2023-04-10 | 155.00 |
| 10 | 2023-04-15 | 200.00 |
| 4 | 2023-04-20 | 120.00 |
| 7 | 2023-04-25 | 190.00 |
| 3 | 2023-05-01 | 140.00 |
| 11 | 2023-05-05 | 220.00 |
| 6 | 2023-05-10 | 75.00 |
| 8 | 2023-05-15 | 130.00 |
| 12 | 2023-05-20 | 165.00 |
| 9 | 2023-05-25 | 180.00 |
| 13 | 2023-06-01 | 150.00 |
| 10 | 2023-06-05 | 190.00 |

### 쿼리 작성

- 구글 시트에 결과물 csv 파일로 저장 -> 피벗 테이블 생성

| cohort_group | 0 | 1 | 2 | 3 | 총계 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2023-01-01** | 3 | 1 | 2 | 1 | **7** |
| **2023-02-01** | 3 | 2 | 2 | | **7** |
| **2023-03-01** | 2 | 2 | | | **4** |
| **2023-04-01** | 2 | 2 | | | **4** |
| **2023-05-01** | 2 | | | | **2** |
| **2023-06-01** | 1 | | | | **1** |
| **총계** | **13** | **7** | **4** | **1** | **25** |

## 10-6. 유저들의 구매 정보를 활용하여 **RFM 분석**하기

- RFM 분석 : 고객의 가치를 평가하는 세 가지 요소인 Recency(최근성), Frequency(빈도), Monetary(금액)를 활용한 고객 세분화 기법
    - Recency : 마지막 구매일로부터 현재까지의 기간 -> 최근에 구매한 고객일수록 가치가 높다
    - Frequency : 총 구매 횟수 -> 자주 구매한 고객일수록 가치가 높다
    - Monetary : 총 구매 금액 -> 많이 구매한 고객일수록 가치가 높다
- 세분화된 고객 그룹에 대해 걸맞는 마케팅 전략을 수립할 수 있다
    
> 유저들의 구매 정보를 기록한 rfm 테이블을 활용하여 RFM 세그먼테이션을 분석한다
> Recency, Frequency, Monetary를 계산하고, 1점에서 5점까지의 척도로 각각 점수를 매긴 후 RFMScore를 구해보자

### 데이터 : rfm

| user_id | order_date | order_value |
| :--- | :--- | :--- |
| C001 | 2023-01-01 10:00:00 | 200.0 |
| C002 | 2023-01-02 12:30:00 | 300.0 |
| C001 | 2023-01-03 15:20:00 | 450.0 |
| C001 | 2023-01-04 11:00:00 | 230.0 |
| C002 | 2023-01-05 13:45:00 | 120.0 |
| C002 | 2023-01-06 14:30:00 | 350.0 |
| C005 | 2023-01-07 16:00:00 | 280.0 |
| C002 | 2023-01-08 17:30:00 | 500.0 |
| C001 | 2023-01-09 18:20:00 | 150.0 |
| C005 | 2023-01-10 19:00:00 | 190.0 |
| C006 | 2023-01-11 20:30:00 | 320.0 |
| C008 | 2023-01-12 21:45:00 | 310.0 |
| C006 | 2023-01-13 22:10:00 | 430.0 |
| C007 | 2023-01-14 23:00:00 | 210.0 |
| C008 | 2023-01-15 08:30:00 | 110.0 |
| C009 | 2023-02-01 09:40:00 | 250.0 |
| C010 | 2023-02-02 10:50:00 | 290.0 |
| C008 | 2023-02-03 11:55:00 | 390.0 |
| C009 | 2023-02-04 12:05:00 | 310.0 |
| C010 | 2023-02-05 13:15:00 | 130.0 |
| C009 | 2023-02-06 14:25:00 | 220.0 |
| C007 | 2023-02-07 15:35:00 | 330.0 |
| C003 | 2023-02-08 16:45:00 | 340.0 |
| C001 | 2023-02-09 17:55:00 | 360.0 |
| C010 | 2023-02-10 18:00:00 | 370.0 |
| C009 | 2023-02-11 19:05:00 | 380.0 |
| C002 | 2023-02-12 20:10:00 | 400.0 |
| C004 | 2023-02-13 21:15:00 | 410.0 |
| C007 | 2023-02-14 22:20:00 | 420.0 |
| C010 | 2023-02-15 23:25:00 | 440.0 |

### 쿼리 작성

```sql
WITH RFM AS (
	SELECT user_id, 
		   DATEDIFF(CURRENT_DATE(), MAX(order_date)) AS Recency,
           COUNT(*) AS Frequency,
           SUM(order_value) AS Monetary
    FROM rfm
    GROUP BY user_id
),
RFMScores AS
(
	SELECT *,
		NTILE(5) OVER (ORDER BY Recency DESC) AS RecencyScore,
		NTILE(5) OVER (ORDER BY Frequency DESC) AS FrequencyScore,
		NTILE(5) OVER (ORDER BY Monetary ASC) AS MonetaryScore
    FROM RFM
)
SELECT *, (RecencyScore + FrequencyScore + MonetaryScore) AS RFMScore
FROM RFMScores
ORDER BY user_id;
```

 
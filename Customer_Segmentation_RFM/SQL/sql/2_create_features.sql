/*
 [Project] RFM 기반 고객 세그먼테이션
 [Step 2] Feature Engineering & RFM Table Creation
 
 - 목적: 전처리된 데이터를 바탕으로 고객별 RFM 지표 및 심화 행동 패턴 피처 생성
 - 주요 피처:
    1. RFM (Recency, Frequency, Monetary)
    2. 파생 변수 (평균 구매 주기, 구매 제품 다양성, 취소율 등)
 - 최종 산출물: `user_data` 테이블
*/

CREATE OR REPLACE TABLE `continual-nomad-479202-p8.test.user_data` AS

WITH 
-- 1. Base Metrics: 기본 RFM 지표 집계
base_metrics AS (
  SELECT
    CustomerID,
    -- Recency: 전체 데이터 중 가장 최근 날짜(2011-12-09)로부터 며칠이 지났는지 계산
    -- (MAX(InvoiceDate) OVER())는 전체 데이터셋의 마지막 날짜를 의미함
    DATE_DIFF(MAX(DATE(InvoiceDate)) OVER(), MAX(DATE(InvoiceDate)), DAY) AS recency,
    
    -- Frequency: 구매 횟수 (고유한 송장 번호 수)
    COUNT(DISTINCT InvoiceNo) AS frequency,
    
    -- Item Count: 총 구매 아이템 수량
    SUM(Quantity) AS total_quantity,
    
    -- Monetary: 총 지출 금액 (소수점 첫째 자리 반올림)
    ROUND(SUM(UnitPrice * Quantity), 1) AS monetary_total
  FROM `continual-nomad-479202-p8.test.data`
  GROUP BY CustomerID
),

-- 2. Product Diversity: 구매 제품의 다양성 (고유 StockCode 수)
product_diversity AS (
  SELECT 
    CustomerID,
    COUNT(DISTINCT StockCode) AS unique_products_count
  FROM `continual-nomad-479202-p8.test.data`
  GROUP BY CustomerID
),

-- 3. Purchase Interval: 평균 구매 주기 (재방문 주기)
purchase_intervals AS (
  SELECT
    CustomerID,
    -- 구매 간격이 없으면(1회 구매) 0, 아니면 평균 간격 반올림
    COALESCE(ROUND(AVG(interval_days), 2), 0) AS average_interval
  FROM (
    SELECT
      CustomerID,
      -- 이전 구매일과의 차이(일수) 계산 (Window Function 활용)
      DATE_DIFF(
        DATE(InvoiceDate), 
        LAG(DATE(InvoiceDate)) OVER (PARTITION BY CustomerID ORDER BY InvoiceDate), 
        DAY
      ) AS interval_days
    FROM `continual-nomad-479202-p8.test.data`
  )
  WHERE interval_days IS NOT NULL -- 첫 구매(NULL) 제외
  GROUP BY CustomerID
),

-- 4. Cancel Behavior: 취소율 및 취소 빈도
cancel_behavior AS (
  SELECT
    CustomerID,
    COUNT(*) AS total_transactions_all, -- 취소 포함 전체 트랜잭션 수
    SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END) AS cancel_frequency
  FROM `continual-nomad-479202-p8.test.data`
  GROUP BY CustomerID
)

-- [Final Select] 모든 CTE를 결합하여 최종 데이터셋 생성
SELECT
  base.CustomerID,
  
  -- RFM Features
  base.recency,
  base.frequency,
  base.monetary_total AS monetary,
  
  -- Additional Features
  base.total_quantity, -- 총 구매 수량
  SAFE_DIVIDE(base.monetary_total, base.frequency) AS avg_monetary_per_purchase, -- 1회당 평균 구매액
  
  div.unique_products_count AS unique_products, -- 다양한 물건을 사는가?
  
  interval.average_interval, -- 얼마나 자주 오는가?
  
  cancel.cancel_frequency, -- 취소 횟수
  ROUND(SAFE_DIVIDE(cancel.cancel_frequency, cancel.total_transactions_all), 2) AS cancel_rate -- 취소율
  
FROM base_metrics base
LEFT JOIN product_diversity div ON base.CustomerID = div.CustomerID
LEFT JOIN purchase_intervals interval ON base.CustomerID = interval.CustomerID
LEFT JOIN cancel_behavior cancel ON base.CustomerID = cancel.CustomerID;
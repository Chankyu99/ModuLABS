-- 1. 회사 일별 매출액 구하기
 
-- 테이블 생성
CREATE TABLE `orders`
(	
	order_id VARCHAR(50) NOT NULL,
	user_id VARCHAR(50) NOT NULL, 
    order_timestamp TIMESTAMP
);

CREATE TABLE `payments`
(
	order_id VARCHAR(50) NOT NULL,
    value FLOAT
);

-- 데이터 삽입

INSERT INTO orders (order_id, user_id, order_timestamp) VALUES
('order_1', 'user_1', TIMESTAMP('2018-01-01 10:00:00')),
('order_2', 'user_2', TIMESTAMP('2018-01-01 12:30:00')),
('order_3', 'user_3', TIMESTAMP('2018-01-02 09:20:00')),
('order_4', 'user_4', TIMESTAMP('2018-01-02 10:15:00')),
('order_5', 'user_5', TIMESTAMP('2018-01-02 14:05:00')),
('order_6', 'user_6', TIMESTAMP('2018-01-03 11:00:00')),
('order_7', 'user_7', TIMESTAMP('2018-01-03 13:45:00')),
('order_8', 'user_8', TIMESTAMP('2018-01-04 15:30:00')),
('order_9', 'user_9', TIMESTAMP('2018-01-04 18:00:00')),
('order_10', 'user_10', TIMESTAMP('2018-01-05 20:30:00')),
('order_11', 'user_11', TIMESTAMP('2018-01-06 09:00:00')),
('order_12', 'user_12', TIMESTAMP('2018-01-06 12:45:00')),
('order_13', 'user_13', TIMESTAMP('2018-01-07 16:20:00')),
('order_14', 'user_14', TIMESTAMP('2018-01-08 17:35:00')),
('order_15', 'user_15', TIMESTAMP('2018-01-09 19:50:00')),
('order_16', 'user_16', TIMESTAMP('2018-01-10 21:15:00')),
('order_17', 'user_17', TIMESTAMP('2018-01-11 22:40:00')),
('order_18', 'user_18', TIMESTAMP('2018-01-12 23:05:00')),
('order_19', 'user_19', TIMESTAMP('2018-01-13 13:15:00')),
('order_20', 'user_20', TIMESTAMP('2018-01-14 14:25:00')),
('order_21', 'user_21', TIMESTAMP('2018-01-15 15:35:00')),
('order_22', 'user_22', TIMESTAMP('2018-01-16 16:45:00')),
('order_23', 'user_23', TIMESTAMP('2018-01-17 17:55:00')),
('order_24', 'user_24', TIMESTAMP('2018-01-18 18:05:00')),
('order_25', 'user_25', TIMESTAMP('2018-01-19 19:15:00')),
('order_26', 'user_26', TIMESTAMP('2018-01-20 20:25:00')),
('order_27', 'user_27', TIMESTAMP('2018-01-21 21:35:00')),
('order_28', 'user_28', TIMESTAMP('2018-01-22 22:45:00')),
('order_29', 'user_29', TIMESTAMP('2018-01-23 23:55:00')),
('order_30', 'user_30', TIMESTAMP('2018-01-24 11:05:00'));


-- 데이터 삽입
INSERT INTO payments (order_id, value) VALUES
('order_1', 100.00),
('order_2', 150.00),
('order_3', 200.00),
('order_4', 110.00),
('order_5', 120.00),
('order_6', 130.00),
('order_7', 140.00),
('order_8', 210.00),
('order_9', 220.00),
('order_10', 230.00),
('order_11', 240.00),
('order_12', 250.00),
('order_13', 260.00),
('order_14', 270.00),
('order_15', 280.00),
('order_16', 290.00),
('order_17', 300.00),
('order_18', 310.00),
('order_19', 320.00),
('order_20', 330.00),
('order_21', 340.00),
('order_22', 350.00),
('order_23', 360.00),
('order_24', 370.00),
('order_25', 380.00),
('order_26', 390.00),
('order_27', 400.00),
('order_28', 410.00),
('order_29', 420.00),
('order_30', 430.00);

-- 일별 매출액 구하기
SELECT DATE(o.order_timestamp) AS dt, SUM(p.value) AS daily_value
FROM orders AS o
JOIN payments AS p
ON o.order_id = p.order_id
GROUP BY dt
;

-- 10-2. PV, UV 계산하기 
-- 데이터 생성
CREATE TABLE `visits`
(
	log_id INT,
    user_id INT,
    page_url VARCHAR(50),
    timestamp TIMESTAMP
);

INSERT INTO `visits` (log_id, user_id, page_url, timestamp)
VALUES
 (1, 101, 'home', TIMESTAMP '2023-10-31 10:00:00'),
 (2, 102, 'product_detail', TIMESTAMP '2023-10-31 10:15:00'),
 (3, 101, 'cart', TIMESTAMP '2023-10-31 10:30:00'),
 (4, 103, 'home', TIMESTAMP '2023-10-31 11:00:00'),
 (5, 104, 'order', TIMESTAMP '2023-10-31 11:15:00'),
 (6, 102, 'product_detail', TIMESTAMP '2023-10-31 11:30:00'),
 (7, 105, 'product_detail', TIMESTAMP '2023-10-31 12:00:00'),
 (8, 101, 'home', TIMESTAMP '2023-10-31 12:15:00'),
 (9, 102, 'cart', TIMESTAMP '2023-10-31 12:30:00'),
 (10, 104, 'order', TIMESTAMP '2023-10-31 13:00:00');
 
 -- 페이지별 PV
SELECT page_url, COUNT(*) AS PV
FROM visits 
GROUP BY page_url;

-- 페이지별 Unique PV
SELECT page_url, COUNT(DISTINCT user_id) AS Unique_PV
FROM visits
GROUP BY page_url;

-- 페이지별 Visits
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

-- 페이지별 Unique Visitors
SELECT page_url, COUNT(DISTINCT user_id) AS UV
FROM visits
GROUP BY page_url;

-- 10-3. ARPU & ARPPU
-- 데이터 생성
CREATE TABLE `arpu` (
 user_id INT,
 purchase_date DATE,
 revenue FLOAT
);


INSERT INTO `arpu` (user_id, purchase_date, revenue) VALUES
(1, '2023-01-01', 10.00),
(2, '2023-01-01', 20.00),
(3, '2023-01-01', 0.00),
(1, '2023-01-02', 15.00),
(2, '2023-01-02', 0.00),
(3, '2023-01-02', 5.00),
(4, '2023-01-03', 20.00),
(5, '2023-01-03', 20.00),
(6, '2023-01-03', 0.00),
(1, '2023-01-04', 10.00),
(2, '2023-01-04', 25.00),
(7, '2023-01-04', 15.00),
(8, '2023-01-05', 40.00),
(5, '2023-01-05', 10.00),
(9, '2023-01-05', 0.00),
(10, '2023-01-06', 50.00),
(11, '2023-01-06', 35.00),
(6, '2023-01-06', 20.00),
(12, '2023-01-07', 15.00),
(13, '2023-01-07', 10.00),
(1, '2023-01-08', 5.00),
(14, '2023-01-08', 25.00),
(15, '2023-01-08', 30.00),
(16, '2023-01-09', 45.00),
(17, '2023-01-09', 0.00),
(18, '2023-01-10', 20.00),
(19, '2023-01-10', 35.00),
(20, '2023-01-11', 20.00),
(21, '2023-01-11', 25.00),
(22, '2023-01-12', 15.00);

-- ARPU & ARPPU 구하기
SELECT SUM(revenue) / COUNT(DISTINCT user_id) AS ARPU, 
SUM(revenue) / COUNT(DISTINCT CASE WHEN revenue > 0 THEN user_id END) AS ARPPU
FROM arpu;

-- 10-4. 퍼널 분석
-- 테이블 생성 
CREATE TABLE funnel (
 user_id INT,
 action VARCHAR(50),
 action_date DATE
);
-- 데이터 삽입
INSERT INTO funnel (user_id, action, action_date) VALUES
(4, 'visit', '2023-01-02'),
(5, 'visit', '2023-01-02'),
(6, 'visit', '2023-01-03'),
(4, 'signup', '2023-01-03'),
(5, 'add_to_cart', '2023-01-04'),
(6, 'purchase', '2023-01-04'),
(7, 'visit', '2023-01-04'),
(7, 'signup', '2023-01-05'),
(7, 'add_to_cart', '2023-01-06'),
(7, 'purchase', '2023-01-07'),
(8, 'visit', '2023-01-07'),
(8, 'signup', '2023-01-08'),
(8, 'add_to_cart', '2023-01-09'),
(9, 'visit', '2023-01-09'),
(9, 'signup', '2023-01-10'),
(10, 'visit', '2023-01-10'),
(10, 'add_to_cart', '2023-01-11'),
(10, 'purchase', '2023-01-12'),
(11, 'visit', '2023-01-12'),
(11, 'signup', '2023-01-13'),
(12, 'visit', '2023-01-13'),
(12, 'add_to_cart', '2023-01-14'),
(13, 'visit', '2023-01-14'),
(13, 'signup', '2023-01-15'),
(13, 'add_to_cart', '2023-01-16'),
(13, 'purchase', '2023-01-17'),
(14, 'visit', '2023-01-17'),
(14, 'signup', '2023-01-18'),
(15, 'visit', '2023-01-18'),
(15, 'signup', '2023-01-19'),
(15, 'add_to_cart', '2023-01-20'),
(15, 'purchase', '2023-01-21'),
(16, 'visit', '2023-01-21'),
(16, 'signup', '2023-01-22'),
(17, 'visit', '2023-01-22'),
(17, 'add_to_cart', '2023-01-23'),
(18, 'visit', '2023-01-23'),
(18, 'signup', '2023-01-24'),
(18, 'add_to_cart', '2023-01-25'),
(18, 'purchase', '2023-01-26'),
(19, 'visit', '2023-01-26'),
(19, 'signup', '2023-01-27'),
(20, 'visit', '2023-01-27'),
(20, 'add_to_cart', '2023-01-28'),
(21, 'visit', '2023-01-28'),
(21, 'signup', '2023-01-29'),
(21, 'add_to_cart', '2023-01-30'),
(21, 'purchase', '2023-01-31');

WITH base AS (
    SELECT
        action, COUNT(DISTINCT user_id) AS curr_user_count
    FROM funnel
    GROUP BY action
),
ordered_funnel AS (
    SELECT *,
        CASE WHEN action = 'visit' THEN 1
             WHEN action = 'signup' THEN 2
             WHEN action = 'add_to_cart' THEN 3
             WHEN action = 'purchase' THEN 4
        END AS step_order
    FROM base
);

WITH Funnel_Counts AS (
  SELECT
    COUNT(DISTINCT CASE WHEN action = 'visit' THEN user_id END) AS Visits,
    COUNT(DISTINCT CASE WHEN action = 'signup' THEN user_id END) AS Signups,
    COUNT(DISTINCT CASE WHEN action = 'add_to_cart' THEN user_id END) AS AddToCarts,
    COUNT(DISTINCT CASE WHEN action = 'purchase' THEN user_id END) AS Purchases
  FROM `funnel`
)
SELECT
  Visits,
  Signups,
  AddToCarts,
  Purchases,

  ROUND(Signups / Visits * 100, 2) AS SignupRate,
  ROUND(AddToCarts / Visits * 100, 2) AS AddToCartRate,
  ROUND(Purchases / Visits * 100, 2) AS PurchaseRate
FROM Funnel_Counts;

-- 10-5. 리텐션 분석
-- 테이블 생성
CREATE TABLE retention (
 user_id INT,
 invoice_date DATE,
 sales_amount FLOAT
);
-- 데이터 삽입
INSERT INTO retention (user_id, invoice_date, sales_amount) VALUES
(1, '2023-01-05', 120.00),
(2, '2023-01-15', 200.00),
(1, '2023-01-20', 80.00),
(3, '2023-01-25', 150.00),
(4, '2023-02-01', 100.00),
(2, '2023-02-05', 220.00),
(1, '2023-02-15', 50.00),
(3, '2023-02-18', 130.00),
(5, '2023-02-20', 175.00),
(6, '2023-02-25', 90.00),
(4, '2023-03-01', 110.00),
(5, '2023-03-05', 160.00),
(7, '2023-03-10', 200.00),
(6, '2023-03-15', 85.00),
(8, '2023-03-20', 140.00),
(2, '2023-03-25', 210.00),
(1, '2023-04-01', 95.00),
(9, '2023-04-05', 180.00),
(5, '2023-04-10', 155.00),
(10, '2023-04-15', 200.00),
(4, '2023-04-20', 120.00),
(7, '2023-04-25', 190.00),
(3, '2023-05-01', 140.00),
(11, '2023-05-05', 220.00),
(6, '2023-05-10', 75.00),
(8, '2023-05-15', 130.00),
(12, '2023-05-20', 165.00),
(9, '2023-05-25', 180.00),
(13, '2023-06-01', 150.00),
(10, '2023-06-05', 190.00);

-- 리텐션 분석
WITH user_cohort AS (
  -- 1. 유저별 최초 구매일(cohort_day) 찾기
  SELECT 
    *,
    MIN(invoice_date) OVER (PARTITION BY user_id) AS cohort_day
  FROM retention
),
cohort_index_added AS (
  -- 2. 코호트 그룹(월 단위) 및 인덱스(경과 월수) 계산
  SELECT
    *,
    STR_TO_DATE(DATE_FORMAT(cohort_day, '%Y-%m-01'), '%Y-%m-%d') AS cohort_group,
    TIMESTAMPDIFF(MONTH, cohort_day, invoice_date) AS cohort_index
  FROM user_cohort
)
-- 3. 코호트 그룹과 인덱스별 유니크 유저 수 집계
SELECT
  cohort_group,
  cohort_index,
  COUNT(DISTINCT user_id) AS user_count
FROM cohort_index_added
GROUP BY cohort_group, cohort_index
ORDER BY cohort_group, cohort_index;

-- 10-6. RFM 분석
-- 테이블 생성
CREATE TABLE rfm (
 user_id VARCHAR(50),
 order_date TIMESTAMP,
 order_value FLOAT
);

-- 데이터 삽입
INSERT INTO rfm (user_id, order_date, order_value) VALUES
('C001', TIMESTAMP('2023-01-01 10:00:00'), 200.0),
('C002', TIMESTAMP('2023-01-02 12:30:00'), 300.0),
('C001', TIMESTAMP('2023-01-03 15:20:00'), 450.0),
('C001', TIMESTAMP('2023-01-04 11:00:00'), 230.0),
('C002', TIMESTAMP('2023-01-05 13:45:00'), 120.0),
('C002', TIMESTAMP('2023-01-06 14:30:00'), 350.0),
('C005', TIMESTAMP('2023-01-07 16:00:00'), 280.0),
('C002', TIMESTAMP('2023-01-08 17:30:00'), 500.0),
('C001', TIMESTAMP('2023-01-09 18:20:00'), 150.0),
('C005', TIMESTAMP('2023-01-10 19:00:00'), 190.0),
('C006', TIMESTAMP('2023-01-11 20:30:00'), 320.0),
('C008', TIMESTAMP('2023-01-12 21:45:00'), 310.0),
('C006', TIMESTAMP('2023-01-13 22:10:00'), 430.0),
('C007', TIMESTAMP('2023-01-14 23:00:00'), 210.0),
('C008', TIMESTAMP('2023-01-15 08:30:00'), 110.0),
('C009', TIMESTAMP('2023-02-01 09:40:00'), 250.0),
('C010', TIMESTAMP('2023-02-02 10:50:00'), 290.0),
('C008', TIMESTAMP('2023-02-03 11:55:00'), 390.0),
('C009', TIMESTAMP('2023-02-04 12:05:00'), 310.0),
('C010', TIMESTAMP('2023-02-05 13:15:00'), 130.0),
('C009', TIMESTAMP('2023-02-06 14:25:00'), 220.0),
('C007', TIMESTAMP('2023-02-07 15:35:00'), 330.0),
('C003', TIMESTAMP('2023-02-08 16:45:00'), 340.0),
('C001', TIMESTAMP('2023-02-09 17:55:00'), 360.0),
('C010', TIMESTAMP('2023-02-10 18:00:00'), 370.0),
('C009', TIMESTAMP('2023-02-11 19:05:00'), 380.0),
('C002', TIMESTAMP('2023-02-12 20:10:00'), 400.0),
('C004', TIMESTAMP('2023-02-13 21:15:00'), 410.0),
('C007', TIMESTAMP('2023-02-14 22:20:00'), 420.0),
('C010', TIMESTAMP('2023-02-15 23:25:00'), 440.0);

-- RFM 분석
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


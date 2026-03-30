-- 고객 세그먼테이션(RFM) 프로젝트 — 데이터 전처리 및 피쳐 추출
-- DBMS : MySQL 8.0+

SET SQL_SAFE_UPDATES = 0;

-- 0. 테이블 생성 및 데이터 적재

-- InvoiceNo 에 'C' 접두어(취소)가 포함되므로 VARCHAR로 설정
ALTER TABLE data MODIFY COLUMN InvoiceNo VARCHAR(20);
TRUNCATE TABLE data;

LOAD DATA LOCAL INFILE '/Users/chankyulee/Downloads/data.csv'
INTO TABLE data
CHARACTER SET latin1                             -- 문자열 인코딩 에러 방지
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    InvoiceDate,
    UnitPrice,
    @CustomerID,                                 -- 빈칸 처리를 위해 변수 사용
    Country
)
SET CustomerID = NULLIF(@CustomerID, '');         -- 빈칸은 NULL로 치환


-- 1. EDA : 데이터 탐색 (실행 전 확인용)

-- 1-1. 스키마 및 샘플 확인
SELECT * FROM data LIMIT 5;

-- 1-2. 컬럼별 결측치 비율
SELECT
    column_name,
    ROUND((total - non_null_cnt) / total * 100, 2) AS missing_pct
FROM (
    SELECT 'InvoiceNo'   AS column_name, COUNT(InvoiceNo)   AS non_null_cnt, COUNT(*) AS total FROM data UNION ALL
    SELECT 'StockCode',                  COUNT(StockCode),                    COUNT(*)         FROM data UNION ALL
    SELECT 'Description',                COUNT(Description),                  COUNT(*)         FROM data UNION ALL
    SELECT 'Quantity',                   COUNT(Quantity),                     COUNT(*)         FROM data UNION ALL
    SELECT 'InvoiceDate',                COUNT(InvoiceDate),                  COUNT(*)         FROM data UNION ALL
    SELECT 'UnitPrice',                  COUNT(UnitPrice),                    COUNT(*)         FROM data UNION ALL
    SELECT 'CustomerID',                 COUNT(CustomerID),                   COUNT(*)         FROM data UNION ALL
    SELECT 'Country',                    COUNT(Country),                      COUNT(*)         FROM data
) AS col_stats;

-- 1-3. 취소 건(InvoiceNo LIKE 'C%') 비율 : 약 2.2%
SELECT
    ROUND(SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS canceled_ratio
FROM data;

-- 1-4. StockCode 내 숫자가 0~1개인 비정상 코드 탐색
WITH UniqueStockCodes AS (
    SELECT DISTINCT StockCode
    FROM data
)
SELECT StockCode,
       LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', '')) AS digit_cnt
FROM UniqueStockCodes
WHERE LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', '')) <= 1;

-- 1-5. UnitPrice 통계
SELECT
    MIN(UnitPrice)              AS min_price,
    MAX(UnitPrice)              AS max_price,
    ROUND(AVG(UnitPrice), 2)    AS avg_price
FROM data;


-- 2. 데이터 클리닝

-- 2-1. 중복 행 제거
CREATE TABLE data_clean AS
SELECT DISTINCT * FROM data;

DROP TABLE data;
RENAME TABLE data_clean TO data;

-- 2-2. 결측치(CustomerID, Description) 제거
DELETE FROM data
WHERE CustomerID IS NULL
   OR Description IS NULL;

-- 2-3. 비정상 StockCode 제거 (숫자 0~1개 : POST, D, C2, M 등)
DELETE FROM data
WHERE StockCode IN (
    SELECT StockCode FROM (
        SELECT DISTINCT StockCode
        FROM data
        WHERE LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', '')) <= 1
    ) AS tmp
);

-- 2-4. 서비스 관련 Description 제거 & 대문자 표준화
DELETE FROM data
WHERE Description IN ('Next Day Carriage', 'High Resolution Image');

UPDATE data
SET Description = UPPER(Description);

-- 2-5. UnitPrice 이상값(0 이하) 제거
DELETE FROM data
WHERE UnitPrice <= 0;

-- 2-6. 최종 행 수 확인
SELECT COUNT(*) AS cleaned_row_count FROM data;


-- 3. RFM 피쳐 추출 & 추가 파생 변수 → 최종 user_data 테이블 생성

CREATE TABLE user_data AS
WITH

-- (R) Recency : 마지막 구매일 ↔ 데이터 내 최신 날짜 간 차이(일)
recency_cte AS (
    SELECT
        CustomerID,
        DATEDIFF(
            (SELECT MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) FROM data),
            MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i')))
        ) AS recency
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
),

-- (F) Frequency : 고유 주문 건수, 총 구매 수량
frequency_cte AS (
    SELECT
        CustomerID,
        COUNT(DISTINCT InvoiceNo) AS purchase_cnt,
        SUM(Quantity)             AS item_cnt
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
),

-- (M) Monetary : 총 지출액, 건당 평균 지출액
monetary_cte AS (
    SELECT
        CustomerID,
        ROUND(SUM(UnitPrice * Quantity), 1) AS user_total
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
),

-- 제품 다양성 : 고유 StockCode 수
diversity_cte AS (
    SELECT
        CustomerID,
        COUNT(DISTINCT StockCode) AS unique_products
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
),

-- 평균 구매 주기(일) : LAG 윈도우 함수 활용
interval_cte AS (
    SELECT
        CustomerID,
        IFNULL(ROUND(AVG(interval_days), 2), 0) AS average_interval
    FROM (
        SELECT
            CustomerID,
            DATEDIFF(
                DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i')),
                LAG(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i')))
                    OVER (PARTITION BY CustomerID
                          ORDER BY STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))
            ) AS interval_days
        FROM data
        WHERE CustomerID IS NOT NULL
    ) AS sub
    GROUP BY CustomerID
),

-- 취소 경향성 : 취소 빈도 / 전체 거래
cancel_cte AS (
    SELECT
        CustomerID,
        COUNT(*)                                                      AS total_transactions,
        SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END)         AS cancel_frequency,
        ROUND(
            SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )                                                             AS cancel_rate
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
)

-- 최종 조합
SELECT
    r.CustomerID,
    r.recency,
    f.purchase_cnt,
    f.item_cnt,
    m.user_total,
    ROUND(m.user_total / f.purchase_cnt, 1)  AS user_average,
    d.unique_products,
    i.average_interval,
    c.total_transactions,
    c.cancel_frequency,
    c.cancel_rate
FROM recency_cte     AS r
JOIN frequency_cte   AS f ON r.CustomerID = f.CustomerID
JOIN monetary_cte    AS m ON r.CustomerID = m.CustomerID
JOIN diversity_cte   AS d ON r.CustomerID = d.CustomerID
LEFT JOIN interval_cte   AS i ON r.CustomerID = i.CustomerID
LEFT JOIN cancel_cte     AS c ON r.CustomerID = c.CustomerID;


-- 4. 최종 결과 확인

-- 총 4,362명 고유 고객의 RFM + 파생 변수 테이블
SELECT * FROM user_data LIMIT 10;
SELECT COUNT(*) AS total_customers FROM user_data;

-- 중간 테이블 정리 (존재 시)
DROP TABLE IF EXISTS user_r;
DROP TABLE IF EXISTS user_rf;
DROP TABLE IF EXISTS user_rfm;

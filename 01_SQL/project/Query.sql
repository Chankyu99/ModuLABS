LOAD DATA LOCAL INFILE '/Users/chankyulee/Downloads/data.csv' 
INTO TABLE data
CHARACTER SET latin1                 -- ✨ 핵심: 이 줄을 꼭 추가해 주세요!
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n' 
IGNORE 1 ROWS
(
    @InvoiceNo, 
    @StockCode, 
    @Description, 
    @Quantity, 
    @InvoiceDate, 
    @UnitPrice, 
    @CustomerID, 
    @Country
)
SET 
    InvoiceNo = @InvoiceNo,
    StockCode = @StockCode,
    Description = @Description,
    Quantity = @Quantity,
    InvoiceDate = @InvoiceDate,
    UnitPrice = @UnitPrice,
    CustomerID = NULLIF(@CustomerID, ''),
    Country = @Country;
    
-- Schema 확인
SELECT *
FROM data
LIMIT 1;

-- 데이터 수 세기
SELECT COUNT(InvoiceNO) AS COUNT_InvoiceNO,
       COUNT(StockCode) AS COUNT_StockCode,
       COUNT(Description) AS COUNT_Description,
       COUNT(Quantity) AS COUNT_Quantity,
       COUNT(InvoiceDate) AS COUNT_InvoiceDate,
       COUNT(UnitPrice) AS COUNT_UnitPrice,
       COUNT(CustomerID) AS COUNT_CustomerID,
       COUNT(Country) AS COUNT_Country
FROM data;

-- 1. 결측치 제거
-- 결측치 비율 확인 -> 24.93%
SELECT
    'InvoiceNo' AS column_name,
    ROUND(SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM data;

SELECT column_name, ROUND((total - column_value) / total * 100, 2)
FROM
(
    SELECT 'InvoiceNo' AS column_name, COUNT(InvoiceNo) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'StockCode' AS column_name, COUNT(StockCode) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'Description' AS column_name, COUNT(Description) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'Quantity' AS column_name, COUNT(Quantity) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'InvoiceDate' AS column_name, COUNT(InvoiceDate) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'UnitPrice' AS column_name, COUNT(UnitPrice) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'CustomerID' AS column_name, COUNT(CustomerID) AS column_value, COUNT(*) AS total FROM data UNION ALL
    SELECT 'Country' AS column_name, COUNT(Country) AS column_value, COUNT(*) AS total FROM data
) AS column_data;

-- Description : 같은 제품(StockCode)이 항상 같은 상세설명(Description)을 가지고 있지 않아 데이터 일관성 문제 발생
SELECT Description
FROM data
WHERE StockCode = '85123A'
GROUP BY Description;

-- 결측치 삭제
DELETE FROM data
WHERE CustomerID IS NULL OR Description IS NULL;

-- 2. 중복값 처리
-- 중복값 확인
WITH temp AS (
	SELECT *, COUNT(*) AS cnt_row
    FROM data
    GROUP BY InvoiceNO, StockCode, Description, QUantity, InvoiceDate, UnitPrice, CustomerID, Country
	HAVING cnt_row > 1
)
-- 중복개수 확인 : 4837개
SELECT COUNT(*) FROM temp;

-- 중복 없는 데이터만 모아 임시 테이블 생성
CREATE TABLE data_tmp AS
SELECT DISTINCT * FROM data;

-- 기존 원본 테이블 삭제 (혹은 데이터를 비움)
DROP TABLE data;

-- 임시 테이블 이름을 원본 이름으로 변경
RENAME TABLE data_tmp TO data;

-- 최종 남은 행 개수 확인
SELECT COUNT(*) FROM data;

-- 3. 컬럼별 오류값 처리
-- 3-1. InvoiceNo 
-- 고유 개수 확인 : 18537개
SELECT COUNT(DISTINCT InvoiceNo) FROM data;

-- 데이터 잘못넣어서 다시 처음부터 시도

ALTER TABLE data MODIFY COLUMN InvoiceNo VARCHAR(20);
SET SQL_SAFE_UPDATES = 0;
TRUNCATE TABLE data;

LOAD DATA LOCAL INFILE '/Users/chankyulee/Downloads/data.csv' 
INTO TABLE data
CHARACTER SET latin1                 -- 문자열 에러 방지 
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
    @CustomerID,                     -- 빈칸 처리를 위해 변수 사용 
    Country
)
SET CustomerID = NULLIF(@CustomerID, ''); -- 빈칸은 NULL로 치환

SELECT COUNT(*) FROM data WHERE CustomerID IS NULL;
-- 대략 13만 개 정도 나오면 정상입니다.

-- 임시 테이블 생성 (중복 제거본)
CREATE TABLE data_clean AS 
SELECT DISTINCT * FROM data;

-- 원본 삭제 및 이름 변경
DROP TABLE data;
RENAME TABLE data_clean TO data;

-- 다시 중복값, 결측치 제거

SELECT COUNT(*) FROM data WHERE InvoiceNo LIKE 'C%';

-- 취소건의 비율 : 2.2 -> RFM 분석의 취지에 맞게 취소건도 분석
SELECT ROUND(SUM(CASE WHEN InvoiceNO LIKE "C%" THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS canceled_ratio
FROM data;

-- 3-2. StockCode
SELECT COUNT(DISTINCT StockCode) 
FROM data;

-- 어떤 제품이 가장 많이 팔렸지?
SELECT StockCode, COUNT(*) AS sell_cnt
FROM data
GROUP BY StockCode
ORDER BY sell_cnt DESC
LIMIT 10;

-- 문자열 내 숫자 길이
WITH UniqueStockCodes AS (
  SELECT DISTINCT StockCode
  FROM data
)
SELECT
  LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', '')) AS number_count,
  COUNT(*) AS stock_cnt
FROM UniqueStockCodes
GROUP BY number_count
ORDER BY stock_cnt DESC;

-- 숫자가 0~1개인 값들에는 어떤 코드들이 들어가 있는지를 확인
WITH StockCount AS (
    SELECT StockCode,
           LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', '')) AS number_count
    FROM data
)
SELECT DISTINCT StockCode, number_count
FROM StockCount
WHERE number_count <= 1;

-- 해당 코드 값들을 가지고 있는 데이터 수는 전체 데이터 수 대비 비율?
SELECT StockCode, ROUND (COUNT(*) / (SELECT COUNT(*) FROM data) * 100, 2) AS ratio
FROM data
GROUP BY StockCode
HAVING StockCode IN ('POST','D','C2','M','BANK CHARGES','PADS','DOT','CRUK');

-- 제품과 관련되지 않은 거래 기록을 제거
DELETE FROM data 
WHERE StockCode IN (
    SELECT StockCode FROM (
        -- 서브쿼리를 한 번 더 감싸서 별칭(tmp)을 붙입니다.
        SELECT StockCode 
        FROM data 
        WHERE (LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, '[0-9]', ''))) <= 1
    ) AS tmp
);

-- 3-3. Description
SELECT Description, COUNT(*) AS description_cnt
FROM data
GROUP BY Description
ORDER BY description_cnt DESC
LIMIT 30;

-- 대소문자가 혼합된 Description이 있는지 확인 : 19개
SELECT DISTINCT Description
FROM data
-- 컬럼 자체를 바이너리화하여 비교하면 대소문자를 엄격히 구분합니다.
WHERE REGEXP_LIKE(BINARY Description, '[a-z]');

-- 1. 'Next Day Carriage'와 'High Resolution Image'와 같은 서비스 관련 정보를 포함하는 행들을 제거
-- 2. 대소문자를 혼합해서 사용하는 경우, 대문자로 표준화하여 데이터셋 전체에서 일관성을 유지할 수 있다. 이는 대소문자에 의한 중복 항목의 가능성을 줄이는 데에도 도움

-- 서비스 관련 정보를 포함하는 행들을 제거
DELETE FROM data
WHERE Description IN ('Next Day Carriage','High Resolution Image');

-- 대소문자를 혼합하고 있는 데이터를 대문자로 표준화
UPDATE data 
SET Description = UPPER(Description);

-- 3-4. Unitprice
-- 최솟값, 최댓값, 평균
SELECT MAX(Unitprice) AS max_price, MIN(Unitprice) AS min_price, AVG(Unitprice) AS avg_price
FROM data;

-- 단가가 0원인 거래의 개수, 구매 수량(Quantity)의 최솟값, 최댓값, 평균
SELECT COUNT(*) AS cnt_quantity,  
       MIN(Quantity) AS min_quantity, 
       MAX(Quantity) AS max_quantity, 
       AVG(Quantity) AS avg_quantity
FROM data
WHERE UnitPrice = 0;

-- 33개의 오류값 제거
-- 1. 삭제 전, 마지막으로 대상 데이터(33건 예상) 확인
SELECT * FROM data WHERE UnitPrice <= 0;

-- 2. 안전 모드 해제 및 데이터 삭제
SET SQL_SAFE_UPDATES = 0;
DELETE FROM data WHERE UnitPrice <= 0;

-- 3. 최종 남은 데이터 행 개수 확인 (포트폴리오 기록용)
SELECT COUNT(*) AS final_row_count FROM data;

-- 4. RFM 분석

-- 4-1. Recency
-- 시/분/초 제외하기
SELECT 
    DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y')) AS InvoiceDay, 
    t.* 
FROM data AS t;

-- 가장 최근 구매 일자
SELECT MAX(InvoiceDate) AS most_recent_date
FROM data;

-- 유저 별 가장 큰 InvoiceDay를 찾아 가장 최근 구매일로 저장
SELECT
  CustomerID,
  MAX(InvoiceDate) AS InvoiceDay
FROM data
GROUP BY CUstomerID
ORDER BY CustomerID;

-- 가장 최근 일자(most_recent_date)와 유저별 마지막 구매일(InvoiceDay)간의 차이를 계산
SELECT 
    CustomerID,
    -- 전체 데이터 중 가장 최근 날짜와 고객별 마지막 날짜의 차이(일수) 계산
    DATEDIFF(
        (SELECT MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) FROM data), 
        MaxInvoiceDay
    ) AS recency
FROM (
    SELECT 
        CustomerID,
        MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) AS MaxInvoiceDay
    FROM data
    WHERE CustomerID IS NOT NULL -- 분석의 정확도를 위해 결측치 제외
    GROUP BY CustomerID
) AS t;

-- 최종 데이터 셋에 필요한 데이터들을 각각 정제해서 이어붙이도록 하겠습니다. 지금까지의 결과를 user_r이라는 이름의 테이블로 저장
CREATE TABLE user_r AS
SELECT 
    CustomerID,
    -- 전체 최신 날짜와 각 고객의 최신 날짜 차이(일수) 계산
    DATEDIFF(
        (SELECT MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) FROM data), 
        MaxInvoiceDay
    ) AS recency
FROM (
    SELECT 
        CustomerID, 
        MAX(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) AS MaxInvoiceDay
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
) AS t;

-- 4-2. Frequency
-- 4-2-1. 전체 거래 건수 계산
SELECT
  CustomerID,
  COUNT(DISTINCT InvoiceNo) AS purchase_cnt
FROM data
GROUP BY CustomerID
ORDER BY purchase_cnt DESC;

-- 4-2-2. 구매한 아이템의 총 수량 계산
SELECT  CustomerID,
  SUM(Quantity) AS item_cnt
FROM data
GROUP BY CustomerID;

-- 합쳐서 user_rf 라는 테이블에 저장
CREATE TABLE user_rf AS

WITH purchase_cnt AS ( 
  -- (1) 전체 거래 건수 (F: Frequency)
  SELECT
    CustomerID,
    COUNT(DISTINCT InvoiceNo) AS purchase_cnt
  FROM data
  GROUP BY CustomerID
),

item_cnt AS (
  -- (2) 총 수량 (M: Monetary 관련)
  SELECT 
    CustomerID,
    SUM(Quantity) AS item_cnt
  FROM data
  GROUP BY CustomerID
)

SELECT
  pc.CustomerID,
  pc.purchase_cnt,          
  ic.item_cnt,
  ur.recency
FROM purchase_cnt AS pc
JOIN item_cnt AS ic
  ON pc.CustomerID = ic.CustomerID
JOIN user_r AS ur
  ON pc.CustomerID = ur.CustomerID;
  
-- 4-3. Monetary
-- 4-3-1. 고객별 총 지출액 계산
SELECT CustomerID, ROUND(SUM(Unitprice * Quantity),1) AS user_total
FROM data
GROUP BY CustomerID;

-- 4-3-2. 고객별 평균 거래 금액 계산
-- 고객별 총 지출액 계산
SELECT CustomerID, ROUND(SUM(UnitPrice * Quantity),1) AS user_total
FROM data
GROUP BY CustomerID;

-- 고객별 평균 거래 금액 계산
CREATE TABLE user_rfm AS
SELECT 
  rf.CustomerID AS CustomerId,
  rf.purchase_cnt,
  rf.item_cnt,
  rf.recency,
  ut.user_total,
  ROUND(ut.user_total / rf.purchase_cnt, 1) AS user_average
FROM user_rf AS rf
LEFT JOIN (
  -- 고객 별 총 지출액
  SELECT CustomerID, ROUND(SUM(UnitPrice * Quantity),1) AS user_total
FROM data
GROUP BY CustomerID
) AS ut
ON rf.CustomerID = ut.CustomerID;

-- 4.4 RFM 통합 테이블 출력하기
SELECT *
FROM user_rfm;

-- 총 4362명 고유 고객의 RFM 테이블 완성

-- 5. 추가 Feature 추출
-- RFM 이외에 유저별 구매 패턴 찾기
-- 5-1. 구매하는 제품의 다양성
CREATE TABLE user_data AS  
WITH unique_products AS (
  SELECT
    CustomerID,
    COUNT(DISTINCT StockCode) AS unique_products
  FROM data
  GROUP BY CustomerID
)
SELECT ur.*, up.unique_products
FROM user_rfm AS ur
JOIN unique_products AS up
ON ur.CustomerID = up.CustomerID;

-- 5-2. 평균 구매 주기
CREATE TABLE user_data_final AS  
WITH purchase_intervals AS (
    SELECT
        CustomerID,
        -- (2) 평균 소요 일수 계산 (NULL이면 0으로 처리)
        IFNULL(ROUND(AVG(interval_days), 2), 0) AS average_interval
    FROM (
        -- (1) 이전 구매일(LAG)과의 차이 계산
        SELECT
            CustomerID,
            DATEDIFF(
                DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i')), 
                LAG(DATE(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) OVER (PARTITION BY CustomerID ORDER BY STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))
            ) AS interval_days
        FROM data
        WHERE CustomerID IS NOT NULL
    ) AS sub
    GROUP BY CustomerID
)
SELECT 
    u.*, 
    pi.average_interval -- EXCEPT 대신 필요한 컬럼만 선택
FROM user_data AS u
LEFT JOIN purchase_intervals AS pi
ON u.CustomerID = pi.CustomerID;

-- 기존 테이블 교체
DROP TABLE user_data;
RENAME TABLE user_data_final TO user_data;

-- 5-3. 구매 취소 경향성
-- 취소 빈도, 취소 비율을 계산하고 통합
CREATE TABLE user_data_updated AS
WITH TransactionInfo AS (
    SELECT
        CustomerID,
        COUNT(*) AS total_transactions,
        -- 취소 건수(C로 시작) 계산
        SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END) AS cancel_frequency
    FROM data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
)
SELECT 
    u.*, 
    t.total_transactions,
    t.cancel_frequency,
    ROUND(t.cancel_frequency / NULLIF(t.total_transactions, 0), 2) AS cancel_rate
FROM user_data AS u
LEFT JOIN TransactionInfo AS t
ON u.CustomerID = t.CustomerID;

DROP TABLE user_data;
RENAME TABLE user_data_updated TO user_data;


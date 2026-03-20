/*
 [Project] RFM 기반 고객 세그먼테이션
 [Step 1] Exploratory Data Analysis (EDA) & Data Cleaning
 
 - 목적: 원본 데이터의 결측치, 중복값, 이상치(Outlier)를 식별하고 정제하여 분석 가능한 상태로 만듦
 - 주요 작업:
    1. 데이터 프로파일링 (누락값 비율, 중복 확인)
    2. 결측치 및 중복 제거
    3. 비즈니스 로직에 맞지 않는 데이터(테스트 코드, 관리비용 등) 제거
    4. 데이터 표준화 (대문자 변환 등)
*/

-- ========================================================
-- 1. Data Profiling (데이터 품질 진단)
-- ========================================================

-- 1-1. 컬럼별 결측치(Null) 비율 계산
SELECT 'InvoiceNo' AS column_name, ROUND(SUM(CASE WHEN InvoiceNo IS NULL THEN 1 ELSE 0 END) / COUNT (*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'StockCode' AS column_name, ROUND(SUM(CASE WHEN StockCode IS NULL THEN 1 ELSE 0 END) / COUNT (*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'Description' AS column_name, ROUND(SUM(CASE WHEN Description IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'Quantity' AS column_name, ROUND(SUM(CASE WHEN Quantity IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'InvoiceDate' AS column_name, ROUND(SUM(CASE WHEN InvoiceDate IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'UnitPrice' AS column_name, ROUND(SUM(CASE WHEN UnitPrice IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'CustomerID' AS column_name, ROUND(SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`
UNION ALL
SELECT 'Country' AS column_name, ROUND(SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM `continual-nomad-479202-p8.test.data`;

-- 1-2. 이상한 StockCode 탐지 (숫자가 0~1개만 포함된 코드)
-- 정규표현식을 사용하여 제품 코드가 아닌 불필요한 데이터(POST, 수수료 등) 식별
SELECT DISTINCT StockCode, number_count
FROM (
  SELECT StockCode,
    LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count
  FROM `continual-nomad-479202-p8.test.data`
) 
WHERE number_count <= 1;


-- ========================================================
-- 2. Data Cleaning (데이터 정제 실행)
-- ========================================================

-- 2-1. 결측치 제거
-- 고객 식별이 불가능하거나(CustomerID Null), 상품 설명이 없는(Description Null) 데이터 삭제
DELETE FROM `continual-nomad-479202-p8.test.data`
WHERE CustomerID IS NULL OR Description IS NULL;

-- 2-2. 중복값 제거
-- 모든 컬럼의 값이 완벽히 동일한 중복 행 제거 (DISTINCT 활용하여 테이블 재생성)
CREATE OR REPLACE TABLE `continual-nomad-479202-p8.test.data` AS
SELECT DISTINCT *
FROM `continual-nomad-479202-p8.test.data`;

-- 2-3. 유효하지 않은 StockCode 제거
-- 제품 판매와 관련 없는 배송비(POST), 수수료(BANK CHARGES), 테스트 코드(Test 등) 제거
DELETE FROM `continual-nomad-479202-p8.test.data`
WHERE StockCode IN ('POST', 'D', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT', 'CRUK');

-- 2-4. 불필요한 서비스 내역 제거
DELETE FROM `continual-nomad-479202-p8.test.data`
WHERE Description IN ('Next Day Carriage', 'High Resolution Image');

-- 2-5. 데이터 표준화
-- Description 컬럼을 모두 대문자로 변환하여 일관성 확보
CREATE OR REPLACE TABLE `continual-nomad-479202-p8.test.data` AS
SELECT 
  * EXCEPT (Description),
  UPPER(Description) AS Description
FROM `continual-nomad-479202-p8.test.data`;

-- 2-6. 가격(UnitPrice) 이상치 제거
-- 가격이 0원인(매출에 기여하지 않는) 데이터 제거
DELETE FROM `continual-nomad-479202-p8.test.data`
WHERE UnitPrice = 0;
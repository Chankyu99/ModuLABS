-- 테이블 생성   
CREATE TABLE `ex1`(
  order_id VARCHAR(50),
  user_id VARCHAR(50),
  item_id VARCHAR(50),
  price DOUBLE
);


-- 데이터 삽입
INSERT INTO ex1 (order_id, user_id, item_id, price)
VALUES
   ('order_001', 'customer_01', 'product_001', 100.0), 
   ('order_001', 'customer_01', 'product_002', 150.0), 
   ('order_002', 'customer_02', 'product_003', 200.0), 
   ('order_003', 'customer_03', 'product_004', 80.0), 
   ('order_004', 'customer_04', 'product_005', 220.0), 
   ('order_004', 'customer_04', 'product_006', 90.0), 
   ('order_005', 'customer_05', 'product_007', 140.0), 
   ('order_006', 'customer_01', 'product_008', 110.0), 
   ('order_007', 'customer_06', 'product_009', 300.0), 
   ('order_008', 'customer_07', 'product_010', 130.0), 
   ('order_009', 'customer_03', 'product_0011', 250.0), 
   ('order_0010', 'customer_08', 'product_012', 90.0);
   
   
-- 총 주문 금액이 높은 상위 3명의 손님 리스트 출력
SELECT user_id, SUM(price) AS total_spent 
FROM ex1
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 3;

-- 8-3. 
-- 테이블 생성   
CREATE TABLE `ex2` (
   table_id VARCHAR(50),
   total_bill DOUBLE,
   tip DOUBLE,
   gender VARCHAR(50),
   party_size INT,
   day VARCHAR(50),
   time VARCHAR(50)
);


-- 데이터 삽입
INSERT INTO ex2 (table_id, total_bill, tip, gender, party_size, day, time)
VALUES
   ('T01', 24.59, 3.61, 'Female', 2, 'Sun', 'Dinner'),
   ('T02', 21.01, 3.50, 'Male', 3, 'Sun', 'Dinner'),
   ('T03', 23.68, 3.31, 'Male', 2, 'Sun', 'Dinner'),
   ('T04', 24.59, 3.61, 'Female', 4, 'Sun', 'Dinner'),
   ('T05', 25.29, 4.71, 'Male', 4, 'Sun', 'Dinner'),
   ('T06', 8.77, 2.00, 'Male', 2, 'Sun', 'Dinner'),
   ('T07', 26.88, 3.12, 'Male', 2, 'Sun', 'Dinner'),
   ('T08', 15.04, 1.96, 'Male', 2, 'Sun', 'Dinner'),
   ('T09', 14.78, 3.23, 'Male', 2, 'Sun', 'Dinner'),
   ('T10', 10.27, 1.71, 'Male', 2, 'Sun', 'Dinner'),
   ('T11', 35.26, 5.00, 'Female', 4, 'Sun', 'Dinner'),
   ('T12', 15.42, 1.57, 'Male', 2, 'Sun', 'Dinner');

SELECT *
FROM ex2
WHERE total_bill > (SELECT AVG(total_bill) FROM ex2);

WITH AverageBill AS (
  SELECT AVG(total_bill) AS avg_bill
  FROM ex2
)
SELECT *
FROM ex2, AverageBill
WHERE ex2.total_bill > AverageBill.avg_bill;

--- 8-4
-- 테이블 생성   
CREATE TABLE `ex3` (
   PRODUCT_ID INT NOT NULL,
   PRODUCT_LINE VARCHAR(50) NOT NULL,
   TOTAL_ORDER INT NOT NULL
);
-- 데이터 삽입
INSERT INTO `ex3` (PRODUCT_ID, PRODUCT_LINE, TOTAL_ORDER)
VALUES
(101, 'Sneakers', 3200),
(102, 'Boots', 2500),
(103, 'Sandals', 1800),
(104, 'Running Shoes', 2100),
(105, 'Sneakers', 3000),
(106, 'Boots', 2700),
(107, 'Sandals', 1600),
(108, 'Running Shoes', 2200),
(109, 'Sneakers', 3100),
(110, 'Boots', 2600),
(111, 'Sandals', 1500),
(112, 'Running Shoes', 2000),
(113, 'Sneakers', 3300),
(114, 'Boots', 2400),
(115, 'Sandals', 1700),
(116, 'Running Shoes', 2300),
(117, 'Sneakers', 3400),
(118, 'Boots', 2800),
(119, 'Sandals', 1900),
(120, 'Running Shoes', 2500);

SELECT PRODUCT_LINE, SUM(TOTAL_ORDER) AS TOTAL_ORDERS
FROM ex3
GROUP BY PRODUCT_LINE
ORDER BY TOTAL_ORDERS DESC
LIMIT 1;

-- 테이블 생성   
CREATE TABLE `ex4-1` (
   MEMBER_ID VARCHAR(50) NOT NULL,
   MEMBER_NAME VARCHAR(50) NOT NULL,
   TLNO VARCHAR(50),
   GENDER VARCHAR(50),
   DATE_OF_BIRTH DATE
);
CREATE TABLE `ex4-2` (
   REVIEW_ID VARCHAR(50) NOT NULL,
   REST_ID VARCHAR(50),
   MEMBER_ID VARCHAR(50),
   REVIEW_SCORE INT,
   REVIEW_TEXT VARCHAR(50),
   REVIEW_DATE DATE
);
-- 데이터 삽입
INSERT INTO `ex4-1` (MEMBER_ID, MEMBER_NAME, TLNO, GENDER, DATE_OF_BIRTH)
VALUES
('kevin@gmail.com', 'Kevin', '01076432111', 'M', '1992-02-12'),
('james@gmail.com', 'James', '01032324117', 'M', '1992-02-22'),
('alice@gmail.com', 'Alice', '01023258688', 'W', '1993-02-23'),
('maria@gmail.com', 'Maria', '01076482209', 'W', '1993-03-16'),
('duke@gmail.com', 'Duke', '01017626711', 'M', '1990-11-30');


INSERT INTO `ex4-2` (REVIEW_ID, REST_ID, MEMBER_ID, REVIEW_SCORE, REVIEW_TEXT, REVIEW_DATE)
VALUES
('R000000065', '00028', 'alice@gmail.com', 5, 'The broth for the shabu-shabu was clean and tasty', '2022-04-12'),
('R000000066', '00039', 'duke@gmail.com', 5, 'The kimchi stew was the best', '2022-02-12'),
('R000000067', '00028', 'duke@gmail.com', 5, 'Loved the generous amount of ham', '2022-02-22'),
('R000000068', '00035', 'kevin@gmail.com', 5, 'The aged sashimi was fantastic', '2022-02-15'),
('R000000069', '00035', 'maria@gmail.com', 4, 'No fishy smell at all', '2022-04-16'),
('R000000070', '00040', 'kevin@gmail.com', 4, 'Cozy atmosphere and great experience', '2022-05-10'),
('R000000071', '00041', 'kevin@gmail.com', 5, 'Top-notch service and taste', '2022-05-12'),
('R000000072', '00042', 'kevin@gmail.com', 3, 'Average taste but friendly staff', '2022-05-14'),
('R000000073', '00043', 'james@gmail.com', 5, 'Both the taste and service were satisfying', '2022-05-15'),
('R000000074', '00044', 'alice@gmail.com', 4, 'The ingredients were fresh', '2022-05-16');

-- 8-5
SELECT A.MEMBER_NAME, B.REVIEW_DATE, B.REVIEW_TEXT
FROM `ex4-1` AS A
JOIN `ex4-2` AS B
ON A.MEMBER_ID = B.MEMBER_ID
WHERE A.MEMBER_ID = (SELECT MEMBER_ID FROM `ex4-2` GROUP BY MEMBER_ID ORDER BY COUNT(*) DESC LIMIT 1) 
ORDER BY B.REVIEW_DATE;

-- 테이블 생성  
CREATE TABLE `ex5_patient` (
   PATIENT_NO VARCHAR(50),
   PATIENT_NAME VARCHAR(50),
   GENDER VARCHAR(50),
   AGE INT
);
CREATE TABLE `ex5_apnt` (
   APNT_YMD TIMESTAMP,
   APNT_NO INT,
   PATIENT_NO VARCHAR(50),
   APNT_CANCEL_YN VARCHAR(50),
   TREATMENT_STATUS VARCHAR(50)
);
-- 데이터 삽입
INSERT INTO `ex5_patient` (PATIENT_NO, PATIENT_NAME, GENDER, AGE) VALUES
('PT22000024', '영희', 'W', 30),
('PT22000035', '철수', 'M', 45),
('PT22000046', '은지', 'W', 20),
('PT22000057', '준호', 'M', 35),
('PT22000068', '수민', 'W', 28),
('PT22000079', '현준', 'M', 52),
('PT22000080', '서연', 'W', 22),
('PT22000091', '지후', 'M', 40),
('PT22000102', '민서', 'W', 33),
('PT22000113', '예준', 'M', 47);


INSERT INTO `ex5_apnt` (APNT_YMD, APNT_NO, PATIENT_NO, APNT_CANCEL_YN, TREATMENT_STATUS) VALUES
(TIMESTAMP '2024-01-01 09:00:00', 49, 'PT22000068', 'Y', 'Completed'),
(TIMESTAMP '2024-01-01 09:30:00', 44, 'PT22000024', 'N', 'Completed'),
(TIMESTAMP '2024-01-01 10:00:00', 50, 'PT22000079', 'N', 'Completed'),
(TIMESTAMP '2024-01-01 10:30:00', 45, 'PT22000035', 'N', ''),
(TIMESTAMP '2024-01-01 11:00:00', 51, 'PT22000080', 'N', ''),
(TIMESTAMP '2024-01-01 11:30:00', 47, 'PT22000046', 'N', ''),
(TIMESTAMP '2024-01-01 13:00:00', 52, 'PT22000091', 'N', ''),
(TIMESTAMP '2024-01-01 14:30:00', 48, 'PT22000057', 'N', ''),
(TIMESTAMP '2024-01-01 15:00:00', 53, 'PT22000102', 'N', ''),
(TIMESTAMP '2024-01-01 16:00:00', 54, 'PT22000113', 'Y', '');

-- 8-6.

SELECT p.PATIENT_NAME
FROM ex5_patient AS p
JOIN ex5_apnt AS a
ON p.PATIENT_NO = a.PATIENT_NO
WHERE a.TREATMENT_STATUS != 'Completed' AND a.APNT_CANCEL_YN = 'N'
ORDER BY a.APNT_YMD
LIMIT 1
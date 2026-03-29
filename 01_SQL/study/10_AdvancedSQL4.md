# 10. SQL 심화 - SQL로 데이터 분석하기

> 인터넷 쇼핑몰에 근무하는 데이터 사이언티스트가 할 수 있는일 연습하기

## 학습 내용

- 회사의 **일별 매출액** 구하기
- 유저가 조회한 페이지 정보를 활용하여 **PV**, **Unique PV**, **UV** 계산하기
- 유저들의 구매 정보를 활용하여 **ARPU**, **ARPPU** 구하기
- 유저들의 액션 로그 정보를 활용하여 **퍼널 분석**하기
- 유저들의 구매 정보를 활용하여 **리텐션 분석**하기
- 유저들의 구매 정보를 활용하여 **RFM 분석**하기

> 사용할 데이터는 다음 쿼리문을 통해 생성

<table>
<tr>
<th>데이터 : orders</th>
<th>데이터 : payments</th>
</tr>
<tr>
<td>
| order_id | user_id | order_timestamp |
| :--- | :--- | :--- |
| order_1 | user_1 | 2018-01-01 10:00:00 |
| order_2 | user_2 | 2018-01-01 12:30:00 |
| order_3 | user_3 | 2018-01-02 09:20:00 |
| order_4 | user_4 | 2018-01-02 10:15:00 |
| order_5 | user_5 | 2018-01-03 11:00:00 |
| order_6 | user_6 | 2018-01-03 13:45:00 |
</td>
<td>
| order_id | value |
| :--- | :--- |
| order_1 | 100 |
| order_2 | 150 |
| order_3 | 200 |
| order_4 | 110 |
| order_5 | 120 |
| order_6 | 130 |
</td>
</tr>
</table>

```sql

-- 테이블 생성
CREATE TABLE `project_orders` (
   order_id STRING NOT NULL,
   user_id STRING NOT NULL,
   order_timestamp TIMESTAMP
);


-- 테이블 생성
CREATE TABLE `project_payments` (
   order_id STRING NOT NULL,
   value FLOAT
);


-- 데이터 삽입
INSERT INTO project_orders (order_id, user_id, order_timestamp) VALUES
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
INSERT INTO project_payments (order_id, value) VALUES
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

```

## 회사의 일별 매출액 구하기

회사의 일별 매출액 합계를 구해야 한다. 

주문이 일어난 시점에 대한 정보를 기록한 `project_orders`와 결제 금액에 대한 정보를 기록한 `project_payments` 테이블을 활용해 일별 매출액 총합을 구한다.

```sql



```


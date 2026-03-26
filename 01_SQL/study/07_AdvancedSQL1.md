# 7. SQL 심화 - 빅데이터 기술 톺아보기

## 학습 내용

- 빅데이터 톺아보기
    - 빅데이터를 지탱하는 툴과 기술
    - 빅쿼리(BigQuery) 실습 환경 설정하기


## 7-1. 빅데이터 톺아보기

### 빅데이터의 반댓말은 스몰데이터

스몰 데이터 = ["개인의 취향과 필요, 생활 양식 등의 사소한 행동에서 나오는 데이터"](https://en.wikipedia.org/wiki/Small_Data)

고객에 대한 사소한 정보까지 담을 수 있는 데이터로, 획기적인 아이디어가 탄생하고 브랜드를 변화시킬 가능성을 품은 것

### 수직 확장과 수평 확장

서비스 운영 ing -> 데이터는 쌓이고, 서비스는 업데이트 되므로 데이터베이스 확장이 필요하다. 확장 방법 2가지가 있는데

1. 수평 확장(수평 스케일링, Scale Out)
    - 기존 DB에 새로운 서버 추가
    - 쉽고 비용이 저렴하다
    - 버그 해결이 어렵고, 서버관리가 어려움
    - 대부분의 웹(앱) 서비스에서 사용하는 서버에서 많이 쓰임 (사용자 급증 시 서버를 확장하고, 안정되면 서버를 줄이면서 유연하게 대처)

2. 수직 확장(수직 스케일링, Scale Up)
    - 기존 서버의 성능을 업그레이드
    - 확장과정이 간단하고, 서버관리가 쉬움
    - 비용 부담이 크고, 업그레이드에 한계가 있고 다른 제품과 통합이 어려움
    - SPOF(Single Point of Failure) : 서버가 다운되면 서비스 전체가 중단됨
    - 데이터 일관성과 무결성이 중요한 서버에 적합하다

### 빅쿼리(BigQuery) 사용하기

> 저는 MySQL WorkBench 사용했습니다.

```sql
-- 테이블 생성   
CREATE TABLE `ex1`(     -- MySQL은 OR REPLACE 기능이 없다
  order_id VARCHAR(50), -- MySQL은 STRING -> VARCHAR
  user_id VARCHAR(50),
  item_id VARCHAR(50),
  price DOUBLE          -- MySQL은 FLOAT64 -> DOUBLE
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
```


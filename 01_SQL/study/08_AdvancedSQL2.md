# 8. SQL 심화 - SQL 기초 체력 키우기

> 기초 문법 다지기!

## 학습 내용

- 우리 쇼핑몰의 VIP 고객은 누구일까?
- 레스토랑 손님 중 대식가는 누구일까?
- 모두몰의 베스트 상품은?
- 베스트 리뷰어에게 경품을 주자!
- 다음 환자분, 진료실로 들어오세요!

## 8-1. 모두몰의 VIP 고객은 누구일까?

앞서 만들어두었던 ex1 테이블에 각 주문에 포함된 상품들에 대한 데이터가 들어있다. 
**총 주문 금액이 높은 상위 3명의 손님** 리스트를 출력하라.

```sql
-- 총 주문 금액이 높은 상위 3명의 손님 리스트 출력
SELECT user_id, SUM(price) AS total_spent 
FROM ex1
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 3
```

## 8-2. 레스토랑 손님 중 대식가는 누구일까?
"ex2"라는 이름의 테이블에는 각 테이블의 식사 금액, 팁, 손님 수, 결제자 성별, 요일, 시간대 등에 대한 데이터가 들어있다.
**각 테이블의 식사 금액이 전체 평균 식사 금액보다 많은 경우**를 찾아내어, 특히 많은 양의 음식을 주문한 큰손 손님들을 식별하는 쿼리문을 작성해주세요. 
결과에는 ex2 테이블의 모든 컬럼이 포함되어야 합니다.

```sql
-- 1. Subquery를 사용한 방법
SELECT *
FROM ex2
WHERE total_bill > (SELECT AVG(total_bill) FROM ex2);

-- 2. WITH를 사용한 방법
WITH AverageBill AS (
  SELECT AVG(total_bill) AS avg_bill
  FROM ex2
)
SELECT *
FROM ex2, AverageBill
WHERE ex2.total_bill > AverageBill.avg_bill;
```

> WITH 구문
- 쿼리문을 임시 테이블처럼 활용할 수 있는 기능, 아래와 같은 구조를 가진다.
- 가독성 좋지만, 쿼리추출결과가 많을때는 효율이 좋지 못함.

```sql
WITH 임시 테이블 명 AS (
   WITH 문으로 저장하고 싶은 SQL 쿼리문
)
SELECT * 
FROM 임시 테이블 명 
```

## 8-3. 모두몰의 베스트 상품은?

"ex3" 라는 이름의 테이블에는 제품별 판매 정보가 담겨 있습니다. 즉 제품번호(PRODUCT_ID)와 제품라인(PRODUCT_LINE), 그리고 각 PRODUCT 별로 판매된 판매량 정보가 들어있죠.

ex3 테이블을 사용하여 판매량 합계가 가장 많은 ‘제품 라인(Product Line) 을 찾아주세요.

```sql
SELECT PRODUCT_LINE, SUM(TOTAL_ORDER) AS TOTAL_ORDERS
FROM ex3
GROUP BY PRODUCT_LINE
ORDER BY TOTAL_ORDERS DESC
LIMIT 1;
```

## 8-4. 베스트 리뷰어에게 경품을 주자!

"ex4_member", "ex4_review" 라는 이름의 테이블에는 각각 회원에 대한 정보와, 회원이 남긴 리뷰에 대한 정보가 담겨 있습니다.

ex4_member 테이블과 ex4_review 테이블을 사용하여 리뷰를 가장 많이 남긴 회원의 리뷰를 조회해 주세요.

최종 출력은 회원 이름, 리뷰 텍스트, 리뷰 작성일이 포함되어야 합니다. 정렬은 아래와 같이 해주세요.

리뷰 작성일: 오름차순
동일 날짜에 적힌 리뷰의 경우에는 리뷰 텍스트를 기준으로 오름차순 정렬

```sql
SELECT A.MEMBER_NAME, B.REVIEW_DATE, B.REVIEW_TEXT
FROM `ex4_review` AS A
JOIN `ex4_member` AS B
ON A.MEMBER_ID = B.MEMBER_ID
WHERE A.MEMBER_ID = (SELECT MEMBER_ID FROM `ex4_review` GROUP BY MEMBER_ID ORDER BY COUNT(*) DESC LIMIT 1) 
ORDER BY B.REVIEW_DATE
```

## 8-5. 다음 환자분, 진료실로 들어오세요!

"ex5_patient" 라는 이름의 테이블에는 환자들에 대한 정보가, "ex5_apnt" 테이블에는 진료 예약 정보가 담겨 있습니다.

테이블들을 활용하여 아직 진료 완료(TREATMENT_STATUS = 'Completed')가 되지 않은 예약 건들에 대하여 '취소 되지 않은 다음 진료 예약' 의 예약자명을 찾아주세요.

```sql
SELECT p.PATIENT_NAME
FROM ex5_patient AS p
JOIN ex5_apnt AS a ON p.PATIENT_NO = a.PATIENT_NO
WHERE a.TREATMENT_STATUS != 'Completed'  -- 진료 완료가 아님
  AND a.APNT_CANCEL_YN = 'N'             -- 취소되지 않음
  AND a.APNT_YMD >= NOW()                -- (선택) '다음' 예약이므로 현재 시간 이후
ORDER BY a.APNT_YMD ASC                  -- 가장 가까운 예약 순
LIMIT 1;                                 -- 그중 첫 번째 한 명
```

> WHERE 1=1
> 항상 참인 명제를 맨 앞에 두어서 조건의 추가와 삭제를 편리하게 할 수 있음!
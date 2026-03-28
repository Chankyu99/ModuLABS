# 9. SQL 심화 - 실무에서 많이 쓰이는 고급 SQL 기술

## 학습 내용

- 윈도우 함수 (1) : 함수 구조와 순위 함수
    - 윈도우 함수란? 함수 구조 살펴보기
    - 순위 함수 : RANK, DENSE_RANK, ROW_NUMBER

- 윈도우 함수 (2) : 집계 함수
    - 집계 함수 : SUM, AVG, MIN, MAX
    - 행 순서 집계 함수 : LAG, LEAD, FIRST_VALUE, LAST_VALUE

- 그룹 함수 
    - 그룹함수란? 함수 구조 살펴보기
    - 그룹 함수 : ROLLUP, CUBE, GROUPING SETS

- 복잡한 데이터 처리 (JSON Formatting)
    - 로그 데이터란? JSON 구조의 로그 데이터 살펴보기
    - JSON 뜯어서 분석하기

## 9-1. 윈도우 함수 (1) : 함수 구조와 순위 함수

### 윈도우 함수 

- 행과 행 사이의 관계를 정의하여 집계나 분석을 수행하는 함수
- 일반적인 GROUP BY와 다르게 기존 행의 세부 정보를 유지하며 계산 결과를 추가할 수 있음

```sql
-- 함수 구조
SELECT 컬럼 명, WINDOW_FUNCTION(ARGUMENTS) 
OVER ([PARTITION BY 절] [ORDER BY 절] [WINDOWING 절])
FROM 테이블 명;
```

- ARGUMENTS : 윈도우 함수에 전달하는 인자
- PARTITION BY : 전체 집합에 대해 소그룹으로 나누는 기준
- ORDER BY : 소그룹에 대한 정렬 기준
- WINDOWING : 행에 대한 범위 기준

### 순위 함수

```sql
-- 함수 구조
SELECT 컬럼 명,
RANK() OVER ([PARTITION BY 컬럼] [ORDER BY 절] [WINDOWING 절])
FROM 테이블 명;
```

> 인수가 별도로 없으며, WINDOWING 절은 거의 쓰이지 않음 

- RANK : 공동 순위 인정, 다음 순위는 건너뜀
- DENSE_RANK : 공동 순위 인정, 다음 순위는 건너뛰지 않음
- ROW_NUMBER : 공동 순위 인정하지 않음, 무조건 1씩 증가 (통상적으로 Primary Key 오름차순으로 순위 부여)

> MySQL에서는 컬럼명 별칭에 예약어를 쓰면 실행되지 않는다는 것을 알았음!

## 9-2 윈도우 함수 (2) : 집계 함수

- SUM, AVG, MIN, MAX : 일반적인 집계 함수와 동일한 문법을 사용하지만, 윈도우 함수로 사용하면 기존 행의 세부 정보를 유지하며 계산 결과를 추가할 수 있음

```sql
-- 함수 구조
SELECT 컬럼 명,
집계 함수(컬럼) OVER ([PARTITION BY 절] [ORDER BY 절] [WINDOWING 절]) 
FROM 테이블 명;
```

- FIRST_VALUE(기준 컬럼) : 윈도우 내 첫 번째 행의 값
- LAST_VALUE(기준 컬럼) : 윈도우 내 마지막 행의 값
- LAG(기준 컬럼, n) : 현재 행의 n번째 이전 행의 값
- LEAD(기준 컬럼, n) : 현재 행의 n번째 다음 행의 값

- WINDOWING 절이 여기서 쓰인다
    - ROWS BETWEEN A AND B : A부터 B까지의 행을 윈도우로 설정
    - CURRENT ROW : 현재 행 (누적 합 or 이동 평균 구할 때 많이 쓴다고 함..)
    - UNBOUNDED PRECEDING : 윈도우의 시작
    - UNBOUNDED FOLLOWING : 윈도우의 끝
    - n PRECEDING : 현재 행의 n번째 이전 행
    - n FOLLOWING : 현재 행의 n번째 다음 행

## 9-3. 그룹 함수

### 그룹 함수란?

- 다양한 차원에서 데이터를 분석할 수 있는 함수로 데이터를 통계 내기 위한 소계, 중계를 구하는 함수이다.

```sql
-- 함수 구조
SELECT 컬럼 명, 
집계 함수(인수) 
FROM 테이블 명
GROUP 그룹 함수(컬럼1, 컬럼2)
```

- 기존 GROUP BY의 결과

```sql
SELECT 상품ID, 기준월, SUM(매출액) AS 매출액_합계
FROM 월별매출
GROUP BY 상품ID, 기준월;
```

| 상품ID | 기준월 | 매출액_합계 |
| :--- | :--- | :--- |
| 1001 | 2024.01 | 1000 |
| 1001 | 2024.02 | 1000 |
| 1001 | 2024.03 | 2000 |
| 1002 | 2024.01 | 1500 |
| 1002 | 2024.02 | 1500 |
| 1002 | 2024.03 | 2500 |

- ROLLUP : 지정한 컬럼 순서에 따라 소그룹간 소계를 생성

```sql
SELECT 상품ID, 기준월, SUM(매출액) AS 매출액_합계
FROM 월별매출
GROUP BY ROLLUP (상품ID, 기준월);
```

| 상품ID | 기준월 | 매출액_합계 | 비고 |
| :--- | :--- | :--- | :--- |
| 1001 | 2024.01 | 1000 | |
| 1001 | 2024.02 | 1000 | |
| 1001 | 2024.03 | 2000 | |
| **1001** | **NULL** | **4000** | **[1001 소계]** |
| 1002 | 2024.01 | 1500 | |
| 1002 | 2024.02 | 1500 | |
| 1002 | 2024.03 | 2500 | |
| **1002** | **NULL** | **5500** | **[1002 소계]** |
| **NULL** | **NULL** | **9500** | **[전체 총계]** |

- CUBE : 지정한 컬럼의 가능한 모든 조합에 대한 소계를 생성

```sql
SELECT 상품ID, 기준월, SUM(매출액) AS 매출액_합계
FROM 월별매출
GROUP BY CUBE (상품ID, 기준월);
```

| 상품ID | 기준월 | 매출액_합계 | 비고 |
| :--- | :--- | :--- | :--- |
| 1001 | 2024.01 | 1000 | |
| 1001 | 2024.02 | 1000 | |
| 1001 | 2024.03 | 2000 | |
| **1001** | **NULL** | **4000** | **[상품 1001 소계]** |
| 1002 | 2024.01 | 1500 | |
| 1002 | 2024.02 | 1500 | |
| 1002 | 2024.03 | 2500 | |
| **1002** | **NULL** | **5500** | **[상품 1002 소계]** |
| **NULL** | **2024.01** | **2500** | **[1월 전체 소계]** |
| **NULL** | **2024.02** | **2500** | **[2월 전체 소계]** |
| **NULL** | **2024.03** | **4500** | **[3월 전체 소계]** |
| **NULL** | **NULL** | **9500** | **[전체 총계]** |

- GROUPING SETS : 원하는 그룹의 합계만 골라서 생성

```sql
SELECT 상품ID, 기준월, SUM(매출액) AS 매출액_합계
FROM 월별매출
GROUP BY GROUPING SETS (상품ID, 기준월);
```

| 상품ID | 기준월 | 매출액_합계 | 비고 |
| :--- | :--- | :--- | :--- |
| **1001** | **NULL** | **4000** | **[상품 1001 총합]** |
| **1002** | **NULL** | **5500** | **[상품 1002 총합]** |
| **NULL** | **2024.01** | **2500** | **[1월 전체 총합]** |
| **NULL** | **2024.02** | **2500** | **[2월 전체 총합]** |
| **NULL** | **2024.03** | **4500** | **[3월 전체 총합]** |

> GROUPING SETS의 마지막에 ()을 넣으면 전체 총계가 생성된다. (예시에서는 마지막단에 NULL, NULL 전체 총계가 추가됨)

## 9-4. 복잡한 데이터 처리(JSON Formatting)

### 로그 데이터란?

- 웹 페이지, 어플리케이션, 응용 프로그램 등에서 수집된 동작 및 활동 정보를 말한다
- 예시 : 웹사이트에서 상품을 클릭했을 때 발생하는 로그는 대략 다음과 같은 구조를 가진다

```json
{
  "timestamp": "2026-03-29T01:14:03Z",
  "event_type": "click_item",
  "user": {
    "id": "user_123",
    "grade": "VIP"
  },
  "device": {
    "os": "iOS",
    "version": "17.4"
  },
  "item_id": "book_999",
  "price": 15000
}
```

### JSON 뜯어서 분석하기

- DBMS에 따라 지원되는 JSON 함수가 다르다. 

- MySQL JSON 데이터 추출 방식

1. JSON_EXTRACT() 함수 활용 (BigQuery 방식과 유사, 가장 표준적인 방법)

```sql
SELECT 
    JSON_EXTRACT(log, '$.user') AS user,
    JSON_EXTRACT(log, '$.action') AS action,
    JSON_EXTRACT(log, '$.timestamp') AS timestamp
FROM table;
```

> 이 방법으로 하면 결과에 따옴표가 붙어서 나오는데, JSON_UNQUOTE() 함수를 사용하면 따옴표를 제거할 수 있다.

```sql
SELECT 
    JSON_UNQUOTE(JSON_EXTRACT(log, '$.user')) AS user,
    JSON_UNQUOTE(JSON_EXTRACT(log, '$.action')) AS action,
    JSON_UNQUOTE(JSON_EXTRACT(log, '$.timestamp')) AS timestamp
FROM table;
```

2. 인라인 경로 연산자 (->) 활용 (PostgreSQL 방식과 유사, MySQL 5.7 버전부터 지원하며, 코드가 더 간결해진다)

```sql
SELECT 
    log->"$.user" AS user,
    log->"$.action" AS action,
    log->"$.timestamp" AS timestamp
FROM table;
```

> 따옴표 제거 연산자 (->>) 활용
> 추출된 값에서 큰따옴표(" ")를 제거하고 순수한 텍스트만 가져오고 싶을 때 사용한다

```sql
SELECT log->>"$.user" AS user FROM table;
```
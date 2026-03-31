# 📊 Pandas 핵심 메서드 총정리

> 데이터 사이언티스트가 반드시 알아야 할 Pandas 메서드를 카테고리별로 정리

---

## 목차

1. [데이터 입출력](#1-데이터-입출력-io)
2. [데이터 확인](#2-데이터-확인)
3. [데이터 선택 및 인덱싱](#3-데이터-선택-및-인덱싱)
4. [데이터 정렬](#4-데이터-정렬)
5. [결측치 처리](#5-결측치-처리)
6. [중복 데이터 처리](#6-중복-데이터-처리)
7. [데이터 타입 변환](#7-데이터-타입-변환)
8. [데이터 변환 및 가공](#8-데이터-변환-및-가공)
9. [통계 및 집계](#9-통계-및-집계)
10. [그룹 연산](#10-그룹-연산)
11. [데이터 병합 및 결합](#11-데이터-병합-및-결합)
12. [피벗 및 재구조화](#12-피벗-및-재구조화)
13. [문자열 처리](#13-문자열-처리)
14. [날짜/시간 처리](#14-날짜시간-처리)
15. [윈도우 함수](#15-윈도우-함수)

---

## 1. 데이터 입출력 (I/O)

### 읽기

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `pd.read_csv()` | CSV 파일 읽기 | `pd.read_csv("data.csv", encoding="utf-8")` |
| `pd.read_excel()` | Excel 파일 읽기 | `pd.read_excel("data.xlsx", sheet_name="Sheet1")` |
| `pd.read_json()` | JSON 파일 읽기 | `pd.read_json("data.json")` |
| `pd.read_sql()` | SQL 쿼리 결과 읽기 | `pd.read_sql("SELECT * FROM table", conn)` |
| `pd.read_parquet()` | Parquet 파일 읽기 | `pd.read_parquet("data.parquet")` |
| `pd.read_clipboard()` | 클립보드에서 읽기 | `pd.read_clipboard()` |
| `pd.read_html()` | 웹 테이블 읽기 | `pd.read_html("https://example.com")` |

### `read_csv()` 주요 파라미터

```python
pd.read_csv(
    "data.csv",
    sep=",",              # 구분자 (기본: 쉼표)
    header=0,             # 헤더 행 번호 (None이면 헤더 없음)
    index_col=0,          # 인덱스로 사용할 열
    usecols=["A", "B"],   # 특정 열만 읽기
    nrows=100,            # 앞에서 n행만 읽기
    skiprows=[1, 2],      # 건너뛸 행
    na_values=["?", "N/A"], # 결측치로 인식할 값
    encoding="utf-8",     # 인코딩 (한글: "cp949" 또는 "euc-kr")
    dtype={"Age": int},   # 열별 데이터 타입 지정
    parse_dates=["Date"], # 날짜로 파싱할 열
)
```

### 저장

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `df.to_csv()` | CSV로 저장 | `df.to_csv("output.csv", index=False)` |
| `df.to_excel()` | Excel로 저장 | `df.to_excel("output.xlsx", index=False)` |
| `df.to_json()` | JSON으로 저장 | `df.to_json("output.json")` |
| `df.to_parquet()` | Parquet으로 저장 | `df.to_parquet("output.parquet")` |
| `df.to_sql()` | DB 테이블로 저장 | `df.to_sql("table", conn)` |

---

## 2. 데이터 확인

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `df.head(n)` | 앞 n행 출력 (기본 5) | DataFrame |
| `df.tail(n)` | 뒤 n행 출력 (기본 5) | DataFrame |
| `df.shape` | (행 수, 열 수) | tuple |
| `df.info()` | 열별 타입, 결측치 수, 메모리 | None (출력) |
| `df.describe()` | 기술 통계량 (수치형) | DataFrame |
| `df.dtypes` | 열별 데이터 타입 | Series |
| `df.columns` | 열 이름 목록 | Index |
| `df.index` | 인덱스 정보 | Index |
| `df.values` | 값을 NumPy 배열로 반환 | ndarray |
| `df.nunique()` | 열별 고유값 수 | Series |
| `df.sample(n)` | 랜덤 n행 추출 | DataFrame |
| `len(df)` | 행 수 | int |

### 예시

```python
print(f"데이터 크기: {df.shape}")       # (891, 12)
print(f"열 목록: {df.columns.tolist()}") # ['PassengerId', 'Survived', ...]
df.describe(include="all")               # 수치형 + 범주형 모두 포함
```

---

## 3. 데이터 선택 및 인덱싱

### 열 선택

```python
df['Name']              # Series 반환 (열 1개)
df[['Name', 'Age']]     # DataFrame 반환 (열 여러 개)
```

### 행 선택

```python
df[0:5]                 # 슬라이싱 (0~4행)
df[df['Age'] > 30]      # 불리언 인덱싱 (조건 필터링)
```

### loc vs iloc

| 메서드 | 기준 | 설명 |
|--------|------|------|
| `df.loc[]` | **라벨(이름)** 기반 | 행/열 이름으로 선택 |
| `df.iloc[]` | **정수(위치)** 기반 | 행/열 번호로 선택 |

```python
# loc - 라벨 기반 (끝 포함)
df.loc[0:5, 'Name']              # 0~5행, Name 열
df.loc[0:5, 'Name':'Age']        # 0~5행, Name부터 Age까지 모든 열
df.loc[df['Age'] > 30, 'Name']   # 조건 + 열 선택

# iloc - 위치 기반 (끝 미포함)
df.iloc[0:5, 0]                  # 0~4행, 0번째 열
df.iloc[0:5, 0:3]                # 0~4행, 0~2번째 열
df.iloc[[0, 2, 4], [1, 3]]      # 특정 행/열 번호 지정
```

### 조건 필터링

```python
# 단일 조건
df[df['Age'] > 30]

# 복합 조건 (& = AND, | = OR, ~ = NOT)
df[(df['Age'] > 30) & (df['Survived'] == 1)]
df[(df['Pclass'] == 1) | (df['Pclass'] == 2)]
df[~df['Embarked'].isna()]

# isin() - 여러 값 중 하나에 해당
df[df['Embarked'].isin(['C', 'Q'])]

# between() - 범위 조건
df[df['Age'].between(20, 30)]

# query() - 문자열로 조건 작성
df.query("Age > 30 and Survived == 1")
```

---

## 4. 데이터 정렬

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `df.sort_values()` | 값 기준 정렬 | `df.sort_values('Age', ascending=False)` |
| `df.sort_index()` | 인덱스 기준 정렬 | `df.sort_index()` |
| `df.nlargest(n, col)` | 상위 n개 추출 | `df.nlargest(10, 'Fare')` |
| `df.nsmallest(n, col)` | 하위 n개 추출 | `df.nsmallest(10, 'Age')` |
| `df.rank()` | 순위 매기기 | `df['Age'].rank(method='min')` |

```python
# 다중 열 정렬
df.sort_values(['Pclass', 'Age'], ascending=[True, False])

# 정렬 후 인덱스 리셋
df.sort_values('Age').reset_index(drop=True)
```

---

## 5. 결측치 처리

### 결측치 확인

```python
df.isnull()              # 전체 True/False 표
df.isnull().sum()        # 열별 결측치 수
df.isnull().sum().sum()  # 전체 결측치 수
df.notnull()             # isnull()의 반대

# 결측치 비율 확인
(df.isnull().sum() / len(df) * 100).round(2)
```

### 결측치 제거

```python
df.dropna()                          # 결측치 있는 행 모두 제거
df.dropna(axis=1)                    # 결측치 있는 열 제거
df.dropna(subset=['Age', 'Cabin'])   # 특정 열 기준으로 제거
df.dropna(thresh=5)                  # 비결측치가 5개 이상인 행만 유지
df.dropna(how='all')                 # 모든 값이 결측인 행만 제거
```

### 결측치 채우기

```python
df.fillna(0)                            # 특정 값으로 채우기
df['Age'].fillna(df['Age'].mean())      # 평균값으로 채우기
df['Age'].fillna(df['Age'].median())    # 중앙값으로 채우기
df['Cabin'].fillna('Unknown')           # 문자열로 채우기
df.fillna(method='ffill')               # 앞의 값으로 채우기 (forward fill)
df.fillna(method='bfill')               # 뒤의 값으로 채우기 (backward fill)

# 열별로 다른 값으로 채우기
df.fillna({'Age': 0, 'Cabin': 'Unknown', 'Embarked': 'S'})

# 결측치 보간 (선형)
df['Age'].interpolate(method='linear')
```

---

## 6. 중복 데이터 처리

```python
df.duplicated()                      # 중복 행 True/False (Series)
df.duplicated().sum()                # 중복 행 수
df[df.duplicated()]                  # 중복 행 추출
df[df.duplicated(subset=['Name'])]   # 특정 열 기준 중복

df.drop_duplicates()                          # 중복 행 제거 (첫 번째 유지)
df.drop_duplicates(keep='last')               # 마지막 행 유지
df.drop_duplicates(subset=['Name', 'Age'])    # 특정 열 기준 중복 제거
```

---

## 7. 데이터 타입 변환

```python
df['Age'].astype(int)                # 정수로 변환
df['Age'].astype(float)              # 실수로 변환
df['Age'].astype(str)                # 문자열로 변환
df['Pclass'].astype('category')      # 범주형으로 변환

# 숫자로 변환 (에러 무시)
pd.to_numeric(df['Age'], errors='coerce')    # 변환 불가 → NaN
pd.to_numeric(df['Age'], errors='ignore')    # 변환 불가 → 원본 유지

# 날짜로 변환
pd.to_datetime(df['Date'])
pd.to_datetime(df['Date'], format='%Y-%m-%d')
```

---

## 8. 데이터 변환 및 가공

### 열 추가/삭제/이름변경

```python
# 열 추가
df['New_Col'] = df['Age'] * 2
df['Category'] = df['Age'].apply(lambda x: '성인' if x >= 18 else '미성년')

# 열 삭제
df.drop('Cabin', axis=1, inplace=True)           # 단일 열
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True) # 여러 열
df.drop(columns=['Cabin', 'Ticket'], inplace=True)  # columns 파라미터 사용

# 행 삭제
df.drop(index=[0, 1, 2], inplace=True)  # 특정 행 삭제

# 열 이름 변경
df.rename(columns={'Name': '이름', 'Age': '나이'})
df.columns = ['col1', 'col2', 'col3']   # 전체 변경
```

### apply / map / applymap

| 메서드 | 적용 대상 | 설명 |
|--------|-----------|------|
| `apply()` | Series 또는 DataFrame | 행/열 단위 함수 적용 |
| `map()` | Series 전용 | 요소별 매핑 또는 함수 적용 |
| `applymap()` | DataFrame 전용 | 모든 요소에 함수 적용 |

```python
# apply - 열에 함수 적용
df['Age'].apply(lambda x: '성인' if x >= 18 else '미성년')
df.apply(lambda row: row['Fare'] / row['Age'], axis=1)  # 행 단위

# map - 값 매핑 (딕셔너리 또는 함수)
df['Sex'].map({'male': 0, 'female': 1})
df['Name'].map(len)    # 이름 글자 수

# applymap - 전체 요소에 적용 (pandas 2.1+에서는 map으로 통합)
df[['Age', 'Fare']].applymap(lambda x: round(x, 1))
```

### replace

```python
df['Sex'].replace({'male': 0, 'female': 1})
df.replace(to_replace=0, value=np.nan)     # 전체에서 0을 NaN으로
df.replace([0, -1], np.nan)                # 여러 값 한번에 교체
```

### 행/열 전환

```python
df.T                     # 행과 열을 바꿈 (전치)
```

### 기타 변환

```python
df['Age'].clip(0, 100)               # 범위 제한 (0~100)
df['Age'].abs()                      # 절대값
df['Fare'].round(2)                  # 반올림
df.reset_index(drop=True)           # 인덱스 리셋
df.set_index('PassengerId')         # 특정 열을 인덱스로
```

---

## 9. 통계 및 집계

### 기본 통계

| 메서드 | 설명 |
|--------|------|
| `df.mean()` | 평균 |
| `df.median()` | 중앙값 |
| `df.mode()` | 최빈값 |
| `df.std()` | 표준편차 |
| `df.var()` | 분산 |
| `df.min()` / `df.max()` | 최솟값 / 최댓값 |
| `df.sum()` | 합계 |
| `df.count()` | 비결측치 수 |
| `df.quantile(0.25)` | 분위수 (25%) |
| `df.cumsum()` | 누적 합 |
| `df.cumprod()` | 누적 곱 |
| `df.cummax()` | 누적 최대 |
| `df.cummin()` | 누적 최소 |
| `df.pct_change()` | 변화율 |
| `df.diff()` | 차분 (이전 값과의 차이) |

### 빈도 및 상관

```python
df['Embarked'].value_counts()              # 값별 빈도수
df['Embarked'].value_counts(normalize=True) # 비율로 출력

df.corr()                        # 상관계수 행렬
df['Age'].corr(df['Fare'])       # 두 열 간 상관계수
df.cov()                         # 공분산 행렬

# idxmax / idxmin - 최대/최소값의 인덱스
df['Age'].idxmax()    # 가장 나이 많은 사람의 인덱스
df['Age'].idxmin()    # 가장 어린 사람의 인덱스
```

### agg (여러 집계 한번에)

```python
df['Age'].agg(['mean', 'median', 'std'])

df.agg({
    'Age': ['mean', 'max'],
    'Fare': ['sum', 'min']
})
```

---

## 10. 그룹 연산

### groupby

```python
# 기본 사용법
df.groupby('Pclass')['Survived'].mean()

# 여러 열로 그룹
df.groupby(['Pclass', 'Sex'])['Survived'].mean()

# 여러 집계 함수
df.groupby('Pclass').agg({
    'Survived': 'mean',
    'Age': ['mean', 'median'],
    'Fare': 'sum'
})

# 그룹별 크기
df.groupby('Pclass').size()

# 그룹별 반복
for name, group in df.groupby('Pclass'):
    print(f"Pclass: {name}, 행 수: {len(group)}")

# 그룹별 커스텀 함수
df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))
```

### transform vs apply (groupby에서)

| 메서드 | 반환 크기 | 용도 |
|--------|-----------|------|
| `transform()` | 원본과 동일 | 그룹별 결과를 원본 크기로 브로드캐스트 |
| `apply()` | 그룹별 결과 | 그룹별로 자유로운 연산 |

```python
# transform - 그룹 평균으로 결측치 채우기 (원본 크기 유지)
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

# apply - 그룹별 상위 2명 추출
df.groupby('Pclass').apply(lambda x: x.nlargest(2, 'Fare'))
```

---

## 11. 데이터 병합 및 결합

### merge (SQL JOIN과 동일)

```python
pd.merge(df1, df2, on='key')                  # 공통 열 기준 (inner join)
pd.merge(df1, df2, on='key', how='left')       # LEFT JOIN
pd.merge(df1, df2, on='key', how='right')      # RIGHT JOIN
pd.merge(df1, df2, on='key', how='outer')      # FULL OUTER JOIN

# 열 이름이 다를 때
pd.merge(df1, df2, left_on='id', right_on='user_id')

# 인덱스 기준 병합
pd.merge(df1, df2, left_index=True, right_index=True)
```

### concat (단순 이어 붙이기)

```python
pd.concat([df1, df2])                   # 행 방향 (위아래로)
pd.concat([df1, df2], axis=1)           # 열 방향 (좌우로)
pd.concat([df1, df2], ignore_index=True) # 인덱스 리셋
```

### join

```python
df1.join(df2, how='left')              # 인덱스 기준 결합
```

---

## 12. 피벗 및 재구조화

### pivot_table

```python
df.pivot_table(
    values='Survived',      # 집계할 값
    index='Pclass',         # 행
    columns='Sex',          # 열
    aggfunc='mean'          # 집계 함수
)

# 여러 집계 함수
df.pivot_table(
    values='Age',
    index='Pclass',
    aggfunc=['mean', 'count']
)
```

### melt (wide → long 변환)

```python
pd.melt(df, id_vars=['Name'], value_vars=['Age', 'Fare'],
        var_name='variable', value_name='value')
```

### crosstab (교차표)

```python
pd.crosstab(df['Pclass'], df['Survived'])
pd.crosstab(df['Pclass'], df['Survived'], normalize='index')  # 행 비율
```

### stack / unstack

```python
df.stack()       # 열 → 행 (wide → long)
df.unstack()     # 행 → 열 (long → wide)
```

### cut / qcut (구간 분할)

```python
# cut - 동일 간격으로 분할
pd.cut(df['Age'], bins=5)                        # 5등분
pd.cut(df['Age'], bins=[0, 18, 35, 60, 100],
       labels=['미성년', '청년', '중년', '노년'])

# qcut - 동일 개수로 분할 (분위수 기반)
pd.qcut(df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

## 13. 문자열 처리

> `str` 접근자를 통해 Series의 문자열 메서드 사용

```python
df['Name'].str.lower()              # 소문자 변환
df['Name'].str.upper()              # 대문자 변환
df['Name'].str.strip()              # 양쪽 공백 제거
df['Name'].str.len()                # 글자 수
df['Name'].str.contains('Mr')       # 포함 여부 (True/False)
df['Name'].str.startswith('A')      # 시작 문자
df['Name'].str.endswith('s')        # 끝 문자

# 분리 및 추출
df['Name'].str.split(',')           # 쉼표로 분리 (리스트 반환)
df['Name'].str.split(',').str[0]    # 분리 후 첫 번째 요소
df['Name'].str.extract(r'(\w+)')    # 정규식 추출

# 교체
df['Name'].str.replace('Mr.', 'Mister', regex=False)
df['Name'].str.replace(r'\d+', '', regex=True)  # 숫자 제거

# 패딩
df['ID'].str.zfill(5)               # 5자리 0 패딩 → "00042"
df['Name'].str.pad(10, side='left')  # 왼쪽 공백 패딩

# 카테고리 변환
df['Name'].str.get_dummies(sep=',')  # 원핫 인코딩
```

---

## 14. 날짜/시간 처리

### 변환 및 생성

```python
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 날짜 범위 생성
pd.date_range('2024-01-01', periods=30, freq='D')   # 일 단위
pd.date_range('2024-01-01', '2024-12-31', freq='M')  # 월 단위
```

### dt 접근자

```python
df['Date'].dt.year           # 연도
df['Date'].dt.month          # 월
df['Date'].dt.day            # 일
df['Date'].dt.hour           # 시
df['Date'].dt.dayofweek      # 요일 (0=월요일)
df['Date'].dt.day_name()     # 요일 이름
df['Date'].dt.quarter        # 분기
df['Date'].dt.is_month_end   # 월말 여부
df['Date'].dt.date           # 날짜만 추출
df['Date'].dt.time           # 시간만 추출
```

### 시간 연산

```python
# 시간 차이
df['Duration'] = df['End'] - df['Start']
df['Duration'].dt.days       # 일수로 변환

# 시간 오프셋
df['Date'] + pd.Timedelta(days=7)     # 7일 후
df['Date'] - pd.DateOffset(months=1)  # 1달 전
```

### 리샘플링 (시계열)

```python
df.set_index('Date').resample('M').mean()    # 월별 평균
df.set_index('Date').resample('W').sum()     # 주별 합계
df.set_index('Date').resample('Q').first()   # 분기별 첫 값
```

---

## 15. 윈도우 함수

```python
# 이동 평균 (Moving Average)
df['MA_7'] = df['Value'].rolling(window=7).mean()    # 7일 이동 평균
df['MA_30'] = df['Value'].rolling(window=30).mean()   # 30일 이동 평균

# 이동 표준편차
df['Value'].rolling(window=7).std()

# 이동 최대/최소
df['Value'].rolling(window=7).max()

# 지수 이동 평균 (EMA)
df['Value'].ewm(span=7).mean()

# expanding (누적 윈도우)
df['Value'].expanding().mean()     # 누적 평균
df['Value'].expanding().max()      # 누적 최대

# shift (행 이동)
df['Prev_Value'] = df['Value'].shift(1)    # 이전 값
df['Next_Value'] = df['Value'].shift(-1)   # 다음 값
```

---

## 🔑 자주 쓰는 패턴 Quick Reference

### 데이터 로드 → 탐색

```python
df = pd.read_csv("data.csv")
df.shape                    # 크기 확인
df.head()                   # 상위 5행
df.info()                   # 타입, 결측치
df.describe()               # 기술 통계
df.isnull().sum()           # 결측치 수
df.duplicated().sum()       # 중복 수
```

### 전처리 파이프라인

```python
df = (df
    .drop(columns=['불필요열1', '불필요열2'])          # 불필요 열 제거
    .drop_duplicates()                               # 중복 제거
    .dropna(subset=['중요열'])                        # 중요 열 결측치 행 제거
    .fillna({'Age': df['Age'].median()})             # 결측치 채우기
    .rename(columns={'old': 'new'})                  # 열 이름 변경
    .astype({'Pclass': 'category'})                  # 타입 변환
    .reset_index(drop=True)                          # 인덱스 리셋
)
```

### 불리언 인덱싱 패턴

```python
# 조건 → True/False Series → 대괄호에 넣으면 필터링
mask = (df['Age'] > 30) & (df['Survived'] == 1)
df[mask]
```

---

## ⚠️ 주의사항

### inplace 파라미터
```python
# inplace=True → 원본 수정, 반환값 None
df.drop('col', axis=1, inplace=True)    # ✅ OK
result = df.drop('col', axis=1, inplace=True)  # ❌ result는 None!

# inplace=False (기본값) → 새 객체 반환
df = df.drop('col', axis=1)    # ✅ 이 방식을 권장
```

### SettingWithCopyWarning

```python
# ❌ 경고 발생 가능
df[df['Age'] > 30]['Name'] = 'test'

# ✅ .loc 사용
df.loc[df['Age'] > 30, 'Name'] = 'test'
```

### axis 방향

```
axis=0 : 행 방향 (↓ 위에서 아래로)
axis=1 : 열 방향 (→ 왼쪽에서 오른쪽으로)
```

---

> 📌 Pandas 공식 문서: https://pandas.pydata.org/docs/
>
> 📌 버전: 이 문서는 Pandas 2.x 기준으로 작성되었습니다.

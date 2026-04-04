# GDELT & CAMEO 코드북 가이드 (팀 공용)

> 원본: `GDELT-Data_Format_Codebook.pdf` + `CAMEO.Manual.1.1b3.pdf`
> 이 문서는 팀원 전원이 데이터를 동일한 수준으로 이해하기 위한 요약본입니다.

---

## 1. GDELT 데이터란?

GDELT(Global Database of Events, Language, and Tone)는 전 세계 뉴스 기사를 자동으로 읽고, **"누가, 누구에게, 무엇을 했는가"**를 구조화한 데이터베이스입니다.

- 100개국 이상, 수십 개 언어의 뉴스를 매 15분마다 수집
- 1979년부터 현재까지 수억 건의 이벤트 기록
- 우리 데이터: **2013.04 ~ 2026.03, 약 2,660만 건**

---

## 2. 컬럼 구조 (gdelt_main_final.parquet)

데이터는 크게 **5개 영역**으로 나뉩니다.

### 2.1 행위자 정보 (Actor)

모든 이벤트는 **Actor1(주체)이 Actor2(대상)에게 어떤 행동을 했는가**로 기록됩니다.

| 컬럼 | 타입 | 설명 | 예시 |
|:---|:---:|:---|:---|
| `Actor1Code` | 문자 | Actor1의 전체 CAMEO 코드 | IRNMIL (이란 군) |
| `Actor1Name` | 문자 | Actor1의 실제 이름 | IRAN, GEORGE W BUSH |
| `Actor1CountryCode` | 문자 | Actor1의 국가 코드 (3자리) | IRN, USA, ISR |
| `Actor1KnownGroupCode` | 문자 | 알려진 조직 코드 (UN, NATO 등) | IGO, NGO |
| `Actor1EthnicCode` | 문자 | 민족 코드 | KUR (쿠르드) |
| `Actor1Religion1Code` | 문자 | 종교 코드 | SHI (시아파) |
| `Actor1Type1Code` | 문자 | 역할 코드 (정부, 군, 경찰, 반군 등) | GOV, MIL, REB |

> Actor2도 동일한 구조로 반복됩니다.

**주요 CountryCode**:
- `IRN` = 이란
- `USA` = 미국
- `ISR` = 이스라엘
- `IRQ` = 이라크
- `LBN` = 레바논

**주요 Type Code**:
| 코드 | 의미 |
|:---:|:---|
| GOV | 정부 |
| MIL | 군대 |
| REB | 반군 |
| OPP | 야당/반대파 |
| COP | 경찰 |
| JUD | 사법부 |
| SPY | 정보기관 |
| MED | 언론 |
| EDU | 교육 |
| CVL | 시민 |

### 2.2 이벤트 속성 (Event Action)

| 컬럼 | 타입 | 설명 | 핵심 |
|:---|:---:|:---|:---:|
| `IsRootEvent` | 0/1 | 기사의 첫 문단에 나온 사건인가? | ⭐ |
| `EventCode` | 문자 | CAMEO 상세 이벤트 코드 (3~4자리) | ⭐ |
| `EventBaseCode` | 문자 | 중간 단위 코드 (2~3자리) | |
| `EventRootCode` | 문자 | 최상위 분류 코드 (2자리) | ⭐ |
| `QuadClass` | 정수 | 4대 분류 | ⭐ |
| `GoldsteinScale` | 실수 | 안정성 점수 (-10 ~ +10) | ⭐ |
| `NumMentions` | 정수 | 뉴스 언급 횟수 | ⭐ |
| `NumSources` | 정수 | 뉴스 출처 수 | |
| `NumArticles` | 정수 | 기사 수 | |
| `AvgTone` | 실수 | 뉴스 감성 점수 (-100 ~ +100, 보통 -10 ~ +10) | ⭐ |

**IsRootEvent** = 1이면 기사의 핵심 사건. 0이면 부수적 언급. 중요도 필터로 사용.

**QuadClass (4대 분류)**:
| 값 | 의미 | Root Code 범위 |
|:---:|:---|:---|
| 1 | 언어적 협력 (Verbal Cooperation) | 01~05 |
| 2 | 물질적 협력 (Material Cooperation) | 06~09 |
| 3 | 언어적 갈등 (Verbal Conflict) | 10~14 |
| 4 | **물질적 갈등 (Material Conflict)** | **15~20** |

> QuadClass = 4가 우리의 핵심 분석 대상입니다.

### 2.3 이벤트 날짜

| 컬럼 | 타입 | 설명 |
|:---|:---:|:---|
| `SQLDATE` | 정수 | 사건 발생일 (YYYYMMDD 형식) |
| `MonthYear` | 정수 | 연-월 (YYYYMM 형식) |
| `Year` | 정수 | 연도 |
| `FractionDate` | 실수 | 소수점 연도 (2024.5 = 2024년 7월) |
| `DATEADDED` | 정수 | DB 등록일 |

### 2.4 위치 정보 (Geography)

각 이벤트에 3가지 위치가 기록됩니다:
- **Actor1Geo_**: Actor1과 관련된 위치
- **Actor2Geo_**: Actor2와 관련된 위치
- **ActionGeo_**: **사건이 실제 발생한 위치** ← 가장 중요

| 컬럼 | 타입 | 설명 | 핵심 |
|:---|:---:|:---|:---:|
| `ActionGeo_Type` | 정수 | 위치 해상도 | ⭐ |
| `ActionGeo_FullName` | 문자 | 전체 지명 | ⭐ |
| `ActionGeo_CountryCode` | 문자 | 국가 코드 (2자리 FIPS) | |
| `ActionGeo_ADM1Code` | 문자 | 행정구역 코드 | |
| `ActionGeo_Lat` | 실수 | 위도 | ⭐ |
| `ActionGeo_Long` | 실수 | 경도 | ⭐ |
| `ActionGeo_FeatureID` | 정수 | 지명 고유 식별자 (GNS/GNIS) | ⭐ |

**ActionGeo_Type 해상도**:
| 값 | 의미 | FeatureID |
|:---:|:---|:---:|
| 1 | 국가 (Country) | 없음 |
| 2 | 미국 주 (US State) | 없음 |
| 3 | 미국 도시 (US City) | 있음 |
| 4 | **해외 도시 (World City)** | **있음** |
| 5 | 해외 행정구역 (World State) | 없음 |

> **ActionGeo_Type = 4이면서 FeatureID가 있는 것**이 가장 정밀한 위치 데이터입니다.

> ⚠️ **중요**: 같은 도시는 항상 같은 좌표(도시 중심점)입니다. 테헤란 이벤트 10만 건이 전부 (35.75, 51.51)입니다. GDELT는 건물/시설 단위 좌표를 제공하지 않습니다.

> ⚠️ **중요**: 같은 장소의 이름이 다르게 표기될 수 있습니다 (Tehran vs Teheran, Mecca vs Makkah). 따라서 위치를 비교할 때는 `FeatureID`를 사용해야 합니다.

### 2.5 고유 식별자

| 컬럼 | 타입 | 설명 |
|:---|:---:|:---|
| `GLOBALEVENTID` | 정수 | 이벤트 고유 ID (전 세계 유일) |

---

## 3. CAMEO 이벤트 코드 체계

CAMEO(Conflict and Mediation Event Observations)는 국제 관계 이벤트를 **20개 Root Code**로 분류하는 표준 체계입니다.

### 3.1 전체 Root Code 목록

| Root | 이름 | 한국어 | QuadClass | 성격 |
|:---:|:---|:---|:---:|:---:|
| 01 | Make public statement | 공개 성명 | 1 | 협력 |
| 02 | Appeal | 호소 | 1 | 협력 |
| 03 | Express intent to cooperate | 협력 의사 표현 | 1 | 협력 |
| 04 | Consult | 협의 | 1 | 협력 |
| 05 | Engage in diplomatic cooperation | 외교적 협력 | 1 | 협력 |
| 06 | Engage in material cooperation | 물질적 협력 | 2 | 협력 |
| 07 | Provide aid | 원조 제공 | 2 | 협력 |
| 08 | Yield | 양보 | 2 | 협력 |
| 09 | Investigate | 조사 | 1 | 중립 |
| 10 | Demand | 요구 | 3 | 갈등 |
| 11 | Disapprove | 비난 | 3 | 갈등 |
| 12 | Reject | 거부 | 3 | 갈등 |
| 13 | Threaten | 위협 | 3 | 갈등 |
| 14 | Protest | 시위 | 3 | 갈등 |
| **15** | **Exhibit military posture** | **군사 태세** | **4** | **물질적 갈등** |
| **16** | **Reduce relations** | **관계 축소** | **4** | **물질적 갈등** |
| **17** | **Coerce** | **강압** | **4** | **물질적 갈등** |
| **18** | **Assault** | **폭행** | **4** | **물질적 갈등** |
| **19** | **Fight** | **교전/무력 사용** | **4** | **물질적 갈등** |
| **20** | **Unconventional mass violence** | **대량 폭력** | **4** | **물질적 갈등** |

> Root 15~20 (QuadClass = 4)이 우리의 핵심 분석 대상입니다.

### 3.2 핵심 Root Code 상세 (15, 18, 19, 20)

#### Root 15: EXHIBIT MILITARY POSTURE (군사 태세)

군사력을 과시하거나 경계 태세를 강화하는 행위. **아직 싸우지는 않았지만 싸울 준비를 하는 단계**.

| EventCode | 설명 | 예시 |
|:---:|:---|:---|
| 150 | 군/경 무력 시위 (미분류) | 군사 퍼레이드 |
| 151 | 경찰 경계 격상 | 테러 경보 상향 |
| 152 | 군사 경계 격상 | 방공 시스템 가동 |
| 153 | 경찰력 증강 | 국경 경비 강화 |
| **154** | **병력/무기 증강 배치** | **걸프만 항모 전개** |
| 155 | 사이버 전력 동원 | 사이버 공격 태세 |

> **GoldsteinScale**: 대체로 -7.0 ~ -9.4

#### Root 18: ASSAULT (폭행/비정규 폭력)

정규군이 아닌 비정규적 폭력. 테러, 납치, 암살 등.

| EventCode | 설명 | 위성 촬영 가치 |
|:---:|:---|:---:|
| 180 | 비정규 폭력 (미분류) | 🔶 애매 |
| **181** | **납치, 하이재킹, 인질** | ❌ 촬영 불가 |
| 182 | 물리적 폭행 | ❌ |
| **183** | **자살 폭탄/차량 폭탄** | ✅ 폭발 흔적 |
| 184 | 인간 방패로 사용 | ❌ |
| **185** | **암살 시도** | ❌ 촬영 불가 |
| **186** | **암살** | ❌ 촬영 불가 |

> **GoldsteinScale**: 대체로 -9.0 ~ -10.0

#### Root 19: FIGHT (교전/무력 사용) ← 가장 중요

정규 군사력을 사용한 교전. 우리 데이터에서 가장 빈도가 높고, 위성 촬영 가치가 가장 높은 카테고리.

| EventCode | 설명 | 위성 촬영 가치 |
|:---:|:---|:---:|
| **190** | **통상 무력 사용 (미분류)** | 🔶 추가 검증 필요 |
| 191 | 봉쇄, 이동 제한 | ✅ 바리케이드 등 |
| 192 | 영토 점령 | ✅ 점령 흔적 |
| 193 | 소화기 교전 | ❌ 흔적 미미 |
| **194** | **포병/전차 교전** | ✅✅ 파괴 흔적 명확 |
| **195** | **항공 공격/폭격** | ✅✅ 폭격 흔적 명확 |
| 1951 | 정밀 유도 무기 | ✅✅ |
| 1952 | 드론 공격 | ✅✅ |
| 196 | 휴전 위반 | ✅ |

> **GoldsteinScale**: -10.0 고정
> 이벤트 코드 194, 195가 위성 촬영에 가장 유의미합니다.

#### Root 20: UNCONVENTIONAL MASS VIOLENCE (대량 폭력)

대규모 비정규 폭력. 대량 학살, 화학무기 등.

| EventCode | 설명 | 위성 촬영 가치 |
|:---:|:---|:---:|
| 200 | 대규모 비정규 폭력 (미분류) | ✅ |
| 201 | 대량 추방 | ✅ 난민 이동 |
| 202 | 대량 학살 | ✅ |
| 203 | 화학/생물/방사능 무기 | ✅✅ |
| 204 | 핵무기 사용 또는 폭발 | ✅✅✅ |

> **GoldsteinScale**: -10.0 고정

### 3.3 코드 계층 구조

```
EventRootCode = "19"           ← 최상위 (Fight)
EventBaseCode = "195"          ← 중간 (Aerial weapons)
EventCode     = "1952"         ← 최하위 (Drone attack)
```

---

## 4. 핵심 지표 해석 가이드

### GoldsteinScale (-10 ~ +10)

| 값 | 의미 | 해당 이벤트 예시 |
|:---|:---|:---|
| +10.0 | 극도의 협력 | 군사 동맹 체결 |
| +7.0 | 강한 협력 | 인도적 원조 |
| +3.0 | 가벼운 협력 | 외교 회담 |
| 0.0 | 중립 | 공식 성명 |
| -3.0 | 가벼운 갈등 | 외교적 비난 |
| -7.0 | 심각한 갈등 | 관계 축소, 제재 |
| **-10.0** | **극도의 갈등** | **군사 공격, 대량 폭력** |

> 이벤트 코드에 따라 **고정값**입니다. 같은 코드의 사건은 규모와 무관하게 같은 점수.

### AvgTone (-100 ~ +100, 보통 -10 ~ +10)

GoldsteinScale이 "사건 유형의 심각도"라면, AvgTone은 **"뉴스 보도의 분위기"**입니다.

- 같은 폭동이라도 AvgTone이 -2이면 경미한 보도, -9이면 심각한 보도
- **AvgTone이 극단적으로 부정적**이면 실제로 심각한 사건일 가능성이 높음

### NumMentions

- 해당 이벤트가 뉴스에서 **총 몇 번 언급**되었는가
- 같은 기사 내 반복 언급도 포함
- **사건의 "중요도" 대리 지표**로 사용 가능
- ⚠️ 시기별 뉴스 총량이 다르므로, 절대값보다 상대적 비교에 사용

### NumSources

- 해당 이벤트를 보도한 **뉴스 출처 수**
- NumMentions보다 "얼마나 다양한 매체가 보도했는가"를 반영
- 값이 높으면 국제적 관심이 큰 사건

---

## 5. gdelt_url_final.parquet

| 컬럼 | 설명 |
|:---|:---|
| `GLOBALEVENTID` | 이벤트 고유 ID (main 파일과 조인 키) |
| `SOURCEURL` | 해당 이벤트가 발견된 뉴스 기사의 URL |

> 하나의 이벤트가 여러 기사에서 언급될 수 있으므로, 같은 GLOBALEVENTID에 여러 URL이 있을 수 있습니다.

---

## 6. EDA에서 확인해야 할 핵심 질문 8개

| # | 질문 | 사용 컬럼 |
|:---:|:---|:---|
| 1 | 데이터에 컬럼이 몇 개이고, 각각 무슨 의미인가? | 전체 |
| 2 | IRN/USA/ISR 이벤트가 전체의 몇 %인가? | Actor1/2CountryCode |
| 3 | EventRootCode 분포는? 15/18/19/20은 몇 %? | EventRootCode |
| 4 | GoldsteinScale 분포는? 어디에 몰려 있는가? | GoldsteinScale |
| 5 | AvgTone 분포는? | AvgTone |
| 6 | 시간에 따른 이벤트 수 추이는? 피크는 언제? | SQLDATE |
| 7 | ActionGeo_FullName 상위 도시는? | ActionGeo_* |
| 8 | NumMentions, NumSources 분포는? | NumMentions, NumSources |

# PyTorch 평점 예측 모델의 Vertex AI 서빙 검증 프로젝트

## 1. 프로젝트 개요

이 프로젝트는 Netflix Prize 평점 데이터를 활용해 PyTorch 기반 평점 예측 모델을 학습하고, 학습된 모델이 Vertex AI Endpoint에서 실제 예측 API로 동작하기까지의 배포 흐름을 검증한 MLOps 실습 프로젝트입니다.

단순히 모델을 학습하는 것에서 끝내지 않고, 학습된 모델이 서비스 환경에서 예측 요청을 받을 수 있으려면 어떤 구성 요소가 필요한지 확인하는 데 초점을 두었습니다.

특히 다음 질문을 중심으로 프로젝트를 진행했습니다.

* 학습된 PyTorch 모델은 어떻게 서빙 가능한 artifact가 되는가?
* 로컬 학습 코드와 클라우드 서빙 환경은 무엇이 다른가?
* Endpoint가 생성되었는데도 예측 요청이 실패하는 이유는 무엇인가?
* request schema, handler, container, model artifact는 어떻게 맞물려야 하는가?

## 2. 프로젝트 목적

모델 성능을 높이는 것보다, 학습 모델이 실제 예측 API로 제공되기까지의 과정을 이해하는 것이 핵심 목적이었습니다.

일반적인 모델 학습 프로젝트에서는 loss 감소나 예측 성능에 집중하지만, 실제 서비스에서는 학습된 모델이 안정적으로 호출되고, 입력 형식에 맞게 예측 결과를 반환하는 과정이 중요합니다.

따라서 이 프로젝트에서는 다음 과정을 직접 확인했습니다.

1. Netflix 평점 데이터를 학습 가능한 형태로 전처리
2. 사용자별 영화 평가 이력을 시퀀스 데이터로 구성
3. PyTorch 기반 평점 예측 모델 학습
4. 학습된 모델을 서빙 가능한 artifact로 패키징
5. Vertex AI Endpoint 배포
6. raw-predict 및 batch request를 통한 예측 응답 검증
7. 예측 요청 실패 시 Cloud Logging과 request schema를 기준으로 원인 추적

## 3. 사용 데이터

### Netflix Prize Data

Netflix Prize 데이터는 사용자, 영화, 평점, 날짜 정보를 포함한 대규모 평점 데이터셋입니다.

본 프로젝트에서는 전체 데이터를 그대로 사용하는 대신, 학습 시간과 실습 환경의 제약을 고려해 사용자 기준 샘플링을 적용했습니다.

| 항목    | 내용                                      |
| ----- | --------------------------------------- |
| 데이터셋  | Netflix Prize Data                      |
| 주요 컬럼 | `Cust_Id`, `Movie_Id`, `Rating`, `Date` |
| 문제 유형 | 사용자 시퀀스 기반 평점 예측                        |
| 입력    | 사용자별 영화 시청/평가 이력                        |
| 출력    | 예측 평점                                   |
| 실험 제약 | 전체 데이터 학습 시간이 커 사용자 기준 `1/30` 샘플링 적용    |

## 4. 데이터 전처리

원본 데이터는 영화 ID와 사용자 평점이 하나의 텍스트 파일 구조 안에 섞여 있어, 바로 학습에 사용할 수 없습니다.

전처리 과정에서는 다음 작업을 수행했습니다.

1. `combined_data_1.txt` ~ `combined_data_4.txt` 파일 로드
2. 영화 ID와 사용자 평점 행 분리
3. `Cust_Id`, `Movie_Id`, `Rating`, `Date` 형태의 테이블로 변환
4. 사용자별 평가 이력을 날짜 기준으로 정렬
5. 고정 길이 sequence를 생성해 모델 입력으로 변환

사용자별 평가 이력은 날짜순으로 정렬한 뒤, 일정 길이의 영화 ID 시퀀스를 모델 입력으로 사용했습니다.

```python
Movie_Sequence = [movie_1, movie_2, ..., movie_n]
Rating_Sequence = [rating_1, rating_2, ..., rating_n]
```

이 구조를 통해 모델은 사용자의 과거 영화 평가 흐름을 바탕으로 마지막 위치의 평점을 예측하도록 구성했습니다.

## 5. 모델 구조

모델은 PyTorch 기반 Transformer Encoder 구조를 사용했습니다.

다만 이 프로젝트의 핵심은 추천 성능 최적화가 아니라, 학습된 모델을 클라우드 환경에서 서빙 가능한 형태로 연결하는 것이었습니다. 따라서 모델 구조 자체보다는 학습 결과물이 배포 환경에서 어떻게 사용되는지에 더 초점을 두었습니다.

### 모델 구성

| 구성 요소               | 설명                           |
| ------------------- | ---------------------------- |
| Movie Embedding     | 영화 ID를 dense vector로 변환      |
| Positional Encoding | 시퀀스 내 위치 정보를 반영              |
| Transformer Encoder | 사용자별 영화 평가 시퀀스의 패턴 학습        |
| Linear Layer        | 마지막 hidden state를 기반으로 평점 예측 |

### 입력과 출력

| 구분   | 내용                  |
| ---- | ------------------- |
| 입력   | 사용자별 영화 ID sequence |
| 출력   | 예측 평점               |
| Loss | 평점 회귀 기준 손실 함수      |

## 6. 배포 구조

이 프로젝트에서 가장 중점적으로 확인한 부분은 모델 학습 이후의 배포 과정입니다.

학습된 모델이 실제 예측 API로 동작하려면 단순히 `.pth` 파일만 저장하는 것으로는 부족합니다. 모델 artifact, handler, container, request schema가 서로 일관되게 맞아야 합니다.

### 전체 흐름

```text
Netflix Prize Data
        ↓
Data Preprocessing
        ↓
PyTorch Model Training
        ↓
Model Artifact Packaging
        ↓
TorchServe model.mar 생성
        ↓
Vertex AI Endpoint 배포
        ↓
raw-predict / batch request 검증
        ↓
Cloud Logging 기반 오류 추적
```

## 7. Vertex AI 서빙 검증

Vertex AI Endpoint 배포 이후, 실제 예측 요청을 보내며 다음 사항을 확인했습니다.

* Endpoint 생성 여부만으로는 배포 성공을 판단할 수 없음
* 모델이 요청을 받더라도 입력 JSON 구조가 handler와 맞지 않으면 예측 실패
* serving 환경에서는 학습 코드보다 request schema와 handler의 일관성이 중요
* Cloud Logging을 통해 예측 실패 원인을 추적할 수 있음

### 검증한 요청 방식

| 요청 방식       | 목적                                         |
| ----------- | ------------------------------------------ |
| 단일 입력 요청    | 하나의 사용자 sequence에 대한 예측 응답 확인              |
| Batch 요청    | 여러 개 입력을 한 번에 전달했을 때 응답 구조 확인              |
| raw-predict | Vertex AI Endpoint에서 모델 서버의 실제 입력 처리 방식 검증 |

## 8. 오류 해결 과정

배포 과정에서 가장 중요하게 확인한 문제는 request schema 불일치였습니다.

Endpoint가 생성되었더라도, 클라이언트가 보내는 JSON 구조와 handler가 기대하는 입력 구조가 다르면 예측 요청은 실패합니다.

문제를 해결하기 위해 다음 순서로 원인을 확인했습니다.

1. Vertex AI Endpoint 생성 여부 확인
2. 예측 요청 JSON 구조 확인
3. Cloud Logging에서 오류 메시지 확인
4. handler가 기대하는 입력 형태와 실제 요청 형태 비교
5. `instances` 기반 입력 구조로 수정
6. 단일 요청 및 batch 요청에서 `predictions` 응답 반환 확인

이 과정을 통해 모델 배포에서 중요한 것은 모델 파일 자체가 아니라, 모델이 입력을 받아 처리하고 응답을 반환하기까지의 전체 인터페이스라는 점을 확인했습니다.

## 9. 학습한 점

이 프로젝트를 통해 가장 크게 배운 점은 모델 학습과 모델 서빙이 서로 다른 문제라는 것입니다.

학습 단계에서는 데이터셋, 모델 구조, loss, optimizer가 중요합니다. 반면 서빙 단계에서는 다음 요소가 더 중요하게 작용했습니다.

* 모델 artifact가 올바르게 저장되었는가
* handler가 입력을 모델이 받을 수 있는 tensor로 변환하는가
* request schema가 handler와 일치하는가
* container 환경에서 필요한 의존성이 모두 충족되는가
* Endpoint는 실제 요청에 대해 안정적으로 응답하는가
* 오류 발생 시 Cloud Logging을 통해 원인을 좁힐 수 있는가

즉, 좋은 모델을 만드는 것과 그 모델을 실제 서비스에서 호출 가능한 형태로 만드는 것은 별도의 역량이 필요하다는 점을 확인했습니다.

## 10. 기술 스택

| 분류              | 기술                       |
| --------------- | ------------------------ |
| Language        | Python                   |
| Modeling        | PyTorch                  |
| Data Processing | Pandas, NumPy            |
| Dataset         | Netflix Prize Data       |
| Cloud Storage   | Google Cloud Storage     |
| Data Warehouse  | BigQuery                 |
| Pipeline        | Vertex AI Pipelines, KFP |
| Serving         | TorchServe               |
| Deployment      | Vertex AI Endpoint       |
| Debugging       | Cloud Logging            |

## 11. 프로젝트 한계

이 프로젝트는 추천 모델 성능을 고도화하기 위한 프로젝트라기보다, 교육 과정에서 제시된 아키텍처를 바탕으로 모델 배포 흐름을 이해하기 위한 실습 프로젝트입니다.

따라서 다음과 같은 한계가 있습니다.

* 전체 Netflix Prize 데이터를 끝까지 학습하지 않고 사용자 기준 샘플링을 적용
* RMSE, MAE 등 모델 성능 지표를 중심으로 비교하지 않음
* 추천 ranking 성능 지표인 NDCG@K, HitRate@K는 측정하지 않음
* CI/CD, 자동 재학습, 모델 모니터링까지는 구현하지 않음
* 실서비스 운영이 아니라 Endpoint 배포 및 예측 응답 검증에 초점을 둠

## 12. 후속 개선 방향

이 프로젝트를 실제 MLOps 프로젝트로 확장한다면 다음 작업을 보완할 수 있습니다.

1. 모델 성능 평가 추가

   * Global Mean baseline
   * User Mean baseline
   * RMSE / MAE 비교

2. 추천 시스템 평가로 확장

   * NDCG@K
   * HitRate@K
   * Recall@K

3. 배포 자동화 개선

   * GitHub Actions 기반 pipeline 실행
   * 학습 artifact 버전 관리
   * 모델 registry 연동

4. 운영 관점 보완

   * Endpoint latency 측정
   * request / response logging
   * 입력 데이터 drift 모니터링
   * 실패 요청 케이스 관리

## 13. 정리

이 프로젝트는 “좋은 추천 모델을 만들었다”는 것을 주장하기 위한 프로젝트가 아닙니다.

대신, 학습된 모델이 실제 예측 API로 동작하기 위해 필요한 배포 구성 요소를 직접 확인한 프로젝트입니다.

특히 PyTorch 모델을 Vertex AI Endpoint에 배포하는 과정에서 model artifact, handler, container, request schema가 일관되어야 하며, Endpoint 생성 이후에도 실제 요청 검증과 로그 기반 디버깅이 필요하다는 점을 확인했습니다.

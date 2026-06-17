# Evaluation Files

이 폴더는 RAG 챗봇 평가용 데이터셋과 결과 파일을 모아둔 공간입니다.

## datasets

- `eval_dataset.csv`: 슬롯 추출, DB 매핑, 검색 평가에 사용하는 기본 평가 데이터셋
- `eval_dataset_robustness.csv`: 오타, 은어, 어순 변경 등 robustness 확인용 평가 데이터셋

## results

- `eval_results.csv`: `eval_rag.py` 실행 결과
- `eval_summary.csv`: 전체 평가 지표 요약
- `eval_results_baseline.csv`: 리팩토링 전 또는 기준 버전 평가 결과
- `eval_summary_baseline.csv`: 기준 버전 평가 요약
- `eval_summary_by_noise_type.csv`: noise type별 평가 요약

## results/ragas

- `ragas_eval_*.csv`: RAGAS Faithfulness 평가 결과
- 무료 Gemini API quota 제한 때문에 하루에 소량 샘플만 누적 평가합니다.

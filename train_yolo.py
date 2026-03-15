"""
train_yolo.py
=============
YOLOv8 파인튜닝 학습 스크립트 (1개월 차 3주 차)

사용법:
    # 기본 실행 (nano 모델, 50 epochs)
    python train_yolo.py

    # 커스텀 옵션
    python train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16

학습 결과는 runs/detect/train_<실험명>/ 폴더에 저장됩니다.
    - weights/best.pt   : 검증 mAP 기준 최고 성능 가중치
    - weights/last.pt   : 마지막 epoch 가중치
    - results.csv       : epoch별 loss/mAP 수치
    - val_batch*.jpg    : validation 시각화
"""

import argparse
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 파인튜닝 학습 스크립트")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="베이스 모델 (yolov8n.pt / yolov8s.pt / yolov8m.pt / yolo11n.pt 등) [default: yolov8n.pt]"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="yolo_master_dataset/data.yaml",
        help="data.yaml 경로 [default: yolo_master_dataset/data.yaml]"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 에폭 수 [default: 50]"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="입력 이미지 크기 [default: 640]"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기 (-1 이면 AutoBatch) [default: 16]"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="초기 학습률 [default: 0.01]"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience (epochs) [default: 15]"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="학습 디바이스 (cpu / 0 / mps 등, 비워두면 자동 감지)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="실험 이름 (비워두면 자동 생성)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="last.pt 에서 학습 재시작"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="빠른 결과 확인 모드: imgsz=320, epochs=20으로 강제 설정 (결과를 빨리 보고 싶을 때)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="데이터 로더 워커 수 [default: 4]"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 의존성 확인 ───────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics 패키지가 없습니다. 다음 명령어로 설치하세요:")
        print("   pip install ultralytics")
        return

    # ── 경로 확인 ─────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {data_path.resolve()}")
        print("   merge_yolo.py를 먼저 실행했는지 확인하세요.")
        return

    # ── --fast 모드 적용 ─────────────────────────────────────────
    if args.fast:
        print("⚡ [FAST MODE] imgsz=320, epochs=20 으로 강제 설정합니다.")
        args.imgsz = 320
        args.epochs = 20
        args.patience = 10

    # ── 실험 이름 자동 생성 ──────────────────────────────────────
    model_tag = Path(args.model).stem  # e.g. "yolov8n"
    mode_tag = "_fast" if args.fast else ""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_name = args.name if args.name else f"{model_tag}_ep{args.epochs}{mode_tag}_{timestamp}"

    # ── 모델 로드 ─────────────────────────────────────────────────
    print(f"\n🚀 YOLO 파인튜닝 학습 시작")
    print(f"   Base model : {args.model}")
    print(f"   Data       : {data_path.resolve()}")
    print(f"   Epochs     : {args.epochs}")
    print(f"   Image size : {args.imgsz}px  {'(FAST MODE)' if args.fast else ''}")
    print(f"   Batch      : {args.batch}")
    print(f"   Workers    : {args.workers}")
    print(f"   Experiment : {exp_name}")
    print(f"   Resume     : {args.resume}")
    print("-" * 50)

    if args.resume:
        # last.pt에서 이어서 학습
        resume_path = Path(f"runs/detect/{exp_name}/weights/last.pt")
        if not resume_path.exists():
            print(f"❌ Resume할 last.pt를 찾을 수 없습니다: {resume_path}")
            return
        model = YOLO(str(resume_path))
    else:
        model = YOLO(args.model)

    # ── 학습 실행 ─────────────────────────────────────────────────
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device if args.device else None,
        workers=args.workers,
        project="runs/detect",
        name=exp_name,
        exist_ok=True,              # 같은 이름 실험 덮어쓰기 허용
        resume=args.resume,
        # ----- Augmentation -----
        # ⚠️ mosaic/mixup을 끔: 빈 라벨 파일과 충돌하여
        #    'shape mismatch' RuntimeError 유발 (ultralytics 알려진 이슈)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,                # 색조 조정 (조명 변화 대응)
        hsv_s=0.7,                  # 채도 조정
        hsv_v=0.4,                  # 명도 조정
        flipud=0.05,                # 상하 뒤집기
        fliplr=0.5,                 # 좌우 뒤집기
        # ----- NMS 속도 최적화 (epoch 12 이후 validation 폭발적 지연 방지) -----
        # 기본 val_conf=0.001 → 저신뢰도 박스 수만 개 생성 → NMS 타임아웃
        conf=0.01,                  # inference confidence threshold (높일수록 NMS 박스 수 감소)
        iou=0.7,                    # NMS IoU threshold
        max_det=300,                # 이미지 당 최대 탐지 박스 수 제한 (기본 300)
        # ----- 저장 설정 -----
        save=True,
        save_period=-1,             # 중간 체크포인트 저장 안 함 (속도 향상)
        val=True,                   # 매 epoch 후 validation 실행
        plots=True,                 # loss curve / confusion matrix 그래프 저장
    )

    # ── 학습 완료 메시지 ──────────────────────────────────────────
    best_path = Path(f"runs/detect/{exp_name}/weights/best.pt")
    print("\n" + "=" * 50)
    print(f"✅ 학습 완료!")
    print(f"   최고 성능 가중치  : {best_path.resolve()}")
    print(f"   결과 폴더        : runs/detect/{exp_name}/")
    print("\n📊 다음 단계: 성능 평가")
    print(f"   python evaluate_yolo.py --weights {best_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

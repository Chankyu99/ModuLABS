"""
evaluate_yolo.py
================
학습된 YOLO 모델의 성능 평가 + 오탐지(FP/FN) 시각화 스크립트 (1개월 차 4주 차)

사용법:
    # 기본 실행 (best.pt 경로를 직접 지정)
    python evaluate_yolo.py --weights runs/detect/yolov8n_ep50_XXXX/weights/best.pt

    # 저장 폴더 지정
    python evaluate_yolo.py --weights <path/to/best.pt> --output eval_results

출력:
    - 콘솔: 클래스별 Precision / Recall / mAP50 / mAP50-95 테이블
    - eval_results/
        ├── confusion_matrix.png    : confusion matrix 시각화
        ├── fp_samples/             : 오탐지(FP) 샘플 이미지 (최대 12장)
        └── fn_samples/             : 미탐지(FN) 샘플 이미지 (최대 12장)
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml


CLASS_NAMES = ["power_bank", "liquid_cosmetics", "lighter", "laptop", "tablet"]
COLORS = {
    "gt":   (0,   200, 0),    # ground truth: 초록
    "pred": (0,   0,   220),  # prediction:   파랑
    "fp":   (220, 0,   0),    # false positive: 빨강
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 성능 평가 및 오탐지 시각화")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="평가할 best.pt 경로 (예: runs/detect/train_xxx/weights/best.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="yolo_master_dataset/data.yaml",
        help="data.yaml 경로 [default: yolo_master_dataset/data.yaml]"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="추론 이미지 크기 [default: 640]"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="탐지 confidence threshold [default: 0.25]"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold [default: 0.5]"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="평가 결과 저장 폴더 [default: eval_results]"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=12,
        help="FP/FN 시각화 최대 샘플 수 [default: 12]"
    )
    return parser.parse_args()


def iou(boxA, boxB):
    """두 bbox (x1,y1,x2,y2) 간의 IoU 계산"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx, cy, bw, bh, W, H):
    """YOLO 정규화 좌표 → 픽셀 절대 좌표 (x1,y1,x2,y2)"""
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return x1, y1, x2, y2


def draw_boxes(img, boxes, color, label_prefix=""):
    """이미지에 bounding box와 레이블 텍스트를 그림"""
    for cls_id, x1, y1, x2, y2, conf in boxes:
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{label_prefix}{name}" + (f" {conf:.2f}" if conf >= 0 else "")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, max(y1 - 18, 0)), (x1 + tw + 2, max(y1, 18)), color, -1)
        cv2.putText(img, label, (x1 + 1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def save_sample_grid(samples, out_dir, title, max_samples=12):
    """샘플 이미지 리스트를 그리드로 저장"""
    if not samples:
        print(f"   ℹ️  {title}: 해당 케이스 없음")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(samples), max_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten()
    for i, (img_rgb, subtitle) in enumerate(samples[:n]):
        axes[i].imshow(img_rgb)
        axes[i].set_title(subtitle, fontsize=8)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = out_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   💾 저장됨: {out_path}")


def analyze_errors(model, test_img_dir, test_lbl_dir, conf_thres, iou_thres, max_samples):
    """
    test 셋에 대해 FP / FN 이미지 샘플을 직접 수집합니다.
    - FP: 모델이 탐지했지만 실제로는 없는 박스 (오탐)
    - FN: 실제로 있지만 모델이 놓친 물체 (미탐)
    """
    fp_samples = []
    fn_samples = []

    img_paths = list(Path(test_img_dir).glob("*.jpg")) + \
                list(Path(test_img_dir).glob("*.png")) + \
                list(Path(test_img_dir).glob("*.jpeg"))
    random.shuffle(img_paths)

    print(f"\n🔍 FP/FN 분석 중... (test 이미지 {len(img_paths)}장)")

    for img_path in img_paths:
        if len(fp_samples) >= max_samples and len(fn_samples) >= max_samples:
            break

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        # ── Ground Truth 파싱 ──
        lbl_path = Path(test_lbl_dir) / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
                gt_boxes.append((cls_id, x1, y1, x2, y2, -1))

        # ── 모델 추론 ──
        results = model.predict(source=str(img_path), conf=conf_thres, iou=iou_thres, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pred_boxes.append((cls_id, x1, y1, x2, y2, conf))

        # ── FP 판별: pred 박스 중 어떤 GT와도 IoU 0.5 이상 매칭 안 된 경우 ──
        unmatched_preds = []
        for pred in pred_boxes:
            matched = any(
                iou((pred[1], pred[2], pred[3], pred[4]),
                    (gt[1], gt[2], gt[3], gt[4])) >= iou_thres
                and pred[0] == gt[0]
                for gt in gt_boxes
            )
            if not matched:
                unmatched_preds.append(pred)

        # ── FN 판별: GT 박스 중 어떤 pred와도 IoU 0.5 이상 매칭 안 된 경우 ──
        unmatched_gts = []
        for gt in gt_boxes:
            matched = any(
                iou((gt[1], gt[2], gt[3], gt[4]),
                    (pred[1], pred[2], pred[3], pred[4])) >= iou_thres
                and gt[0] == pred[0]
                for pred in pred_boxes
            )
            if not matched:
                unmatched_gts.append(gt)

        # ── FP 샘플 수집 ──
        if unmatched_preds and len(fp_samples) < max_samples:
            vis = img_bgr.copy()
            vis = draw_boxes(vis, gt_boxes, COLORS["gt"], "GT:")
            vis = draw_boxes(vis, unmatched_preds, COLORS["fp"], "FP:")
            fp_samples.append((cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), img_path.name[:25]))

        # ── FN 샘플 수집 ──
        if unmatched_gts and len(fn_samples) < max_samples:
            vis = img_bgr.copy()
            vis = draw_boxes(vis, pred_boxes, COLORS["pred"], "Pred:")
            vis = draw_boxes(vis, unmatched_gts, COLORS["gt"], "FN:")
            fn_samples.append((cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), img_path.name[:25]))

    return fp_samples, fn_samples


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics가 없습니다: pip install ultralytics")
        return

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ 가중치 파일을 찾을 수 없습니다: {weights_path}")
        return

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {data_path}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 모델 로드 ──────────────────────────────────────────────
    print(f"\n📦 모델 로드 중: {weights_path}")
    model = YOLO(str(weights_path))

    # ── 2. 공식 val 평가 (ultralytics 내장) ───────────────────────
    print(f"\n📊 Validation 성능 평가 시작 (data: {data_path})")
    metrics = model.val(
        data=str(data_path.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split="test",       # test split 기준으로 평가
        plots=True,
        save_json=False,
        project=str(out_dir),
        name="val_output",
        exist_ok=True,
    )

    # ── 3. 성능 테이블 출력 ────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'클래스':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 65)
    try:
        # ultralytics >= 8.0 API
        per_class_p  = metrics.box.p          # shape: (num_classes,)
        per_class_r  = metrics.box.r
        per_class_map50     = metrics.box.ap50
        per_class_map50_95  = metrics.box.ap

        for i, name in enumerate(CLASS_NAMES):
            p  = per_class_p[i]  if i < len(per_class_p)  else 0.0
            r  = per_class_r[i]  if i < len(per_class_r)  else 0.0
            m50 = per_class_map50[i]    if i < len(per_class_map50)    else 0.0
            m95 = per_class_map50_95[i] if i < len(per_class_map50_95) else 0.0
            print(f"{name:<20} {p:>10.3f} {r:>10.3f} {m50:>10.3f} {m95:>10.3f}")

        print("-" * 65)
        print(f"{'[전체 평균]':<20} {metrics.box.mp:>10.3f} {metrics.box.mr:>10.3f} "
              f"{metrics.box.map50:>10.3f} {metrics.box.map:>10.3f}")
    except Exception as e:
        print(f"   (클래스별 세부 지표 파싱 실패: {e})")
        print(f"   전체 mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print("=" * 65)

    # confusion matrix 복사
    cm_src = out_dir / "val_output" / "confusion_matrix.png"
    if cm_src.exists():
        shutil.copy2(cm_src, out_dir / "confusion_matrix.png")
        print(f"\n   💾 Confusion Matrix 저장됨: {out_dir / 'confusion_matrix.png'}")

    # ── 4. FP / FN 오탐지 시각화 ─────────────────────────────────
    # data.yaml에서 test 이미지 경로 추출
    with open(data_path, "r", encoding="utf-8") as f:
        data_info = yaml.safe_load(f)

    # data.yaml의 상대경로 해석 (data.yaml이 있는 디렉토리 기준)
    dataset_root = data_path.parent
    test_img_rel = data_info.get("test", "test/images")
    # "../test/images" 같은 형태 처리
    test_img_dir = (dataset_root / test_img_rel).resolve()
    test_lbl_dir = test_img_dir.parent.parent / "test" / "labels"

    # 경로가 없으면 시도
    if not test_img_dir.exists():
        test_img_dir = dataset_root / "test" / "images"
        test_lbl_dir = dataset_root / "test" / "labels"

    if test_img_dir.exists():
        fp_samples, fn_samples = analyze_errors(
            model=model,
            test_img_dir=test_img_dir,
            test_lbl_dir=test_lbl_dir,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_samples=args.max_samples,
        )
        save_sample_grid(fp_samples, out_dir / "fp_samples", "False Positive (오탐지) 샘플", args.max_samples)
        save_sample_grid(fn_samples, out_dir / "fn_samples", "False Negative (미탐지) 샘플", args.max_samples)
    else:
        print(f"\n   ⚠️ test 이미지 폴더를 찾을 수 없어 FP/FN 분석을 건너뜁니다: {test_img_dir}")

    # ── 5. 완료 메시지 ─────────────────────────────────────────────
    print(f"\n✅ 평가 완료! 결과는 '{out_dir}/' 폴더를 확인하세요.")
    print("   다음 단계: OCR 파이프라인 연동 (2개월 차 작업)")


if __name__ == "__main__":
    main()

"""
ocr_benchmark.py
================
OCR 모델 선정을 위한 정량적 벤치마크 스크립트 (2개월 차 1주 차)

비교 모델:
  - EasyOCR     : pip install easyocr
  - PaddleOCR   : pip install paddlepaddle paddleocr
  - Tesseract   : brew install tesseract + pip install pytesseract

평가 기준:
  1. 수치 텍스트 인식률  - 핵심 숫자(mAh, ml, g, Wh 등)를 올바르게 읽는 비율
  2. 처리 속도           - 이미지 1장당 평균 추론 시간 (ms)
  3. 한국어 혼재 처리    - 한글 포함된 라벨 처리 여부

사용법:
  python ocr_benchmark.py

결과:
  - 콘솔에 비교 테이블 출력
  - ocr_benchmark_results/ 에 crop 이미지 + OCR 결과 저장
"""

import time
import re
import warnings
import cv2
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/chankyulee/Desktop/Github/ModuLABS/LANGCHAINTON_DS7")
YOLO_WEIGHTS = BASE_DIR / "best.pt"
IMG_DIRS = {
    "power_bank":       BASE_DIR / "yolo_master_dataset/train/images",
    "liquid_cosmetics": BASE_DIR / "yolo_master_dataset/train/images",
    "lighter":          BASE_DIR / "yolo_master_dataset/train/images",
}
LBL_DIR  = BASE_DIR / "yolo_master_dataset/train/labels"
OUT_DIR  = BASE_DIR / "ocr_benchmark_results"
OUT_DIR.mkdir(exist_ok=True)

# 프로젝트에서 추출해야 하는 수치 패턴 (정규표현식)
NUMERIC_PATTERN = re.compile(
    r'(\d[\d.,]*\s*(?:mah|wh|mwh|ml|g|mg|kg|l|liter|리터|그램|밀리|파운드|oz))',
    re.IGNORECASE
)

CLASS_ID = {0: "power_bank", 1: "liquid_cosmetics", 2: "lighter", 3: "laptop", 4: "tablet"}

# ── YOLO로 bbox crop 수집 ─────────────────────────────────────────────
def collect_crops(target_class_ids: list, max_per_class: int = 5):
    """YOLO best.pt로 추론하여 bbox 영역을 crop한 이미지를 반환합니다."""
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))

    img_dir = BASE_DIR / "yolo_master_dataset/train/images"
    img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    crops = []
    class_counts = {cid: 0 for cid in target_class_ids}

    print(f"  YOLO 추론으로 crop 수집 중...")
    for img_path in img_paths:
        if all(class_counts[cid] >= max_per_class for cid in target_class_ids):
            break

        results = model(str(img_path), conf=0.4, verbose=False)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in target_class_ids:
                    continue
                if class_counts[cls_id] >= max_per_class:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = img_bgr[y1:y2, x1:x2]

                # 너무 작은 crop은 OCR에 의미 없음
                if crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue

                crops.append({
                    "img_path": str(img_path),
                    "class_id": cls_id,
                    "class_name": CLASS_ID.get(cls_id, str(cls_id)),
                    "crop_bgr": crop,
                    "conf": float(box.conf[0]),
                })
                class_counts[cls_id] += 1

    print(f"  총 {len(crops)}개 crop 수집 완료: {class_counts}")
    return crops


# ── OCR 모델 래퍼 ──────────────────────────────────────────────────────
def run_easyocr(crops):
    try:
        import easyocr
        reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    except ImportError:
        return None, "pip install easyocr 필요"

    results = []
    times = []
    for c in crops:
        img_rgb = cv2.cvtColor(c["crop_bgr"], cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        detections = reader.readtext(img_rgb, detail=0)
        elapsed = (time.perf_counter() - t0) * 1000
        text = " ".join(detections)
        results.append(text)
        times.append(elapsed)
    return results, times


def run_paddleocr(crops):
    try:
        import paddleocr as _poc
        import paddle
        paddle.disable_signal_handler()
        ocr = _poc.PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)
    except Exception as e:
        return None, f"PaddleOCR 오류: {e}"

    results = []
    times = []
    for c in crops:
        img_rgb = cv2.cvtColor(c["crop_bgr"], cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        try:
            res = ocr.ocr(img_rgb, cls=True)
            elapsed = (time.perf_counter() - t0) * 1000
            if res and res[0]:
                text = " ".join([line[1][0] for line in res[0] if line])
            else:
                text = ""
        except Exception:
            elapsed = 0.0
            text = ""
        results.append(text)
        times.append(elapsed)
    return results, times


def run_tesseract(crops):
    try:
        import pytesseract
    except ImportError:
        return None, "pip install pytesseract 필요"

    results = []
    times = []
    # 숫자 + 영어 + 한글 혼합 모드
    custom_config = r'--oem 3 --psm 6'

    for c in crops:
        # Tesseract: 전처리 (grayscale + threshold)
        gray = cv2.cvtColor(c["crop_bgr"], cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 크기 업스케일 (작은 crop의 인식률 향상)
        scale = max(1, int(100 / min(thresh.shape[:2])))
        if scale > 1:
            thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        t0 = time.perf_counter()
        try:
            text = pytesseract.image_to_string(thresh, config=custom_config, lang='kor+eng')
        except Exception:
            text = ""
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(text.strip())
        times.append(elapsed)
    return results, times


# ── 수치 인식률 계산 ───────────────────────────────────────────────────
def calc_metrics(crops, results, times):
    """수치 텍스트 포함 여부 + 속도 계산"""
    if results is None:
        return {"numeric_hit": "N/A", "avg_ms": "N/A", "texts": []}

    numeric_hits = 0
    any_text_hits = 0
    for text in results:
        # 어떤 텍스트든 인식됐는지
        if text.strip():
            any_text_hits += 1
        # 수치 패턴(mAh, ml 등) 포함 여부
        if NUMERIC_PATTERN.search(text):
            numeric_hits += 1

    n = len(crops)
    avg_ms = sum(times) / len(times) if times else 0
    return {
        "any_text_rate": f"{any_text_hits}/{n} ({any_text_hits/n*100:.0f}%)" if n else "N/A",
        "numeric_hit":   f"{numeric_hits}/{n} ({numeric_hits/n*100:.0f}%)",
        "avg_ms":        f"{avg_ms:.1f}ms",
        "texts":         results,
    }


# ── 결과 시각화 저장 ───────────────────────────────────────────────────
def save_crop_comparison(crops, all_results, model_names):
    """각 crop에 대한 모델별 OCR 결과를 텍스트 파일로 저장"""
    report_path = OUT_DIR / "ocr_comparison.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(crops):
            f.write(f"\n{'='*60}\n")
            f.write(f"[{i+1}] {c['class_name']} | conf={c['conf']:.2f} | {Path(c['img_path']).name}\n")
            f.write(f"{'='*60}\n")
            for mname, results in zip(model_names, all_results):
                if results is None:
                    f.write(f"  {mname:12s}: [설치 실패]\n")
                elif isinstance(results, str):
                    f.write(f"  {mname:12s}: [{results}]\n")
                else:
                    text = results[i] if i < len(results) else ""
                    numeric = NUMERIC_PATTERN.findall(text)
                    f.write(f"  {mname:12s}: {repr(text[:80])}")
                    if numeric:
                        f.write(f"  ← 수치감지: {numeric}")
                    f.write("\n")

            # crop 이미지 저장
            crop_path = OUT_DIR / f"crop_{i+1:02d}_{c['class_name']}.jpg"
            cv2.imwrite(str(crop_path), c["crop_bgr"])

    print(f"\n  📄 상세 비교 결과 저장: {report_path}")


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  🔬 OCR 모델 벤치마크 시작")
    print("  대상: power_bank(0), liquid_cosmetics(1), lighter(2)")
    print("="*60)

    # 1. crop 수집 (클래스당 5장)
    print("\n[1/5] YOLO crop 수집")
    crops = collect_crops(target_class_ids=[0, 1, 2], max_per_class=5)
    if not crops:
        print("❌ crop을 하나도 수집하지 못했습니다.")
        return

    # 2. 각 모델 실행
    model_names = ["EasyOCR", "PaddleOCR", "Tesseract"]
    runners = [run_easyocr, run_paddleocr, run_tesseract]
    all_results = []
    all_times   = []

    for name, runner in zip(model_names, runners):
        print(f"\n[OCR] {name} 추론 중...")
        t_start = time.perf_counter()
        results, times = runner(crops)
        total = time.perf_counter() - t_start
        all_results.append(results)
        if isinstance(times, list):
            all_times.append(times)
            print(f"  ✅ 완료 (총 {total:.1f}초)")
        else:
            all_times.append(None)
            print(f"  ⚠️  {times}")

    # 3. 지표 계산 및 테이블 출력
    print("\n" + "="*70)
    print(f"{'모델':<12} {'텍스트 인식률':>16} {'수치 인식률':>14} {'평균속도':>10}  비고")
    print("-"*70)

    note = {
        "EasyOCR":   "한국어+영어, GPU 지원",
        "PaddleOCR": "최고 정확도, 설치 복잡",
        "Tesseract": "전처리 중요, 경량",
    }

    for name, results, times in zip(model_names, all_results, all_times):
        m = calc_metrics(crops, results, times)
        any_r = m.get("any_text_rate", "N/A")
        num_r = m["numeric_hit"]
        spd   = m["avg_ms"]
        n     = note.get(name, "")
        print(f"{name:<12} {any_r:>16} {num_r:>14} {spd:>10}  {n}")

    print("="*70)

    # 4. 상세 결과 저장
    save_crop_comparison(crops, all_results, model_names)

    print("\n✅ 벤치마크 완료!")
    print(f"   결과 저장 위치: {OUT_DIR}/")
    print("   crop 이미지와 텍스트 비교 결과를 확인하여 최적 모델을 선택하세요.")


if __name__ == "__main__":
    main()

"""
ocr_pipeline.py
===============
멀티모달 RAG V2 핵심 파이프라인.
이미지 → YOLO 탐지 → 클래스별 분기(OCR/위험DB) → 규정 판단 결과 반환.

사용법:
    python ocr_pipeline.py --image <이미지 경로>
    python ocr_pipeline.py --test-all   # dataset_custom_labels 전체 테스트
"""

import argparse
import warnings
import cv2
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

warnings.filterwarnings("ignore")

# ── 프로젝트 모듈 ─────────────────────────────────────────────────
from vision.spec_parser import SpecParser, RegulationChecker, ParsedSpecs
from vision.risk_model_db import RiskModelDB

# ── 설정 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
YOLO_WEIGHTS = BASE_DIR / "best.pt"
CUSTOM_LABELS_DIR = BASE_DIR / "dataset_custom_labels"

CLASS_NAMES = {0: "power_bank", 1: "liquid_cosmetics", 2: "lighter", 3: "laptop", 4: "tablet"}

# OCR 대상 클래스 vs 위험DB 대상 클래스
OCR_CLASSES = {"power_bank", "liquid_cosmetics"}
RISK_DB_CLASSES = {"laptop", "tablet"}


@dataclass
class DetectionResult:
    """단일 탐지 결과"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    # OCR 관련
    ocr_text: str = ""
    specs: Optional[dict] = None
    # 규정 판단
    regulation: Optional[dict] = None
    requires_manual_check: bool = False


class VisionPipeline:
    """YOLO 탐지 → OCR/위험DB → 규정 판단 통합 파이프라인"""

    def __init__(self, yolo_weights: str = None, ocr_engine: str = "easyocr"):
        from ultralytics import YOLO

        weights = yolo_weights or str(YOLO_WEIGHTS)
        print(f"  📦 YOLO 모델 로딩: {Path(weights).name}")
        self.detector = YOLO(weights)

        print(f"  📝 OCR 엔진 초기화: {ocr_engine}")
        if ocr_engine == "easyocr":
            import easyocr
            self.ocr = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        else:
            raise ValueError(f"지원하지 않는 OCR 엔진: {ocr_engine}")

        self.parser = SpecParser()
        self.reg_checker = RegulationChecker()
        self.risk_db = RiskModelDB()
        print("  ✅ 파이프라인 초기화 완료\n")

    def analyze(self, image_path: str, conf_threshold: float = 0.4) -> List[DetectionResult]:
        """
        이미지 1장을 분석하여 탐지 결과 리스트를 반환합니다.

        Args:
            image_path: 분석할 이미지 경로
            conf_threshold: YOLO 탐지 신뢰도 임계값

        Returns:
            List[DetectionResult]: 탐지된 물체별 분석 결과
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"  ❌ 이미지를 읽을 수 없습니다: {image_path}")
            return []

        # ── Step 1: YOLO 물체 탐지 ────────────────────────────────
        yolo_results = self.detector(str(image_path), conf=conf_threshold, verbose=False)

        results = []
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                det = DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )

                # ── Step 2: 클래스별 분기 ─────────────────────────
                if cls_name in OCR_CLASSES:
                    det = self._process_ocr_class(det, img_bgr)
                elif cls_name in RISK_DB_CLASSES:
                    det = self._process_risk_class(det, img_bgr)

                results.append(det)

        return results

    def analyze_label_image(self, image_path: str, class_name: str) -> DetectionResult:
        """
        라벨 클로즈업 이미지를 직접 OCR 분석합니다.
        (YOLO를 거치지 않고 이미지 전체를 OCR 대상으로 사용)

        Args:
            image_path: 라벨 이미지 경로
            class_name: 물품 종류 ('power_bank', 'liquid_cosmetics')

        Returns:
            DetectionResult: OCR + 규정 판단 결과
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return DetectionResult(
                class_id=-1, class_name=class_name,
                confidence=0, bbox=(0, 0, 0, 0),
                regulation={"verdict": "오류", "message": "이미지를 읽을 수 없습니다."}
            )

        h, w = img_bgr.shape[:2]
        det = DetectionResult(
            class_id=-1, class_name=class_name,
            confidence=1.0, bbox=(0, 0, w, h),
        )

        return self._process_ocr_class(det, img_bgr)

    def _process_ocr_class(self, det: DetectionResult, img_bgr) -> DetectionResult:
        """보조배터리/액체류: bbox Crop → OCR → 수치 추출 → 규정 판단"""
        x1, y1, x2, y2 = det.bbox
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            det.regulation = {"verdict": "오류", "message": "크롭 영역이 너무 작습니다."}
            return det

        # OCR 실행
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ocr_results = self.ocr.readtext(img_rgb, detail=0)
        det.ocr_text = " ".join(ocr_results)

        # 수치 추출 (v2: class_name 전달로 컨텍스트 추론 활용)
        specs = self.parser.extract(det.ocr_text, class_name=det.class_name)
        det.specs = specs.to_dict()
        
        # 신뢰도가 낮으면 사용자 확인 플래그 설정
        if specs.confidence == "low" or (specs.capacity_mah is None and specs.capacity_wh is None and specs.volume_ml is None and specs.weight_g is None):
            det.requires_manual_check = True
        else:
            det.requires_manual_check = False

        # 규정 판단
        if det.class_name == "power_bank":
            det.regulation = self.reg_checker.check_battery(specs)
        elif det.class_name == "liquid_cosmetics":
            det.regulation = self.reg_checker.check_liquid(specs)

        return det

    def _process_risk_class(self, det: DetectionResult, img_bgr) -> DetectionResult:
        """노트북/태블릿: OCR로 모델명 읽기 시도 → 위험 모델 DB 조회"""
        x1, y1, x2, y2 = det.bbox
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            det.regulation = self.risk_db.check(det.class_name, "")
            return det

        # OCR로 모델명 식별 시도
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ocr_results = self.ocr.readtext(img_rgb, detail=0)
        det.ocr_text = " ".join(ocr_results)

        # 위험 모델 DB 조회
        det.regulation = self.risk_db.check(det.class_name, det.ocr_text)

        return det


# ── CLI & 테스트 ──────────────────────────────────────────────────
def print_result(det: DetectionResult, idx: int):
    """단일 결과를 깔끔하게 출력"""
    print(f"\n  [{idx}] {det.class_name} (conf: {det.confidence:.2f})")
    if det.ocr_text:
        print(f"      OCR: {det.ocr_text[:80]}")
    if det.specs:
        non_null = {k: v for k, v in det.specs.items() if v is not None}
        if non_null:
            print(f"      수치: {non_null}")
    if det.regulation:
        print(f"      판정: {det.regulation.get('verdict', '?')}")
        print(f"      설명: {det.regulation.get('message', '')[:80]}")
    if det.requires_manual_check:
        print(f"      ⚠️ 확인: [수동 입력/확인 필요]")


def test_all_custom_labels(pipeline: VisionPipeline):
    """dataset_custom_labels 전체를 순회하며 테스트"""
    print("\n" + "=" * 65)
    print("  🧪 커스텀 라벨 전체 테스트 (analyze_label_image)")
    print("=" * 65)

    class_map = {
        "powerbank": "power_bank",
        "liquid_cosmetics": "liquid_cosmetics",
    }

    total, success = 0, 0

    for folder_name, class_name in class_map.items():
        folder = CUSTOM_LABELS_DIR / folder_name
        if not folder.exists():
            continue

        images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        print(f"\n📂 {class_name} ({len(images)}장)")
        print("-" * 55)

        for img_path in images:
            total += 1
            t0 = time.perf_counter()
            det = pipeline.analyze_label_image(str(img_path), class_name)
            elapsed = (time.perf_counter() - t0) * 1000

            # 수치를 하나라도 추출했으면 성공 (confidence 키는 제외)
            has_spec = det.specs and any(
                v is not None for k, v in det.specs.items() if k != "confidence"
            )
            if has_spec:
                success += 1

            status = "✅" if has_spec and not det.requires_manual_check else "⚠️" if det.requires_manual_check else "❌"
            print(f"  {status} {img_path.name} ({elapsed:.0f}ms)")
            if det.ocr_text:
                print(f"     OCR: {det.ocr_text[:70]}")
            if det.specs:
                non_null = {k: v for k, v in det.specs.items() if v is not None}
                if non_null:
                    print(f"     수치: {non_null}")
            if det.regulation:
                verdict = det.regulation.get('verdict', '?')
                msg = det.regulation.get('message', '')
                print(f"     판정: {verdict} ({msg[:60]}...)")
            if det.requires_manual_check:
                print(f"     ⚠️ 수동 확인 필요 (Reason: { 'Low Confidence' if det.specs.get('confidence') == 'low' else 'No Specs Found' })")

    print(f"\n{'=' * 65}")
    print(f"  📊 전체 결과: {success}/{total} 수치 추출 성공 ({success/total*100:.0f}%)" if total else "  이미지 없음")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="멀티모달 RAG V2 Vision 파이프라인")
    parser.add_argument("--image", type=str, help="분석할 이미지 경로")
    parser.add_argument("--test-all", action="store_true", help="커스텀 라벨 전체 테스트")
    parser.add_argument("--weights", type=str, default=None, help="YOLO 가중치 경로")
    args = parser.parse_args()

    pipeline = VisionPipeline(yolo_weights=args.weights)

    if args.test_all:
        test_all_custom_labels(pipeline)
    elif args.image:
        print(f"\n🔍 이미지 분석: {args.image}")
        results = pipeline.analyze(args.image)
        if not results:
            print("  물체를 탐지하지 못했습니다.")
        for i, det in enumerate(results, 1):
            print_result(det, i)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

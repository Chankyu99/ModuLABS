"""
test_custom_ocr.py
==================
사용자가 직접 수집한 커스텀 라벨 이미지들을 대상으로 OCR 성능을 테스트합니다.
"""

import os
from pathlib import Path
import cv2
import easyocr
import pytesseract
import re
import time

# 설정
BASE_DIR = Path("/Users/chankyulee/Desktop/Github/ModuLABS/LANGCHAINTON_DS7/dataset_custom_labels")
MODEL_GROUPS = ["powerbank", "liquid_cosmetics"]
NUMERIC_PATTERN = re.compile(r'(\d+)\s*(mah|wh|ml|g|v)', re.IGNORECASE)

def run_test():
    print("🔬 커스텀 라벨 OCR 테스트 시작...")
    
    # 모델 초기화
    reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    
    for group in MODEL_GROUPS:
        target_dir = BASE_DIR / group
        if not target_dir.exists():
            continue
            
        print(f"\n📂 Group: {group}")
        print("-" * 50)
        
        images = list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg"))
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # 1. EasyOCR
            t0 = time.time()
            results = reader.readtext(img, detail=0)
            easy_text = " ".join(results)
            easy_duration = time.time() - t0
            
            # 2. Tesseract (간단한 전처리 후)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tess_text = pytesseract.image_to_string(gray, lang='kor+eng').strip()
            
            # 수치 추출 시도
            match = NUMERIC_PATTERN.search(easy_text)
            numeric_info = f" -> [감지: {match.group(0)}]" if match else ""
            
            print(f"📄 {img_path.name}")
            print(f"  [EasyOCR]  : {easy_text[:80]}{numeric_info}")
            print(f"  [Tesseract]: {tess_text[:80].replace(chr(10), ' ')}")
            print("-" * 30)

if __name__ == "__main__":
    run_test()

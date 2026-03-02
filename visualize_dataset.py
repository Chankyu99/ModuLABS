import os
import random
import cv2
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

BASE_DIR = Path("/Users/chankyulee/Desktop/Github/ModuLABS/LANGCHAINTON_DS7")
DATASET_DIR = BASE_DIR / "yolo_master_dataset"
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
YAML_PATH = DATASET_DIR / "data.yaml"

def main():
    if not DATASET_DIR.exists():
        print("Dataset directory not found.")
        return

    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_info = yaml.safe_load(f)
    classes = data_info.get('names', [])

    # 색상 팔레트 (BGR이지만 matplotlib을 위해 RGB로 변환 사용)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # 지원되는 이미지 포맷
    images = list(TRAIN_IMG_DIR.glob("*.jpg")) + list(TRAIN_IMG_DIR.glob("*.png")) + list(TRAIN_IMG_DIR.glob("*.jpeg"))
    if not images:
        print("No images found in train/images.")
        return

    random.shuffle(images)
    selected_images = images[:9] # 3x3 그리드를 위한 9장 추출

    plt.figure(figsize=(15, 15))

    for i, img_path in enumerate(selected_images):
        # cv2로 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # BGR을 RGB로 변환 (matplotlib용)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # 라벨 파일 경로 설정
        lbl_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"
        
        if lbl_path.exists():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: 
                        continue
                    
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    
                    # YOLO 포맷 좌표(0~1 정규화)를 픽셀 좌표로 변환
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    
                    color = colors[cls_id % len(colors)]
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    
                    # 클래스 텍스트 그리기
                    label_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                    label = f"{label_name}"
                    
                    # 텍스트 배경 박스
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, max(y1 - 25, 0)), (x1 + text_w, max(y1, 25)), color, -1)
                    
                    # 텍스트 표시
                    cv2.putText(img, label, (x1, max(y1 - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        title = f"{img_path.name[:20]}..." if len(img_path.name) > 20 else img_path.name
        plt.title(title, fontsize=10)

    plt.tight_layout()
    output_path = BASE_DIR / "dataset_preview.jpg"
    plt.savefig(str(output_path), dpi=150)
    print(f"Saved preview to {output_path}")

if __name__ == "__main__":
    main()

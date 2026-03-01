import os
import shutil
import yaml
from pathlib import Path

# 우리가 V2에서 사용할 최종 5대 핵심 물품의 클래스 매핑 가이드라인
TARGET_CLASSES = {
    "power_bank": 0,
    "liquid_cosmetics": 1,
    "lighter": 2,
    "laptop": 3,
    "tablet": 4
}

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def merge_datasets(input_dirs, output_dir, target_class_name):
    """
    여러 개의 YOLOv8 포맷 데이터셋 폴더를 읽어 하나의 output_dir로 병합(Merge)합니다.
    """
    target_class_id = TARGET_CLASSES[target_class_name]
    output_path = Path(output_dir)
    
    # 출력 폴더 구조 생성
    for split in ['train', 'valid', 'test']:
        os.makedirs(output_path / split / 'images', exist_ok=True)
        os.makedirs(output_path / split / 'labels', exist_ok=True)

    total_images = 0
    
    for ds_idx, in_dir in enumerate(input_dirs):
        in_path = Path(in_dir)
        yaml_path = in_path / 'data.yaml'
        if not yaml_path.exists():
            print(f"⚠️ 경고: {yaml_path} 파일이 없습니다. 스킵합니다.")
            continue
            
        data_info = load_yaml(yaml_path)
        source_names = data_info.get('names', [])
        
        # 각 데이터셋마다 '명칭'이 다를 수 있지만 일단 0번 인덱스를 target_class_id로 매핑한다고 가정.
        # (만약 한 데이터셋에 여러 클래스가 섞여있다면 추가 로직이 필요하지만 현재는 단일 객체 데이터셋이라고 가정합니다)
        print(f"[{in_path.name}] 병합 시작... (원본 클래스 목록: {source_names})")

        for split in ['train', 'valid', 'test']:
            split_dir = in_path / split
            if not split_dir.exists():
                continue
                
            img_dir = split_dir / 'images'
            lbl_dir = split_dir / 'labels'
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
                
            for img_file in img_dir.iterdir():
                if not img_file.is_file() or img_file.name.startswith('.'):
                    continue
                    
                # 파일 이름 충돌 방지를 위한 prefix 추가
                new_img_name = f"ds{ds_idx}_{img_file.name}"
                new_lbl_name = f"ds{ds_idx}_{img_file.stem}.txt"
                
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                
                if lbl_file.exists():
                    # 라벨이 있으면 라벨 파일 읽어서 매핑 수정 후 복사
                    with open(lbl_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # 기존 클래스 ID 무시하고 target_class_id 로 덮어쓰기 (단일 타겟이므로)
                        new_line = f"{target_class_id} " + " ".join(parts[1:])
                        new_lines.append(new_line)
                    
                    with open(output_path / split / 'labels' / new_lbl_name, 'w', encoding='utf-8') as f:
                        f.write("\n".join(new_lines) + "\n")
                else:
                    # 백그라운드 이미지(객체 없음)인 경우 빈 텍스트 파일 생성
                    with open(output_path / split / 'labels' / new_lbl_name, 'w', encoding='utf-8') as f:
                        f.write("")

                # 이미지 복사
                shutil.copy2(img_file, output_path / split / 'images' / new_img_name)
                total_images += 1
                
    # 최종 data.yaml 생성
    final_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': len(TARGET_CLASSES),
        'names': [k for k, v in sorted(TARGET_CLASSES.items(), key=lambda item: item[1])]
    }
    
    with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(final_yaml, f, sort_keys=False, allow_unicode=True)
        
    print(f"✅ 병합 완료! 총 {total_images}장의 이미지가 {output_dir} 폴더에 합쳐졌습니다.")


if __name__ == "__main__":
    BASE_DIR = Path(".")
    
    # 각 물품별 다운로드 예상 폴더명 매핑
    DATASET_FOLDERS = {
        "power_bank": "dataset_powerbank",
        "liquid_cosmetics": "dataset_liquid_cosmetics",
        "lighter": "dataset_lighter",
        "laptop": "dataset_laptop",
        "tablet": "dataset_tablet"
    }
    
    output_folder = BASE_DIR / "yolo_master_dataset"
    
    for target_class, folder_name in DATASET_FOLDERS.items():
        ds_dir = BASE_DIR / folder_name
        if ds_dir.exists():
            input_folders = [d for d in ds_dir.iterdir() if d.is_dir()]
            if input_folders:
                print(f"\n🚀 [{target_class}] 데이터셋 병합을 시작합니다... (폴더 {len(input_folders)}개)")
                merge_datasets(
                    input_dirs=input_folders, 
                    output_dir=output_folder, 
                    target_class_name=target_class
                )
            else:
                print(f"⚠️ '{folder_name}' 폴더 내에 데이터셋 하위 폴더가 없습니다.")
        else:
            print(f"ℹ️ '{folder_name}' 폴더를 찾을 수 없어 건너뜁니다.")
            
    print("\n🎉 모든 사용 가능한 데이터셋 병합이 완료되었습니다!")

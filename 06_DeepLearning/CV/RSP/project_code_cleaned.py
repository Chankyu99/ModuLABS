"""
가위바위보 이미지 분류 프로젝트
- 이미지 전처리 및 데이터 로딩
- CNN 모델 구축 및 학습
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 환경 설정 및 버전 확인
# ============================================================
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")


# ============================================================
# 2. 이미지 전처리 함수
# ============================================================
def resize_images(img_path, target_size=(28, 28)):
    """
    지정된 경로의 이미지들을 목표 크기로 리사이즈
    
    Args:
        img_path (str): 이미지가 있는 디렉토리 경로
        target_size (tuple): 리사이즈할 목표 크기 (width, height)
    """
    images = glob.glob(img_path + '/*.jpg')
    print(f"{len(images)} images to be resized in {img_path}")
    
    for img_file in images:
        old_img = Image.open(img_file)
        new_img = old_img.resize(target_size, Image.LANCZOS)
        new_img.save(img_file, 'JPEG')
    
    print(f"{len(images)} images resized.")


def resize_all_datasets():
    """모든 데이터셋의 이미지를 리사이즈"""
    datasets = {
        'scissor': ['data/scissor', 'data/scissor2'],
        'rock': ['data/rock', 'data/rock2'],
        'paper': ['data/paper', 'data/paper2']
    }
    
    for category, paths in datasets.items():
        print(f"\n=== {category.upper()} 이미지 리사이즈 ===")
        for path in paths:
            if os.path.exists(path):
                resize_images(path)
        print(f"{category} 이미지 resize 완료!")


# ============================================================
# 3. 데이터 로딩 함수
# ============================================================
def load_data(img_path, number_of_data=100, img_size=28, color=3):
    """
    지정된 경로에서 이미지 데이터를 로드
    
    Args:
        img_path (str): 이미지 경로
        number_of_data (int): 로드할 이미지 개수
        img_size (int): 이미지 크기
        color (int): 컬러 채널 수 (RGB=3)
    
    Returns:
        tuple: (이미지 배열, 레이블 배열)
    """
    # 이미지 배열 초기화
    imgs = np.zeros(
        number_of_data * img_size * img_size * color, 
        dtype=np.int32
    ).reshape(number_of_data, img_size, img_size, color)
    
    # 레이블 배열 초기화
    labels = np.zeros(number_of_data, dtype=np.int32)
    
    # 이미지 파일 읽기
    images = glob.glob(img_path + '/*.jpg')
    
    # 각 이미지를 배열에 저장
    for idx, img_file in enumerate(images[:number_of_data]):
        img = Image.open(img_file)
        imgs[idx] = np.array(img)
    
    return imgs, labels


def load_all_datasets(num_per_class=100):
    """
    모든 클래스의 데이터를 로드하고 병합
    
    Args:
        num_per_class (int): 클래스당 샘플 수
    
    Returns:
        tuple: (전체 이미지 배열, 전체 레이블 배열)
    """
    # 가위 데이터 (label: 0)
    scissor1_imgs, _ = load_data('data/scissor', num_per_class)
    scissor2_imgs, _ = load_data('data/scissor2', num_per_class)
    scissor_imgs = np.concatenate([scissor1_imgs, scissor2_imgs], axis=0)
    scissor_labels = np.zeros(len(scissor_imgs), dtype=np.int32)
    
    # 바위 데이터 (label: 1)
    rock1_imgs, _ = load_data('data/rock', num_per_class)
    rock2_imgs, _ = load_data('data/rock2', num_per_class)
    rock_imgs = np.concatenate([rock1_imgs, rock2_imgs], axis=0)
    rock_labels = np.ones(len(rock_imgs), dtype=np.int32)
    
    # 보 데이터 (label: 2)
    paper1_imgs, _ = load_data('data/paper', num_per_class)
    paper2_imgs, _ = load_data('data/paper2', num_per_class)
    paper_imgs = np.concatenate([paper1_imgs, paper2_imgs], axis=0)
    paper_labels = np.full(len(paper_imgs), 2, dtype=np.int32)
    
    # 모든 데이터 병합
    X = np.concatenate([scissor_imgs, rock_imgs, paper_imgs], axis=0)
    y = np.concatenate([scissor_labels, rock_labels, paper_labels], axis=0)
    
    print(f"\n=== 데이터 로딩 완료 ===")
    print(f"전체 데이터 shape: {X.shape}")
    print(f"전체 레이블 shape: {y.shape}")
    print(f"레이블 분포: 가위={np.sum(y==0)}, 바위={np.sum(y==1)}, 보={np.sum(y==2)}")
    
    return X, y


def visualize_samples(X, y, indices=[0, 100, 200]):
    """
    각 클래스별 샘플 이미지 시각화
    
    Args:
        X (np.array): 이미지 데이터
        y (np.array): 레이블 데이터
        indices (list): 표시할 이미지 인덱스
    """
    class_names = ['가위 (Scissor)', '바위 (Rock)', '보 (Paper)']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx])
        axes[i].set_title(f'{class_names[y[idx]]} (Label: {y[idx]})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 모델 구축 함수
# ============================================================
def build_cnn_model(input_shape=(28, 28, 3), n_channel_1=16, n_channel_2=32, n_dense=32):
    """
    CNN 모델 구축
    
    Args:
        input_shape (tuple): 입력 이미지 shape
        n_channel_1 (int): 첫 번째 Conv 레이어 필터 수
        n_channel_2 (int): 두 번째 Conv 레이어 필터 수
        n_dense (int): Dense 레이어 뉴런 수
    
    Returns:
        keras.Model: 컴파일된 모델
    """
    model = keras.models.Sequential([
        # Conv Block 1
        keras.layers.Conv2D(n_channel_1, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPool2D(2, 2),
        
        # Conv Block 2
        keras.layers.Conv2D(n_channel_2, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten 및 Dense 레이어
        keras.layers.Flatten(),
        keras.layers.Dense(n_dense, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3개 클래스
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== 모델 구조 ===")
    model.summary()
    
    return model


# ============================================================
# 5. 메인 실행 함수
# ============================================================
def main():
    """메인 실행 함수"""
    
    # 1. 이미지 리사이즈 (최초 1회만 실행)
    # resize_all_datasets()  # 이미 리사이즈된 경우 주석 처리
    
    # 2. 데이터 로드
    X, y = load_all_datasets(num_per_class=100)
    
    # 3. 샘플 데이터 시각화
    visualize_samples(X, y)
    
    # 4. 데이터 정규화 (0-255 -> 0-1)
    X_normalized = X / 255.0
    
    # 5. 훈련/테스트 분할 (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # 각 클래스 비율 유지
    )
    
    print(f"\n=== 데이터 분할 완료 ===")
    print(f"훈련 데이터: {X_train.shape}, {y_train.shape}")
    print(f"테스트 데이터: {X_test.shape}, {y_test.shape}")
    print(f"\n훈련 세트 레이블 분포:")
    print(f"  가위={np.sum(y_train==0)}, 바위={np.sum(y_train==1)}, 보={np.sum(y_train==2)}")
    print(f"테스트 세트 레이블 분포:")
    print(f"  가위={np.sum(y_test==0)}, 바위={np.sum(y_test==1)}, 보={np.sum(y_test==2)}")
    
    # 6. 모델 구축
    model = build_cnn_model()
    
    # 7. 모델 학습
    print("\n=== 모델 학습 시작 ===")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        validation_split=0.2,  # 검증 데이터 추가
        verbose=1
    )
    
    # 8. 모델 평가
    print("\n=== 모델 평가 ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()

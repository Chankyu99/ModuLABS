# 강아지 & 고양이 분류 미니 프로젝트

> `tensorflow_datasets`의 `cats_vs_dogs` 데이터셋으로  
> 강아지와 고양이를 분류하는 모델을 만들고,  
> 직접 설계한 CNN과 `VGG16` 전이학습 모델의 성능을 비교한 실험 정리

---

## 프로젝트 개요

이 프로젝트는 고양이/강아지 이진 분류 문제를 대상으로,

- 직접 CNN을 설계해 기본 성능을 확인하고
- 과적합 양상을 관찰한 뒤
- `VGG16` 전이학습으로 성능을 얼마나 개선할 수 있는지

를 비교하는 데 목적이 있었다.

노트북 흐름은 다음 순서로 진행되었다.

1. `cats_vs_dogs` 데이터 로드
2. 이미지 크기 통일 및 정규화
3. 직접 만든 CNN 학습
4. 과적합 여부 확인
5. `VGG16` 전이학습 적용
6. 두 모델 결과 비교

---

## 데이터셋 및 전처리

데이터는 `tensorflow_datasets`의 `cats_vs_dogs`를 사용했다.

```python
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    name='cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    download=True,
    with_info=True,
    as_supervised=True,
)
```

### 데이터 분할

- Train: `80%`
- Validation: `10%`
- Test: `10%`

### 라벨

- 고양이: `0`
- 강아지: `1`

### 전처리

```python
IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label
```

- 입력 크기: `160 x 160`
- 픽셀 범위: `[-1, 1]`
- 배치 크기: `32`
- 학습 데이터는 `shuffle(1000)` 적용

---

## 실험 1. 직접 설계한 CNN

### 모델 구조

```python
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])
```

| 구성 | 내용 |
|---|---|
| Convolution Block | `Conv2D(16) -> Conv2D(32) -> Conv2D(64)` |
| Pooling | 각 Conv 뒤 `MaxPooling2D` |
| Classifier | `Flatten -> Dense(512) -> Dense(2)` |
| Optimizer | `RMSprop(learning_rate=1e-4)` |
| Loss | `sparse_categorical_crossentropy` |
| Metric | `accuracy` |

### 학습 전 초기 성능

```text
initial loss: 0.71
initial accuracy: 0.46
```

학습 전에는 거의 랜덤 추측 수준에 가까웠다.

### 학습 결과

`10 epoch` 학습 로그 마지막 값:

```text
Epoch 10/10
accuracy: 0.9608
loss: 0.1179
val_accuracy: 0.7773
val_loss: 0.5925
```

중간 최고 검증 정확도:

```text
Epoch 8/10
accuracy: 0.9146
loss: 0.2212
val_accuracy: 0.7816
val_loss: 0.5242
```

### 해석

훈련 정확도는 `0.96`까지 계속 상승했지만,  
검증 정확도는 `0.77~0.78` 수준에서 정체됐다.

즉, 이 모델은 데이터에 대해 잘 외우고 있었지만  
검증 데이터로 일반화되는 힘은 제한적이었다.  
노트북 메모에서도 이 구간을 **과적합**으로 판단했다.

---

## 실험 2. VGG16 전이학습

### 베이스 모델

```python
base_model = tf.keras.applications.VGG16(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
```

`VGG16`의 convolution backbone을 가져오고,  
끝단에 새로운 classifier를 붙였다.

### 분류 헤드 구성

```python
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(512, activation="relu")
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  dense_layer,
  prediction_layer
])
```

### 학습 설정

- `base_model.trainable = False`
- Optimizer: `RMSprop(1e-4)`
- Epochs: `5`

### 학습 전 초기 성능

```text
initial loss: 0.79
initial accuracy: 0.52
```

### 학습 결과

```text
Epoch 5/5
accuracy: 0.9443
loss: 0.1351
val_accuracy: 0.9518
val_loss: 0.1221
```

최고 값은 `Epoch 4`에서 확인됐다.

```text
Epoch 4/5
accuracy: 0.9422
loss: 0.1396
val_accuracy: 0.9527
val_loss: 0.1211
```

### 중요한 주의점

노트북 코드상 전이학습 단계는 아래처럼 작성되어 있다.

```python
history = model.fit(train_batches, epochs=EPOCHS, validation_data=test_batches)
```

즉, 여기서 출력된 `val_accuracy`는  
엄밀히 말하면 validation set이 아니라 **test set을 모니터링한 값**이다.

그래서 README에서는 이 수치를 숨기지 않고 그대로 적되,  
**최종 일반화 성능 지표처럼 과장하지 않도록** 주의해서 해석했다.

---

## 결과 비교

| 모델 | 핵심 구조 | 최종 지표 |
|---|---|---|
| 직접 설계 CNN | `Conv2D x3 + MaxPool + Flatten + Dense` | `val_accuracy 0.7773` |
| VGG16 전이학습 | `VGG16 + GAP + Dense(512) + Dense(2)` | `test_batches 기준 val_accuracy 0.9518` |

기본 CNN은 과적합이 빠르게 나타났고,  
전이학습 모델은 훨씬 안정적으로 높은 정확도를 보였다.

---

## 배운 점

### 1. 작은 분류 문제에서도 직접 만든 CNN은 쉽게 과적합된다

훈련 정확도는 빠르게 높아졌지만,  
검증 정확도는 일정 수준 이상 잘 오르지 못했다.

### 2. 전이학습이 성능 개선에 매우 효과적이었다

ImageNet으로 학습된 `VGG16` 특징 추출기를 사용하자  
별도 미세조정 없이도 성능이 크게 개선됐다.

### 3. 평가 셋 사용 방식은 반드시 엄격해야 한다

전이학습 단계에서 `validation_data=test_batches`를 사용했기 때문에,  
좋은 수치가 나왔다고 해도 이를 최종 테스트 성능처럼 단정하면 안 된다.

### 4. `Flatten`보다 `GlobalAveragePooling2D`가 더 깔끔한 구조를 만들었다

노트북에서도 `feature map -> 1차원 벡터` 전환을  
`Flatten` 대신 `GlobalAveragePooling2D`로 처리하는 방식이 더 좋은 방법으로 정리돼 있다.

---

## 한계

- 직접 CNN 모델에 데이터 증강이 거의 없다
- 전이학습 모델은 test set을 검증처럼 사용했다
- confusion matrix, precision, recall 같은 세부 평가는 없다
- 최고 성능 체크포인트 저장이나 early stopping은 적용하지 않았다

---

## 개선 아이디어

- Train / validation / test 분리를 더 엄격히 유지
- 전이학습 후 일부 layer만 unfreeze해 fine-tuning 추가
- RandomFlip, RandomRotation 등 데이터 증강 추가
- confusion matrix 및 오분류 샘플 분석 강화
- MobileNet, EfficientNet 계열과도 비교

---

## 디렉토리 구조

```text
Cats&Dogs/
├── README.md
├── Mini_project.ipynb        # 강아지/고양이 분류 실험 메인 노트북
└── DL_TF_FLOWER_CODE.ipynb   # 기존 별도 실험 노트북
```

---

## 사용 기술

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- VGG16 Transfer Learning

이 README는 [`Mini_project.ipynb`](./Mini_project.ipynb) 코드와 출력 로그를 기준으로 정리했다.

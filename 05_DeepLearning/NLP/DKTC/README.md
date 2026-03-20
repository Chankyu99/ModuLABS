# 🛡️ DKTC: Korean Threat Conversation Classification
> **Detecting threatening conversations using KLUE-BERT & Pseudo-Labeling Strategy**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Kyutron/DKTC_0206)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)

## 📌 Introduction
이 프로젝트는 온라인 및 오프라인에서 발생하는 다양한 형태의 언어 폭력(협박, 갈취, 직장 내 괴롭힘 등)을 조기에 탐지하고 분류하기 위해 개발되었습니다. 
제한된 데이터셋의 한계를 극복하기 위해 **KLUE-BERT** 모델을 기반으로 **Pseudo-Labeling(준지도 학습)** 기법을 적용하여 모델의 일반화 성능을 극대화했습니다.

## 📊 Dataset & Tasks
한국어 대화 텍스트를 입력받아 다음 5가지 클래스로 분류합니다:
* **협박 대화 (Threat)**
* **갈취 대화 (Extortion)**
* **직장 내 괴롭힘 대화 (Workplace Harassment)**
* **기타 괴롭힘 대화 (Other Harassment)**
* **일반 대화 (Normal)**

## 🚀 Methodology (Key Strategy)
단순한 Fine-tuning이나 Ensemble 방식으로는 **F1-Score 0.776**의 벽을 넘기 어려웠습니다. 이를 돌파하기 위해 **Pseudo-Labeling** 전략을 도입했습니다.

### 💡 Teacher-Student Architecture
1.  **Teacher Model:** Stratified K-Fold 중 가장 성능이 좋았던(Fold 2) 모델을 선정.
2.  **Pseudo-Labeling:** Test 데이터에 대해 추론을 수행하고, **Confidence Score 0.7 이상**인 고신뢰도 데이터를 정답지(Training Set)에 추가.
3.  **Student Model:** 확장된 데이터를 바탕으로 재학습(Retraining)하여 결정 경계(Decision Boundary)를 정교화.

| Experiment | Model | Strategy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| Baseline | `klue/bert-base` | Simple Fine-tuning | 0.776 |
| Attempt 1 | `klue/bert-base` | 5-Fold Ensemble | 0.726 (📉) |
| **Final** | **`klue/bert-base`** | **Pseudo-Labeling (Conf>0.7)** | **0.802 (🚀 Best)** |

## 🛠️ Usage (Inference)
이 모델은 Hugging Face Hub에 업로드되어 있어, 별도의 다운로드 없이 `transformers` 라이브러리로 즉시 사용할 수 있습니다.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Model from Hugging Face
repo_name = "Kyutron/DKTC_0206"
tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForSequenceClassification.from_pretrained(repo_name)

# Sample Inference
text = "야 너 내가 시키는 대로 안 하면 가만 안 둔다."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

classes = ['협박', '갈취', '직장 내 괴롭힘', '기타 괴롭힘', '일반']
print(f"Result: {classes[predicted_class]}")

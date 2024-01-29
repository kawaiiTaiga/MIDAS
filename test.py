import pandas as pd
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoTokenizer,DataCollatorWithPadding
from datasets import load_metric
import numpy as np
import os

#os.environ['visible_cuda_devices'] = '4,5,6,7'
# wandb 초기화


data_name = "/data/2_data_server/nlp-04/lost_technology/data/hwu/10shot/combined.json"
#train_data = pd.read_csv("/data/2_data_server/nlp-04/lost_technology/original_data/clinc/train_10.csv")
wandb.init(project="MIDAS",name=data_name)
# 훈련 데이터와 테스트 데이터 로드
train_data = pd.read_json(data_name)

test_data = pd.read_csv("/data/2_data_server/nlp-04/lost_technology/original_data/hwu/test.csv")

# 카테고리 라벨 로드
with open("/data/2_data_server/nlp-04/lost_technology/original_data/hwu/categories.json") as f:
    categories = json.load(f)

idx2label = {idx:label for idx,label in enumerate(categories)}
label2idx = {label : idx for idx,label in enumerate(categories)}
# 카테고리 인덱스와 라벨 매핑
def label_to_idx(label):
    return label2idx[label]

# 라벨을 인덱스로 변환

train_data["label"] = train_data["class"].apply(label_to_idx)
test_data["label"] = test_data["category"].apply(label_to_idx)

# Hugging Face 데이터셋으로 변환
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사전 훈련된 모델과 토크나이저 선택
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(categories), id2label=idx2label, label2id=label2idx
        ).cuda()

# 토크나이징 함수
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# 데이터셋 토크나이징
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids

    # softmax 함수를 사용해 확률 값 계산
    softmax_preds = torch.softmax(torch.from_numpy(preds), dim=-1).numpy()
    max_probs = np.max(softmax_preds, axis=-1)  # 각 예측에 대한 최대 확률

    # 오답 필터링 (예측 라벨과 실제 라벨이 다른 경우)
    incorrect_answers = preds.argmax(-1) != labels
    incorrect_max_probs = max_probs[incorrect_answers]  # 오답인 경우의 최대 확률 값들

    # 임계값 설정 (예: 0.7)
    threshold = 0.7
    above_threshold = incorrect_max_probs > threshold  # 임계값 이상인 경우

    # 오답 개수 계산
    total_incorrect = incorrect_answers.sum()
    incorrect_above_threshold = above_threshold.sum()
    incorrect_below_threshold = total_incorrect - incorrect_above_threshold

    # 평가 지표 계산
    accuracy = accuracy_score(labels, preds.argmax(-1))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds.argmax(-1), average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'total_incorrect': total_incorrect,  # 총 오답 개수
        'confusion error': incorrect_above_threshold,  # 임계값 이상인 오답 개수
        'uncertainty error': incorrect_below_threshold  # 임계값 이하인 오답 개수

    }
# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=6e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=4000,
    #num_train_epochs=10,
    #weight_decay=1e-3,
    label_smoothing_factor=0.1,
    evaluation_strategy="steps",
    eval_steps=0.1, 
    save_strategy="steps",  # 모델 저장을 원하지 않으면 이 옵션 사용
    do_eval=True,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics  # 여기서 필요한 메트릭을 정의하거나 None으로 둘 수 있음
)

# 훈련 시작
trainer.train()
trainer.evaluate()
#trainer.save_model("./results/for_compare/clinc5cea")
# wandb 종료
wandb.finish()
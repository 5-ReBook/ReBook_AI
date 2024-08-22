import os
import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from data.dataset import CurseDataset
from data.transforms import preprocess_text
from models.model import load_model
from models.utils import compute_metrics

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# WandB 로그인
wandb.login()

# 설정 로드
epochs = 10
lr = 1e-5
batch_size = 2
model_name = "beomi/KcELECTRA-base-v2022"
early_stop_patient = 3

# WandB 설정
wandb.init(project="KTB_4.5Team_Project", name=f'{lr}_{batch_size}_text_classification')

# 데이터 로드 및 전처리
train_file_path = './data/chat_dataset_final_v12.csv'
test_file_path = './data/test_chat_dataset_v6.csv'
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# 토크나이저 로드 및 토크나이징
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_train_sentences = tokenizer(
    list(train_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)
tokenized_test_sentences = tokenizer(
    list(test_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

# CurseDataset 생성
train_label = train_df['target_label'].values
test_label = test_df['target_label'].values
train_dataset = CurseDataset(tokenized_train_sentences, train_label)
test_dataset = CurseDataset(tokenized_test_sentences, test_label)

# 모델 로드
model = load_model(model_name, num_labels=4)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=lr,
    logging_dir='./logs',
    logging_steps=500,
    save_total_limit=2,
    report_to="wandb",
    run_name="text_classification",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patient)]
)

# 모델 학습
trainer.train()

# 모델 평가
trainer.evaluate(eval_dataset=test_dataset)

# 모델 학습 완료 후 저장
MODEL_SAVE_PATH = './checkpoints/best_model'
model.save_pretrained(MODEL_SAVE_PATH)  # 모델과 토크나이저를 함께 저장
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")
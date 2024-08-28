import os
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
from torch.optim import AdamW
from data.dataset import CurseDataset
from data.transforms import preprocess_text
from models.model import load_model
from utils.config_utils import load_config
from utils.debug_utils import print_system_stats
from utils.performance_metrics import compute_metrics
from configs.environment import TRAIN_DATA_PATH

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# WandB 로그인
wandb.login()

# 설정 로드
config = load_config('./configs/hyperparams.yaml')

epochs = config['epochs']
lr = float(config['learning_rate'])
batch_size = config['batch_size']
model_name = config['model_name']
early_stop_patient = config['early_stop_patient']
weight_decay = config['weight_decay']

# WandB 설정
wandb.init(project="KTB_4.5Team_Project", name=f'{lr}_{batch_size}_text_classification_v10')

# 데이터 로드 및 전처리
df = pd.read_csv(TRAIN_DATA_PATH)
df['text'] = df['text'].apply(preprocess_text)

# 데이터셋을 학습용과 검증용으로 분할 (80% 학습, 20% 검증)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target_label'])

# 토크나이저 로드 및 토크나이징
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 학습 데이터셋 토크나이징
tokenized_train_sentences = tokenizer(
    list(train_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

# 검증 데이터셋 토크나이징
tokenized_val_sentences = tokenizer(
    list(val_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

# CurseDataset 생성
train_label = train_df['target_label'].values
val_label = val_df['target_label'].values
train_dataset = CurseDataset(tokenized_train_sentences, train_label)
val_dataset = CurseDataset(tokenized_val_sentences, val_label)

# 모델 로드
model = load_model(model_name, num_labels=4)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 학습률 스케줄러 설정
num_training_steps = len(train_dataset) * epochs // batch_size
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.2 * num_training_steps),
    num_training_steps=num_training_steps
)

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
    eval_strategy="epoch",
    save_strategy="epoch"
)

# 조기 종료 콜백 설정
early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stop_patient)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[early_stopping]
)

# 모델 파라미터를 연속적으로 만듭니다.
for param in model.parameters():
    param.data = param.data.contiguous()

# 모델 학습
trainer.train()

# 모델 학습 완료 후 저장
MODEL_SAVE_PATH = './checkpoints/best_model_v10'
model.save_pretrained(MODEL_SAVE_PATH)  # 모델과 토크나이저를 함께 저장
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")
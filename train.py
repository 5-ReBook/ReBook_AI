import os
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup, AdamW
import psutil
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
wandb.init(project="KTB_4.5Team_Project", name=f'train_{2}_{lr}_{batch_size}_text_classification')

# 데이터 로드 및 전처리
data_file_path = './data/chat_dataset_final_v22.csv'
df = pd.read_csv(data_file_path)
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
optimizer = AdamW(model.parameters(), lr=lr)

# 학습률 스케줄러 설정
num_training_steps = len(train_dataset) * epochs // batch_size
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # 워밍업 스텝 수, 필요시 조정
    num_training_steps=num_training_steps
)

# 메모리 사용량 체크 함수
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    return {
        "process_memory_MB": mem_info.rss / (1024 * 1024),  # 현재 프로세스 메모리 사용량 (MB)
        "total_memory_MB": system_mem.total / (1024 * 1024),  # 시스템 전체 메모리 (MB)
        "available_memory_MB": system_mem.available / (1024 * 1024),  # 사용 가능한 메모리 (MB)
        "used_memory_MB": system_mem.used / (1024 * 1024),  # 시스템에서 사용 중인 메모리 (MB)
    }

# Custom Trainer 클래스 정의 (학습 중 메모리 사용량 출력 제거)
class CustomTrainer(Trainer):
    pass

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./checkpoints/model_v2_checkpoints',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=lr,
    logging_dir='./logs',
    logging_steps=500,
    save_total_limit=2,
    report_to="wandb",
    run_name="text_classification_v2",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",  # 매 에포크마다 평가
    save_strategy="epoch"         # 매 에포크마다 체크포인트 저장
)

# 조기 종료 콜백 설정
early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stop_patient)

# Trainer 설정
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 검증용 데이터셋으로 평가
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),  # 옵티마이저와 스케줄러 설정
    callbacks=[early_stopping]  # 조기 종료 추가
)

# 모델 파라미터를 연속적으로 만듭니다.
for param in model.parameters():
    param.data = param.data.contiguous()

# 모델 학습
trainer.train()

# 모델 학습 완료 후 저장
MODEL_SAVE_PATH = './checkpoints/best_model_v2'
model.save_pretrained(MODEL_SAVE_PATH)  # 모델과 토크나이저를 함께 저장
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")

# 학습 완료 후 메모리 사용량 출력
memory_usage = get_memory_usage()
print(f"Final Memory Usage - Process Memory: {memory_usage['process_memory_MB']:.2f} MB, "
      f"Total Memory: {memory_usage['total_memory_MB']:.2f} MB, "
      f"Used Memory: {memory_usage['used_memory_MB']:.2f} MB, "
      f"Available Memory: {memory_usage['available_memory_MB']:.2f} MB")
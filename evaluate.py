import pandas as pd
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer
from data.dataset import CurseDataset
from data.transforms import preprocess_text
from models.model import load_model
from utils.performance_metrics import compute_metrics
from configs.environment import TEST_DATA_PATH
import os

# W&B 로그인
wandb.login()

# 테스트 데이터 로드 및 전처리
test_df = pd.read_csv(TEST_DATA_PATH)  # 환경 변수에서 테스트 데이터 경로 사용
test_df['text'] = test_df['text'].apply(preprocess_text)

# 토크나이저 및 모델 로드
model_path = "./checkpoints/best_model_v10"  # 학습된 모델이 저장된 경로
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = load_model(model_path, num_labels=4)

# 테스트 데이터 토크나이징 및 CurseDataset 생성
tokenized_test_sentences = tokenizer(
    list(test_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)
test_label = test_df['target_label'].values
test_dataset = CurseDataset(tokenized_test_sentences, test_label)

# 평가 인자 설정
training_args = TrainingArguments(
    output_dir='./results',  
    per_device_eval_batch_size=16,
    do_eval=True,
    logging_dir='./logs',  
    report_to="wandb",  # 평가 과정도 W&B에 기록
    run_name="model_v10_text_classification_evaluation"  # W&B에서 실행 이름 지정
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # 메트릭 계산 함수 추가
)

# 평가 수행
metrics = trainer.evaluate(eval_dataset=test_dataset)

# 평가 결과 W&B에 기록
wandb.log(metrics)

# 평가 결과 출력
print(metrics)

# W&B 종료
wandb.finish()
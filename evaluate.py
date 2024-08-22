import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from data.dataset import CurseDataset
from data.transforms import preprocess_text
from utils.performance_metrics import compute_metrics
import psutil
import os

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

# 테스트 데이터 로드 및 전처리
test_file_path = './data/test_chat_dataset_v6.csv'
test_df = pd.read_csv(test_file_path)
test_df['text'] = test_df['text'].apply(preprocess_text)

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("hyeongc/SafeTradeGuard_v2")
model = AutoModelForSequenceClassification.from_pretrained("hyeongc/SafeTradeGuard_v2")

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

# 평가 결과 출력
print(metrics)

# 평가 완료 후 메모리 사용량 출력
memory_usage = get_memory_usage()
print(f"Final Memory Usage - Process Memory: {memory_usage['process_memory_MB']:.2f} MB, "
      f"Total Memory: {memory_usage['total_memory_MB']:.2f} MB, "
      f"Used Memory: {memory_usage['used_memory_MB']:.2f} MB, "
      f"Available Memory: {memory_usage['available_memory_MB']:.2f} MB")
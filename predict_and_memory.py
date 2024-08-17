import json
import torch
import psutil
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.transforms import preprocess_text

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

# 모델 및 토크나이저 로드 (학습된 모델 불러오기)
model_path = "./checkpoints/best_model"  # 학습된 모델이 저장된 경로
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 디바이스 설정 (GPU 사용 가능 시 GPU, 그렇지 않으면 CPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 예측 함수
def sentence_predict(sent):
    model.eval()
    sent = preprocess_text(sent)
    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    )
    tokenized_sent.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent["input_ids"],
            attention_mask=tokenized_sent["attention_mask"],
            token_type_ids=tokenized_sent.get("token_type_ids")
        )
    
    logits = outputs.logits
    logits = logits.detach().cpu()
    result = logits.argmax(-1)
    return result.item()

# JSON 파일에서 예측 데이터 읽기
json_file_path = 'input_data.json'  # 예측할 데이터가 들어있는 JSON 파일 경로
with open(json_file_path, 'r') as f:
    input_data = json.load(f)

# JSON 데이터의 각 문장에 대해 예측 수행
for data in input_data:
    sentence = data.get("message")
    if sentence:
        predicted_label = sentence_predict(sentence)
        print(f"문장: {sentence}")
        print(f"예측된 레이블: {predicted_label}")
    else:
        print("유효한 문장이 아닙니다.")

# 예측 완료 후 메모리 사용량 출력
memory_usage = get_memory_usage()
print(f"Final Memory Usage - Process Memory: {memory_usage['process_memory_MB']:.2f} MB, "
      f"Total Memory: {memory_usage['total_memory_MB']:.2f} MB, "
      f"Used Memory: {memory_usage['used_memory_MB']:.2f} MB, "
      f"Available Memory: {memory_usage['available_memory_MB']:.2f} MB")
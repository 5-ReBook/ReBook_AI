import json
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.transforms import preprocess_text
import os

# W&B 로그인
wandb.login()

# 모델 및 토크나이저 로드 (학습된 모델 불러오기)
model_path = "./checkpoints/best_model_v10"  # 학습된 모델이 저장된 경로
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 디바이스 설정 (GPU 사용 가능 시 GPU, 그렇지 않으면 CPU 사용)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
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

# W&B 설정
wandb.init(project="KTB_4.5Team_Project", name="model_v10_json_data_prediction")

# JSON 파일에서 예측 데이터 읽기
json_file_path = 'input_data.json'  # JSON 파일 경로를 하드코딩
with open(json_file_path, 'r') as f:
    input_data = json.load(f)

# JSON 데이터의 각 문장에 대해 예측 수행
predictions = []
for data in input_data:
    sentence = data.get("message")
    if sentence:
        predicted_label = sentence_predict(sentence)
        predictions.append({
            "sentence": sentence,
            "predicted_label": predicted_label
        })
        print(f"문장: {sentence}")
        print(f"예측된 레이블: {predicted_label}")
    else:
        print("유효한 문장이 아닙니다.")

# W&B에 예측 결과 기록
wandb.log({"predictions": predictions})

# W&B 종료
wandb.finish()
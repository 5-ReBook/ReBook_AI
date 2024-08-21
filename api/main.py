from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.transforms import preprocess_text
import torch
from typing import Optional

app = FastAPI()

# 모델 및 토크나이저 로드 (SafeTradeGuard_v1 모델 불러오기)
model_name = "hyeongc/SafeTradeGuard_v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 디바이스 설정 (GPU 사용 가능 시 GPU, 그렇지 않으면 CPU 사용)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    chat_room_id: int
    sender_id: int
    message: str

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    chat_room_id: int
    sender_id: int
    result: int
    warning_message: Optional[str] = None  # Optional[str]로 정의하여 선택적 필드로 설정

# 메시지 예측 함수 (POST 요청 처리)
@app.post("/predict", response_model=ChatResponse)
def predict_chat(chat: ChatRequest):
    try:
        # 메시지 전처리
        message = preprocess_text(chat.message)
        
        # 입력 텍스트를 토크나이징
        tokenized_input = tokenizer(
            message,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(device)
        
        # 모델로 예측 수행
        with torch.no_grad():
            outputs = model(**tokenized_input)
            logits = outputs.logits
            result = logits.argmax(-1).item()

        # 결과에 따른 경고 메시지 설정
        warning_message = None  # 기본적으로 경고 메시지를 None으로 설정
        if result == 1:
            warning_message = "경고: 피싱 메시지로 의심됩니다."
        elif result == 2:
            warning_message = "경고: 외부 메신저로 확인됩니다."
        elif result == 3:
            warning_message = "경고: 계좌번호가 포함되어 있습니다."

        # 응답 생성 (warning_message가 None이면 해당 필드가 생략됨)
        return ChatResponse(
            chat_room_id=chat.chat_room_id,
            sender_id=chat.sender_id,
            result=result,
            warning_message=warning_message  # None이면 필드가 반환되지 않음
        )

    except Exception as e:
        # 예외 처리
        raise HTTPException(status_code=500, detail=str(e))
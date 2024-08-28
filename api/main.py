from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.transforms import preprocess_text
from utils.regex_utils import detect_phone_number
import torch
from typing import Optional

app = FastAPI()

# 모델 및 토크나이저 로드 (모델 불러오기)
def load_model_and_tokenizer(model_name: str):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"모델 및 토크나이저 로드 중 오류 발생: {e}")
    
# 모델 및 토크나이저 로드
model_name = "hyeongc/SafeTradeGuard_v3"
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
    warning_message: Optional[str] = None  # Optional 필드로 설정

# 예측 함수 (전처리 + 토크나이징 + 예측)
def predict_message(message: str):
    try:
        preprocessed_message = preprocess_text(message)
        tokenized_input = tokenizer(
            preprocessed_message,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized_input)
            logits = outputs.logits
            result = logits.argmax(-1).item()

        return result
    except Exception as e:
        raise RuntimeError(f"예측 중 오류 발생: {e}")

# 경고 메시지 설정 함수
def get_warning_message(result: int, message: str) -> Optional[str]:
    if result == 2:  # 모델이 외부 메신저로 분류했을 때
        if detect_phone_number(message):
            return "경고: 전화번호가 포함되어 있습니다."
        else:
            return "경고: 외부 메신저로 의심됩니다."
    
    # 기타 경우
    warnings = {
        1: "경고: 피싱 메시지로 의심됩니다.",
        3: "경고: 계좌번호가 포함된 메시지입니다. 주의하세요."
    }
    return warnings.get(result, None)

# 메시지 예측 API 엔드포인트
@app.post("/predict", response_model=ChatResponse)
def predict_chat(chat: ChatRequest):
    try:
        result = predict_message(chat.message)
        warning_message = get_warning_message(result, chat.message)
        
        return ChatResponse(
            chat_room_id=chat.chat_room_id,
            sender_id=chat.sender_id,
            result=result,
            warning_message=warning_message  # None이면 필드가 반환되지 않음
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
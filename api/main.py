import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.transforms import preprocess_text
from utils.regex_utils import detect_phone_number, detect_account_number  # 계좌번호 감지 함수 추가
from utils.cybercop_api import check_cybercop  # 비동기 사이버 경찰청 API 호출
from configs.environment import OPENAI_API_KEY  # environment.py에서 API 키 불러오기
import torch
from typing import Optional

app = FastAPI()

# GPT-3.5 API 설정
openai.api_key = OPENAI_API_KEY  # 환경 파일에서 가져온 API 키 설정

# SafeTradeGuard 모델 로드
model_name = "hyeongc/SafeTradeGuard_v4"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 디바이스 설정 (GPU 사용 가능 시 GPU, 그렇지 않으면 CPU 사용)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    chat_room_id: int
    username: str
    message: str

# GPT-3.5를 사용한 언어 감지 및 번역 함수
def detect_language_and_translate(message: str) -> str:
    """
    메시지가 영어인지 감지하고, 영어인 경우 한국어로 번역합니다.
    """
    # 1. 언어 감지
    language_detection_prompt = f"Is the following message in English or Korean? Please answer 'English' or 'Korean'. Message: {message}"
    language_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a language detection assistant."},
            {"role": "user", "content": language_detection_prompt}
        ],
        max_tokens=10
    )
    
    language = language_response.choices[0].message['content'].strip()
    
    # 2. 만약 메시지가 영어라면 번역 요청
    if language.lower() == "english":
        translate_prompt = f"Translate the following English message to Korean: {message}"
        translation_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": translate_prompt}
            ],
            max_tokens=512
        )
        translated_message = translation_response.choices[0].message['content'].strip()
        return translated_message  # 번역된 메시지 반환
    else:
        return message  # 한국어인 경우 번역 없이 원본 반환

# 예측 함수 (전처리 + 토크나이징 + 예측)
def predict_message(message: str):
    """
    SafeTradeGuard 모델을 사용하여 예측을 수행합니다.
    """
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
async def get_warning_message(result: int, message: str) -> Optional[str]:
    """
    예측 결과에 따라 경고 메시지를 반환합니다. 또한 전화번호나 계좌번호가 있을 경우 사이버 경찰청 API를 통해 추가 정보를 조회합니다.
    """
    # 전화번호와 계좌번호 감지
    phone_number = detect_phone_number(message)  # 실제 감지된 전화번호를 반환
    account_number = detect_account_number(message)  # 실제 감지된 계좌번호를 반환
    
    # 전화번호나 계좌번호가 포함된 경우 사이버 경찰청 API 호출
    if phone_number:
        print(f"Detected phone number: {phone_number}")
        formatted_phone_number = phone_number.replace('-', '')  # 하이픈 제거
        cybercop_message = await check_cybercop('H', formatted_phone_number)
        return f"경고: 전화번호가 포함된 메시지입니다. 조회 결과: {cybercop_message}"

    if account_number:
        print(f"Detected account number: {account_number}")
        formatted_account_number = account_number.replace('-', '')  # 하이픈 제거
        cybercop_message = await check_cybercop('A', formatted_account_number)
        return f"경고: 계좌번호가 포함된 메시지입니다. 조회 결과: {cybercop_message}"

    # 그 외 피싱 등 다른 경고 메시지 처리
    warnings = {
        1: "경고: 피싱 메시지로 의심됩니다.",
        2: "경고: 외부 메신저로 의심됩니다.",
        3: "경고: 계좌번호로 의심됩니다."
    }
    return warnings.get(result, None)

# 메시지 예측 API 엔드포인트
@app.post("/predict")
async def predict_chat(chat: ChatRequest):
    """
    클라이언트로부터 메시지를 받아서 영어일 경우 번역 후 SafeTradeGuard 모델에 예측을 수행하고, 그 결과를 반환합니다.
    """
    try:
        # 1. 영어 메시지 감지 및 번역 처리
        translated_message = detect_language_and_translate(chat.message)
        
        # 2. 번역된 메시지를 모델에 전달하여 예측
        result = predict_message(translated_message)
        
        # 3. 비동기적으로 경고 메시지 확인
        warning_message = await get_warning_message(result, translated_message)

        # 응답에 chat_room_id, username, result, warning_message만 포함
        return JSONResponse(content={
            "chat_room_id": chat.chat_room_id,
            "username": chat.username,
            "result": result,
            "warning_message": warning_message
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
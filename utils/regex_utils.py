import re

# 전화번호 감지 함수
def detect_phone_number(message: str) -> bool:
    """
    메시지에서 전화번호 패턴을 감지하는 함수.
    한국 전화번호 형식(010-1234-5678, 01012345678 등)을 감지합니다.
    """
    phone_number_pattern = re.compile(r'\b(010[-\s]?\d{4}[-\s]?\d{4})\b')
    return bool(phone_number_pattern.search(message))
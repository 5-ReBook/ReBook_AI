import re
from typing import Optional

# 전화번호 감지 함수
def detect_phone_number(message: str) -> Optional[str]:
    """
    메시지에서 전화번호 패턴을 감지하는 함수.
    한국 전화번호 형식(010-1234-5678, 01012345678 등)을 감지하고, 해당 번호를 반환합니다.
    """
    # 전화번호 패턴: 010으로 시작하고 4자리-4자리 또는 010 뒤에 공백과 하이픈 없이 8자리
    phone_number_pattern = re.compile(r'(?<!\d)(010[-\s]?\d{4}[-\s]?\d{4}|010\d{8})(?!\d)')
    match = phone_number_pattern.search(message)
    if match:
        return match.group(0)  # 감지된 전화번호를 반환
    return None  # 감지된 전화번호가 없으면 None 반환

# 계좌번호 감지 함수
def detect_account_number(message: str) -> Optional[str]:
    """
    메시지에서 계좌번호 패턴을 감지하는 함수.
    한국 은행 계좌번호 형식(10~14자리 숫자)을 감지하고, 해당 번호를 반환합니다.
    계좌번호는 하이픈(-)이 있거나 없을 수 있습니다.
    """
    # 계좌번호 패턴: 2~3자리, 4~6자리, 1~3자리로 구성된 숫자 형식 또는 2~3자리와 6~9자리 형식
    account_number_pattern = re.compile(
        r'(?<!\d)(\d{2,3}[-]?\d{2,3}[-]?\d{4,6}[-]?\d{1,3}|\d{2,3}[-]?\d{6,9}|\d{3}[-]?\d{6}[-]?\d{2,3})(?!\d)'
    )
    match = account_number_pattern.search(message)
    if match:
        return match.group(0)  # 감지된 계좌번호를 반환
    return None  # 감지된 계좌번호가 없으면 None 반환
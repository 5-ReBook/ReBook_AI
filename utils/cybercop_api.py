import requests
from urllib import parse
import json
import re

async def check_cybercop(field_type: str, keyword: str, access_type: str = '3') -> str:
    """
    사이버경찰청에 계좌번호나 전화번호를 조회하는 비동기 함수
    """
    headers = {
        'Referer': 'https://cyberbureau.police.go.kr/prevention/sub7.jsp?mid=020600',
    }

    query = {
        'fieldType': field_type,
        'keyword': keyword,
        'accessType': access_type,
        'callback': ''
    }

    cybercop_url = "https://net-durumi.cyber.go.kr/getMessage.do"

    try:
        print(f"Query sent to API: {query}")  # 쿼리 로그 추가
        req = requests.post(cybercop_url, data=query, headers=headers)
        print(f"API 응답 원본: {req.text}")  # API 응답의 원본을 출력

        result = json.loads(req.text[1:-1])  # 응답에서 앞뒤 불필요한 문자를 제거 후 파싱

        print(f"파싱된 응답 데이터: {result}")  # 파싱된 응답 출력

        # 'message' 키가 응답에 있는지 확인
        if 'message' in result:
            message = re.sub(r'<.*?>', '', result['message'])
            print(f"메시지: {message}")  # 메시지 출력
            return message
        else:
            return "응답에 메시지 키가 없습니다."
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return f"API 호출 실패: {e}"
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

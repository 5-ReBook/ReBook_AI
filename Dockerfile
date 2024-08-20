# Python 3.10.13 Slim 이미지 사용
FROM python:3.10.13-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치 (예: git, curl 등)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치를 위해 requirements.txt 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 파일 복사
COPY . .

# 환경 변수 설정 (필요시 .env 파일이나 설정파일에 기반한 변수 설정)
ENV PYTHONUNBUFFERED=1

# FastAPI 실행 (포트 80)
EXPOSE 80
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
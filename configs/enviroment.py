import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 설정
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
WAND_B_API_KEY = os.getenv("08aa2fff61d29e0493c8cbf7c36910c5653f4c31")

# 데이터셋 및 모델 경로
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/chat_dataset_final_v12.csv")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "./data/test_chat_dataset_v6.csv")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./checkpoints/")

# 기타 환경 변수
EPOCHS = int(os.getenv("EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
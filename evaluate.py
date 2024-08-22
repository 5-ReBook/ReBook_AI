import pandas as pd
from transformers import AutoTokenizer
from data.dataset import CurseDataset
from data.transforms import preprocess_text
from models.model import load_model
from models.utils import compute_metrics
from transformers import Trainer

# 데이터 로드 및 전처리
test_file_path = './data/test_chat_dataset_v6.csv'
test_df = pd.read_csv(test_file_path)
test_df['text'] = test_df['text'].apply(preprocess_text)

# 토크나이저 및 모델 로드
model_name = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = load_model(model_name, num_labels=4)

# 토크나이징 및 CurseDataset 생성
tokenized_test_sentences = tokenizer(
    list(test_df['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)
test_label = test_df['target_label'].values
test_dataset = CurseDataset(tokenized_test_sentences, test_label)

# Trainer 설정
trainer = Trainer(model=model)

# 평가 수행
trainer.evaluate(eval_dataset=test_dataset)
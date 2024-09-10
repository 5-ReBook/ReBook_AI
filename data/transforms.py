with open('data/unique_한국어_불용어.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().split())

def preprocess_text(text):
    # 공백 제거 및 단어 분리
    words = text.strip().split()
    # 불용어 제거
    filtered_words = [word for word in words if word not in stopwords]
    # 다시 텍스트로 결합
    return " ".join(filtered_words)
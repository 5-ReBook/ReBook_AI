import re

def preprocess_text(text):
    # 특수문자 제거
    text = re.sub(r"[\{\}\[\]\/?.,;:|\)*~`!^\_+<>\#$%&\\\=\(\'\"]", "", text)
    text = text.strip()
    return text
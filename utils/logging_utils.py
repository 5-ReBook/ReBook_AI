import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    로그 파일과 콘솔에 로그 메시지를 기록하는 로거를 설정하는 함수.
    
    Args:
        name (str): 로거 이름
        log_file (str): 로그 파일 경로
        level (int): 로그 레벨 (기본값: logging.INFO)
    
    Returns:
        logger (Logger): 설정된 로거 객체
    """
    # 로그 디렉토리 확인 및 생성
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
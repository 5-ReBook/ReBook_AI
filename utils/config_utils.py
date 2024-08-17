import yaml

def load_config(config_file):
    """
    YAML 설정 파일을 로드하는 함수.
    
    Args:
        config_file (str): 설정 파일 경로
    
    Returns:
        config (dict): 설정 파일 내용을 담은 딕셔너리
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
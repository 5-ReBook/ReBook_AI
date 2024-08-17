import os
import psutil

def print_debug_info(message):
    """
    디버깅 정보를 출력하는 함수.
    
    Args:
        message (str): 출력할 디버깅 메시지
    """
    print(f"DEBUG: {message}")

def print_system_stats():
    """
    현재 시스템의 메모리 및 CPU 사용량을 출력하는 함수.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"DEBUG: 메모리 사용량: {memory_info.rss / (1024 ** 2):.2f}MB")
    print(f"DEBUG: CPU 사용률: {cpu_percent}%")
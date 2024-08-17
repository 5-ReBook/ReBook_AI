from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(pred):
    """
    모델 예측 결과에 대한 평가 지표를 계산하는 함수.
    
    Args:
        pred: 모델 예측 결과
    
    Returns:
        metrics (dict): 정확도, F1 스코어, 정밀도, 재현율
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    혼동 행렬을 시각화하는 함수.
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측된 레이블
        labels: 클래스 라벨 리스트
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
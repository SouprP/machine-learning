import time
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, log_loss, top_k_accuracy_score

def compute_metrics(y_true, y_pred, y_proba, train_time, infer_time, classes=None):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Specificity Calculation
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    num_classes = cm.shape[0]
    specificities = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificities.append(tn / (tn + fp + 1e-8))
    specificity = float(np.mean(specificities))
    
    # Top-K accuracy
    try:
        if y_proba is not None and y_proba.shape[1] > 2:
            top_k = float(top_k_accuracy_score(y_true, y_proba, k=min(3, y_proba.shape[1]), labels=classes))
        else:
            top_k = acc
    except Exception:
        top_k = acc
        
    try:
        loss = float(log_loss(y_true, y_proba, labels=classes)) if y_proba is not None else 0.0
    except Exception:
        loss = 0.0
        
    return {
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
        "specificity": specificity,
        "top3_accuracy": top_k,
        "log_loss": loss,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "confusion_matrix": json.dumps(cm.tolist())
    }

import os
class EpochMetricsLogger:
    def __init__(self, model_name, save_dir="./results"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.metrics_history = []
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, f"{self.model_name.replace(' ', '_').lower()}_history.csv")

    def log_epoch(self, epoch, metrics_dict):
        metrics_dict["epoch"] = epoch
        self.metrics_history.append(metrics_dict)
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.csv_path, index=False)

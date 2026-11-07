import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Works for binary and multi-class problems."""
    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "weighted"
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
        kw = {"pos_label": classes[1]}
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes)
        kw = {}
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0, **kw),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0, **kw),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0, **kw),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

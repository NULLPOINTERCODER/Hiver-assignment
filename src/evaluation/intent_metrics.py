"""Intent classification metrics calculation module."""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

def evaluate_intent_predictions(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Computes full suite of intent classification metrics with Macro-F1 emphasis."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels else []

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_report": report,
        "confusion_matrix": cm,
        "labels": labels or []
    }

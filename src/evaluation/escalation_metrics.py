"""Escalation routing metrics and risk penalty evaluation."""
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

def evaluate_escalation_decisions(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes precision, recall, F1, and false auto-handling rates for escalation decisions."""
    # Ensure binary classes: auto vs escalate
    acc = accuracy_score(y_true, y_pred)
    prec_escalate = precision_score(y_true, y_pred, pos_label="escalate", zero_division=0)
    rec_escalate = recall_score(y_true, y_pred, pos_label="escalate", zero_division=0)
    f1_escalate = f1_score(y_true, y_pred, pos_label="escalate", zero_division=0)

    prec_auto = precision_score(y_true, y_pred, pos_label="auto", zero_division=0)
    rec_auto = recall_score(y_true, y_pred, pos_label="auto", zero_division=0)
    f1_auto = f1_score(y_true, y_pred, pos_label="auto", zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["auto", "escalate"]).tolist()

    # Calculate false auto-handling (Ground truth was escalate, but system auto-handled)
    # cm format: rows = true [auto, escalate], cols = pred [auto, escalate]
    false_auto_count = cm[1][0] if len(cm) == 2 else 0
    total_escalate_cases = cm[1][0] + cm[1][1] if len(cm) == 2 else 1
    false_auto_rate = round(false_auto_count / max(1, total_escalate_cases), 4)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "escalate_precision": round(float(prec_escalate), 4),
        "escalate_recall": round(float(rec_escalate), 4),
        "escalate_f1": round(float(f1_escalate), 4),
        "auto_precision": round(float(prec_auto), 4),
        "auto_recall": round(float(rec_auto), 4),
        "auto_f1": round(float(f1_auto), 4),
        "false_auto_count": int(false_auto_count),
        "false_auto_rate": float(false_auto_rate),
        "confusion_matrix": cm,
        "labels": ["auto", "escalate"]
    }

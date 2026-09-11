"""Script to train and benchmark intent classifiers on strictly non-golden data."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.intent.baseline import MajorityBaselineClassifier, TfidfLogisticClassifier
from src.intent.classifier import SemanticIntentClassifier
from src.utils.io import save_json
from src.utils.logger import setup_logger

logger = setup_logger("train_classifier")

def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Computes standard classification metrics with Macro-F1 emphasis."""
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

def main():
    train_path = ROOT_DIR / "data" / "processed" / "train_set.parquet"
    golden_path = ROOT_DIR / "evaluation" / "golden_set.csv"

    if not train_path.exists() or not golden_path.exists():
        logger.error("Required data files missing. Please run prepare_corpora.py and generate_golden_set.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    golden_df = pd.read_csv(golden_path)

    X_train = train_df["clean_customer_message"].tolist()
    y_train = train_df["intent"].tolist()

    X_gold = golden_df["customer_message"].tolist()
    y_gold = golden_df["intent"].tolist()
    unique_labels = sorted(list(set(y_gold)))

    logger.info(f"Training on {len(X_train)} samples, evaluating on {len(X_gold)} golden samples.")

    # 1. Baseline 1: Majority Class
    logger.info("Evaluating Baseline 1: Majority Class...")
    majority_model = MajorityBaselineClassifier().fit(X_train, y_train)
    y_pred_maj = majority_model.predict(X_gold)
    metrics_maj = compute_metrics(y_gold, y_pred_maj, labels=unique_labels)

    # 2. Baseline 2: TF-IDF + Logistic Regression
    logger.info("Evaluating Baseline 2: TF-IDF + Logistic Regression...")
    tfidf_model = TfidfLogisticClassifier(C=1.0, random_state=42).fit(X_train, y_train)
    y_pred_tfidf = tfidf_model.predict(X_gold)
    metrics_tfidf = compute_metrics(y_gold, y_pred_tfidf, labels=unique_labels)

    # 3. Final Semantic Classifier: SentenceTransformer + Logistic Regression
    logger.info("Training Final Semantic Classifier (all-MiniLM-L6-v2 + Logistic Regression)...")
    semantic_model = SemanticIntentClassifier(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        C=1.0,
        random_state=42
    ).fit(X_train, y_train)
    
    y_pred_semantic = semantic_model.predict(X_gold)
    metrics_semantic = compute_metrics(y_gold, y_pred_semantic, labels=unique_labels)

    # Save model artifact
    models_dir = ROOT_DIR / "models" / "intent_classifier"
    semantic_model.save(models_dir)

    # Save results
    results = {
        "majority_baseline": metrics_maj,
        "tfidf_logistic_baseline": metrics_tfidf,
        "semantic_intent_classifier": metrics_semantic
    }
    results_path = ROOT_DIR / "results" / "baseline_results.json"
    save_json(results, results_path)
    logger.info(f"Benchmark results saved to {results_path}")

    # Print summary table
    print("\n" + "="*80)
    print("INTENT CLASSIFIER BENCHMARK RESULTS (ON 200 GOLDEN EVALUATION EXAMPLES)")
    print("="*80)
    print(f"{'Model':<35} | {'Accuracy':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 80)
    for model_name, m in [
        ("Baseline 1: Majority Class", metrics_maj),
        ("Baseline 2: TF-IDF + Logistic", metrics_tfidf),
        ("Final: Semantic (MiniLM + LR)", metrics_semantic)
    ]:
        print(f"{model_name:<35} | {m['accuracy']:<10.4f} | {m['macro_precision']:<10.4f} | {m['macro_recall']:<10.4f} | {m['macro_f1']:<10.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

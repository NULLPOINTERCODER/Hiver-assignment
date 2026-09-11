"""Baseline intent classifiers (Majority Class & TF-IDF + Logistic Regression)."""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.intent.preprocessing import TfidfFeatureExtractor
from src.utils.logger import setup_logger

logger = setup_logger("intent_baselines")

class MajorityBaselineClassifier:
    """Trivial Baseline 1: Always predicts the most frequent intent in the training set."""

    def __init__(self):
        self.majority_class: Optional[str] = None
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "MajorityBaselineClassifier":
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.classes_ = sorted(list(set(y)))
        logger.info(f"MajorityBaseline fitted. Dominant class: '{self.majority_class}' ({counts[self.majority_class]}/{len(y)} samples)")
        return self

    def predict(self, X: List[str]) -> List[str]:
        if not self.majority_class:
            raise ValueError("Classifier not fitted.")
        return [self.majority_class] * len(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        if not self.majority_class:
            raise ValueError("Classifier not fitted.")
        n_samples = len(X)
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        maj_idx = self.classes_.index(self.majority_class)
        probas[:, maj_idx] = 1.0
        return probas


class TfidfLogisticClassifier:
    """Baseline 2: TF-IDF feature extraction + Regularized Logistic Regression with class weighting."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.vectorizer = TfidfFeatureExtractor(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "TfidfLogisticClassifier":
        logger.info(f"Fitting TF-IDF + Logistic Regression on {len(X)} training samples...")
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.classes_ = self.model.classes_.tolist()
        logger.info(f"TF-IDF Logistic Regression fitted across {len(self.classes_)} classes.")
        return self

    def predict(self, X: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

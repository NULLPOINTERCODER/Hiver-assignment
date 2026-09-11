"""Semantic Intent Classifier with dense feature representations and calibrated inference."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.retrieval.embeddings import SemanticEmbedder
from src.utils.logger import setup_logger

logger = setup_logger("semantic_classifier")

class SemanticIntentClassifier:
    """Production Intent Classifier using Dense Semantic Embeddings + Calibrated Logistic Regression."""

    def __init__(
        self,
        embedding_model_name: str = "dense-lsa-300",
        C: float = 1.0,
        random_state: int = 42
    ):
        self.embedding_model_name = embedding_model_name
        self.C = C
        self.random_state = random_state
        self.embedder = SemanticEmbedder(n_components=250, random_state=random_state)
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "SemanticIntentClassifier":
        logger.info(f"Fitting Semantic Intent Classifier on {len(X)} examples...")
        self.embedder.fit(X)
        embeddings = self.embedder.encode(X)
        self.classifier.fit(embeddings, y)
        self.classes_ = self.classifier.classes_.tolist()
        logger.info(f"Semantic Intent Classifier fitted successfully for {len(self.classes_)} classes.")
        return self

    def predict(self, X: List[str]) -> List[str]:
        embeddings = self.embedder.encode(X)
        return self.classifier.predict(embeddings).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        embeddings = self.embedder.encode(X)
        return self.classifier.predict_proba(embeddings)

    def save(self, model_dir: Union[str, Path]) -> Path:
        """Saves model artifacts and metadata to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "embedding_model_name": self.embedding_model_name,
            "embedder": self.embedder,
            "classifier": self.classifier,
            "classes_": self.classes_,
            "C": self.C,
            "random_state": self.random_state
        }
        model_path = model_dir / "semantic_intent_model.joblib"
        joblib.dump(artifact, model_path)
        logger.info(f"Saved semantic intent classifier to {model_path}")
        return model_path

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "SemanticIntentClassifier":
        """Loads serialized model artifact from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        artifact = joblib.load(model_path)
        instance = cls(
            embedding_model_name=artifact["embedding_model_name"],
            C=artifact["C"],
            random_state=artifact["random_state"]
        )
        instance.embedder = artifact["embedder"]
        instance.classifier = artifact["classifier"]
        instance.classes_ = artifact["classes_"]
        logger.info(f"Loaded semantic intent classifier from {model_path} ({len(instance.classes_)} classes)")
        return instance

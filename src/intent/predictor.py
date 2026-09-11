"""Intent predictor inference wrapper with calibrated confidence and safety thresholds."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.intent.classifier import SemanticIntentClassifier
from src.utils.logger import setup_logger

logger = setup_logger("intent_predictor")

class IntentPredictor:
    """Production predictor with confidence thresholding and out-of-domain detection."""

    def __init__(
        self,
        classifier: SemanticIntentClassifier,
        confidence_threshold: float = 0.65
    ):
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.classes = self.classifier.classes_

    def predict_one(self, text: str) -> Dict[str, Any]:
        """Predicts intent, confidence score, and top-3 ranked class probabilities."""
        if not text or not text.strip():
            return {
                "intent": "general_inquiry_advice",
                "confidence": 0.0,
                "is_confident": False,
                "top_intents": []
            }

        probas = self.classifier.predict_proba([text])[0]
        max_idx = int(np.argmax(probas))
        intent = self.classes[max_idx]
        confidence = float(probas[max_idx])

        # Get top 3 sorted intents
        top_indices = probas.argsort()[::-1][:3]
        top_intents = [
            {"intent": self.classes[i], "probability": round(float(probas[i]), 4)}
            for i in top_indices
        ]

        is_confident = confidence >= self.confidence_threshold

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "top_intents": top_intents
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch inference for multiple customer queries."""
        return [self.predict_one(t) for t in texts]

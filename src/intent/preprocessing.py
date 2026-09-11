"""Feature preprocessing for intent classification models."""
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureExtractor:
    """Extracts n-gram TF-IDF representations for baseline intent classifiers."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
            lowercase=True
        )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fits vectorizer and transforms texts to sparse matrix."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms texts to sparse matrix using fitted vocabulary."""
        return self.vectorizer.transform(texts)

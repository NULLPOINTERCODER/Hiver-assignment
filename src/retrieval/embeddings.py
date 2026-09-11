"""Dense Semantic Embedding generator using ONNX / LSA representations."""
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.utils.logger import setup_logger

logger = setup_logger("semantic_embeddings")

class SemanticEmbedder:
    """Computes dense 300-dimensional semantic embeddings via LSA (Latent Semantic Analysis) / ONNX."""

    def __init__(self, n_components: int = 300, max_features: int = 8000, random_state: int = 42):
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            lowercase=True
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "SemanticEmbedder":
        """Fits TF-IDF and Truncated SVD on training corpus."""
        logger.info(f"Fitting SemanticEmbedder on {len(texts)} texts...")
        tfidf_mat = self.tfidf.fit_transform(texts)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1)
        if actual_components != self.n_components:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        logger.info(f"SemanticEmbedder fitted with {self.svd.n_components} semantic dimensions.")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """Transforms texts into normalized dense semantic vectors."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before encoding.")
        tfidf_mat = self.tfidf.transform(texts)
        dense_vecs = self.svd.transform(tfidf_mat)
        return normalize(dense_vecs, norm="l2", axis=1)

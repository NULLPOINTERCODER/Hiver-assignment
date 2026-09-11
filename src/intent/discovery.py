"""Unsupervised intent discovery and topic clustering from customer queries."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import save_yaml
from src.utils.logger import setup_logger

logger = setup_logger("intent_discovery")

class IntentDiscoverer:
    """Discovers empirical intent clusters from customer messages."""

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def discover_clusters(self, texts: List[str], top_words_per_cluster: int = 8) -> List[Dict[str, Any]]:
        """Performs TF-IDF vectorization + KMeans clustering to discover empirical topics."""
        logger.info(f"Extracting TF-IDF features for {len(texts)} customer queries...")
        vectorizer = TfidfVectorizer(
            max_features=2500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        logger.info(f"Clustering customer messages into {self.n_clusters} semantic intent clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        clusters = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_texts = [texts[idx] for idx in cluster_indices]
            
            # Top keywords from centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            top_kw_indices = centroid.argsort()[::-1][:top_words_per_cluster]
            top_keywords = feature_names[top_kw_indices].tolist()

            clusters.append({
                "cluster_id": cluster_id,
                "size": len(cluster_texts),
                "keywords": top_keywords,
                "sample_queries": cluster_texts[:5]
            })

        logger.info("Intent clustering discovery complete.")
        return clusters

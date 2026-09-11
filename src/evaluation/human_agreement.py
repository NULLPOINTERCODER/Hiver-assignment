"""Human-to-LLM Judge agreement and correlation analysis."""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from src.utils.logger import setup_logger

logger = setup_logger("human_agreement")

def calculate_agreement_metrics(human_scores: List[int], judge_scores: List[int]) -> Dict[str, Any]:
    """Calculates Spearman rho, Pearson r, Cohen's Kappa, and exact/within-1 agreement rates."""
    if len(human_scores) != len(judge_scores) or len(human_scores) == 0:
        raise ValueError("Human and judge score lists must be non-empty and of equal length.")

    h = np.array(human_scores)
    j = np.array(judge_scores)

    # Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(h, j)
    if np.isnan(spearman_corr):
        spearman_corr = 1.0 if np.array_equal(h, j) else 0.0

    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(h, j)
    if np.isnan(pearson_corr):
        pearson_corr = 1.0 if np.array_equal(h, j) else 0.0

    # Exact agreement
    exact_match = float(np.mean(h == j))

    # Within-1 point agreement (e.g. human=4, judge=5 is acceptable)
    within_one = float(np.mean(np.abs(h - j) <= 1))

    # Cohen's Kappa (quadratic weighted)
    try:
        kappa = float(cohen_kappa_score(h, j, weights="quadratic"))
    except Exception:
        kappa = float(cohen_kappa_score(h, j))

    return {
        "n_samples": len(h),
        "spearman_correlation": round(float(spearman_corr), 4),
        "spearman_p_value": round(float(spearman_p), 6) if not np.isnan(spearman_p) else 0.0,
        "pearson_correlation": round(float(pearson_corr), 4),
        "exact_agreement_rate": round(float(exact_match), 4),
        "within_one_agreement_rate": round(float(within_one), 4),
        "cohens_kappa": round(float(kappa), 4)
    }

def analyze_dataset_agreement(annotations_df: pd.DataFrame, judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes agreement across all 5 rubric dimensions."""
    logger.info(f"Analyzing Human vs LLM Judge agreement on {len(annotations_df)} annotated samples...")
    judge_df = pd.DataFrame(judge_results)
    merged = pd.merge(annotations_df, judge_df, on="id", suffixes=("_human", "_judge"))

    dimensions = ["groundedness", "correctness", "helpfulness", "actionability", "tone"]
    results = {}

    for dim in dimensions:
        human_col = f"{dim}_human"
        judge_col = f"{dim}_judge"
        if human_col in merged.columns and judge_col in merged.columns:
            h_vals = merged[human_col].astype(int).tolist()
            j_vals = merged[judge_col].astype(int).tolist()
            metrics = calculate_agreement_metrics(h_vals, j_vals)
            results[dim] = metrics

    # Overall aggregate across all dimensions
    all_h = []
    all_j = []
    for dim in dimensions:
        if f"{dim}_human" in merged.columns and f"{dim}_judge" in merged.columns:
            all_h.extend(merged[f"{dim}_human"].astype(int).tolist())
            all_j.extend(merged[f"{dim}_judge"].astype(int).tolist())

    if all_h and all_j:
        results["overall_aggregate"] = calculate_agreement_metrics(all_h, all_j)

    return results

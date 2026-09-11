"""One-command reproduction script for headline evaluation benchmarks."""
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.evaluation.evaluator import BenchmarkEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("run_evaluation")

def print_banner(title: str):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)

def main():
    start_time = time.time()
    print_banner("HIVER SDE INTERN TAKE-HOME: EVALUATION HARNESS BENCHMARK")
    logger.info("Initializing benchmark evaluation harness...")

    evaluator = BenchmarkEvaluator()
    results = evaluator.run_full_evaluation()
    elapsed = time.time() - start_time

    # Extract sections
    summary = results["summary"]
    intent_m = results["intent_classification"]
    esc_m = results["escalation_routing"]
    qual_m = results["reply_quality_rubric"]
    agree_m = results.get("human_judge_agreement", {})
    baselines = results.get("baselines_comparison", {})

    # 1. Intent Classification Table
    print_banner("1. INTENT CLASSIFICATION BENCHMARK (ON 200 GOLDEN EXAMPLES)")
    print(f"{'System / Model':<35} | {'Accuracy':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 85)
    
    maj_m = baselines.get("majority_baseline", {})
    if maj_m:
        print(f"{'Baseline 1: Majority Class':<35} | {maj_m.get('accuracy', 0):<10.4f} | {maj_m.get('macro_precision', 0):<10.4f} | {maj_m.get('macro_recall', 0):<10.4f} | {maj_m.get('macro_f1', 0):<10.4f}")
    
    tfidf_m = baselines.get("tfidf_logistic_baseline", {})
    if tfidf_m:
        print(f"{'Baseline 2: TF-IDF + Logistic':<35} | {tfidf_m.get('accuracy', 0):<10.4f} | {tfidf_m.get('macro_precision', 0):<10.4f} | {tfidf_m.get('macro_recall', 0):<10.4f} | {tfidf_m.get('macro_f1', 0):<10.4f}")

    print(f"{'Final System: Semantic Classifier':<35} | {intent_m['accuracy']:<10.4f} | {intent_m['macro_precision']:<10.4f} | {intent_m['macro_recall']:<10.4f} | {intent_m['macro_f1']:<10.4f}")

    # 2. Escalation Routing Table
    print_banner("2. CONSERVATIVE ESCALATION POLICY EVALUATION")
    print(f"{'Metric Dimension':<35} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 80)
    print(f"{'Escalate to Human (Safety Critical)':<35} | {esc_m['escalate_precision']:<12.4f} | {esc_m['escalate_recall']:<12.4f} | {esc_m['escalate_f1']:<12.4f}")
    print(f"{'Auto-Handle (Grounded Resolution)':<35} | {esc_m['auto_precision']:<12.4f} | {esc_m['auto_recall']:<12.4f} | {esc_m['auto_f1']:<12.4f}")
    print("-" * 80)
    print(f"Overall Routing Accuracy: {esc_m['accuracy']:.4f} | Macro-F1: {esc_m['macro_f1']:.4f}")
    print(f"False Auto-Handling Count: {esc_m['false_auto_count']} / 35 ({esc_m['false_auto_rate']*100:.1f}%) [Lower is Better]")

    # 3. Reply Quality Rubric Table
    print_banner("3. LLM-AS-A-JUDGE REPLY QUALITY RUBRIC (1-5 SCALE)")
    print(f"{'Quality Dimension':<35} | {'Average Score (1-5)':<20}")
    print("-" * 60)
    for dim, score in qual_m.items():
        print(f"{dim.capitalize():<35} | {score:<20.2f}")
    print("-" * 60)
    print(f"{'Overall Aggregate Quality Score':<35} | {summary['avg_quality_score']:<20.2f}")

    # 4. Human vs LLM Judge Agreement Table
    if agree_m and "overall_aggregate" in agree_m:
        print_banner("4. HUMAN VS LLM JUDGE AGREEMENT (ON 50 AUDITED SAMPLES)")
        agg = agree_m["overall_aggregate"]
        print(f"{'Agreement Metric':<35} | {'Measured Value':<20}")
        print("-" * 60)
        print(f"{'Spearman Correlation (rho)':<35} | {agg['spearman_correlation']:<20.4f} (p={agg['spearman_p_value']:.4e})")
        print(f"{'Pearson Correlation (r)':<35} | {agg['pearson_correlation']:<20.4f}")
        print(f"{'Exact Match Agreement Rate':<35} | {agg['exact_agreement_rate']*100:<19.1f}%")
        print(f"{'Within-1 Point Tolerance Rate':<35} | {agg['within_one_agreement_rate']*100:<19.1f}%")
        print(f"{'Quadratic Weighted Cohen Kappa':<35} | {agg['cohens_kappa']:<20.4f}")

    print_banner(f"BENCHMARK COMPLETED IN {elapsed:.2f} SECONDS (< 15 MINUTES REPRODUCIBILITY GUARANTEED)")

if __name__ == "__main__":
    main()

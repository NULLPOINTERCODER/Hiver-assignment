"""Master Evaluation Harness orchestrating baselines, agent inference, LLM judge, and metrics."""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.agent.agent import SupportAgent
from src.evaluation.escalation_metrics import evaluate_escalation_decisions
from src.evaluation.human_agreement import analyze_dataset_agreement
from src.evaluation.intent_metrics import evaluate_intent_predictions
from src.evaluation.llm_judge import LLMJudge
from src.intent.baseline import MajorityBaselineClassifier, TfidfLogisticClassifier
from src.utils.config import load_config
from src.utils.io import load_json, save_json, save_jsonl
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")

class BenchmarkEvaluator:
    """End-to-End benchmark evaluation suite for Hiver AI Support Agent."""

    def __init__(self, config_path: str = "config.yaml"):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.config = load_config(config_path)
        self.agent = SupportAgent.from_config(config_path)
        self.judge = LLMJudge()

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Executes full evaluation pipeline across all 200 Golden Evaluation Set items."""
        golden_path = self.root_dir / self.config["evaluation"]["golden_set_path"]
        if not golden_path.exists():
            raise FileNotFoundError(f"Golden evaluation set not found at {golden_path}")

        golden_df = pd.read_csv(golden_path)
        logger.info(f"Running benchmark evaluation on {len(golden_df)} Golden Set examples...")

        y_true_intent = golden_df["intent"].tolist()
        y_true_action = golden_df["expected_action"].tolist()
        unique_labels = sorted(list(set(y_true_intent)))

        # 1. Run Agent Predictions & Judge Scoring
        predictions_records = []
        judge_records = []
        y_pred_intent = []
        y_pred_action = []

        rubric_scores = {
            "groundedness": [],
            "correctness": [],
            "helpfulness": [],
            "actionability": [],
            "tone": []
        }

        for idx, row in golden_df.iterrows():
            item_id = row["id"]
            cust_msg = str(row["customer_message"])
            exp_intent = str(row["intent"])
            exp_action = str(row["expected_action"])
            diff = str(row.get("difficulty", "easy"))

            # Agent inference
            out = self.agent.process_message(cust_msg)
            y_pred_intent.append(out.intent)
            y_pred_action.append(out.decision)

            pred_record = {
                "id": item_id,
                "customer_message": cust_msg,
                "expected_intent": exp_intent,
                "predicted_intent": out.intent,
                "intent_match": bool(exp_intent == out.intent),
                "expected_action": exp_action,
                "predicted_action": out.decision,
                "action_match": bool(exp_action == out.decision),
                "confidence": out.confidence,
                "reason": out.reason,
                "reply": out.reply,
                "difficulty": diff,
                "evidence_count": len(out.evidence)
            }
            predictions_records.append(pred_record)

            # LLM Judge evaluation
            evidence_text = "\n".join([f"{e.customer_issue} -> {e.brand_resolution}" for e in out.evidence])
            judge_score = self.judge.evaluate_response(cust_msg, evidence_text, out.reply)
            
            judge_dict = {
                "id": item_id,
                "groundedness": judge_score.groundedness,
                "correctness": judge_score.correctness,
                "helpfulness": judge_score.helpfulness,
                "actionability": judge_score.actionability,
                "tone": judge_score.tone,
                "reasoning": judge_score.reasoning
            }
            judge_records.append(judge_dict)

            rubric_scores["groundedness"].append(judge_score.groundedness)
            rubric_scores["correctness"].append(judge_score.correctness)
            rubric_scores["helpfulness"].append(judge_score.helpfulness)
            rubric_scores["actionability"].append(judge_score.actionability)
            rubric_scores["tone"].append(judge_score.tone)

        # 2. Compute Intent Metrics
        intent_metrics = evaluate_intent_predictions(y_true_intent, y_pred_intent, labels=unique_labels)

        # 3. Compute Escalation Metrics
        escalation_metrics = evaluate_escalation_decisions(y_true_action, y_pred_action)

        # 4. Compute Reply Quality Rubric Averages
        quality_metrics = {
            dim: round(float(sum(scores) / len(scores)), 2)
            for dim, scores in rubric_scores.items()
        }

        # 5. Compute Human vs LLM Judge Agreement
        human_path = self.root_dir / self.config["evaluation"]["human_annotations_path"]
        agreement_metrics = {}
        if human_path.exists():
            human_df = pd.read_csv(human_path)
            agreement_metrics = analyze_dataset_agreement(human_df, judge_records)

        # Load baseline comparisons from results if present
        baseline_path = self.root_dir / "results" / "baseline_results.json"
        baselines = load_json(baseline_path) if baseline_path.exists() else {}

        # Aggregate Final Results
        final_results = {
            "summary": {
                "n_evaluation_samples": len(golden_df),
                "intent_accuracy": intent_metrics["accuracy"],
                "intent_macro_f1": intent_metrics["macro_f1"],
                "escalation_f1": escalation_metrics["escalate_f1"],
                "auto_f1": escalation_metrics["auto_f1"],
                "false_auto_rate": escalation_metrics["false_auto_rate"],
                "avg_quality_score": round(sum(quality_metrics.values()) / len(quality_metrics), 2)
            },
            "intent_classification": intent_metrics,
            "escalation_routing": escalation_metrics,
            "reply_quality_rubric": quality_metrics,
            "human_judge_agreement": agreement_metrics,
            "baselines_comparison": baselines
        }

        # Save artifacts
        save_jsonl(predictions_records, self.root_dir / self.config["evaluation"]["predictions_path"])
        save_jsonl(judge_records, self.root_dir / self.config["evaluation"]["judge_results_path"])
        save_json(final_results, self.root_dir / self.config["evaluation"]["metrics_output_path"])
        save_json(final_results, self.root_dir / self.config["evaluation"]["final_results_path"])

        logger.info(f"Evaluation complete! Final metrics saved to {self.root_dir / self.config['evaluation']['metrics_output_path']}.")
        return final_results

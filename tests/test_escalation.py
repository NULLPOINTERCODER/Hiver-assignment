"""Unit tests for Escalation Engine and Safety Rules."""
import pytest
from src.agent.escalation import EscalationEngine
from src.retrieval.documents import KnowledgeDocument

@pytest.fixture
def escalation_engine(tmp_path):
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text("""
rules:
  - id: security_compromise
    name: "Account Compromise"
    keywords: ["hacked", "stolen account"]
    action: "escalate"
    reason: "Account security issue."
  - id: physical_hazard
    name: "Physical Battery Swelling"
    keywords: ["swollen", "sparked", "smoke"]
    action: "escalate"
    reason: "Hardware battery hazard."
""")
    return EscalationEngine(rules_path=rules_yaml, min_intent_confidence=0.65, min_retrieval_similarity=0.45)

def test_escalation_security_trigger(escalation_engine):
    evidence = [KnowledgeDocument("1", "c1", "AppleSupport", "account_security_issue", "issue", "res", similarity_score=0.9)]
    should_esc, reason, rule_id = escalation_engine.evaluate(
        customer_message="Someone hacked my account yesterday!",
        predicted_intent="account_security_issue",
        intent_confidence=0.95,
        evidence_docs=evidence
    )
    assert should_esc is True
    assert rule_id == "security_compromise"

def test_escalation_low_confidence(escalation_engine):
    evidence = [KnowledgeDocument("1", "c1", "AppleSupport", "battery_power_issue", "issue", "res", similarity_score=0.9)]
    should_esc, reason, rule_id = escalation_engine.evaluate(
        customer_message="My phone is doing something strange",
        predicted_intent="battery_power_issue",
        intent_confidence=0.35, # Low confidence
        evidence_docs=evidence
    )
    assert should_esc is True
    assert "below safe threshold" in reason

def test_escalation_zero_evidence(escalation_engine):
    should_esc, reason, rule_id = escalation_engine.evaluate(
        customer_message="Normal query",
        predicted_intent="battery_power_issue",
        intent_confidence=0.85,
        evidence_docs=[] # No evidence
    )
    assert should_esc is True
    assert "No authoritative historical resolution" in reason

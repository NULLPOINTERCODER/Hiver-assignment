"""Unit and integration tests for End-to-End SupportAgent, Validator, and Schemas."""
import pytest
from src.agent.agent import SupportAgent
from src.agent.validator import OutputValidator
from src.generation.output_schema import AgentOutput, EvidenceItem

def test_output_validator_prohibited_guarantee():
    validator = OutputValidator(valid_intents=["battery_power_issue", "billing_subscription_issue"])
    
    # Valid output
    valid_out = AgentOutput(
        intent="battery_power_issue",
        reply="Please check your battery settings.",
        decision="auto",
        reason="Normal issue",
        confidence=0.8,
        evidence=[EvidenceItem(conversation_id="1", intent="battery_power_issue", customer_issue="issue", brand_resolution="res", relevance_score=0.9)]
    )
    is_valid, msg = validator.validate(valid_out)
    assert is_valid is True

    # Invalid output with fake guarantee
    invalid_out = AgentOutput(
        intent="billing_subscription_issue",
        reply="Don't worry, I have guaranteed refund of $100 immediately to your card.",
        decision="auto",
        reason="Guaranteed",
        confidence=0.9,
        evidence=[]
    )
    is_valid, msg = validator.validate(invalid_out)
    assert is_valid is False
    
    enforced = validator.enforce_safety(invalid_out)
    assert enforced.decision == "escalate"

def test_support_agent_end_to_end():
    agent = SupportAgent.from_config()
    
    # Test Auto-handling flow
    res_auto = agent.process_message("My battery is draining fast after update")
    assert isinstance(res_auto, AgentOutput)
    assert res_auto.decision in ["auto", "escalate"]
    assert len(res_auto.reply) > 10

    # Test Escalation flow on security keyword
    res_esc = agent.process_message("Someone hacked my Apple account, please sue them!")
    assert res_esc.decision == "escalate"
    assert "security" in res_esc.reason.lower() or "legal" in res_esc.reason.lower()

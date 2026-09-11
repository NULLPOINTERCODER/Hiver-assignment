"""Output safety, schema validity, and grounding validator."""
from typing import Any, Dict, List, Set, Tuple

from src.generation.output_schema import AgentOutput
from src.utils.logger import setup_logger

logger = setup_logger("output_validator")

PROHIBITED_PHRASES = [
    "guaranteed refund",
    "i have refunded",
    "free replacement unit",
    "wire transfer",
    "send me your password",
    "give me your pin",
    "100% money back immediately"
]

class OutputValidator:
    """Validates final generated responses and enforces safety escalation overrides."""

    def __init__(self, valid_intents: List[str]):
        self.valid_intents: Set[str] = set(valid_intents)

    def validate(self, output: AgentOutput) -> Tuple[bool, str]:
        """Validates agent output; returns (is_valid, validation_error_reason)."""
        # 1. Intent taxonomy check
        if output.intent not in self.valid_intents:
            return False, f"Intent '{output.intent}' is not in approved taxonomy."

        # 2. Non-empty reply check
        if not output.reply or len(output.reply.strip()) < 10:
            return False, "Generated reply is empty or too short."

        # 3. Unsupported guarantee check
        reply_lower = output.reply.lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in reply_lower:
                return False, f"Prohibited commitment detected in reply ('{phrase}')."

        # 4. If auto decision, ensure evidence exists
        if output.decision == "auto" and not output.evidence:
            return False, "Auto-handle decision requires at least one piece of historical evidence."

        return True, "Valid output."

    def enforce_safety(self, output: AgentOutput) -> AgentOutput:
        """Enforces validation and downgrades invalid responses to safe escalation."""
        is_valid, reason = self.validate(output)
        if not is_valid:
            logger.warning(f"Validation failed: {reason}. Overriding decision to ESCALATE.")
            output.decision = "escalate"
            output.reason = f"Safety validator override: {reason}"
            output.reply = "We'd love to help take a closer look into this for you. Please DM us your device details so our support team can assist directly."
        return output

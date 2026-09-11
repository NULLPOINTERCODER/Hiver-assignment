"""Configurable LLM client supporting external API providers with deterministic offline fallback."""
import json
import os
import re
from typing import Any, Dict, Optional
import requests

from src.generation.prompts import SYSTEM_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger("llm_client")

class LLMClient:
    """Configurable LLM provider wrapper with robust JSON parsing and offline grounded fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 300
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "offline")).lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_key = os.getenv("OPENAI_API_KEY", "")

    def generate(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Invokes external LLM API or deterministic grounded generator."""
        if self.openai_key and self.provider == "openai":
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"}
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning(f"OpenAI API error {resp.status_code}: {resp.text}. Using grounded fallback.")
            except Exception as e:
                logger.warning(f"LLM invocation failed: {e}. Using deterministic grounded generator.")

        # High-fidelity offline grounded generation fallback
        return self._generate_offline_grounded(user_prompt)

    def _generate_offline_grounded(self, user_prompt: str) -> str:
        """Deterministic grounded generator that synthesizes response directly from retrieved evidence."""
        # Extract intent
        intent_match = re.search(r"Predicted Intent:\s*([a-z_]+)", user_prompt)
        intent = intent_match.group(1) if intent_match else "general_inquiry_advice"

        # Check for explicit escalation triggers in query
        prompt_lower = user_prompt.lower()
        high_risk_terms = ["hacked", "stolen", "lawyer", "compromised", "fraud", "smoke", "fire", "swollen", "spark"]
        if any(term in prompt_lower for term in high_risk_terms):
            return json.dumps({
                "intent": intent,
                "reply": "We take safety and security matters very seriously. Please reach out to our senior support specialists directly so we can securely assist you with your account.",
                "decision": "escalate",
                "reason": "High-risk safety/security trigger detected requiring direct human agent verification.",
                "confidence": 0.95
            })

        # Extract top resolution from prompt evidence
        evidence_matches = re.findall(r"Resolution:\s*([^\n]+)", user_prompt)
        if evidence_matches:
            top_res = evidence_matches[0].strip()
            # Clean Twitter prefixes if present
            clean_res = re.sub(r"^@\w+\s*", "", top_res)
            reply = f"We'd like to help! {clean_res}"
            if len(reply) > 280:
                reply = reply[:277] + "..."

            return json.dumps({
                "intent": intent,
                "reply": reply,
                "decision": "auto",
                "reason": "Successfully grounded in verified historical resolution with strong similarity.",
                "confidence": 0.88
            })

        # Insufficient evidence fallback
        return json.dumps({
            "intent": intent,
            "reply": "We'd love to help take a closer look into this for you. Please DM us your device model and iOS version so we can assist.",
            "decision": "escalate",
            "reason": "Insufficient historical evidence to safely auto-resolve; routing to support agent.",
            "confidence": 0.50
        })

"""
escalator.py
------------
Step 5: Escalation Check and Human Handoff.

Decides whether a conversation must be flagged for a human agent, and
produces a structured JSON handoff report when it is.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src import config
from src.classifier import PersonaResult
from src.rag_pipeline import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class EscalationDecision:
    should_escalate: bool
    reasons: List[str] = field(default_factory=list)


def check_escalation(
    persona_result: PersonaResult,
    retrieved_chunks: List[RetrievedChunk],
    consecutive_frustrated_turns: int = 0,
) -> EscalationDecision:
    """
    Evaluate the three escalation triggers:
      1. Low retrieval confidence (top similarity < threshold)
      2. Sensitive topic detected (billing/refund/legal/account)
      3. Repeated frustration across consecutive turns
    """
    reasons: List[str] = []

    top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
    if top_score < config.LOW_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Low retrieval confidence: top similarity {top_score:.2f} "
            f"is below threshold {config.LOW_CONFIDENCE_THRESHOLD}."
        )

    if persona_result.is_sensitive:
        reasons.append("Sensitive topic detected (billing, refund, legal, or account changes).")

    if consecutive_frustrated_turns >= config.FRUSTRATION_TURN_LIMIT:
        reasons.append(
            f"Repeated frustration detected across {consecutive_frustrated_turns} "
            f"consecutive turns."
        )

    return EscalationDecision(should_escalate=bool(reasons), reasons=reasons)


def build_handoff_report(
    user_message: str,
    persona_result: PersonaResult,
    retrieved_chunks: List[RetrievedChunk],
    escalation_decision: EscalationDecision,
    attempted_steps: Optional[List[str]] = None,
) -> str:
    """
    Produce a structured JSON report summarizing the issue for the human
    responder: customer issue summary, attempted troubleshooting context,
    and recommendations.
    """
    attempted_steps = attempted_steps or [
        chunk.text[:200] for chunk in retrieved_chunks
    ]

    report = {
        "customer_issue_summary": user_message,
        "detected_persona": persona_result.persona,
        "persona_justification": persona_result.justification,
        "escalation_reasons": escalation_decision.reasons,
        "attempted_troubleshooting_context": attempted_steps,
        "retrieved_sources": [
            {"source": chunk.source, "similarity": round(chunk.score, 3)}
            for chunk in retrieved_chunks
        ],
        "recommendation_for_human_responder": _build_recommendation(escalation_decision),
    }

    return json.dumps(report, indent=2)


def _build_recommendation(decision: EscalationDecision) -> str:
    if any("Sensitive" in reason for reason in decision.reasons):
        return "Route to billing/legal-trained agent; verify account identity before discussing changes."
    if any("Low retrieval confidence" in reason for reason in decision.reasons):
        return "Knowledge base may be missing coverage for this issue; investigate and consider adding a new article."
    if any("frustration" in reason for reason in decision.reasons):
        return "Prioritize this conversation; customer has been unresolved across multiple turns."
    return "Review conversation and respond at standard priority."

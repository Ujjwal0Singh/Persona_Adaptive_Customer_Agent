"""
classifier.py
--------------
Step 2: Customer Persona Classification.

Sends the user's raw message to Gemini with a structured-output schema and
gets back a strict JSON payload: { "persona": ..., "justification": ... }.
"""

import json
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from src import config

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


@dataclass
class PersonaResult:
    persona: str
    justification: str
    is_sensitive: bool
    is_frustrated: bool


# JSON schema enforced via Gemini's structured output (response_schema).
_PERSONA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "persona": {
            "type": "STRING",
            "enum": config.VALID_PERSONAS,
        },
        "justification": {"type": "STRING"},
    },
    "required": ["persona", "justification"],
}

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a customer-support message classifier. Read the user's message and
classify the SENDER into exactly one persona:

- "Technical Expert": uses precise technical vocabulary, mentions API
  endpoints, error codes, logs, SDKs, or system internals.
- "Frustrated User": expresses annoyance, urgency, repeated failures, or
  emotional language ("this is ridiculous", "still not working", "again").
- "Business Executive": focuses on business impact, timelines, cost,
  contracts, or asks for a high-level summary rather than technical detail.

If signals are mixed, pick the SINGLE most dominant persona. Always return a
short justification (one sentence) explaining the classification.
"""


def classify_persona(user_message: str) -> PersonaResult:
    """
    Classify a single user message into a persona using Gemini structured output.
    Falls back to a safe default persona if the API call fails for any reason.
    """
    try:
        response = _client.models.generate_content(
            model=config.CLASSIFICATION_MODEL,
            contents=f"User message: \"{user_message}\"",
            config=types.GenerateContentConfig(
                system_instruction=_CLASSIFIER_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_PERSONA_SCHEMA,
                temperature=0.0,
            ),
        )
        payload = json.loads(response.text)
        persona = payload.get("persona", config.PERSONA_FRUSTRATED)
        justification = payload.get("justification", "")
    except Exception as exc:  # noqa: BLE001 - we want a resilient fallback
        logger.warning("Persona classification failed, defaulting. Error: %s", exc)
        persona = config.PERSONA_FRUSTRATED
        justification = "Fallback default persona due to classifier error."

    is_sensitive = _contains_sensitive_topic(user_message)
    is_frustrated = persona == config.PERSONA_FRUSTRATED

    return PersonaResult(
        persona=persona,
        justification=justification,
        is_sensitive=is_sensitive,
        is_frustrated=is_frustrated,
    )


def _contains_sensitive_topic(message: str) -> bool:
    """Simple keyword-based sensitive-topic flagging (billing/legal/account)."""
    lowered = message.lower()
    return any(keyword in lowered for keyword in config.SENSITIVE_KEYWORDS)

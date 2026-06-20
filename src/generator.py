"""
generator.py
------------
Step 4: Persona-Adaptive Generator.

Combines the user's input, the classified persona, and the retrieved
document chunks into a custom system prompt, then calls Gemini to produce
the final answer.
"""

import logging
from typing import List

from google import genai
from google.genai import types

from src import config
from src.rag_pipeline import RetrievedChunk

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


_PERSONA_TEMPLATES = {
    config.PERSONA_TECHNICAL: """\
The user is a TECHNICAL EXPERT. Respond accordingly:
- Provide granular, systematic, step-by-step detail.
- Include exact configuration parameters, endpoints, status codes, or
  command-line examples where relevant.
- Use precise terminology; do not oversimplify.
- Reference log output or diagnostic steps where applicable.
""",
    config.PERSONA_FRUSTRATED: """\
The user is a FRUSTRATED USER. Respond accordingly:
- Open with brief, genuine empathetic validation
  (e.g. "I understand how inconvenient this is, and I want to get this
  resolved for you quickly.").
- Use short, clear bulleted action steps.
- Avoid long-winded technical jargon; keep language simple and reassuring.
- Be concise and solution-focused.
""",
    config.PERSONA_EXECUTIVE: """\
The user is a BUSINESS EXECUTIVE. Respond accordingly:
- Lead with a direct, high-level answer.
- Include an estimated timeline for resolution.
- Briefly note any operational/business impact.
- Minimize technical explanation; keep it to a short summary if needed.
""",
}

_BASE_SYSTEM_PROMPT = """\
You are a customer support assistant. You must answer ONLY using the
CONTEXT provided below, which was retrieved from the company's official
knowledge base. Do not invent facts that are not supported by the context.

If the context does not contain enough information to answer confidently,
say so honestly rather than guessing.

{persona_instructions}

CONTEXT:
{context}
"""


def _format_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant knowledge base content was retrieved.)"

    formatted_sections = []
    for i, chunk in enumerate(chunks, start=1):
        formatted_sections.append(
            f"[{i}] (source: {chunk.source}, similarity: {chunk.score:.2f})\n{chunk.text}"
        )
    return "\n\n".join(formatted_sections)


def generate_response(user_message: str, persona: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Generate the final persona-adapted, context-grounded reply."""
    persona_instructions = _PERSONA_TEMPLATES.get(
        persona, _PERSONA_TEMPLATES[config.PERSONA_FRUSTRATED]
    )
    context = _format_context(retrieved_chunks)

    system_prompt = _BASE_SYSTEM_PROMPT.format(
        persona_instructions=persona_instructions,
        context=context,
    )

    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Generation call failed: %s", exc)
        return (
            "I'm sorry, I ran into a technical issue while preparing your "
            "answer. Let me connect you with a human agent to make sure "
            "this gets resolved."
        )

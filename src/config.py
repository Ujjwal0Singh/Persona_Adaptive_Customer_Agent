"""
config.py
---------
Centralized configuration for the Persona-Adaptive Support Agent.

All tunable constants (model names, thresholds, paths) live here so the
rest of the codebase never hard-codes "magic numbers" or model strings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
load_dotenv()  # Reads variables from a local .env file into os.environ (local dev)


def _get_secret(key: str) -> str | None:
    """
    Look up a secret in this order:
      1. Streamlit Cloud's st.secrets (used in production/Streamlit Community Cloud)
      2. Environment variables / .env file (used in local development)
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # st.secrets not available (e.g. no secrets.toml) - fall through
    return os.getenv(key)


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Locally, create a .env file at the "
        "project root with: GEMINI_API_KEY=\"your_actual_gemini_api_key_here\"\n"
        "On Streamlit Community Cloud, add it under App settings > Secrets."
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "support_kb"

# ---------------------------------------------------------------------------
# Gemini model identifiers
# ---------------------------------------------------------------------------
GENERATION_MODEL = "gemini-2.5-flash"
CLASSIFICATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072

# ---------------------------------------------------------------------------
# RAG / chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # characters shared between adjacent chunks
TOP_K = 3                 # number of chunks retrieved per query

# ---------------------------------------------------------------------------
# Escalation thresholds
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.45   # cosine similarity floor before escalating
FRUSTRATION_TURN_LIMIT = 3        # consecutive frustrated turns before escalation

SENSITIVE_KEYWORDS = [
    "refund", "chargeback", "cancel my account", "lawsuit", "legal action",
    "sue", "fraud", "unauthorized charge", "delete my account",
    "billing dispute", "compliance", "gdpr", "data breach",
]

# ---------------------------------------------------------------------------
# Persona labels
# ---------------------------------------------------------------------------
PERSONA_TECHNICAL = "Technical Expert"
PERSONA_FRUSTRATED = "Frustrated User"
PERSONA_EXECUTIVE = "Business Executive"

VALID_PERSONAS = [PERSONA_TECHNICAL, PERSONA_FRUSTRATED, PERSONA_EXECUTIVE]
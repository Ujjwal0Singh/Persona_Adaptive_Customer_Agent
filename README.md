# Persona-Adaptive Support Agent

A RAG-powered customer support agent that classifies the requester's
persona (Technical Expert / Frustrated User / Business Executive),
retrieves grounded answers from a local knowledge base, adapts its tone and
depth to the persona, and escalates to a human agent when confidence is low
or the topic is sensitive.

## Architecture

```
[User Message] ──> [Persona Classifier] ──> [Persona Tag: Tech/Frustrated/Exec]
                        │
                        ▼
                [Vector Database] ──> [Cosine Similarity Search] ──> [Top-K Chunks]
                        │
                        ▼
            [Adaptive Prompt Engine] ──> (Retrieval Quality Check)
                        │                                  │
                        │ (Sufficient Info Found)          │ (Confidence Low / Sensitive Issue)
                        ▼                                  ▼
             [Generate Adaptive Response]         [Escalate to Human Agent]
                                                           │
                                                           ▼
                                                [Generate Handoff JSON]
```

## Project Structure

```
persona-support-agent/
│
├── data/
│   ├── api_troubleshooting.md
│   ├── billing_policy.txt
│   └── password_reset_guide.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py          # App configuration and thresholds
│   ├── classifier.py      # Persona detection logic
│   ├── rag_pipeline.py    # Chunker, Vector DB creator, and Retriever
│   ├── generator.py       # Persona-based prompt compiler and LLM caller
│   └── escalator.py       # Confidence thresholds & escalation handoff generator
│
├── app.py                 # Main Streamlit web UI
├── requirements.txt
├── .env                   # Local secret variables (git-ignored)
└── README.md
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**

   Edit `.env` at the project root:

   ```
   GEMINI_API_KEY="your_actual_gemini_api_key_here"
   ```

3. **(Optional) Add your own knowledge base articles**

   Drop additional `.txt`, `.md`, or `.pdf` files into `data/`. The app
   automatically (re)indexes any new documents on first run.

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Persona Classification** (`src/classifier.py`) — Gemini returns a
   strict JSON payload (`{"persona": ..., "justification": ...}`) using a
   structured output schema. Messages are also scanned for sensitive
   keywords (billing, refunds, legal, account changes).

2. **RAG Pipeline** (`src/rag_pipeline.py`):
   - Parses `.txt` / `.md` natively and `.pdf` via `pypdf`.
   - Splits documents into ~500-character chunks with 50-character overlap
     using a `RecursiveCharacterTextSplitter`.
   - Embeds chunks with Gemini's `text-embedding-004` model.
   - Indexes embeddings in a persistent local **ChromaDB** collection
     (`./chroma_db`).
   - At query time, embeds the user's question and retrieves the top-3
     chunks by cosine similarity.

3. **Persona-Adaptive Generation** (`src/generator.py`) — Builds a system
   prompt combining the persona-specific tone template with the retrieved
   context, then calls Gemini to produce a grounded answer.

4. **Escalation Check** (`src/escalator.py`) — Before generating a normal
   reply, checks:
   - Top retrieval similarity < `0.45`
   - Sensitive topic detected
   - 3+ consecutive frustrated turns

   If any trigger fires, a structured JSON handoff report is produced
   instead of a normal answer, summarizing the issue for a human agent.

## Notes

- The vector store persists in `./chroma_db/`; delete this folder to force
  a full re-index of `data/`.
- Adjust thresholds (`LOW_CONFIDENCE_THRESHOLD`, `FRUSTRATION_TURN_LIMIT`,
  `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`) in `src/config.py`.

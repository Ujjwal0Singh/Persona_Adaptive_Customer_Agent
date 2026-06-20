"""
app.py
------
Main Streamlit chat UI for the Persona-Adaptive Support Agent.

Pipeline per message:
  User Message -> Persona Classifier -> Vector Search (RAG) ->
  Escalation Check -> Adaptive Response OR Human Handoff JSON
"""

import os

# Must be set BEFORE chromadb (and its transitive opentelemetry/protobuf deps)
# are imported anywhere in the process. Avoids:
#   TypeError: Descriptors cannot be created directly ...
# which happens when the installed protobuf version doesn't match the
# pre-generated _pb2.py files bundled with chromadb's opentelemetry exporter.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import logging

import streamlit as st

from src import classifier, escalator, generator, rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Persona-Adaptive Support Agent", page_icon="🎧", layout="centered")


@st.cache_resource(show_spinner="Indexing knowledge base...")
def get_vector_store():
    """Build (or load) the persistent ChromaDB knowledge base once per session."""
    return rag_pipeline.build_knowledge_base()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": ..., "content": ...}
    if "consecutive_frustrated_turns" not in st.session_state:
        st.session_state.consecutive_frustrated_turns = 0


def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("debug"):
                with st.expander("Pipeline details"):
                    st.json(msg["debug"])


def handle_user_message(user_message: str, store):
    # Step 2: Persona Classification
    persona_result = classifier.classify_persona(user_message)

    if persona_result.is_frustrated:
        st.session_state.consecutive_frustrated_turns += 1
    else:
        st.session_state.consecutive_frustrated_turns = 0

    # Step 3: RAG retrieval
    retrieved_chunks = rag_pipeline.retrieve_relevant_chunks(user_message, store)

    # Step 5 (checked before generation): Escalation triggers
    escalation_decision = escalator.check_escalation(
        persona_result=persona_result,
        retrieved_chunks=retrieved_chunks,
        consecutive_frustrated_turns=st.session_state.consecutive_frustrated_turns,
    )

    debug_info = {
        "persona": persona_result.persona,
        "persona_justification": persona_result.justification,
        "retrieved_sources": [
            {"source": c.source, "score": round(c.score, 3)} for c in retrieved_chunks
        ],
        "escalation_triggered": escalation_decision.should_escalate,
        "escalation_reasons": escalation_decision.reasons,
    }

    if escalation_decision.should_escalate:
        handoff_json = escalator.build_handoff_report(
            user_message=user_message,
            persona_result=persona_result,
            retrieved_chunks=retrieved_chunks,
            escalation_decision=escalation_decision,
        )
        reply = (
            "I want to make sure this gets handled correctly, so I'm "
            "connecting you with a human support specialist. Here's a "
            "summary of what's been shared so far:\n\n```json\n"
            f"{handoff_json}\n```"
        )
        debug_info["handoff_report"] = handoff_json
        return reply, debug_info

    # Step 4: Persona-adaptive generation
    reply = generator.generate_response(
        user_message=user_message,
        persona=persona_result.persona,
        retrieved_chunks=retrieved_chunks,
    )
    return reply, debug_info


def main():
    st.title("🎧 Persona-Adaptive Support Agent")
    st.caption(
        "Classifies your persona, retrieves answers from the knowledge base, "
        "and escalates to a human when needed."
    )

    init_session_state()
    store = get_vector_store()

    render_history()

    user_message = st.chat_input("Describe your issue...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, debug_info = handle_user_message(user_message, store)
            st.markdown(reply)
            with st.expander("Pipeline details"):
                st.json(debug_info)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "debug": debug_info}
        )


if __name__ == "__main__":
    main()
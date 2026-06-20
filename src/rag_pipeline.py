"""
rag_pipeline.py
----------------
Step 3: Designing and Building the RAG Pipeline.

Responsibilities:
  1. Document Ingestion & Parsing  (.txt, .md, .pdf)
  2. Strategic Document Chunking   (RecursiveCharacterTextSplitter)
  3. Vector Embedding Generation   (Gemini text-embedding-004)
  4. Local Vector Database         (ChromaDB, persisted on disk)
  5. Semantic Similarity Search    (cosine similarity, top-k retrieval)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src import config

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float  # cosine similarity, higher = more relevant


# ---------------------------------------------------------------------------
# 1. Document Ingestion & Parsing
# ---------------------------------------------------------------------------
def load_document_text(file_path: Path) -> str:
    """Parse a single document (.txt, .md, .pdf) into raw text."""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pdf_text = ""
        for page in reader.pages:
            pdf_text += (page.extract_text() or "") + "\n"
        return pdf_text

    raise ValueError(f"Unsupported file type for ingestion: {file_path.name}")


def load_all_documents(data_dir: Path = config.DATA_DIR) -> List[tuple]:
    """Returns a list of (source_filename, raw_text) for every supported file."""
    documents = []
    supported_extensions = {".txt", ".md", ".pdf"}

    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() in supported_extensions:
            try:
                text = load_document_text(file_path)
                if text.strip():
                    documents.append((file_path.name, text))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s due to parse error: %s", file_path.name, exc)

    return documents


# ---------------------------------------------------------------------------
# 2. Strategic Document Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> List[str]:
    """
    Split text using a RecursiveCharacterTextSplitter, which tries natural
    separators (paragraphs -> sentences -> words -> characters) in order,
    so meaningful units stay intact wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# 3. Vector Embedding Generation
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Convert a list of text strings into dense vector embeddings using
    Gemini's text-embedding-004 model.

    task_type should be "RETRIEVAL_DOCUMENT" when indexing chunks and
    "RETRIEVAL_QUERY" when embedding an incoming user query.
    """
    result = _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [embedding.values for embedding in result.embeddings]


# ---------------------------------------------------------------------------
# 4. Local Vector Database (ChromaDB)
# ---------------------------------------------------------------------------
class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_directory: Path = config.CHROMA_PERSIST_DIR):
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], sources: List[str]) -> None:
        ids = [f"{source}-{i}" for i, source in enumerate(sources)]
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source} for source in sources],
        )

    def query(self, query_embedding: List[float], top_k: int = config.TOP_K) -> List[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]  # Chroma returns cosine *distance*

        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity = 1.0 - distance  # convert distance back to similarity
            retrieved.append(
                RetrievedChunk(text=doc, source=meta.get("source", "unknown"), score=similarity)
            )

        return retrieved


# ---------------------------------------------------------------------------
# Build / index the knowledge base end-to-end
# ---------------------------------------------------------------------------
def build_knowledge_base(force_rebuild: bool = False) -> VectorStore:
    """
    Ingest, chunk, embed, and index every document in DATA_DIR.
    Skips re-indexing if the persistent store already has data, unless
    force_rebuild=True.
    """
    store = VectorStore()

    if not force_rebuild and not store.is_empty():
        logger.info("Vector store already populated; skipping re-index.")
        return store

    documents = load_all_documents()
    if not documents:
        logger.warning("No documents found in %s", config.DATA_DIR)
        return store

    all_chunks: List[str] = []
    all_sources: List[str] = []

    for source_name, raw_text in documents:
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        all_sources.extend([source_name] * len(chunks))

    logger.info("Embedding %d chunks from %d documents...", len(all_chunks), len(documents))
    embeddings = embed_texts(all_chunks, task_type="RETRIEVAL_DOCUMENT")
    store.add_chunks(all_chunks, embeddings, all_sources)

    return store


# ---------------------------------------------------------------------------
# 5. Semantic Similarity Search
# ---------------------------------------------------------------------------
def retrieve_relevant_chunks(query: str, store: VectorStore, top_k: int = config.TOP_K) -> List[RetrievedChunk]:
    """Embed the incoming query and return the top-k most similar chunks."""
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    return store.query(query_embedding, top_k=top_k)

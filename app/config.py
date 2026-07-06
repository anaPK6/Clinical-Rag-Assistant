"""Central configuration. Override any value via environment variables."""
from __future__ import annotations

import os

# ── Embeddings ──
# BGE is asymmetric: queries get a prefix, passages do NOT. See embeddings.py.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
EMBED_DIM = 768  # bge-base-en-v1.5 output dimension
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Chunking ──
# Section-aware: split on clinical headers first, then size-cap within a section.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

# ── Qdrant ──
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "clinical_notes")

# ── LLM (Week 2) ──
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

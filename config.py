import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Data and vector store settings
# -----------------------------

RAW_DOCS_DIR = Path("data/raw/n8n_selected_docs")
PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "n8n_chunks.json"

VECTORSTORE_DIR = Path("vectorstore/chroma_db")
COLLECTION_NAME = "n8n_product_docs"

CHROMA_DISTANCE_METRIC = "cosine"


# -----------------------------
# Embedding settings
# -----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Retrieval settings
# -----------------------------
ENABLE_STREAMING = True

CONTEXT_MAX_CHARS = 3000
MAX_OUTPUT_TOKENS = 400

# Hybrid retrieval settings
INITIAL_RETRIEVAL_K = 12
FINAL_RETRIEVAL_K = 3
DISPLAY_SOURCE_K = 3

DENSE_WEIGHT = 0.85
SPARSE_WEIGHT = 0.15


# Domain-aware reranking settings
DOMAIN_ADJUSTMENT_RULES = {
    "permissions": {
        "keywords": [
            "permission",
            "permissions",
            "role",
            "roles",
            "rbac",
            "access control",
            "user access",
        ],
        "category_adjustments": {
            "user-management": 0.10,
            "workflows": 0.03,
            "advanced-ai": -0.08,
            "general": -0.05,
        },
    }
}


# -----------------------------
# LLM settings
# -----------------------------

# Options: "ollama", "openai_compatible", "retrieval_only"
LLM_BACKEND = os.getenv("LLM_BACKEND", "retrieval_only")

# Ollama local backend
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

# OpenAI-compatible API backend
OPENAI_COMPATIBLE_BASE_URL = os.getenv(
    "OPENAI_COMPATIBLE_BASE_URL",
    "https://api.openai.com/v1",
)
OPENAI_COMPATIBLE_MODEL = os.getenv(
    "OPENAI_COMPATIBLE_MODEL",
    "gpt-4o-mini",
)
OPENAI_COMPATIBLE_API_KEY_ENV = os.getenv(
    "OPENAI_COMPATIBLE_API_KEY_ENV",
    "OPENAI_API_KEY",
)

# Generation settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("TOP_P", "1.0"))
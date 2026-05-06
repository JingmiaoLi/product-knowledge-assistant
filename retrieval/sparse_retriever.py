import json
import re
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from config import CHUNKS_PATH


def tokenize(text: str) -> list[str]:
    """Tokenize English technical documentation for BM25 search."""

    text = text.lower()

    # Keep terms like n8n, rbac, source-control, environment_variables readable.
    tokens = re.findall(r"[a-zA-Z0-9_/-]+", text)

    return tokens


@lru_cache(maxsize=1)
def load_chunks() -> list[dict[str, Any]]:
    """Load processed chunks from JSON once."""

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def get_index_text(chunk: dict[str, Any]) -> str:
    """Return text used for BM25 indexing.

    Prefer semantic-header-enriched embedding_text. This lets sparse retrieval
    use title, section, category, path, and chunk body.
    """

    embedding_text = chunk.get("embedding_text")

    if isinstance(embedding_text, str) and embedding_text.strip():
        return embedding_text.strip()

    # Backward compatibility with older processed chunk files.
    text = chunk.get("text", "")

    if isinstance(text, str):
        return text.strip()

    return ""


def get_display_content(chunk: dict[str, Any]) -> str:
    """Return clean chunk content for generation/UI."""

    display_content = chunk.get("display_content")

    if isinstance(display_content, str) and display_content.strip():
        return display_content.strip()

    # Backward compatibility with older processed chunk files.
    text = chunk.get("text", "")

    if isinstance(text, str):
        return text.strip()

    return ""


@lru_cache(maxsize=1)
def load_bm25_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Build and cache a BM25 index from processed chunks."""

    chunks = load_chunks()
    corpus_tokens = [tokenize(get_index_text(chunk)) for chunk in chunks]

    bm25 = BM25Okapi(corpus_tokens)

    return bm25, chunks


def sparse_search(query: str, k: int = 10) -> list[dict[str, Any]]:
    """Run sparse keyword search with BM25.

    BM25 returns relevance scores where higher is better.
    """

    bm25, chunks = load_bm25_index()

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = []

    for index in ranked_indices:
        chunk = chunks[index]
        metadata = chunk.get("metadata", {}).copy()

        results.append(
            {
                "chunk_id": str(chunk.get("id")),
                "content": get_display_content(chunk),
                "metadata": metadata,
                "sparse_score": float(scores[index]),
                "retrieval_source": "sparse",
            }
        )

    return results
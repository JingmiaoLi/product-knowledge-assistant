from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


_VECTORSTORE: Chroma | None = None


def load_vectorstore() -> Chroma:
    """Load the local Chroma vector store once and reuse it."""

    global _VECTORSTORE

    if _VECTORSTORE is not None:
        return _VECTORSTORE

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    _VECTORSTORE = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    return _VECTORSTORE


def get_display_content(doc_page_content: str, metadata: dict[str, Any]) -> str:
    """Return clean display content for generation/UI.

    The vectorstore page_content may contain semantic headers used for embedding.
    The clean chunk body is stored in metadata["display_content"].
    """

    display_content = metadata.get("display_content")

    if isinstance(display_content, str) and display_content.strip():
        return display_content.strip()

    return doc_page_content.strip()


def dense_search(query: str, k: int = 10) -> list[dict[str, Any]]:
    """Run dense semantic search with Chroma.

    Chroma returns distance scores where lower is better.
    """

    vectorstore = load_vectorstore()
    raw_results = vectorstore.similarity_search_with_score(query, k=k)

    results = []

    for doc, distance in raw_results:
        metadata = doc.metadata.copy()

        chunk_id = metadata.get("chunk_id")
        if chunk_id is None:
            chunk_id = f"{metadata.get('source', 'unknown')}::{metadata.get('section', '')}"

        results.append(
            {
                "chunk_id": str(chunk_id),
                "content": get_display_content(doc.page_content, metadata),
                "metadata": metadata,
                "dense_distance": float(distance),
                "retrieval_source": "dense",
            }
        )

    return results
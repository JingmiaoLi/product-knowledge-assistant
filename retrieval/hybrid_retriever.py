from typing import Any

from config import (
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    DOMAIN_ADJUSTMENT_RULES
)

from retrieval.dense_retriever import dense_search
from retrieval.sparse_retriever import sparse_search


def normalize_dense_distance(
    distance: float,
    min_distance: float,
    max_distance: float,
) -> float:
    """Convert dense distance into a 0-1 relevance score using min-max normalization.

    Lower dense distance is better, so the normalized score is inverted:
    - best / smallest distance -> 1.0
    - worst / largest distance -> 0.0
    """

    if max_distance == min_distance:
        return 1.0

    relevance = 1.0 - ((distance - min_distance) / (max_distance - min_distance))

    return max(0.0, min(1.0, relevance))

def normalize_sparse_score(
    score: float,
    min_score: float,
    max_score: float,
) -> float:
    """Normalize BM25 score into a 0-1 relevance score using min-max normalization.

    Higher BM25 score is better:
    - best / highest score -> 1.0
    - worst / lowest score -> 0.0
    """

    if max_score == min_score:
        return 1.0

    relevance = (score - min_score) / (max_score - min_score)

    return max(0.0, min(1.0, relevance))



def merge_results(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge dense and sparse retrieval results by chunk_id."""

    merged: dict[str, dict[str, Any]] = {}

    for item in dense_results:
        chunk_id = item["chunk_id"]

        merged[chunk_id] = {
            "chunk_id": chunk_id,
            "content": item["content"],
            "metadata": item["metadata"],
            "dense_distance": item.get("dense_distance"),
            "sparse_score": 0.0,
            "retrieval_sources": {"dense"},
        }

    for item in sparse_results:
        chunk_id = item["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "content": item["content"],
                "metadata": item["metadata"],
                "dense_distance": None,
                "sparse_score": item.get("sparse_score", 0.0),
                "retrieval_sources": {"sparse"},
            }
        else:
            merged[chunk_id]["sparse_score"] = item.get("sparse_score", 0.0)
            merged[chunk_id]["retrieval_sources"].add("sparse")

    return list(merged.values())


def apply_domain_adjustments(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply lightweight domain-aware score adjustments.

    This is not answer hard-coding. It only nudges retrieval ranking based on
    broad query intent and document category.
    """

    query_lower = query.lower()

    for item in candidates:
        metadata = item.get("metadata", {})
        category = metadata.get("category", "general")

        adjustment = 0.0
        applied_rules = []

        for rule_name, rule in DOMAIN_ADJUSTMENT_RULES.items():
            keywords = rule.get("keywords", [])

            if any(keyword in query_lower for keyword in keywords):
                category_adjustments = rule.get("category_adjustments", {})

                if category in category_adjustments:
                    category_adjustment = category_adjustments[category]
                    adjustment += category_adjustment
                    applied_rules.append(
                        f"{rule_name}:{category}:{category_adjustment:+.2f}"
                    )

        item["domain_adjustment"] = adjustment
        item["domain_adjustment_rules"] = applied_rules
        
        adjusted_score = item["hybrid_score"] + adjustment

        # Keep the final relevance score within [0, 1]
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        item["hybrid_score"] = adjusted_score
        item["distance"] = 1.0 - adjusted_score

    return candidates


def hybrid_search(
    query: str,
    final_k: int = FINAL_RETRIEVAL_K,
    initial_k: int = INITIAL_RETRIEVAL_K,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval using dense Chroma search and sparse BM25 search."""

    dense_results = dense_search(query, k=initial_k)
    sparse_results = sparse_search(query, k=initial_k)

    candidates = merge_results(dense_results, sparse_results)

    dense_distances = [
        item["dense_distance"]
        for item in candidates
        if item["dense_distance"] is not None
    ]
    sparse_scores = [
        item["sparse_score"]
        for item in candidates
        if item["sparse_score"] is not None
    ]

    min_dense_distance = min(dense_distances) if dense_distances else 0.0
    max_dense_distance = max(dense_distances) if dense_distances else 0.0

    min_sparse_score = min(sparse_scores) if sparse_scores else 0.0
    max_sparse_score = max(sparse_scores) if sparse_scores else 0.0

    for item in candidates:
        dense_distance = item["dense_distance"]

        if dense_distance is None:
            dense_relevance = 0.0
        else:
            dense_relevance = normalize_dense_distance(
                dense_distance,
                min_dense_distance,
                max_dense_distance,
            )

        sparse_relevance = normalize_sparse_score(
            item.get("sparse_score", 0.0),
            min_sparse_score,
            max_sparse_score,
        )

        hybrid_score = (
            DENSE_WEIGHT * dense_relevance
            + SPARSE_WEIGHT * sparse_relevance
        )

        item["dense_relevance"] = dense_relevance
        item["sparse_relevance"] = sparse_relevance
        item["hybrid_score"] = hybrid_score
        item["retrieval_sources"] = sorted(item["retrieval_sources"])

        # For compatibility with current RAG source display.
        # Higher hybrid_score is better, but previous code expects "distance".
        item["distance"] = 1.0 - hybrid_score

    candidates = apply_domain_adjustments(query, candidates)

    ranked = sorted(
        candidates,
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    return ranked[:final_k]
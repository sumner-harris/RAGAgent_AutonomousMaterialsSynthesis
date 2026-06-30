from __future__ import annotations

from typing import Iterable

import requests
from langchain_core.documents import Document

from RAG.openai_compat import normalize_base_url, resolve_api_key


def reranking_is_configured(*, model: str | None, base_url: str | None) -> bool:
    """Return whether reranking can run for the current request."""
    return bool((model or "").strip() and normalize_base_url(base_url))


def summarize_reranked_documents(
    *,
    documents: Iterable[Document],
) -> list[dict]:
    """Return UI-friendly details for documents kept by the reranker.

    Args:
        documents: Retrieved documents after optional reranking.

    Returns:
        Ordered dictionaries describing chunks that include a rerank score.
    """
    summarized = []
    for rank, document in enumerate(documents, start=1):
        rerank_score = document.metadata.get("rerank_score")
        if rerank_score is None:
            continue
        summarized.append(
            {
                "rank": len(summarized) + 1,
                "source": document.metadata.get("source", ""),
                "chunk_id": document.metadata.get("chunk_id"),
                "score": float(rerank_score),
                "text": document.page_content,
            }
        )
    return summarized


def keep_top_reranked_documents(
    *,
    documents: list[Document],
    top_k: int | None,
) -> list[Document]:
    """Keep only the highest-ranked reranked documents.

    Args:
        documents: Documents already ordered by rerank relevance.
        top_k: Maximum number of documents to keep.

    Returns:
        Leading reranked documents up to the requested count.
    """
    if top_k is None:
        return documents
    if top_k < 1:
        return []
    return documents[:top_k]


def _build_rerank_endpoint(base_url: str) -> str:
    """Return the rerank endpoint for a normalized OpenAI-compatible base URL."""
    normalized = normalize_base_url(base_url)
    if not normalized:
        raise ValueError("A rerank base URL is required.")
    return normalized.rstrip("/") + "/rerank"


def _build_rerank_headers(api_key: str | None, base_url: str | None) -> dict[str, str]:
    """Build HTTP headers for the rerank request."""
    headers = {"Content-Type": "application/json"}
    resolved_key = resolve_api_key(
        api_key,
        base_url,
        fallback_to_env=False,
    )
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    return headers


def _rerank_payload(
    *,
    query: str,
    documents: Iterable[Document],
    model: str,
) -> dict:
    """Build a Cohere-compatible rerank request payload."""
    return {
        "model": model,
        "query": query,
        "documents": [doc.page_content for doc in documents],
    }


def rerank_documents(
    *,
    query: str,
    documents: list[Document],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[Document]:
    """Rerank retrieved documents with an external reranker.

    Args:
        query: User query text.
        documents: Candidate documents to rerank.
        model: Rerank model ID served by the endpoint.
        api_key: Optional bearer token.
        base_url: OpenAI-compatible base URL for the rerank host.
        timeout: Request timeout in seconds.

    Returns:
        Documents reordered by descending rerank relevance score.

    Raises:
        requests.RequestException: When the rerank endpoint request fails.
        ValueError: When the endpoint response is malformed.
    """
    if not documents or not reranking_is_configured(model=model, base_url=base_url):
        return documents

    response = requests.post(
        _build_rerank_endpoint(base_url),
        headers=_build_rerank_headers(api_key, base_url),
        json=_rerank_payload(
            query=query,
            documents=documents,
            model=model,
        ),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Rerank response did not include a results list.")

    reranked_documents = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(documents):
            continue
        rerank_score = item.get("relevance_score")
        document = documents[index]
        if rerank_score is not None:
            document = Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rerank_score": rerank_score,
                },
            )
        reranked_documents.append(document)

    if not reranked_documents:
        raise ValueError("Rerank response did not return usable document indices.")

    return reranked_documents

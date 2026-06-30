"""Helpers for summarizing retrieval provenance by source document."""

from __future__ import annotations

from collections.abc import Iterable


PROVENANCE_DETAIL_KEYS = (
    "chunk_ids",
    "graph_source_ids",
    "graph_report_ids",
    "graph_entity_ids",
    "graph_relationship_ids",
    "graph_claim_ids",
    "graph_text_unit_ids",
)


def new_source_detail(source: str) -> dict:
    """Create an empty provenance summary for one source document.

    Args:
        source: Source document label, usually a PDF filename.

    Returns:
        Dictionary with stable provenance keys and empty evidence lists.
    """
    detail = {"source": str(source or "").strip()}
    for key in PROVENANCE_DETAIL_KEYS:
        detail[key] = []
    detail["evidence_count"] = 0
    return detail


def add_source_detail_evidence(
    detail: dict,
    *,
    chunk_id=None,
    graph_source_id=None,
    graph_report_id=None,
    graph_entity_id=None,
    graph_relationship_id=None,
    graph_claim_id=None,
    graph_text_unit_id=None,
) -> None:
    """Add one provenance reference to a source detail dictionary."""
    if chunk_id is not None:
        detail["chunk_ids"].append(chunk_id)
    if graph_source_id is not None:
        detail["graph_source_ids"].append(str(graph_source_id))
    if graph_report_id is not None:
        detail["graph_report_ids"].append(str(graph_report_id))
    if graph_entity_id is not None:
        detail["graph_entity_ids"].append(str(graph_entity_id))
    if graph_relationship_id is not None:
        detail["graph_relationship_ids"].append(str(graph_relationship_id))
    if graph_claim_id is not None:
        detail["graph_claim_ids"].append(str(graph_claim_id))
    if graph_text_unit_id is not None:
        detail["graph_text_unit_ids"].append(str(graph_text_unit_id))


def summarize_langchain_documents(documents) -> tuple[list[str], list[dict]]:
    """Summarize final LangChain documents into unique source-document entries.

    Args:
        documents: Iterable of LangChain `Document` objects with `source` and
            optional `chunk_id` metadata.

    Returns:
        Tuple of unique source names and detailed provenance dictionaries.
    """
    detail_map = {}
    for document in documents:
        source = str(document.metadata.get("source", "")).strip()
        if not source:
            continue
        detail = detail_map.setdefault(source, new_source_detail(source))
        add_source_detail_evidence(
            detail,
            chunk_id=document.metadata.get("chunk_id"),
        )
    details = finalize_source_document_details(detail_map.values())
    return [detail["source"] for detail in details], details


def finalize_source_document_details(details: Iterable[dict]) -> list[dict]:
    """Normalize, deduplicate, and sort provenance detail dictionaries."""
    finalized = []
    for raw_detail in details:
        detail = {"source": str(raw_detail.get("source", "")).strip()}
        if not detail["source"]:
            continue

        for key in PROVENANCE_DETAIL_KEYS:
            detail[key] = _dedupe_preserve_order(raw_detail.get(key, []))

        detail["evidence_count"] = sum(len(detail[key]) for key in PROVENANCE_DETAIL_KEYS)
        finalized.append(detail)

    finalized.sort(key=lambda item: (-item["evidence_count"], item["source"].lower()))
    return finalized


def merge_source_document_details(*detail_groups: Iterable[dict]) -> list[dict]:
    """Merge one or more provenance-detail iterables by source document."""
    detail_map = {}
    for group in detail_groups:
        for detail in group:
            source = str(detail.get("source", "")).strip()
            if not source:
                continue
            merged = detail_map.setdefault(source, new_source_detail(source))
            for key in PROVENANCE_DETAIL_KEYS:
                merged[key].extend(detail.get(key, []))
    return finalize_source_document_details(detail_map.values())


def _dedupe_preserve_order(values: Iterable) -> list:
    """Deduplicate iterable values while preserving their first-seen order."""
    seen = set()
    result = []
    for value in values:
        marker = value
        if isinstance(value, list):
            marker = tuple(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result

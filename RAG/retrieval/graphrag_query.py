# RAG/retrieval/graphrag_query.py
"""GraphRAG query helpers with best-effort document provenance resolution."""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.graphrag import _update_graphrag_settings
from graphrag_runtime_patch.openai_local_patch import (
    PATCH_ENV_VAR as GRAPHRAG_QUERY_PATCH_ENV_VAR,
    install as install_graphrag_query_patch,
    is_openai_hosted_base_url,
)
from RAG.openai_compat import (
    LOCAL_API_KEY_PLACEHOLDER,
    make_openai_client,
    resolve_api_key,
    resolve_base_url,
)
from RAG.retrieval.provenance import (
    add_source_detail_evidence,
    finalize_source_document_details,
    new_source_detail,
)
from state.config import (
    GRAPHRAG_GLOBAL_RETRIEVAL_METHOD,
    GRAPHRAG_LOCAL_RETRIEVAL_METHOD,
    LEGACY_RETRIEVAL_METHOD_ALIASES,
)


DEFAULT_GRAPHRAG_COMMUNITY_LEVEL = 2
DEFAULT_GRAPHRAG_RESPONSE_TYPE = "Multiple Paragraphs"
GRAPH_CITATION_PATTERN = re.compile(
    r"(Reports|Sources|Entities|Relationships|Claims)\s*\(([^)]*)\)",
    re.IGNORECASE,
)


class GraphRAGQueryError(RuntimeError):
    """Raised when GraphRAG query fails."""


@dataclass
class GraphRAGQueryResult:
    """GraphRAG answer text plus best-effort source-document provenance.

    Attributes:
        answer_text: Shortened GraphRAG answer text for downstream prompting.
        context_records: Normalized context-data records returned by GraphRAG.
        source_documents: Unique document labels inferred from the GraphRAG run.
        source_document_details: Per-document evidence details and supporting IDs.
        citations: Parsed GraphRAG `[Data: ...]` citation references from the
            unshortened answer text.
    """

    answer_text: str | None
    context_records: dict[str, Any]
    source_documents: list[str]
    source_document_details: list[dict[str, Any]]
    citations: dict[str, list[str]]


def graphrag_query_method_for_retrieval_method(retrieval_method: str) -> str:
    """Map an app retrieval method name to a GraphRAG CLI query method."""
    method = LEGACY_RETRIEVAL_METHOD_ALIASES.get(retrieval_method, retrieval_method)
    if method == GRAPHRAG_GLOBAL_RETRIEVAL_METHOD:
        return "global"
    if method == GRAPHRAG_LOCAL_RETRIEVAL_METHOD:
        return "local"
    raise ValueError(f"Unsupported GraphRAG retrieval method: {retrieval_method}")


def query_graphrag_with_context(
    root_dir,
    query,
    *,
    method="global",
    max_chars=6000,
    api_key: str | None = None,
    base_url: str | None = None,
    chat_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_model: str | None = None,
) -> GraphRAGQueryResult:
    """Run a GraphRAG query and return answer text plus provenance details.

    Args:
        root_dir: GraphRAG workspace root directory.
        query: User query text.
        method: GraphRAG query mode, either `global` or `local`.
        max_chars: Maximum characters retained in the returned answer text.
        api_key: Optional API key for the active prompt route.
        base_url: Optional prompt-model OpenAI-compatible base URL.
        chat_model: Optional prompt model ID for the GraphRAG query LLM.
        embedding_base_url: Optional embedding-model base URL for local search.
        embedding_model: Optional embedding model ID for local search.

    Returns:
        Structured GraphRAG query result with best-effort PDF provenance.

    Raises:
        GraphRAGQueryError: When GraphRAG query execution fails.
        ValueError: When an unsupported query method is requested.
    """
    if method not in {"global", "local"}:
        raise ValueError(f"Unsupported GraphRAG query method: {method}")

    root = Path(root_dir)
    with _temporary_graphrag_query_config(
        root=root,
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    ) as config_filepath, _temporary_graphrag_query_runtime_patch(
        base_url=base_url
    ):
        answer, context_data = _run_graphrag_query(
            root=root,
            query=query,
            method=method,
            config_filepath=config_filepath,
        )
    context_records = _normalize_context_records(context_data)
    citations = _extract_graph_citations(answer or "")
    source_document_details = _resolve_graph_source_document_details(
        root=root,
        method=method,
        context_records=context_records,
        citations=citations,
    )
    return GraphRAGQueryResult(
        answer_text=(
            textwrap.shorten(answer, width=max_chars, placeholder=" ...")
            if answer
            else None
        ),
        context_records=context_records,
        source_documents=[detail["source"] for detail in source_document_details],
        source_document_details=source_document_details,
        citations=citations,
    )


def query_graphrag(root_dir, query, *, method="global", max_chars=6000):
    """Run a GraphRAG query and return a shortened text response."""
    return query_graphrag_with_context(
        root_dir,
        query,
        method=method,
        max_chars=max_chars,
    ).answer_text


def _run_graphrag_query(
    *,
    root: Path,
    query: str,
    method: str,
    config_filepath: Path | None = None,
) -> tuple[str | None, Any]:
    """Run GraphRAG through its Python query helpers and capture context data."""
    try:
        from graphrag.cli.query import run_global_search, run_local_search
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise GraphRAGQueryError(
            "GraphRAG Python query helpers are unavailable in this environment."
        ) from exc

    if method == "global":
        runner = run_global_search
        runner_kwargs = {
            "config_filepath": config_filepath,
            "data_dir": None,
            "root_dir": root,
            "community_level": DEFAULT_GRAPHRAG_COMMUNITY_LEVEL,
            "dynamic_community_selection": False,
            "response_type": DEFAULT_GRAPHRAG_RESPONSE_TYPE,
            "streaming": False,
            "query": query,
            "verbose": False,
        }
    else:
        runner = run_local_search
        runner_kwargs = {
            "config_filepath": config_filepath,
            "data_dir": None,
            "root_dir": root,
            "community_level": DEFAULT_GRAPHRAG_COMMUNITY_LEVEL,
            "response_type": DEFAULT_GRAPHRAG_RESPONSE_TYPE,
            "streaming": False,
            "query": query,
            "verbose": False,
        }

    try:
        answer, context_data = _run_graphrag_runner(
            runner=runner,
            runner_kwargs=runner_kwargs,
        )
    except Exception as exc:
        raise GraphRAGQueryError(str(exc)) from exc

    answer_text = str(answer or "").strip()
    return answer_text or None, context_data


def _run_graphrag_runner(*, runner, runner_kwargs: dict[str, Any]):
    """Execute a GraphRAG query runner safely from sync or async server contexts."""
    if _has_running_event_loop():
        return _run_graphrag_runner_in_worker_thread(
            runner=runner,
            runner_kwargs=runner_kwargs,
        )

    return _invoke_graphrag_runner(runner=runner, runner_kwargs=runner_kwargs)


def _has_running_event_loop() -> bool:
    """Return whether the current thread already owns a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _invoke_graphrag_runner(*, runner, runner_kwargs: dict[str, Any]):
    """Invoke a GraphRAG runner while suppressing its stdout chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        return runner(**runner_kwargs)


def _run_graphrag_runner_in_worker_thread(*, runner, runner_kwargs: dict[str, Any]):
    """Run GraphRAG in an isolated thread when the caller is already async."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="graphrag-query") as pool:
        future = pool.submit(
            _invoke_graphrag_runner,
            runner=runner,
            runner_kwargs=runner_kwargs,
        )
        return future.result()


def _is_prompt_model_id(model_id: str) -> bool:
    """Return whether a discovered model ID looks prompt-capable."""
    normalized = (model_id or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith("text-embedding") or "embedding" in normalized:
        return False
    return True


def _discover_single_prompt_model(
    *,
    api_key: str | None,
    base_url: str | None,
) -> str | None:
    """Return the sole prompt model on an endpoint when discovery is unambiguous."""
    resolved_base_url = resolve_base_url(base_url, fallback_to_env=False)
    if not resolved_base_url:
        return None

    try:
        client = make_openai_client(
            api_key=api_key,
            base_url=resolved_base_url,
            fallback_to_env=False,
        )
        data = client.models.list()
    except Exception:
        return None

    model_ids = sorted(
        {
            str(item.id).strip()
            for item in getattr(data, "data", [])
            if getattr(item, "id", None) and _is_prompt_model_id(str(item.id))
        }
    )
    if len(model_ids) == 1:
        return model_ids[0]
    return None


@contextlib.contextmanager
def _temporary_graphrag_query_runtime_patch(*, base_url: str | None):
    """Enable the local GraphRAG no-thinking patch during query-time runs."""
    resolved_base_url = resolve_base_url(base_url, fallback_to_env=False)
    if not resolved_base_url or is_openai_hosted_base_url(resolved_base_url):
        yield
        return

    previous = os.environ.get(GRAPHRAG_QUERY_PATCH_ENV_VAR)
    os.environ[GRAPHRAG_QUERY_PATCH_ENV_VAR] = "1"
    install_graphrag_query_patch()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(GRAPHRAG_QUERY_PATCH_ENV_VAR, None)
        else:
            os.environ[GRAPHRAG_QUERY_PATCH_ENV_VAR] = previous


@contextlib.contextmanager
def _temporary_graphrag_query_config(
    *,
    root: Path,
    api_key: str | None,
    base_url: str | None,
    chat_model: str | None,
    embedding_base_url: str | None,
    embedding_model: str | None,
):
    """Yield a temporary GraphRAG config that reflects the current query routes."""
    settings_path = root / "settings.yaml"
    resolved_base_url = resolve_base_url(base_url, fallback_to_env=False)
    resolved_embedding_base_url = resolve_base_url(
        embedding_base_url,
        fallback_to_env=False,
    )
    resolved_chat_model = (chat_model or "").strip() or _discover_single_prompt_model(
        api_key=api_key,
        base_url=resolved_base_url,
    )
    resolved_embedding_model = (embedding_model or "").strip() or None
    explicit_api_key = (api_key or "").strip()

    needs_override = any(
        (
            resolved_base_url,
            resolved_embedding_base_url,
            resolved_chat_model,
            resolved_embedding_model,
            explicit_api_key,
        )
    )
    if not settings_path.is_file() or not needs_override:
        yield None
        return

    resolved_api_key = resolve_api_key(
        api_key,
        resolved_base_url,
        fallback_to_env=False,
    )
    with tempfile.TemporaryDirectory(prefix="graphrag-query-") as temp_dir:
        temp_root = Path(temp_dir)
        temp_settings_path = temp_root / "settings.yaml"
        temp_settings_path.write_text(
            settings_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        _update_graphrag_settings(
            temp_settings_path,
            base_url=resolved_base_url,
            embedding_base_url=resolved_embedding_base_url,
            chat_model=resolved_chat_model,
            embedding_model=resolved_embedding_model,
        )
        (temp_root / ".env").write_text(
            "GRAPHRAG_API_KEY="
            f"{resolved_api_key or LOCAL_API_KEY_PLACEHOLDER}\n",
            encoding="utf-8",
        )
        yield temp_settings_path


def _normalize_context_records(context_data: Any) -> dict[str, Any]:
    """Convert GraphRAG context dataframes into plain record dictionaries."""
    if not isinstance(context_data, dict):
        return {}

    normalized = {}
    for key, value in context_data.items():
        if value is None:
            normalized[key] = []
        elif hasattr(value, "to_dict"):
            normalized[key] = value.to_dict(orient="records")
        elif isinstance(value, list):
            normalized[key] = value
        elif isinstance(value, dict):
            normalized[key] = value
        else:
            normalized[key] = []
    return normalized


def _extract_graph_citations(answer_text: str) -> dict[str, list[str]]:
    """Parse GraphRAG `[Data: ...]` citations into normalized ID lists."""
    citations = {
        "reports": [],
        "sources": [],
        "entities": [],
        "relationships": [],
        "claims": [],
    }
    collected = {key: set() for key in citations}

    for label, raw_ids in GRAPH_CITATION_PATTERN.findall(answer_text or ""):
        key = label.lower()
        if key not in collected:
            continue
        for match in re.findall(r"\d+", raw_ids):
            collected[key].add(match)

    for key, values in collected.items():
        citations[key] = sorted(values, key=lambda item: int(item))
    return citations


def _resolve_graph_source_document_details(
    *,
    root: Path,
    method: str,
    context_records: dict[str, Any],
    citations: dict[str, list[str]],
) -> list[dict]:
    """Resolve GraphRAG context records into per-document provenance details."""
    output_dir = root / "output"
    if not output_dir.is_dir():
        return []

    documents_df = _read_graph_table(output_dir / "documents.parquet")
    text_units_df = _read_graph_table(output_dir / "text_units.parquet")
    if documents_df is None or text_units_df is None:
        return []

    detail_map = {}
    document_titles = _build_document_title_map(documents_df)
    source_short_to_docs, source_short_to_text_unit_id = _build_text_unit_short_maps(
        text_units_df
    )
    text_unit_id_to_docs = _build_text_unit_id_map(text_units_df)
    selected_from_citations = False

    report_citations = citations.get("reports", [])
    source_citations = citations.get("sources", [])
    entity_citations = citations.get("entities", [])
    relationship_citations = citations.get("relationships", [])
    claim_citations = citations.get("claims", [])

    if report_citations:
        communities_df = _read_graph_table(output_dir / "communities.parquet")
        if communities_df is not None:
            community_to_text_units = _build_community_text_unit_map(communities_df)
            for report_id in report_citations:
                text_unit_ids = community_to_text_units.get(report_id, [])
                _attach_text_unit_documents(
                    detail_map,
                    text_unit_ids=text_unit_ids,
                    document_titles=document_titles,
                    text_unit_id_to_docs=text_unit_id_to_docs,
                    graph_report_id=report_id,
                )
                selected_from_citations = selected_from_citations or bool(text_unit_ids)

    if source_citations:
        for source_id in source_citations:
            document_ids = source_short_to_docs.get(source_id, [])
            text_unit_id = source_short_to_text_unit_id.get(source_id)
            _attach_documents(
                detail_map,
                document_ids=document_ids,
                document_titles=document_titles,
                graph_source_id=source_id,
                graph_text_unit_id=text_unit_id,
            )
            selected_from_citations = selected_from_citations or bool(document_ids)

    if entity_citations:
        entities_df = _read_graph_table(output_dir / "entities.parquet")
        if entities_df is not None:
            entity_to_text_units = _build_row_text_unit_map(
                entities_df,
                short_id_col="human_readable_id",
                text_unit_ids_col="text_unit_ids",
            )
            for entity_id in entity_citations:
                text_unit_ids = entity_to_text_units.get(entity_id, [])
                _attach_text_unit_documents(
                    detail_map,
                    text_unit_ids=text_unit_ids,
                    document_titles=document_titles,
                    text_unit_id_to_docs=text_unit_id_to_docs,
                    graph_entity_id=entity_id,
                )
                selected_from_citations = selected_from_citations or bool(text_unit_ids)

    if relationship_citations:
        relationships_df = _read_graph_table(output_dir / "relationships.parquet")
        if relationships_df is not None:
            relationship_to_text_units = _build_row_text_unit_map(
                relationships_df,
                short_id_col="human_readable_id",
                text_unit_ids_col="text_unit_ids",
            )
            for relationship_id in relationship_citations:
                text_unit_ids = relationship_to_text_units.get(relationship_id, [])
                _attach_text_unit_documents(
                    detail_map,
                    text_unit_ids=text_unit_ids,
                    document_titles=document_titles,
                    text_unit_id_to_docs=text_unit_id_to_docs,
                    graph_relationship_id=relationship_id,
                )
                selected_from_citations = selected_from_citations or bool(text_unit_ids)

    if claim_citations:
        covariates_df = _read_graph_table(output_dir / "covariates.parquet")
        if covariates_df is not None:
            claim_to_text_units = _build_covariate_text_unit_map(covariates_df)
            for claim_id in claim_citations:
                text_unit_ids = claim_to_text_units.get(claim_id, [])
                _attach_text_unit_documents(
                    detail_map,
                    text_unit_ids=text_unit_ids,
                    document_titles=document_titles,
                    text_unit_id_to_docs=text_unit_id_to_docs,
                    graph_claim_id=claim_id,
                )
                selected_from_citations = selected_from_citations or bool(text_unit_ids)

    if not selected_from_citations:
        if method == "local" and context_records.get("sources"):
            for record in context_records.get("sources", []):
                source_id = _normalize_record_id(record.get("id"))
                document_ids = source_short_to_docs.get(source_id, [])
                text_unit_id = source_short_to_text_unit_id.get(source_id)
                _attach_documents(
                    detail_map,
                    document_ids=document_ids,
                    document_titles=document_titles,
                    graph_source_id=source_id,
                    graph_text_unit_id=text_unit_id,
                )

        if context_records.get("reports"):
            communities_df = _read_graph_table(output_dir / "communities.parquet")
            if communities_df is not None:
                community_to_text_units = _build_community_text_unit_map(communities_df)
                for record in context_records.get("reports", []):
                    report_id = _normalize_record_id(record.get("id"))
                    text_unit_ids = community_to_text_units.get(report_id, [])
                    _attach_text_unit_documents(
                        detail_map,
                        text_unit_ids=text_unit_ids,
                        document_titles=document_titles,
                        text_unit_id_to_docs=text_unit_id_to_docs,
                        graph_report_id=report_id,
                    )

    return finalize_source_document_details(detail_map.values())


def _read_graph_table(path: Path) -> pd.DataFrame | None:
    """Read one GraphRAG parquet table when present."""
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _build_document_title_map(documents_df: pd.DataFrame) -> dict[str, str]:
    """Map GraphRAG document IDs to user-facing source document labels."""
    result = {}
    for row in documents_df.to_dict(orient="records"):
        document_id = _normalize_record_id(row.get("id"))
        if not document_id:
            continue
        result[document_id] = _normalize_graph_document_title(row.get("title"))
    return result


def _build_text_unit_short_maps(
    text_units_df: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Map GraphRAG local-search source IDs to document IDs and text-unit IDs."""
    source_short_to_docs = {}
    source_short_to_text_unit_id = {}
    reset = text_units_df.reset_index(drop=True)
    for index, row in reset.iterrows():
        short_id = str(index)
        source_short_to_docs[short_id] = _coerce_list_of_ids(row.get("document_ids"))
        source_short_to_text_unit_id[short_id] = _normalize_record_id(row.get("id"))
    return source_short_to_docs, source_short_to_text_unit_id


def _build_text_unit_id_map(text_units_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map GraphRAG text-unit IDs to their supporting document IDs."""
    result = {}
    for row in text_units_df.to_dict(orient="records"):
        text_unit_id = _normalize_record_id(row.get("id"))
        if not text_unit_id:
            continue
        result[text_unit_id] = _coerce_list_of_ids(row.get("document_ids"))
    return result


def _build_community_text_unit_map(communities_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map GraphRAG community/report IDs to supporting text-unit IDs."""
    result = {}
    for row in communities_df.to_dict(orient="records"):
        community_id = _normalize_record_id(row.get("community"))
        if not community_id:
            continue
        result[community_id] = _coerce_list_of_ids(row.get("text_unit_ids"))
    return result


def _build_row_text_unit_map(
    df: pd.DataFrame,
    *,
    short_id_col: str,
    text_unit_ids_col: str,
) -> dict[str, list[str]]:
    """Map GraphRAG short IDs to supporting text-unit IDs."""
    result = {}
    for row in df.to_dict(orient="records"):
        short_id = _normalize_record_id(row.get(short_id_col))
        if not short_id:
            continue
        result[short_id] = _coerce_list_of_ids(row.get(text_unit_ids_col))
    return result


def _build_covariate_text_unit_map(covariates_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map GraphRAG claim/covariate short IDs to supporting text-unit IDs."""
    result = {}
    for row in covariates_df.to_dict(orient="records"):
        short_id = _normalize_record_id(row.get("human_readable_id"))
        if not short_id:
            continue
        result[short_id] = _coerce_list_of_ids(row.get("text_unit_id"))
    return result


def _attach_text_unit_documents(
    detail_map: dict[str, dict],
    *,
    text_unit_ids: list[str],
    document_titles: dict[str, str],
    text_unit_id_to_docs: dict[str, list[str]],
    graph_report_id=None,
    graph_entity_id=None,
    graph_relationship_id=None,
    graph_claim_id=None,
) -> None:
    """Attach all source documents referenced by one or more GraphRAG text units."""
    for text_unit_id in text_unit_ids:
        _attach_documents(
            detail_map,
            document_ids=text_unit_id_to_docs.get(str(text_unit_id), []),
            document_titles=document_titles,
            graph_report_id=graph_report_id,
            graph_entity_id=graph_entity_id,
            graph_relationship_id=graph_relationship_id,
            graph_claim_id=graph_claim_id,
            graph_text_unit_id=text_unit_id,
        )


def _attach_documents(
    detail_map: dict[str, dict],
    *,
    document_ids: list[str],
    document_titles: dict[str, str],
    chunk_id=None,
    graph_source_id=None,
    graph_report_id=None,
    graph_entity_id=None,
    graph_relationship_id=None,
    graph_claim_id=None,
    graph_text_unit_id=None,
) -> None:
    """Attach one or more documents to the aggregate provenance map."""
    for document_id in document_ids:
        source = document_titles.get(str(document_id))
        if not source:
            continue
        detail = detail_map.setdefault(source, new_source_detail(source))
        add_source_detail_evidence(
            detail,
            chunk_id=chunk_id,
            graph_source_id=graph_source_id,
            graph_report_id=graph_report_id,
            graph_entity_id=graph_entity_id,
            graph_relationship_id=graph_relationship_id,
            graph_claim_id=graph_claim_id,
            graph_text_unit_id=graph_text_unit_id,
        )


def _coerce_list_of_ids(value) -> list[str]:
    """Normalize a scalar or list-like GraphRAG field into string IDs."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, str):
        return [_normalize_record_id(value)] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [
            _normalize_record_id(item)
            for item in value
            if _normalize_record_id(item)
        ]
    normalized = _normalize_record_id(value)
    return [normalized] if normalized else []


def _normalize_record_id(value) -> str:
    """Normalize GraphRAG record IDs to stable string keys."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    return text


def _normalize_graph_document_title(title) -> str:
    """Convert GraphRAG `.txt` workspace titles back to app-facing PDF names."""
    text = str(title or "").strip()
    if text.lower().endswith(".txt"):
        return text[:-4] + ".pdf"
    return text

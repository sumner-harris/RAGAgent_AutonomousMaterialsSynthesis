import asyncio
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import yaml

from RAG.retrieval.graphrag_query import (
    GRAPHRAG_QUERY_PATCH_ENV_VAR,
    _run_graphrag_query,
    _temporary_graphrag_query_runtime_patch,
    query_graphrag_with_context,
)


def _graph_tables():
    return {
        "documents.parquet": pd.DataFrame(
            [
                {"id": "doc-a", "title": "alpha.txt"},
                {"id": "doc-b", "title": "beta.txt"},
            ]
        ),
        "text_units.parquet": pd.DataFrame(
            [
                {"id": "tu-0", "document_ids": np.array(["doc-a"], dtype=object)},
                {"id": "tu-1", "document_ids": np.array(["doc-b"], dtype=object)},
            ]
        ),
        "communities.parquet": pd.DataFrame(
            [
                {"community": 42, "text_unit_ids": np.array(["tu-1"], dtype=object)},
            ]
        ),
    }


def _patch_graph_tables(monkeypatch, tables: dict[str, pd.DataFrame]):
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._read_graph_table",
        lambda path: tables.get(Path(path).name),
    )


def test_query_graphrag_with_context_resolves_local_source_citations(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "graphrag"
    (root / "output").mkdir(parents=True)
    _patch_graph_tables(monkeypatch, _graph_tables())
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._run_graphrag_query",
        lambda **kwargs: (
            "Local answer [Data: Sources (1)]",
            {"sources": [{"id": 1, "text": "beta chunk"}]},
        ),
    )

    result = query_graphrag_with_context(root, "test query", method="local")

    assert result.answer_text == "Local answer [Data: Sources (1)]"
    assert result.source_documents == ["beta.pdf"]
    assert result.source_document_details == [
        {
            "source": "beta.pdf",
            "chunk_ids": [],
            "graph_source_ids": ["1"],
            "graph_report_ids": [],
            "graph_entity_ids": [],
            "graph_relationship_ids": [],
            "graph_claim_ids": [],
            "graph_text_unit_ids": ["tu-1"],
            "evidence_count": 2,
        }
    ]


def test_query_graphrag_with_context_resolves_global_report_citations(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "graphrag"
    (root / "output").mkdir(parents=True)
    _patch_graph_tables(monkeypatch, _graph_tables())
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._run_graphrag_query",
        lambda **kwargs: (
            "Global answer [Data: Reports (42)]",
            {"reports": [{"id": 42, "title": "community summary"}]},
        ),
    )

    result = query_graphrag_with_context(root, "test query", method="global")

    assert result.source_documents == ["beta.pdf"]
    assert result.source_document_details == [
        {
            "source": "beta.pdf",
            "chunk_ids": [],
            "graph_source_ids": [],
            "graph_report_ids": ["42"],
            "graph_entity_ids": [],
            "graph_relationship_ids": [],
            "graph_claim_ids": [],
            "graph_text_unit_ids": ["tu-1"],
            "evidence_count": 2,
        }
    ]


def test_query_graphrag_with_context_falls_back_to_local_context_records(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "graphrag"
    (root / "output").mkdir(parents=True)
    _patch_graph_tables(monkeypatch, _graph_tables())
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._run_graphrag_query",
        lambda **kwargs: (
            "Local answer without explicit citations",
            {"sources": [{"id": 0, "text": "alpha chunk"}]},
        ),
    )

    result = query_graphrag_with_context(root, "test query", method="local")

    assert result.source_documents == ["alpha.pdf"]
    assert result.source_document_details == [
        {
            "source": "alpha.pdf",
            "chunk_ids": [],
            "graph_source_ids": ["0"],
            "graph_report_ids": [],
            "graph_entity_ids": [],
            "graph_relationship_ids": [],
            "graph_claim_ids": [],
            "graph_text_unit_ids": ["tu-0"],
            "evidence_count": 2,
        }
    ]


def test_query_graphrag_with_context_builds_runtime_query_config(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "graphrag"
    (root / "output").mkdir(parents=True)
    (root / "settings.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "default_chat_model": {"model": "stale-model"},
                    "default_embedding_model": {"model": "stale-embedding"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _patch_graph_tables(monkeypatch, _graph_tables())
    captured = {}

    def fake_run(**kwargs):
        config_filepath = kwargs["config_filepath"]
        captured["config_filepath"] = str(config_filepath)
        captured["settings"] = yaml.safe_load(
            Path(config_filepath).read_text(encoding="utf-8")
        )
        captured["env"] = (Path(config_filepath).parent / ".env").read_text(
            encoding="utf-8"
        )
        return "Global answer [Data: Reports (42)]", {
            "reports": [{"id": 42, "title": "community summary"}]
        }

    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._run_graphrag_query",
        fake_run,
    )

    result = query_graphrag_with_context(
        root,
        "test query",
        method="global",
        api_key="",
        base_url="http://nvidiaspark:8000/v1",
        chat_model="gemma-4-live",
        embedding_base_url="http://nvidiaspark:8001/v1",
        embedding_model="vllm-sfr-embedding-mistral",
    )

    assert captured["config_filepath"].endswith("settings.yaml")
    assert (
        captured["settings"]["models"]["default_chat_model"]["model"]
        == "gemma-4-live"
    )
    assert (
        captured["settings"]["models"]["default_chat_model"]["api_base"]
        == "http://nvidiaspark:8000/v1"
    )
    assert (
        captured["settings"]["models"]["default_embedding_model"]["model"]
        == "vllm-sfr-embedding-mistral"
    )
    assert (
        captured["settings"]["models"]["default_embedding_model"]["api_base"]
        == "http://nvidiaspark:8001/v1"
    )
    assert captured["env"].strip() == "GRAPHRAG_API_KEY=EMPTY"
    assert result.source_documents == ["beta.pdf"]


def test_query_graphrag_with_context_keeps_provenance_when_answer_is_empty(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "graphrag"
    (root / "output").mkdir(parents=True)
    _patch_graph_tables(monkeypatch, _graph_tables())
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query._run_graphrag_query",
        lambda **kwargs: (
            "",
            {"sources": [{"id": 0, "text": "alpha chunk"}]},
        ),
    )

    result = query_graphrag_with_context(root, "test query", method="local")

    assert result.answer_text is None
    assert result.context_records == {"sources": [{"id": 0, "text": "alpha chunk"}]}
    assert result.source_documents == ["alpha.pdf"]
    assert result.source_document_details[0]["graph_source_ids"] == ["0"]


def test_temporary_graphrag_query_runtime_patch_enables_local_no_thinking(
    monkeypatch,
):
    installed = []
    monkeypatch.delenv(GRAPHRAG_QUERY_PATCH_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "RAG.retrieval.graphrag_query.install_graphrag_query_patch",
        lambda: installed.append(True),
    )

    with _temporary_graphrag_query_runtime_patch(
        base_url="http://nvidiaspark:8000/v1"
    ):
        assert installed == [True]
        assert os.environ[GRAPHRAG_QUERY_PATCH_ENV_VAR] == "1"

    assert GRAPHRAG_QUERY_PATCH_ENV_VAR not in os.environ


def test_run_graphrag_query_uses_worker_thread_inside_running_event_loop(
    monkeypatch,
    tmp_path,
):
    import graphrag.cli.query as query_module

    captured = {}

    def fake_run_local_search(**kwargs):
        captured["runner_thread_id"] = threading.get_ident()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            captured["runner_has_loop"] = False
        else:
            captured["runner_has_loop"] = True
        return "Local answer", {"sources": [{"id": 1}]}

    monkeypatch.setattr(query_module, "run_local_search", fake_run_local_search)

    async def _run():
        caller_thread_id = threading.get_ident()
        result = _run_graphrag_query(
            root=tmp_path,
            query="test query",
            method="local",
        )
        return caller_thread_id, result

    caller_thread_id, (answer_text, context_data) = asyncio.run(_run())

    assert answer_text == "Local answer"
    assert context_data == {"sources": [{"id": 1}]}
    assert captured["runner_thread_id"] != caller_thread_id
    assert captured["runner_has_loop"] is False

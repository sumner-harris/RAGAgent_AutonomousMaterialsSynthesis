from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from RAG.services.retrieval_service import retrieve_context
from state.config import (
    DEFAULT_RETRIEVAL_DIVERSITY,
    DEFAULT_RETRIEVAL_METHOD,
    DEFAULT_RERANK_TOP_K,
    TOP_K_TEXT_FAISS,
)


def test_retrieve_context_uses_shared_faiss_and_diversity_defaults(
    dummy_kb,
    monkeypatch,
):
    index_path, meta_path, _ = dummy_kb
    captured = {}

    monkeypatch.setattr(
        "RAG.services.retrieval_service.connection_is_configured",
        lambda api_key, base_url: True,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.create_embedding_function",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service._build_faiss_store",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.reranking_is_configured",
        lambda **kwargs: False,
    )

    def _fake_retrieve_documents_with_method(query, method, **kwargs):
        captured["method"] = method
        captured["top_k"] = kwargs["top_k"]
        captured["diversity"] = kwargs["diversity"]
        return [
            Document(
                page_content="ctx",
                metadata={"source": "dummy.pdf", "chunk_id": 0},
            )
        ]

    monkeypatch.setattr(
        "RAG.services.retrieval_service.retrieve_documents_with_method",
        _fake_retrieve_documents_with_method,
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    assert captured["method"] == DEFAULT_RETRIEVAL_METHOD
    assert captured["top_k"] == TOP_K_TEXT_FAISS
    assert captured["diversity"] == DEFAULT_RETRIEVAL_DIVERSITY
    assert "ctx" in result["context_text"]
    assert result["source_documents"] == ["dummy.pdf"]
    assert result["source_document_details"] == [
        {
            "source": "dummy.pdf",
            "chunk_ids": [0],
            "graph_source_ids": [],
            "graph_report_ids": [],
            "graph_entity_ids": [],
            "graph_relationship_ids": [],
            "graph_claim_ids": [],
            "graph_text_unit_ids": [],
            "evidence_count": 1,
        }
    ]


def test_retrieve_context_uses_shared_rerank_top_k_default(
    dummy_kb,
    monkeypatch,
):
    index_path, meta_path, _ = dummy_kb
    captured = {}

    monkeypatch.setattr(
        "RAG.services.retrieval_service.connection_is_configured",
        lambda api_key, base_url: True,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.create_embedding_function",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service._build_faiss_store",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.retrieve_documents_with_method",
        lambda query, method, **kwargs: [
            Document(
                page_content=f"doc {index}",
                metadata={
                    "source": "dummy.pdf",
                    "chunk_id": index,
                    "rerank_score": 1.0 - (index * 0.01),
                },
            )
            for index in range(20)
        ],
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.reranking_is_configured",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.rerank_documents",
        lambda **kwargs: kwargs["documents"],
    )

    def _fake_keep_top_reranked_documents(*, documents, top_k):
        captured["top_k"] = top_k
        return documents[:top_k]

    monkeypatch.setattr(
        "RAG.services.retrieval_service.keep_top_reranked_documents",
        _fake_keep_top_reranked_documents,
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        rerank_base_url="http://localhost:8002/v1",
        rerank_model="nemotron-rerank-1b-v2",
    )

    assert captured["top_k"] == DEFAULT_RERANK_TOP_K
    assert len(result["retrieved_context_text"].split("\n\n")) == 20
    assert len(result["context_text"].split("\n\n")) == DEFAULT_RERANK_TOP_K
    assert len(result["sources"]) == DEFAULT_RERANK_TOP_K
    assert result["source_documents"] == ["dummy.pdf"]
    assert result["source_document_details"][0]["chunk_ids"] == list(
        range(DEFAULT_RERANK_TOP_K)
    )


def test_retrieve_context_allows_vector_search_with_embedding_route_only(
    dummy_kb,
    monkeypatch,
):
    index_path, meta_path, _ = dummy_kb
    captured = {}

    def fake_connection_is_configured(api_key, base_url):
        return base_url == "http://localhost:8001/v1"

    monkeypatch.setattr(
        "RAG.services.retrieval_service.connection_is_configured",
        fake_connection_is_configured,
    )

    def fake_create_embedding_function(**kwargs):
        captured["embedding_base_url"] = kwargs["base_url"]
        return object()

    monkeypatch.setattr(
        "RAG.services.retrieval_service.create_embedding_function",
        fake_create_embedding_function,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service._build_faiss_store",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.reranking_is_configured",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.retrieve_documents_with_method",
        lambda query, method, **kwargs: [
            Document(
                page_content="ctx",
                metadata={"source": "dummy.pdf", "chunk_id": 0},
            )
        ],
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
        api_key="",
        base_url="",
        embedding_base_url="http://localhost:8001/v1",
        retrieval_method="faiss_mmr",
    )

    assert captured["embedding_base_url"] == "http://localhost:8001/v1"
    assert "ctx" in result["context_text"]


def test_retrieve_context_allows_bm25_without_any_model_route(
    dummy_kb,
    monkeypatch,
):
    index_path, meta_path, _ = dummy_kb
    captured = {}

    monkeypatch.setattr(
        "RAG.services.retrieval_service.connection_is_configured",
        lambda api_key, base_url: False,
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.create_embedding_function",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("BM25 retrieval should not build embeddings.")
        ),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.BM25Retriever.from_documents",
        lambda docs, k: object(),
    )
    monkeypatch.setattr(
        "RAG.services.retrieval_service.reranking_is_configured",
        lambda **kwargs: False,
    )

    def fake_retrieve_documents_with_method(query, method, **kwargs):
        captured["vector_store"] = kwargs["vector_store"]
        return [
            Document(
                page_content="bm25 ctx",
                metadata={"source": "dummy.pdf", "chunk_id": 0},
            )
        ]

    monkeypatch.setattr(
        "RAG.services.retrieval_service.retrieve_documents_with_method",
        fake_retrieve_documents_with_method,
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
        api_key="",
        base_url="",
        embedding_base_url="",
        retrieval_method="bm25",
    )

    assert captured["vector_store"] is None
    assert "bm25 ctx" in result["context_text"]


def test_retrieve_context_requires_prompt_route_for_multi_query_method():
    with pytest.raises(ValueError, match="prompt model endpoint"):
        retrieve_context(
            query="test query",
            index_path="index.faiss",
            meta_path="meta.pkl",
            graphrag_dir=None,
            api_key="",
            base_url="",
            embedding_base_url="http://localhost:8001/v1",
            retrieval_method="hybrid_multi_query",
        )


def test_retrieve_context_routes_graphrag_global_without_model_endpoint(
    dummy_kb,
    monkeypatch,
    tmp_path,
):
    index_path, meta_path, _ = dummy_kb
    graphrag_dir = tmp_path / "graphrag"
    graphrag_dir.mkdir()
    captured = {}

    monkeypatch.setattr(
        "RAG.services.retrieval_service.query_graphrag_with_context",
        lambda root, query, *, method="global", max_chars=6000, **kwargs: (
            captured.update(
                {
                    "root": str(root),
                    "query": query,
                    "method": method,
                    "api_key": kwargs.get("api_key"),
                    "base_url": kwargs.get("base_url"),
                    "chat_model": kwargs.get("chat_model"),
                    "embedding_base_url": kwargs.get("embedding_base_url"),
                    "embedding_model": kwargs.get("embedding_model"),
                }
            )
            or SimpleNamespace(
                answer_text="global graph context",
                source_documents=["graph_doc.pdf"],
                source_document_details=[
                    {
                        "source": "graph_doc.pdf",
                        "chunk_ids": [],
                        "graph_source_ids": [],
                        "graph_report_ids": ["7"],
                        "graph_entity_ids": [],
                        "graph_relationship_ids": [],
                        "graph_claim_ids": [],
                        "graph_text_unit_ids": ["tu-1"],
                        "evidence_count": 2,
                    }
                ],
            )
        ),
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        api_key=None,
        base_url="http://nvidiaspark:8000/v1",
        embedding_base_url="http://nvidiaspark:8001/v1",
        retrieval_method="graphrag_global",
        chat_model="gemma-4-live",
    )

    assert captured == {
        "root": str(graphrag_dir),
        "query": "test query",
        "method": "global",
        "api_key": None,
        "base_url": "http://nvidiaspark:8000/v1",
        "chat_model": "gemma-4-live",
        "embedding_base_url": "http://nvidiaspark:8001/v1",
        "embedding_model": "text-embedding-3-small",
    }
    assert "global graph context" in result["context_text"]
    assert result["sources"] == [{"source": "graphrag", "chunk_id": -1}]
    assert result["source_documents"] == ["graph_doc.pdf"]
    assert result["source_document_details"][0]["graph_report_ids"] == ["7"]


def test_retrieve_context_routes_graphrag_local_without_model_endpoint(
    dummy_kb,
    monkeypatch,
    tmp_path,
):
    index_path, meta_path, _ = dummy_kb
    graphrag_dir = tmp_path / "graphrag"
    graphrag_dir.mkdir()
    captured = {}

    monkeypatch.setattr(
        "RAG.services.retrieval_service.query_graphrag_with_context",
        lambda root, query, *, method="global", max_chars=6000, **kwargs: (
            captured.update(
                {
                    "root": str(root),
                    "query": query,
                    "method": method,
                    "api_key": kwargs.get("api_key"),
                    "base_url": kwargs.get("base_url"),
                    "chat_model": kwargs.get("chat_model"),
                    "embedding_base_url": kwargs.get("embedding_base_url"),
                    "embedding_model": kwargs.get("embedding_model"),
                }
            )
            or SimpleNamespace(
                answer_text="local graph context",
                source_documents=["local_doc.pdf"],
                source_document_details=[
                    {
                        "source": "local_doc.pdf",
                        "chunk_ids": [],
                        "graph_source_ids": ["3"],
                        "graph_report_ids": [],
                        "graph_entity_ids": ["9"],
                        "graph_relationship_ids": [],
                        "graph_claim_ids": [],
                        "graph_text_unit_ids": ["tu-2"],
                        "evidence_count": 3,
                    }
                ],
            )
        ),
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        api_key=None,
        base_url="http://nvidiaspark:8000/v1",
        embedding_base_url="http://nvidiaspark:8001/v1",
        retrieval_method="graphrag_local",
        chat_model="gemma-4-live",
    )

    assert captured == {
        "root": str(graphrag_dir),
        "query": "test query",
        "method": "local",
        "api_key": None,
        "base_url": "http://nvidiaspark:8000/v1",
        "chat_model": "gemma-4-live",
        "embedding_base_url": "http://nvidiaspark:8001/v1",
        "embedding_model": "text-embedding-3-small",
    }
    assert "local graph context" in result["context_text"]
    assert result["sources"] == [{"source": "graphrag", "chunk_id": -1}]
    assert result["source_documents"] == ["local_doc.pdf"]
    assert result["source_document_details"][0]["graph_source_ids"] == ["3"]


def test_retrieve_context_keeps_graphrag_provenance_without_answer_text(
    dummy_kb,
    monkeypatch,
    tmp_path,
):
    index_path, meta_path, _ = dummy_kb
    graphrag_dir = tmp_path / "graphrag"
    graphrag_dir.mkdir()

    monkeypatch.setattr(
        "RAG.services.retrieval_service.query_graphrag_with_context",
        lambda root, query, *, method="global", max_chars=6000, **kwargs: (
            SimpleNamespace(
                answer_text=None,
                source_documents=["local_doc.pdf"],
                source_document_details=[
                    {
                        "source": "local_doc.pdf",
                        "chunk_ids": [],
                        "graph_source_ids": ["3"],
                        "graph_report_ids": [],
                        "graph_entity_ids": ["9"],
                        "graph_relationship_ids": [],
                        "graph_claim_ids": [],
                        "graph_text_unit_ids": ["tu-2"],
                        "evidence_count": 3,
                    }
                ],
            )
        ),
    )

    result = retrieve_context(
        query="test query",
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        api_key=None,
        base_url="http://nvidiaspark:8000/v1",
        embedding_base_url="http://nvidiaspark:8001/v1",
        retrieval_method="graphrag_local",
        chat_model="gemma-4-live",
    )

    assert result["context_text"] == ""
    assert result["source_documents"] == ["local_doc.pdf"]
    assert result["source_document_details"][0]["graph_source_ids"] == ["3"]
    assert result["sources"] == [{"source": "graphrag", "chunk_id": -1}]

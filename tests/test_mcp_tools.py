import pytest

from pydantic import ValidationError

from RAG.mcp.schemas import KBBuildInput, RetrieveAndAnswerInput, RetrieveContextInput
from RAG.mcp.tools import register_tools
from state.config import RETRIEVAL_METHODS


class FakeMCP:
    def __init__(self):
        self.registry = {}

    def tool(self):
        def _decorator(func):
            self.registry[func.__name__] = func
            return func

        return _decorator


def _build_registry():
    mcp = FakeMCP()
    register_tools(mcp)
    return mcp.registry


def test_retrieve_context_tool_passes_rerank_settings(monkeypatch):
    captured = {}

    def fake_retrieve_context(**kwargs):
        captured.update(kwargs)
        return {
            "context_text": "final context",
            "retrieved_context_text": "all retrieved context",
            "rerank_chunks": [
                {
                    "rank": 1,
                    "source": "doc.pdf",
                    "chunk_id": 7,
                    "score": 0.91,
                    "text": "best chunk",
                }
            ],
            "sources": [{"source": "doc.pdf", "chunk_id": 7}],
            "source_documents": ["doc.pdf"],
            "source_document_details": [
                {
                    "source": "doc.pdf",
                    "chunk_ids": [7],
                    "graph_source_ids": [],
                    "graph_report_ids": [],
                    "graph_entity_ids": [],
                    "graph_relationship_ids": [],
                    "graph_claim_ids": [],
                    "graph_text_unit_ids": [],
                    "evidence_count": 1,
                }
            ],
            "embedding_model": "vllm-sfr-embedding-mistral",
            "embedding_profile": "sfr",
        }

    monkeypatch.setattr("RAG.mcp.tools.retrieve_context", fake_retrieve_context)
    tools = _build_registry()
    params = RetrieveContextInput(
        query="q",
        index_path="index.faiss",
        meta_path="meta.pkl",
        rerank_base_url="http://localhost:8002/v1",
        rerank_model="nemotron-rerank-1b-v2",
        rerank_top_k=9,
        top_k_faiss=8,
        retrieval_method="faiss_mmr",
    )

    result = tools["retrieve_context_tool"](params)

    assert captured["rerank_base_url"] == "http://localhost:8002/v1"
    assert captured["rerank_model"] == "nemotron-rerank-1b-v2"
    assert captured["rerank_top_k"] == 9
    assert captured["top_k_faiss"] == 8
    assert result.retrieved_context_text == "all retrieved context"
    assert result.rerank_chunks[0].score == 0.91


def test_retrieve_and_answer_tool_uses_model_alias(monkeypatch):
    captured = {}
    captured_context = {}

    monkeypatch.setattr(
        "RAG.mcp.tools.retrieve_context",
        lambda **kwargs: (
            captured_context.update(kwargs)
            or {
                "context_text": "final context",
                "retrieved_context_text": "all retrieved context",
                "rerank_chunks": [],
                "sources": [],
                "source_documents": [],
                "source_document_details": [],
                "embedding_model": "text-embedding-3-small",
                "embedding_profile": "default",
            }
        ),
    )

    def fake_generate_answer(**kwargs):
        captured.update(kwargs)
        return "answer text"

    monkeypatch.setattr("RAG.mcp.tools.generate_answer", fake_generate_answer)
    tools = _build_registry()
    params = RetrieveAndAnswerInput(
        query="q",
        index_path="index.faiss",
        meta_path="meta.pkl",
        model="gemma-4-31b-it",
        system="You are helpful.",
        enable_web_search=False,
    )

    result = tools["retrieve_and_answer_tool"](params)

    assert captured["model"] == "gemma-4-31b-it"
    assert captured_context["chat_model"] == "gemma-4-31b-it"
    assert result.answer == "answer text"
    assert result.retrieved_context_text == "all retrieved context"


def test_kb_build_allows_embedding_only_route_when_graphrag_disabled(monkeypatch):
    captured = {}

    def fake_connection_is_configured(api_key, base_url):
        return base_url == "http://localhost:8001/v1"

    def fake_build_kb(**kwargs):
        captured.update(kwargs)
        return {
            "total_chunks": 1,
            "total_tokens": 10,
            "estimated_cost": 0.0,
            "warnings": [],
        }

    monkeypatch.setattr(
        "RAG.mcp.tools.connection_is_configured",
        fake_connection_is_configured,
    )
    monkeypatch.setattr(
        "RAG.mcp.tools._make_embeddings",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr("RAG.mcp.tools.build_kb", fake_build_kb)
    monkeypatch.setattr(
        "RAG.mcp.tools.make_openai_client",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Prompt client should not be created.")
        ),
    )
    tools = _build_registry()
    params = KBBuildInput(
        pdf_dir="pdfs",
        index_path="index.faiss",
        meta_path="meta.pkl",
        graphrag_dir="",
        embedding_model="Salesforce/SFR-Embedding-Mistral",
        embedding_profile="sfr",
        api_key="",
        base_url="",
        embedding_base_url="http://localhost:8001/v1",
        run_graphrag=False,
    )

    result = tools["kb_build"](params)

    assert captured["client"] is None
    assert result.total_chunks == 1


def test_kb_build_requires_prompt_route_when_graphrag_enabled(monkeypatch):
    def fake_connection_is_configured(api_key, base_url):
        return base_url == "http://localhost:8001/v1"

    monkeypatch.setattr(
        "RAG.mcp.tools.connection_is_configured",
        fake_connection_is_configured,
    )
    monkeypatch.setattr(
        "RAG.mcp.tools._make_embeddings",
        lambda *args, **kwargs: object(),
    )
    tools = _build_registry()
    params = KBBuildInput(
        pdf_dir="pdfs",
        index_path="index.faiss",
        meta_path="meta.pkl",
        graphrag_dir="graphrag",
        embedding_model="Salesforce/SFR-Embedding-Mistral",
        embedding_profile="sfr",
        api_key="",
        base_url="",
        embedding_base_url="http://localhost:8001/v1",
        run_graphrag=True,
    )

    try:
        tools["kb_build"](params)
    except ValueError as exc:
        assert "prompt model endpoint" in str(exc)
    else:
        raise AssertionError("Expected GraphRAG prompt-route validation to fail.")


def test_mcp_retrieval_inputs_expose_all_gui_methods():
    expected = set(RETRIEVAL_METHODS)

    for model in (RetrieveContextInput, RetrieveAndAnswerInput):
        schema = model.model_json_schema()
        ref = schema["properties"]["retrieval_method"]["$ref"].split("/")[-1]
        assert set(schema["$defs"][ref]["enum"]) == expected

        payload = {
            "query": "q",
            "index_path": "index.faiss",
            "meta_path": "meta.pkl",
        }
        if model is RetrieveAndAnswerInput:
            payload.update(
                {
                    "system": "You are helpful.",
                    "enable_web_search": False,
                }
            )

        for method in expected:
            validated = model.model_validate(
                {
                    **payload,
                    "retrieval_method": method,
                }
            )
            assert validated.retrieval_method.value == method

    with pytest.raises(ValidationError):
        RetrieveContextInput.model_validate(
            {
                "query": "q",
                "index_path": "index.faiss",
                "meta_path": "meta.pkl",
                "retrieval_method": "not_a_real_method",
            }
        )

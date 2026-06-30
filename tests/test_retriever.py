from types import SimpleNamespace

from langchain_core.documents import Document

from RAG.retrieval.retriever import QAContextRetriever


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def test_retriever_keeps_full_display_context_but_truncates_generation_context(
    monkeypatch,
):
    session_state = _SessionState(
        {
            "diversity": 0.7,
            "db": object(),
            "bm25": object(),
            "api_key": "EMPTY",
            "api_base_url": "http://localhost:8000/v1",
        "gpt_model": "gpt-4o-mini",
        "rerank_model": "nemotron-rerank-1b-v2",
            "rerank_api_base_url": "http://localhost:8002/v1",
            "rerank_top_k": 15,
            "upload_meta": [],
            "upload_images": [],
        }
    )

    monkeypatch.setattr(
        "RAG.retrieval.retriever.st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.retrieve_documents_with_method",
        lambda *args, **kwargs: [
            Document(
                page_content=f"doc {index}",
                metadata={
                    "source": "dummy.pdf",
                    "chunk_id": index,
                },
            )
            for index in range(20)
        ],
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.rerank_documents",
        lambda **kwargs: [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rerank_score": 1.0 - (index * 0.01),
                },
            )
            for index, document in enumerate(kwargs["documents"])
        ],
    )

    result = QAContextRetriever().retrieve(
        "test query",
        use_uploads=False,
        retrieval_method="faiss_mmr",
    )

    assert len(result["retrieved_context_text"].split("\n\n")) == 20
    assert len(result["context_text"].split("\n\n")) == 15
    assert len(result["rerank_chunks"]) == 15
    assert result["source_documents"] == ["dummy.pdf"]
    assert result["source_document_details"][0]["chunk_ids"] == list(range(15))


def test_retriever_routes_graphrag_global_queries(monkeypatch):
    session_state = _SessionState(
        {
            "diversity": 0.7,
            "upload_meta": [],
            "upload_images": [],
            "graphrag": "C:/graph",
            "api_key": "EMPTY",
            "api_base_url": "http://nvidiaspark:8000/v1",
            "embedding_api_base_url": "http://nvidiaspark:8001/v1",
            "gpt_model": "gemma-4-live",
            "embedding_model": "vllm-sfr-embedding-mistral",
        }
    )
    captured = {}

    monkeypatch.setattr(
        "RAG.retrieval.retriever.st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.os.path.isdir",
        lambda path: True,
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.query_graphrag_with_context",
        lambda root, query, *, method="global", max_chars=6000, **kwargs: (
            captured.update(
                {
                    "root": root,
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

    result = QAContextRetriever().retrieve(
        "test query",
        use_uploads=False,
        retrieval_method="graphrag_global",
    )

    assert captured == {
        "root": "C:/graph",
        "query": "test query",
        "method": "global",
        "api_key": "EMPTY",
        "base_url": "http://nvidiaspark:8000/v1",
        "chat_model": "gemma-4-live",
        "embedding_base_url": "http://nvidiaspark:8001/v1",
        "embedding_model": "vllm-sfr-embedding-mistral",
    }
    assert "global graph context" in result["context_text"]
    assert result["source_documents"] == ["graph_doc.pdf"]
    assert result["source_document_details"][0]["graph_report_ids"] == ["7"]


def test_retriever_routes_graphrag_local_queries(monkeypatch):
    session_state = _SessionState(
        {
            "diversity": 0.7,
            "upload_meta": [],
            "upload_images": [],
            "graphrag": "C:/graph",
            "api_key": "EMPTY",
            "api_base_url": "http://nvidiaspark:8000/v1",
            "embedding_api_base_url": "http://nvidiaspark:8001/v1",
            "gpt_model": "gemma-4-live",
            "embedding_model": "vllm-sfr-embedding-mistral",
        }
    )
    captured = {}

    monkeypatch.setattr(
        "RAG.retrieval.retriever.st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.os.path.isdir",
        lambda path: True,
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.query_graphrag_with_context",
        lambda root, query, *, method="global", max_chars=6000, **kwargs: (
            captured.update(
                {
                    "root": root,
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

    result = QAContextRetriever().retrieve(
        "test query",
        use_uploads=False,
        retrieval_method="graphrag_local",
    )

    assert captured == {
        "root": "C:/graph",
        "query": "test query",
        "method": "local",
        "api_key": "EMPTY",
        "base_url": "http://nvidiaspark:8000/v1",
        "chat_model": "gemma-4-live",
        "embedding_base_url": "http://nvidiaspark:8001/v1",
        "embedding_model": "vllm-sfr-embedding-mistral",
    }
    assert "local graph context" in result["context_text"]
    assert result["source_documents"] == ["local_doc.pdf"]
    assert result["source_document_details"][0]["graph_source_ids"] == ["3"]


def test_retriever_keeps_graphrag_provenance_without_answer_text(monkeypatch):
    session_state = _SessionState(
        {
            "diversity": 0.7,
            "upload_meta": [],
            "upload_images": [],
            "graphrag": "C:/graph",
            "api_key": "EMPTY",
            "api_base_url": "http://nvidiaspark:8000/v1",
            "embedding_api_base_url": "http://nvidiaspark:8001/v1",
            "gpt_model": "gemma-4-live",
            "embedding_model": "vllm-sfr-embedding-mistral",
        }
    )
    messages = []

    class _Callbacks:
        def info(self, msg):
            messages.append(msg)

    monkeypatch.setattr(
        "RAG.retrieval.retriever.st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.os.path.isdir",
        lambda path: True,
    )
    monkeypatch.setattr(
        "RAG.retrieval.retriever.query_graphrag_with_context",
        lambda *args, **kwargs: SimpleNamespace(
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
        ),
    )

    result = QAContextRetriever().retrieve(
        "test query",
        use_uploads=False,
        retrieval_method="graphrag_local",
        callbacks=_Callbacks(),
    )

    assert result["context_text"] == ""
    assert result["source_documents"] == ["local_doc.pdf"]
    assert result["source_document_details"][0]["graph_source_ids"] == ["3"]
    assert messages == ["GraphRAG returned source provenance, but no answer text."]

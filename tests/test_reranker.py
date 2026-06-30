from langchain_core.documents import Document

from RAG.retrieval.reranker import (
    keep_top_reranked_documents,
    rerank_documents,
    summarize_reranked_documents,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_rerank_documents_reorders_by_endpoint_response(monkeypatch):
    documents = [
        Document(page_content="doc zero", metadata={"source": "a", "chunk_id": 0}),
        Document(page_content="doc one", metadata={"source": "b", "chunk_id": 1}),
    ]

    captured = {}

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }
        )

    monkeypatch.setattr("RAG.retrieval.reranker.requests.post", _fake_post)

    reranked = rerank_documents(
        query="which doc is better?",
        documents=documents,
        model="nemotron-rerank-1b-v2",
        api_key="EMPTY",
        base_url="http://nvidiaspark:8002/v1/rerank",
    )

    assert [doc.page_content for doc in reranked] == ["doc one", "doc zero"]
    assert reranked[0].metadata["rerank_score"] == 0.9
    assert captured["url"] == "http://nvidiaspark:8002/v1/rerank"
    assert captured["json"]["model"] == "nemotron-rerank-1b-v2"
    assert captured["json"]["documents"] == ["doc zero", "doc one"]


def test_rerank_documents_returns_original_when_not_configured():
    documents = [Document(page_content="doc zero", metadata={})]

    assert rerank_documents(
        query="test",
        documents=documents,
        model="",
        base_url=None,
    ) == documents


def test_summarize_reranked_documents_keeps_order_and_scores():
    documents = [
        Document(
            page_content="kept first",
            metadata={"source": "a.pdf", "chunk_id": 2, "rerank_score": 0.91},
        ),
        Document(
            page_content="no score",
            metadata={"source": "b.pdf", "chunk_id": 7},
        ),
        Document(
            page_content="kept second",
            metadata={"source": "c.pdf", "chunk_id": 1, "rerank_score": 0.42},
        ),
    ]

    summarized = summarize_reranked_documents(documents=documents)

    assert summarized == [
        {
            "rank": 1,
            "source": "a.pdf",
            "chunk_id": 2,
            "score": 0.91,
            "text": "kept first",
        },
        {
            "rank": 2,
            "source": "c.pdf",
            "chunk_id": 1,
            "score": 0.42,
            "text": "kept second",
        },
    ]


def test_keep_top_reranked_documents_truncates_to_requested_count():
    documents = [
        Document(page_content="doc zero", metadata={"rerank_score": 0.9}),
        Document(page_content="doc one", metadata={"rerank_score": 0.8}),
        Document(page_content="doc two", metadata={"rerank_score": 0.7}),
    ]

    kept = keep_top_reranked_documents(documents=documents, top_k=2)

    assert [doc.page_content for doc in kept] == ["doc zero", "doc one"]

from ingestion.embedding_batching import (
    ChunkRecord,
    build_chunk_records,
    embed_chunk_records,
)


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(embedding) for embedding in embeddings]


class _RecordingEmbeddingsApi:
    def __init__(self, *, fail_on_batch=False, fail_on_texts=None):
        self.fail_on_batch = fail_on_batch
        self.fail_on_texts = set(fail_on_texts or [])
        self.calls = []

    def create(self, *, input, model):
        texts = list(input) if isinstance(input, list) else [input]
        self.calls.append({"model": model, "texts": texts})
        if self.fail_on_batch and len(texts) > 1:
            raise RuntimeError("batch requests are unavailable")
        for text in texts:
            if text in self.fail_on_texts:
                raise RuntimeError("bad chunk")
        return _FakeEmbeddingResponse(
            [[float(index), float(len(text))] for index, text in enumerate(texts)]
        )


class _FakeEmbeddingClient:
    def __init__(self, *, fail_on_batch=False, fail_on_texts=None):
        self.embeddings = _RecordingEmbeddingsApi(
            fail_on_batch=fail_on_batch,
            fail_on_texts=fail_on_texts,
        )


def test_build_chunk_records_preserves_file_and_chunk_order():
    records = build_chunk_records(
        chunks_by_file={
            "first.pdf": ["a", "b"],
            "second.pdf": ["c"],
        }
    )

    assert records == [
        ChunkRecord(source="first.pdf", chunk_id=0, text="a"),
        ChunkRecord(source="first.pdf", chunk_id=1, text="b"),
        ChunkRecord(source="second.pdf", chunk_id=0, text="c"),
    ]


def test_embed_chunk_records_batches_requests():
    records = [
        ChunkRecord(source="doc.pdf", chunk_id=index, text=f"chunk-{index}")
        for index in range(65)
    ]
    progress = []
    client = _FakeEmbeddingClient()

    embedded = embed_chunk_records(
        chunk_records=records,
        embedding_client=client,
        model_name="test-model",
        profile_name="default",
        batch_size=64,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert len(embedded) == 65
    assert [len(call["texts"]) for call in client.embeddings.calls] == [64, 1]
    assert progress[-1] == (65, 65)


def test_embed_chunk_records_falls_back_to_single_requests():
    records = [
        ChunkRecord(source="doc.pdf", chunk_id=0, text="good-a"),
        ChunkRecord(source="doc.pdf", chunk_id=1, text="bad"),
        ChunkRecord(source="doc.pdf", chunk_id=2, text="good-b"),
    ]
    warnings = []
    client = _FakeEmbeddingClient(
        fail_on_batch=True,
        fail_on_texts={"bad"},
    )

    embedded = embed_chunk_records(
        chunk_records=records,
        embedding_client=client,
        model_name="test-model",
        profile_name="default",
        batch_size=3,
        warning_callback=warnings.append,
    )

    assert len(embedded) == 2
    assert len(client.embeddings.calls) == 4
    assert warnings == ["Embedding failed: doc.pdf, chunk 1 -> bad chunk"]

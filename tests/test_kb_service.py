from pathlib import Path

import pytest
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from RAG.services.kb_service import (
    GraphRAGWorkspaceError,
    _append_graphrag,
    append_kb,
    build_kb,
    load_kb,
)
from state.config import ENC


def _write_pdf(path: Path, text: str):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_kb_dummy(dummy_kb):
    index_path, meta_path, _ = dummy_kb
    result = load_kb(
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=None,
    )
    assert result["status"] == "ok"
    assert result["chunk_count"] == 1


def test_append_graphrag_uses_incremental_update_mode(tmp_path, monkeypatch):
    workspace = tmp_path / "graphrag"
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir()
    (workspace / "settings.yaml").write_text("models: {}\n", encoding="utf-8")

    captured = {}

    def fake_run_graphrag_cli(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        "RAG.services.kb_service.run_graphrag_cli",
        fake_run_graphrag_cli,
    )

    _append_graphrag(
        {"doc1.pdf": "new graph text"},
        str(workspace),
        "test-key",
        "http://nvidiaspark:8000/v1",
        "http://nvidiaspark:8001/v1",
        "gemma-4-31b-it",
        "vllm-sfr-embedding-mistral",
    )

    assert captured["kwargs"]["mode"] == "update"
    assert (input_dir / "doc1.txt").read_text(encoding="utf-8") == "new graph text"


def test_append_graphrag_raises_when_incremental_update_fails(tmp_path, monkeypatch):
    workspace = tmp_path / "graphrag"
    (workspace / "input").mkdir(parents=True)
    (workspace / "output").mkdir()
    (workspace / "settings.yaml").write_text("models: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "RAG.services.kb_service.run_graphrag_cli",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(
        GraphRAGWorkspaceError,
        match="GraphRAG incremental update failed",
    ):
        _append_graphrag(
            {"doc1.pdf": "new graph text"},
            str(workspace),
            "test-key",
            "http://nvidiaspark:8000/v1",
            "http://nvidiaspark:8001/v1",
            "gemma-4-31b-it",
            "vllm-sfr-embedding-mistral",
        )


@pytest.mark.integration
@pytest.mark.requires_openai
def test_kb_build_integration(tmp_path, openai_key):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "doc1.pdf", "Smoke test document for KB build.")

    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.pkl"
    graphrag_dir = tmp_path / "graphrag"

    client = OpenAI(api_key=openai_key)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
    )

    result = build_kb(
        client=client,
        embeddings=embeddings,
        enc=ENC,
        pdf_dir=str(pdf_dir),
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        embedding_model="text-embedding-3-small",
        run_graphrag=False,
        api_key=openai_key,
    )

    assert index_path.is_file()
    assert meta_path.is_file()
    assert result["total_chunks"] > 0


@pytest.mark.integration
@pytest.mark.requires_openai
def test_kb_append_integration(tmp_path, openai_key):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "doc1.pdf", "Initial KB document.")

    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.pkl"
    graphrag_dir = tmp_path / "graphrag"

    client = OpenAI(api_key=openai_key)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
    )

    build_kb(
        client=client,
        embeddings=embeddings,
        enc=ENC,
        pdf_dir=str(pdf_dir),
        index_path=str(index_path),
        meta_path=str(meta_path),
        graphrag_dir=str(graphrag_dir),
        embedding_model="text-embedding-3-small",
        run_graphrag=False,
        api_key=openai_key,
    )

    append_dir = tmp_path / "append"
    append_dir.mkdir()
    _write_pdf(append_dir / "doc2.pdf", "Appended KB document.")

    result = append_kb(
        client=client,
        enc=ENC,
        index_path=str(index_path),
        meta_path=str(meta_path),
        append_folder=str(append_dir),
        graphrag_dir=str(graphrag_dir),
        run_graphrag=False,
        api_key=openai_key,
    )

    assert result["new_chunks"] > 0

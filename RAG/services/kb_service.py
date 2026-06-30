import pickle
from pathlib import Path

import faiss
import numpy as np

from ingestion.chunkers import chunk_text2
from ingestion.cleaners import remove_junk_lines, remove_junk_sections
from ingestion.embedding_batching import build_chunk_records, embed_chunk_records
from ingestion.embedding_profiles import (
    create_embedding_function,
    resolve_embedding_profile_name,
)
from ingestion.embeddings import estimate_embedding_cost
from ingestion.faiss_store import build_faiss_from_embeddings
from ingestion.graphrag import run_graphrag_cli
from ingestion.loaders import extract_text_from_pdf
from RAG.openai_compat import connection_is_configured, make_openai_client
from RAG.retrieval.kb_builder import (
    GraphRAGWorkspaceError,
    KnowledgeBaseAppendError,
    KnowledgeBaseBuildError,
    KnowledgeBaseLoadError,
    _write_graphrag_input_files,
)
from state.config import (
    EMBEDDING_DIMENSIONS,
    KB_EMBEDDING_BATCH_SIZE,
    TOKENS_PER_CHUNK,
    WORDS_PER_CHUNK_OVERLAP,
)
from state.schemas import ChunkMetadata


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _report_status(callbacks, phase: str, message: str):
    """Update a long-running phase status, falling back to info output."""
    status_cb = _get_cb(callbacks, "status")
    if status_cb:
        status_cb(phase, message)
        return
    _call(_get_cb(callbacks, "info"), message)


def _process_pdf(path: Path, enc):
    text = extract_text_from_pdf(path)
    text = remove_junk_lines(remove_junk_sections(text))
    chunks = chunk_text2(
        text,
        max_tokens=TOKENS_PER_CHUNK,
        tokenizer=enc,
        overlap=WORDS_PER_CHUNK_OVERLAP,
    )
    return text, chunks


def _resolve_embedding_dimension(metadata, index=None):
    if metadata:
        dimension = metadata[0].get("embedding_dimension")
        if dimension:
            return int(dimension)

        model = metadata[0].get("embedding_model")
        if model in EMBEDDING_DIMENSIONS:
            return EMBEDDING_DIMENSIONS[model]

    if index is not None and getattr(index, "d", None):
        return int(index.d)

    raise KnowledgeBaseLoadError(
        "Could not determine the embedding dimension from the knowledge base metadata."
    )


def _run_graphrag(
    texts,
    graphrag_dir: str,
    api_key: str | None,
    base_url: str | None,
    embedding_base_url: str | None,
    chat_model: str | None,
    embedding_model: str,
    warnings,
    callbacks=None,
):
    if not graphrag_dir:
        return

    if not connection_is_configured(api_key, base_url):
        warnings.append(
            "Skipping GraphRAG indexing: no model endpoint was configured."
        )
        return

    root = Path(graphrag_dir)
    input_dir = root / "input"
    (root / "output").mkdir(parents=True, exist_ok=True)
    _write_graphrag_input_files(texts=texts, input_dir=input_dir)

    ok = run_graphrag_cli(
        root,
        input_dir,
        root / "output",
        api_key=api_key,
        base_url=base_url,
        embedding_base_url=embedding_base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        status_callback=lambda message: _report_status(
            callbacks,
            "graphrag",
            message,
        ),
    )
    if not ok:
        warnings.append("GraphRAG indexing failed; see logs for details.")


def _append_graphrag(
    texts,
    graphrag_dir: str,
    api_key: str | None,
    base_url: str | None,
    embedding_base_url: str | None,
    chat_model: str | None,
    embedding_model: str,
    callbacks=None,
):
    if not graphrag_dir:
        return

    if not connection_is_configured(api_key, base_url):
        raise GraphRAGWorkspaceError(
            "GraphRAG update requested but no model endpoint was configured."
        )

    root = Path(graphrag_dir)
    input_dir = root / "input"
    output_dir = root / "output"
    if not input_dir.is_dir() or not output_dir.is_dir():
        raise GraphRAGWorkspaceError(
            "GraphRAG workspace is missing input/output folders."
        )
    settings_path = root / "settings.yaml"
    if not settings_path.is_file():
        raise GraphRAGWorkspaceError(
            f"GraphRAG workspace is missing settings.yaml at {settings_path}."
        )

    _write_graphrag_input_files(texts=texts, input_dir=input_dir)

    ok = run_graphrag_cli(
        root,
        input_dir,
        output_dir,
        api_key=api_key,
        base_url=base_url,
        embedding_base_url=embedding_base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        mode="update",
        status_callback=lambda message: _report_status(
            callbacks,
            "graphrag",
            message,
        ),
    )
    if not ok:
        raise GraphRAGWorkspaceError(
            "GraphRAG incremental update failed; see logs for details."
        )


def build_kb(
    *,
    client,
    embeddings,
    enc,
    pdf_dir: str,
    index_path: str,
    meta_path: str,
    graphrag_dir: str | None,
    embedding_model: str,
    embedding_profile: str | None = None,
    run_graphrag: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_base_url: str | None = None,
    chat_model: str | None = None,
    callbacks=None,
):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise KnowledgeBaseBuildError("No PDFs found in the provided directory.")

    texts = {}
    chunks = {}
    warnings = []
    total_chunks = 0
    total_tokens = 0
    total_cost = 0.0

    _call(_get_cb(callbacks, "info"), "Processing PDFs...")
    for i, pdf in enumerate(pdf_files):
        try:
            text, cks = _process_pdf(pdf, enc)
            texts[pdf.name] = text
            chunks[pdf.name] = cks

            token_count = sum(len(enc.encode(c)) for c in cks)
            total_cost += estimate_embedding_cost(token_count, embedding_model)
            total_chunks += len(cks)
            total_tokens += token_count
        except Exception as exc:
            warnings.append(f"Failed on {pdf.name}: {exc}")
        _call(
            _get_cb(callbacks, "progress"),
            "processing_pdfs",
            i + 1,
            max(len(pdf_files), 1),
        )

    if total_chunks == 0:
        raise KnowledgeBaseBuildError("No chunks produced; aborting.")

    embeddings_list = []
    metadata = []
    resolved_profile = resolve_embedding_profile_name(
        profile_name=embedding_profile,
        model_name=embedding_model,
    )
    _call(_get_cb(callbacks, "info"), "Embedding and indexing...")
    embedding_client = make_openai_client(
        api_key=api_key,
        base_url=embedding_base_url,
        fallback_to_env=False,
    )
    chunk_records = build_chunk_records(chunks_by_file=chunks)
    embedded_records = embed_chunk_records(
        chunk_records=chunk_records,
        embedding_client=embedding_client,
        model_name=embedding_model,
        profile_name=resolved_profile,
        warning_callback=warnings.append,
        progress_callback=lambda completed, total: _call(
            _get_cb(callbacks, "progress"),
            "embedding",
            completed,
            total,
        ),
        batch_size=KB_EMBEDDING_BATCH_SIZE,
    )
    for record, emb in embedded_records:
        embeddings_list.append(emb)
        meta = ChunkMetadata(
            source=record.source,
            chunk_id=record.chunk_id,
            text=record.text,
            embedding_model=embedding_model,
            embedding_profile=resolved_profile,
            embedding_dimension=len(emb),
        )
        metadata.append(meta.model_dump())

    if not embeddings_list:
        raise KnowledgeBaseBuildError("No embeddings computed; aborting.")

    dim = len(embeddings_list[0])
    embedding_fn = embeddings
    if getattr(embeddings, "profile_name", None) != resolved_profile:
        embedding_fn = create_embedding_function(
            model=embedding_model,
            profile_name=resolved_profile,
            api_key=api_key,
            base_url=embedding_base_url,
            fallback_to_env=False,
        )

    index, _, _ = build_faiss_from_embeddings(
        embeddings_list, metadata, embedding_fn, dim
    )

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    if run_graphrag and graphrag_dir:
        _run_graphrag(
            texts,
            graphrag_dir,
            api_key,
            base_url,
            embedding_base_url,
            chat_model,
            embedding_model,
            warnings,
            callbacks=callbacks,
        )

    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "estimated_cost": total_cost,
        "warnings": warnings,
    }


def load_kb(*, index_path: str, meta_path: str, graphrag_dir: str | None):
    index_file = Path(index_path)
    meta_file = Path(meta_path)

    if not index_file.is_file():
        raise KnowledgeBaseLoadError(f"FAISS index file not found at {index_path}.")
    if not meta_file.is_file():
        raise KnowledgeBaseLoadError(f"Metadata file not found at {meta_path}.")

    try:
        index = faiss.read_index(str(index_file))
    except Exception as exc:
        raise KnowledgeBaseLoadError(
            f"Failed to read FAISS index at {index_path}."
        ) from exc

    try:
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
    except Exception as exc:
        raise KnowledgeBaseLoadError(
            f"Failed to load metadata file at {meta_path}."
        ) from exc

    if not metadata:
        raise KnowledgeBaseLoadError(f"Metadata file is empty at {meta_path}.")

    model = metadata[0].get("embedding_model")
    if not model:
        raise KnowledgeBaseLoadError(
            f"Missing 'embedding_model' in metadata at {meta_path}."
        )

    dim = _resolve_embedding_dimension(metadata, index)

    return {
        "status": "ok",
        "embedding_model": model,
        "embedding_profile": metadata[0].get("embedding_profile")
        or resolve_embedding_profile_name(model_name=model),
        "dimension": dim,
        "chunk_count": len(metadata),
        "graphrag_dir": graphrag_dir or "",
    }


def append_kb(
    *,
    client,
    enc,
    index_path: str,
    meta_path: str,
    append_folder: str,
    graphrag_dir: str | None,
    run_graphrag: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_base_url: str | None = None,
    chat_model: str | None = None,
    callbacks=None,
):
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    append_folder = Path(append_folder)

    if not index_path.is_file():
        raise KnowledgeBaseAppendError("Existing FAISS index file not found.")
    if not meta_path.is_file():
        raise KnowledgeBaseAppendError("Existing metadata file not found.")
    if not append_folder.is_dir():
        raise KnowledgeBaseAppendError("Append folder does not exist.")

    pdf_files = list(append_folder.glob("*.pdf"))
    if not pdf_files:
        raise KnowledgeBaseAppendError("No PDFs found in the append folder.")

    try:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata_existing = pickle.load(f)
    except Exception as exc:
        raise KnowledgeBaseAppendError(
            "Failed to load existing knowledge base."
        ) from exc

    existing_model = metadata_existing[0].get("embedding_model")
    if not existing_model:
        raise KnowledgeBaseAppendError("Missing embedding model in metadata.")
    existing_profile = metadata_existing[0].get("embedding_profile") or (
        resolve_embedding_profile_name(model_name=existing_model)
    )
    dim = _resolve_embedding_dimension(metadata_existing, index)

    warnings = []
    new_chunks_by_file = {}
    all_texts = {}
    total_new_chunks = 0
    total_new_tokens = 0
    total_new_cost = 0.0

    _call(_get_cb(callbacks, "info"), "Processing NEW PDFs to append...")
    for i, pdf_path in enumerate(pdf_files):
        try:
            text, chunks = _process_pdf(pdf_path, enc)
            all_texts[pdf_path.name] = text
            new_chunks_by_file[pdf_path.name] = chunks
            token_count = sum(len(enc.encode(c)) for c in chunks)
            total_new_cost += estimate_embedding_cost(token_count, existing_model)
            total_new_chunks += len(chunks)
            total_new_tokens += token_count
        except Exception as exc:
            warnings.append(f"Failed on {pdf_path.name}: {exc}")
        _call(
            _get_cb(callbacks, "progress"),
            "processing_new_pdfs",
            i + 1,
            len(pdf_files),
        )

    if total_new_chunks == 0:
        raise KnowledgeBaseAppendError(
            "No new chunks were produced from the append PDFs."
        )

    _call(_get_cb(callbacks, "info"), "Embedding and appending to FAISS...")
    new_embeddings = []
    new_metadata = []
    embedding_client = make_openai_client(
        api_key=api_key,
        base_url=embedding_base_url,
        fallback_to_env=False,
    )
    chunk_records = build_chunk_records(chunks_by_file=new_chunks_by_file)
    embedded_records = embed_chunk_records(
        chunk_records=chunk_records,
        embedding_client=embedding_client,
        model_name=existing_model,
        profile_name=existing_profile,
        warning_callback=warnings.append,
        progress_callback=lambda completed, total: _call(
            _get_cb(callbacks, "progress"),
            "embedding_append",
            completed,
            total,
        ),
        batch_size=KB_EMBEDDING_BATCH_SIZE,
    )
    for record, embedding in embedded_records:
        if len(embedding) != dim:
            raise KnowledgeBaseAppendError(
                "Embedding dimension mismatch. Aborting."
            )
        new_embeddings.append(embedding)
        meta = ChunkMetadata(
            source=record.source,
            chunk_id=record.chunk_id,
            text=record.text,
            embedding_model=existing_model,
            embedding_profile=existing_profile,
            embedding_dimension=dim,
        )
        new_metadata.append(meta.model_dump())

    if not new_embeddings:
        raise KnowledgeBaseAppendError("No new embeddings computed; aborting.")

    new_mat = np.array(new_embeddings, dtype="float32")
    if new_mat.shape[1] != dim:
        raise KnowledgeBaseAppendError("Embedding dimension mismatch. Aborting.")

    try:
        index.add(new_mat)
    except Exception as exc:
        raise KnowledgeBaseAppendError("Failed to append vectors to FAISS.") from exc

    updated_metadata = metadata_existing + new_metadata
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(updated_metadata, f)

    if run_graphrag and graphrag_dir:
        try:
            _append_graphrag(
                all_texts,
                graphrag_dir,
                api_key,
                base_url,
                embedding_base_url,
                chat_model,
                existing_model,
                callbacks=callbacks,
            )
        except GraphRAGWorkspaceError as exc:
            warnings.append(str(exc))

    return {
        "new_chunks": total_new_chunks,
        "new_tokens": total_new_tokens,
        "estimated_cost": total_new_cost,
        "warnings": warnings,
    }

import pickle
from contextlib import nullcontext
from pathlib import Path

import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever

from ingestion.chunkers import chunk_text2
from ingestion.cleaners import remove_junk_lines, remove_junk_sections
from ingestion.embedding_batching import build_chunk_records, embed_chunk_records
from ingestion.embedding_profiles import (
    create_embedding_function,
    resolve_embedding_profile_name,
)
from ingestion.embeddings import estimate_embedding_cost
from ingestion.faiss_store import build_faiss_from_embeddings, load_faiss_from_disk
from ingestion.graphrag import run_graphrag_cli
from ingestion.loaders import extract_text_from_pdf
from RAG.openai_compat import (
    connection_is_configured,
    make_openai_client,
)
from state.config import (
    EMBEDDING_DIMENSIONS,
    KB_EMBEDDING_BATCH_SIZE,
    TOKENS_PER_CHUNK,
    TOP_K_TEXT_BM25,
    WORDS_PER_CHUNK_OVERLAP,
)
from state.schemas import ChunkMetadata


class KnowledgeBaseError(Exception):
    """Base error for knowledge-base operations."""


class KnowledgeBaseBuildError(KnowledgeBaseError):
    """Raised when building a knowledge base fails."""


class KnowledgeBaseLoadError(KnowledgeBaseError):
    """Raised when loading a knowledge base fails."""


class KnowledgeBaseAppendError(KnowledgeBaseError):
    """Raised when appending to a knowledge base fails."""


class GraphRAGWorkspaceError(KnowledgeBaseError):
    """Raised when GraphRAG workspace is invalid or missing."""


def _get_cb(callbacks, name):
    if callbacks is None:
        return None
    return getattr(callbacks, name, None)


def _call(cb, *args, **kwargs):
    if cb:
        cb(*args, **kwargs)


def _spinner(callbacks, message):
    cb = _get_cb(callbacks, "spinner")
    if cb:
        return cb(message)
    return nullcontext()


def _report_status(callbacks, phase: str, message: str):
    """Update a long-running phase status, falling back to info output."""
    status_cb = _get_cb(callbacks, "status")
    if status_cb:
        status_cb(phase, message)
        return
    _call(_get_cb(callbacks, "info"), message)


def _write_graphrag_input_files(*, texts: dict[str, str], input_dir: Path):
    """Write GraphRAG source texts as UTF-8 input files.

    Args:
        texts: Mapping of source filenames to extracted text content.
        input_dir: Target GraphRAG input directory.

    Raises:
        GraphRAGWorkspaceError: When an input file cannot be written.
    """
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GraphRAGWorkspaceError(
            f"Failed to create GraphRAG input directory at {input_dir}."
        ) from exc
    for name, text in texts.items():
        target = input_dir / f"{Path(name).stem}.txt"
        try:
            target.write_text(text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise GraphRAGWorkspaceError(
                f"Failed to write GraphRAG input file at {target} using UTF-8."
            ) from exc


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


def _resolve_embedding_profile(metadata, model_name: str | None = None):
    """Resolve the embedding formatting profile for a knowledge base."""
    if metadata:
        stored = metadata[0].get("embedding_profile")
        if stored:
            return stored
        if metadata[0].get("embedding_model"):
            model_name = metadata[0]["embedding_model"]
    return resolve_embedding_profile_name(model_name=model_name)


class KnowledgeBaseBuilder:
    """Build, load, and register a knowledge base."""

    def __init__(self, client, embeddings=None):
        self.client = client
        self.embeddings = embeddings
        self._embedding_cache = {}
        import streamlit as st

        self._st = st
        self.enc = st.session_state.enc

    def _embedding_function(self, model, profile_name=None):
        resolved_profile = resolve_embedding_profile_name(
            profile_name=profile_name,
            model_name=model,
        )
        cache_key = (model, resolved_profile)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        cached_model = getattr(self.embeddings, "model", None)
        cached_profile = getattr(self.embeddings, "profile_name", None)
        if (
            self.embeddings is not None
            and cached_model == model
            and cached_profile == resolved_profile
        ):
            self._embedding_cache[cache_key] = self.embeddings
            return self.embeddings

        embedding_fn = create_embedding_function(
            model=model,
            profile_name=resolved_profile,
            api_key=self._st.session_state.api_key,
            base_url=self._st.session_state.get("embedding_api_base_url"),
            fallback_to_env=False,
        )
        self._embedding_cache[cache_key] = embedding_fn
        return embedding_fn

    def _embedding_client(self):
        return make_openai_client(
            api_key=self._st.session_state.api_key,
            base_url=self._st.session_state.get("embedding_api_base_url"),
            fallback_to_env=False,
        )

    def build(self, pdf_dir, index_path, meta_path, graphrag_dir, model, callbacks=None):
        texts, chunks = {}, {}
        total_chunks = 0
        total_tokens = 0
        total_cost = 0.0
        embeddings, metadata = [], []
        embedding_profile = resolve_embedding_profile_name(
            profile_name=self._st.session_state.get("embedding_profile"),
            model_name=model,
        )

        _call(_get_cb(callbacks, "info"), "Processing PDFs...")
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))

        for i, pdf in enumerate(pdf_files):
            try:
                text, cks = self._process_pdf(pdf)
                texts[pdf.name] = text
                chunks[pdf.name] = cks

                token_count = sum(len(self.enc.encode(c)) for c in cks)
                total_cost += estimate_embedding_cost(token_count, model)
                total_chunks += len(cks)
                total_tokens += token_count
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"Failed on {pdf.name}: {exc}",
                )
            _call(
                _get_cb(callbacks, "progress"),
                "processing_pdfs",
                i + 1,
                max(len(pdf_files), 1),
            )

        if total_chunks == 0:
            raise KnowledgeBaseBuildError("No chunks produced; aborting.")

        _call(
            _get_cb(callbacks, "info"),
            (
                f"Total chunks: {total_chunks:,}, "
                f"Total tokens: {total_tokens:,}, "
                f"Est. cost: ${total_cost:.4f}"
            ),
        )

        _call(_get_cb(callbacks, "info"), "Embedding and indexing...")
        embedding_client = self._embedding_client()
        chunk_records = build_chunk_records(chunks_by_file=chunks)
        embedded_records = embed_chunk_records(
            chunk_records=chunk_records,
            embedding_client=embedding_client,
            model_name=model,
            profile_name=embedding_profile,
            warning_callback=lambda message: _call(
                _get_cb(callbacks, "warning"),
                message,
            ),
            progress_callback=lambda completed, total: _call(
                _get_cb(callbacks, "progress"),
                "embedding",
                completed,
                total,
            ),
            batch_size=KB_EMBEDDING_BATCH_SIZE,
        )
        for record, emb in embedded_records:
            embeddings.append(emb)
            meta = ChunkMetadata(
                source=record.source,
                chunk_id=record.chunk_id,
                text=record.text,
                embedding_model=model,
                embedding_profile=embedding_profile,
                embedding_dimension=len(emb),
            )
            metadata.append(meta.model_dump())

        if not embeddings:
            raise KnowledgeBaseBuildError("No embeddings computed; aborting.")

        dim = len(embeddings[0])
        index, db, docs = build_faiss_from_embeddings(
            embeddings,
            metadata,
            self._embedding_function(model, embedding_profile),
            dim,
        )

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        self._run_graphrag(texts, graphrag_dir, model, callbacks=callbacks)
        self._register(index, metadata, db, docs, graphrag_dir)

    def load(self, index_path, meta_path, graphrag_dir):
        """Load an existing knowledge base."""
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            model = metadata[0]["embedding_model"]
            profile_name = _resolve_embedding_profile(metadata, model)
            db, docs = load_faiss_from_disk(
                index,
                metadata,
                self._embedding_function(model, profile_name),
            )
            self._register(index, metadata, db, docs, graphrag_dir)

        except Exception as exc:
            raise KnowledgeBaseLoadError("Failed to load knowledge base.") from exc

    def append(self, index_path, meta_path, append_folder, graphrag_dir, callbacks=None):
        """Append new PDFs to an existing knowledge base."""
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

        existing_model = metadata_existing[0].get(
            "embedding_model", self._st.session_state.get("embedding_model")
        )
        existing_profile = _resolve_embedding_profile(
            metadata_existing,
            existing_model,
        )
        self._st.session_state.embedding_model = existing_model
        self._st.session_state.embedding_profile = existing_profile
        dim = _resolve_embedding_dimension(metadata_existing, index)

        _call(_get_cb(callbacks, "info"), "Processing NEW PDFs to append...")
        new_chunks_by_file = {}
        all_texts = {}
        total_new_chunks = 0
        total_new_tokens = 0
        total_new_cost = 0.0

        for i, pdf_path in enumerate(pdf_files):
            try:
                text, chunks = self._process_pdf(pdf_path)
                all_texts[pdf_path.name] = text
                new_chunks_by_file[pdf_path.name] = chunks
                token_count = sum(len(self.enc.encode(c)) for c in chunks)
                total_new_cost += estimate_embedding_cost(token_count, existing_model)
                total_new_chunks += len(chunks)
                total_new_tokens += token_count
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"Failed on {pdf_path.name}: {exc}",
                )
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

        _call(
            _get_cb(callbacks, "info"),
            (
                f"New chunks: {total_new_chunks:,}, "
                f"New tokens: {total_new_tokens:,}, "
                f"Est. cost: ${total_new_cost:.4f}"
            ),
        )

        _call(_get_cb(callbacks, "info"), "Embedding and appending to FAISS...")
        new_embeddings = []
        new_metadata = []
        embedding_client = self._embedding_client()
        chunk_records = build_chunk_records(chunks_by_file=new_chunks_by_file)
        embedded_records = embed_chunk_records(
            chunk_records=chunk_records,
            embedding_client=embedding_client,
            model_name=existing_model,
            profile_name=existing_profile,
            warning_callback=lambda message: _call(
                _get_cb(callbacks, "warning"),
                message,
            ),
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

        db, docs = load_faiss_from_disk(
            index,
            updated_metadata,
            self._embedding_function(existing_model, existing_profile),
        )
        self._register(index, updated_metadata, db, docs, graphrag_dir)

        with _spinner(callbacks, "Updating Knowledge-Graph with new documents..."):
            try:
                self._append_graphrag(
                    all_texts,
                    graphrag_dir,
                    existing_model,
                    callbacks=callbacks,
                )
                _call(
                    _get_cb(callbacks, "success"),
                    "Knowledge-Graph updated successfully.",
                )
            except Exception as exc:
                _call(
                    _get_cb(callbacks, "warning"),
                    f"Knowledge-Graph update failed: {exc}",
                )

    def _process_pdf(self, path):
        text = extract_text_from_pdf(path)
        text = remove_junk_lines(remove_junk_sections(text))
        chunks = chunk_text2(
            text,
            max_tokens=TOKENS_PER_CHUNK,
            tokenizer=self.enc,
            overlap=WORDS_PER_CHUNK_OVERLAP,
        )
        return text, chunks

    def _run_graphrag(self, texts, root, embedding_model, callbacks=None):
        if not root:
            return
        if not connection_is_configured(
            self._st.session_state.api_key,
            self._st.session_state.get("api_base_url"),
        ):
            return

        root = Path(root)
        input_dir = root / "input"
        (root / "output").mkdir(parents=True, exist_ok=True)
        _write_graphrag_input_files(texts=texts, input_dir=input_dir)

        run_graphrag_cli(
            root,
            input_dir,
            root / "output",
            api_key=self._st.session_state.api_key,
            base_url=self._st.session_state.get("api_base_url"),
            embedding_base_url=self._st.session_state.get("embedding_api_base_url"),
            chat_model=self._st.session_state.get("gpt_model"),
            embedding_model=embedding_model,
            status_callback=lambda message: _report_status(
                callbacks,
                "graphrag",
                message,
            ),
        )

    def _append_graphrag(self, texts, root, embedding_model, callbacks=None):
        """Append new docs to an existing GraphRAG workspace."""
        if not root:
            return
        if not connection_is_configured(
            self._st.session_state.api_key,
            self._st.session_state.get("api_base_url"),
        ):
            raise GraphRAGWorkspaceError(
                "GraphRAG update requested but no model endpoint was configured."
            )

        root = Path(root)
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
            api_key=self._st.session_state.api_key,
            base_url=self._st.session_state.get("api_base_url"),
            embedding_base_url=self._st.session_state.get("embedding_api_base_url"),
            chat_model=self._st.session_state.get("gpt_model"),
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

    def _register(self, index, metadata, db, docs, graphrag_dir):
        self._st.session_state.update(
            index=index,
            metadata=metadata,
            db=db,
            all_documents=docs,
            bm25=BM25Retriever.from_documents(docs, k=TOP_K_TEXT_BM25),
            embedding_model=metadata[0]["embedding_model"],
            embedding_profile=_resolve_embedding_profile(metadata),
            dimension=_resolve_embedding_dimension(metadata, index),
            graphrag=graphrag_dir,
        )

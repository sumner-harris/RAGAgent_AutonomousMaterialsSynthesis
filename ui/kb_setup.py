import os
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from RAG.retrieval.kb_builder import KnowledgeBaseBuilder, KnowledgeBaseError
from state.config import (
    DEFAULT_EMBEDDING_MODEL_OPTIONS,
    DEFAULT_EMBEDDING_PROFILE,
    EMBEDDING_PROFILE_LABELS,
    EMBEDDING_PROFILE_OPTIONS,
    SFR_EMBEDDING_MODEL,
    SFR_EMBEDDING_PROFILE,
)
from state.schemas import KBAppendRequest, KBBuildRequest, KBLoadRequest


def _dequote_path(path):
    """Strip accidental shell quotes."""
    if path is None:
        return path
    return path.strip().strip('"').strip("'")


def _embedding_model_selector():
    """Select the embedding model for OpenAI or a custom embedding route."""
    current = st.session_state.get("embedding_model")
    embedding_provider = st.session_state.get("embedding_provider")
    cache_ready = (
        st.session_state.get("available_embedding_models_cache_key") is not None
    )
    discovered_models = list(st.session_state.get("available_embedding_models") or [])

    if embedding_provider == "custom" and cache_ready:
        if len(discovered_models) == 1:
            model_id = discovered_models[0]
            st.session_state.custom_embedding_model = model_id
            st.session_state.embedding_model = model_id
            st.text_input(
                "Embedding model",
                value=model_id,
                disabled=True,
                help="Discovered automatically from the embedding base URL.",
            )
            return model_id

        if discovered_models:
            selector_options = discovered_models + ["Custom model..."]
            if current in discovered_models:
                index = selector_options.index(current)
            elif not current:
                index = 0
            else:
                index = len(selector_options) - 1
            selected = st.selectbox(
                "Embedding model",
                selector_options,
                index=index,
                help=(
                    "This model is used for generating embeddings and affects "
                    "context matching."
                ),
            )
        else:
            selected = "Custom model..."

        if selected == "Custom model...":
            custom_default = st.session_state.get("custom_embedding_model") or (
                current if current and current not in discovered_models else ""
            )
            custom_model = st.text_input(
                "Custom embedding model ID",
                value=custom_default,
                help=(
                    "Enter the served model ID manually when your custom endpoint "
                    "does not support /v1/models discovery."
                ),
            ).strip()
            st.session_state.custom_embedding_model = custom_model
            st.session_state.embedding_model = custom_model
        else:
            st.session_state.custom_embedding_model = ""
            st.session_state.embedding_model = selected

        return st.session_state.embedding_model

    selector_options = list(DEFAULT_EMBEDDING_MODEL_OPTIONS) + ["Custom model..."]
    index = (
        selector_options.index(current)
        if current in DEFAULT_EMBEDDING_MODEL_OPTIONS
        else len(selector_options) - 1
    )
    selected = st.selectbox(
        "Embedding model",
        selector_options,
        index=index,
        help="This model is used for generating embeddings and affects context matching.",
    )

    if selected == "Custom model...":
        custom_default = st.session_state.get("custom_embedding_model") or (
            current if current and current not in DEFAULT_EMBEDDING_MODEL_OPTIONS else ""
        )
        custom_model = st.text_input(
            "Custom embedding model ID",
            value=custom_default,
            help="Use this when your OpenAI-compatible endpoint serves a non-OpenAI embedding model.",
        ).strip()
        st.session_state.custom_embedding_model = custom_model
        st.session_state.embedding_model = custom_model
    else:
        st.session_state.custom_embedding_model = ""
        st.session_state.embedding_model = selected
        st.session_state.embedding_profile = (
            SFR_EMBEDDING_PROFILE
            if selected == SFR_EMBEDDING_MODEL
            else DEFAULT_EMBEDDING_PROFILE
        )

    return st.session_state.embedding_model


def _embedding_profile_selector():
    """Select how the embedding model expects query and document text."""
    current = st.session_state.get(
        "embedding_profile",
        DEFAULT_EMBEDDING_PROFILE,
    )
    options = list(EMBEDDING_PROFILE_OPTIONS)
    index = options.index(current) if current in options else 0
    st.session_state.embedding_profile = st.selectbox(
        "Embedding formatting profile",
        options,
        index=index,
        format_func=lambda value: EMBEDDING_PROFILE_LABELS.get(value, value),
        help=(
            "Choose how queries and document chunks should be formatted before "
            "calling the embedding endpoint. This can differ from the served "
            "model ID when you use vLLM or another OpenAI-compatible server."
        ),
    )
    return st.session_state.embedding_profile


def _render_embedding_model_hint(model: str, profile: str):
    """Show model/profile-specific guidance for embedding configuration."""
    if st.session_state.get("embedding_provider") == "custom":
        discovered = st.session_state.get("available_embedding_models") or []
        if len(discovered) == 1:
            st.caption(
                "The embedding model ID was discovered automatically from the "
                "custom embedding base URL."
            )
        elif st.session_state.get("available_embedding_models_cache_key") is not None:
            st.caption(
                "If your custom embedding endpoint exposes /v1/models, the served "
                "model IDs appear above automatically. Otherwise, enter the model "
                "ID manually."
            )

    if profile == SFR_EMBEDDING_PROFILE:
        st.info(
            "SFR formatting is active. With vLLM, the embedding model ID can be "
            "whatever your server exposes via /v1/models. Keep the formatting "
            "profile set to SFR even if the served model ID is a local alias."
        )
    elif model == SFR_EMBEDDING_MODEL:
        st.info(
            "The selected model ID is the default SFR Hugging Face name. If your "
            "vLLM server exposes a different alias, use Custom model ID instead."
        )


class StreamlitKBCallbacks:
    def __init__(self):
        self._bars = {}
        self._statuses = {}

    def info(self, msg):
        st.write(msg)

    def warning(self, msg):
        st.warning(msg)

    def success(self, msg):
        st.success(msg)

    def progress(self, phase, current, total):
        if total <= 0:
            return
        if phase not in self._bars:
            self._bars[phase] = st.progress(0)
        pct = int(current / max(total, 1) * 100)
        self._bars[phase].progress(pct)

    def status(self, phase, message):
        if phase not in self._statuses:
            self._statuses[phase] = st.empty()
        self._statuses[phase].info(message)

    def spinner(self, msg):
        return st.spinner(msg)


def kb_setup(client, embeddings):
    st.header("Knowledge-Base Setup")
    has_connection = st.session_state.get("api_verified", False)
    if not has_connection:
        st.info("Please configure the endpoint routing above and click Connect.")

    kb = KnowledgeBaseBuilder(client, embeddings) if has_connection else None

    mode = st.radio(
        "Choose setup method:",
        ("Build", "Load", "Append"),
    )

    if mode == "Build":
        model = _embedding_model_selector()
        profile = _embedding_profile_selector()
        _render_embedding_model_hint(model, profile)
        pdf_dir = _dequote_path(
            st.text_input(
                "Input folder path for PDFs",
                value="inputs",
                help="Path to the PDFs used to build the Knowledge-Base.",
                disabled=not has_connection,
            )
        )
        index_path = _dequote_path(
            st.text_input(
                "Output FAISS index file path (.index)",
                value="outputs/test_index.index",
                help="Path to save the FAISS index.",
                disabled=not has_connection,
            )
        )
        meta_path = _dequote_path(
            st.text_input(
                "Output metadata file path (.pkl)",
                value="outputs/test_metadata.pkl",
                help="Path to save metadata for chunks.",
                disabled=not has_connection,
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "GraphRAG workspace directory (optional)",
                value="",
                help="Leave blank to build only the FAISS knowledge base.",
                disabled=not has_connection,
            )
        )

        if st.button("Build", disabled=not has_connection):
            if not model:
                st.error("Enter an embedding model ID before building the knowledge base.")
                st.stop()
            try:
                _ = KBBuildRequest(
                    pdf_dir=pdf_dir,
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                    embedding_model=model,
                    embedding_profile=profile,
                )
            except ValidationError as exc:
                st.error(f"Invalid build parameters: {exc}")
                st.stop()
            if not os.path.isdir(pdf_dir):
                st.error("The provided folder path does not exist.")
                st.stop()
            pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            if not pdf_files:
                st.error("No PDF files found in the selected folder.")
                st.stop()

            callbacks = StreamlitKBCallbacks()
            try:
                kb.build(pdf_dir, index_path, meta_path, graphrag_dir, model, callbacks)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.session_state.graph_viewer_workspace = graphrag_dir or ""

            st.success("Knowledge-Base built.")

    elif mode == "Load":
        _render_embedding_model_hint(
            st.session_state.get("embedding_model", ""),
            st.session_state.get("embedding_profile", DEFAULT_EMBEDDING_PROFILE),
        )
        index_path = _dequote_path(
            st.text_input(
                "Index file path (.index)",
                value="outputs/test_index.index",
                disabled=not has_connection,
            )
        )
        meta_path = _dequote_path(
            st.text_input(
                "Metadata file path (.pkl)",
                value="outputs/test_metadata.pkl",
                disabled=not has_connection,
            )
        )
        graphrag_dir = _dequote_path(
            st.text_input(
                "GraphRAG directory (optional)",
                value="",
                help="Leave blank to load only the FAISS knowledge base.",
                disabled=not has_connection,
            )
        )
        if st.button("Load", disabled=not has_connection):
            try:
                _ = KBLoadRequest(
                    index_path=index_path,
                    meta_path=meta_path,
                    graphrag_dir=graphrag_dir,
                )
            except ValidationError as exc:
                st.error(f"Invalid load parameters: {exc}")
                st.stop()
            if not os.path.isfile(index_path):
                st.error("Index file not found.")
                st.stop()
            if not os.path.isfile(meta_path):
                st.error("Metadata file not found.")
                st.stop()

            effective_graphrag_dir = graphrag_dir or ""
            if graphrag_dir and not os.path.isdir(graphrag_dir):
                st.warning(
                    "The provided GraphRAG directory does not exist. Loading the FAISS knowledge base without GraphRAG."
                )
                effective_graphrag_dir = ""

            try:
                kb.load(index_path, meta_path, effective_graphrag_dir)
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.session_state.graph_viewer_workspace = effective_graphrag_dir or ""

            if effective_graphrag_dir:
                required = [
                    "entities.parquet",
                    "relationships.parquet",
                    "documents.parquet",
                    "communities.parquet",
                    "community_reports.parquet",
                ]

                missing = [
                    filename
                    for filename in required
                    if not (Path(effective_graphrag_dir) / "output" / filename).exists()
                ]

                if missing:
                    st.warning(
                        "Knowledge-Graph directory is missing critical files. The FAISS knowledge base is still loaded."
                    )
                    st.session_state.graphrag = ""
                else:
                    st.session_state.graphrag = effective_graphrag_dir
                    st.success("Knowledge-Graph loaded successfully.")

            st.success("Knowledge-Base loaded.")

    elif mode == "Append":
        _render_embedding_model_hint(
            st.session_state.get("embedding_model", ""),
            st.session_state.get("embedding_profile", DEFAULT_EMBEDDING_PROFILE),
        )
        exist_index_path = st.text_input(
            "Existing index file path (.index)",
            value="outputs/index.index",
            disabled=not has_connection,
        )
        exist_meta_path = st.text_input(
            "Existing metadata file path (.pkl)",
            value="outputs/meta.pkl",
            disabled=not has_connection,
        )
        graphrag_dir = st.text_input(
            "GraphRAG directory (optional existing workspace)",
            value="",
            disabled=not has_connection,
            help="Leave blank to append only to the FAISS knowledge base.",
        )
        append_folder = _dequote_path(
            st.text_input(
                "Input folder path for new PDFs to append",
                value="inputs/new",
                help="Path to the new PDFs to append.",
                disabled=not has_connection,
            )
        )

        if st.button("Append to Knowledge-Base", disabled=not has_connection):
            try:
                _ = KBAppendRequest(
                    index_path=exist_index_path,
                    meta_path=exist_meta_path,
                    append_folder=append_folder,
                    graphrag_dir=graphrag_dir,
                )
            except ValidationError as exc:
                st.error(f"Invalid append parameters: {exc}")
                st.stop()
            if not os.path.isfile(exist_index_path):
                st.error("Existing FAISS index file not found.")
                st.stop()
            if not os.path.isfile(exist_meta_path):
                st.error("Existing metadata file not found.")
                st.stop()
            if not os.path.isdir(append_folder):
                st.error("Append folder does not exist.")
                st.stop()
            effective_graphrag_dir = graphrag_dir or ""
            if graphrag_dir and not os.path.isdir(graphrag_dir):
                st.warning(
                    "The provided GraphRAG directory was not found. Appending only to the FAISS knowledge base."
                )
                effective_graphrag_dir = ""

            callbacks = StreamlitKBCallbacks()
            try:
                kb.append(
                    exist_index_path,
                    exist_meta_path,
                    append_folder,
                    effective_graphrag_dir,
                    callbacks,
                )
            except KnowledgeBaseError as exc:
                st.error(str(exc))
                return

            st.session_state.graph_viewer_workspace = effective_graphrag_dir or ""

            st.success("Knowledge-Base updated.")

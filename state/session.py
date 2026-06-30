# state/session.py
import streamlit as st
from state.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROFILE,
    DEFAULT_RETRIEVAL_DIVERSITY,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_METHOD,
    ENC,
)


def init_session():
    """
    Initialize all Streamlit session_state keys used by the app.
    Safe to call multiple times.
    """

    prompt_provider = (
        "custom" if st.session_state.get("api_base_url") else "openai"
    )
    embedding_provider = (
        "custom"
        if st.session_state.get("embedding_api_base_url")
        else "openai"
    )

    defaults = {
        # API / client
        "api_verified": False,
        "client": None,
        "api_key": "",
        "api_base_url": "",
        "embedding_api_base_url": "",
        "rerank_api_base_url": "",
        "prompt_provider": prompt_provider,
        "embedding_provider": embedding_provider,
        "api_key_error": None,
        "api_connection_warning": None,
        "openai_key_input": "",
        "openai_base_url_input": "",
        "embedding_base_url_input": "",
        "rerank_base_url_input": "",
        "prompt_provider_input": prompt_provider,
        "embedding_provider_input": embedding_provider,
        "available_models": None,
        "available_models_cache_key": None,
        "available_prompt_models": None,
        "available_prompt_models_cache_key": None,
        "available_embedding_models": None,
        "available_embedding_models_cache_key": None,
        "available_rerank_models": None,
        "available_rerank_models_cache_key": None,
        "custom_gpt_model": "",
        "custom_embedding_model": "",
        "custom_rerank_model": "",
        "rerank_model": "",

        # Models
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_profile": DEFAULT_EMBEDDING_PROFILE,
        "gpt_model": None,
        "diversity": DEFAULT_RETRIEVAL_DIVERSITY,
        "rerank_top_k": DEFAULT_RERANK_TOP_K,
        "enc": ENC,

        # Knowledge base
        "dimension": None,
        "index": None,
        "metadata": None,
        "db": None,
        "bm25": None,
        "all_documents": None,
        "graphrag": None,

        # Uploads
        "upload_db": None,
        "upload_meta": [],
        "upload_images": [],

        # Q&A
        "last_query": "",
        "last_answer": "",
        "context_meta": "",
        "rerank_meta": [],
        "use_graphrag": False,
        "retrieval_method": DEFAULT_RETRIEVAL_METHOD,

        # Graph viewer
        "graph_viewer_workspace": "",
        "graph_viewer_selected_community": "",
        "graph_viewer_search_text": "",
        "graph_viewer_max_nodes": 80,
        "graph_viewer_max_edges": 160,
        "graph_viewer_mode": "Interactive",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

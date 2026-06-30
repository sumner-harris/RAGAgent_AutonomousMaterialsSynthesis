import streamlit as st

from RAG.openai_compat import make_openai_client, normalize_base_url
from state.config import DEFAULT_EMBEDDING_MODEL_OPTIONS

DEFAULT_MODEL_OPTIONS = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-08-06",
    "gpt-5",
    "gpt-5-thinking",
    "gpt-5-pro",
    "o4-mini-2025-04-16",
    "o4-mini-deep-research-2025-06-26",
]
PROMPT_MODELS_SESSION_KEY = "available_prompt_models"
PROMPT_MODELS_CACHE_SESSION_KEY = "available_prompt_models_cache_key"
EMBEDDING_MODELS_SESSION_KEY = "available_embedding_models"
EMBEDDING_MODELS_CACHE_SESSION_KEY = "available_embedding_models_cache_key"
RERANK_MODELS_SESSION_KEY = "available_rerank_models"
RERANK_MODELS_CACHE_SESSION_KEY = "available_rerank_models_cache_key"


def _session_store(session=None):
    """Return the active state store."""
    return st.session_state if session is None else session


def _is_allowed_prompt_model(model_id: str) -> bool:
    """Hide clearly unsupported entries while allowing custom local model IDs."""
    mid = (model_id or "").lower()
    if not mid:
        return False
    if "codex" in mid:
        return False
    if mid.startswith("text-embedding") or "embedding" in mid:
        return False
    return True


def _is_allowed_embedding_model(model_id: str) -> bool:
    """Allow any non-empty model ID for embedding endpoints."""
    return bool((model_id or "").strip())


def _is_allowed_rerank_model(model_id: str) -> bool:
    """Allow any non-empty model ID for rerank endpoints."""
    return bool((model_id or "").strip())


def _prompt_model_cache_key(client=None, *, session=None) -> str:
    store = _session_store(session)
    if client is not None and getattr(client, "base_url", None):
        return str(client.base_url).rstrip("/")
    return normalize_base_url(store.get("api_base_url")) or "openai-default"


def _embedding_model_cache_key(
    client=None,
    *,
    base_url: str | None = None,
    session=None,
) -> str:
    store = _session_store(session)
    if client is not None and getattr(client, "base_url", None):
        return str(client.base_url).rstrip("/")
    return (
        normalize_base_url(base_url)
        or normalize_base_url(store.get("embedding_api_base_url"))
        or "openai-embeddings-default"
    )


def _rerank_model_cache_key(
    client=None,
    *,
    base_url: str | None = None,
    session=None,
) -> str:
    store = _session_store(session)
    if client is not None and getattr(client, "base_url", None):
        return str(client.base_url).rstrip("/")
    return (
        normalize_base_url(base_url)
        or normalize_base_url(store.get("rerank_api_base_url"))
        or "openai-rerank-default"
    )


def _discover_model_options(client, *, model_filter) -> list[str]:
    """Return unique, sorted model IDs from an OpenAI-compatible endpoint."""
    data = client.models.list()
    models = [m.id for m in getattr(data, "data", [])]
    return sorted({model_id for model_id in models if model_filter(model_id)})


def discover_prompt_model_options(client) -> list[str]:
    """Return prompt-capable model IDs from the configured route."""
    return _discover_model_options(client, model_filter=_is_allowed_prompt_model)


def discover_embedding_model_options(client) -> list[str]:
    """Return embedding model IDs from the configured route."""
    return _discover_model_options(client, model_filter=_is_allowed_embedding_model)


def discover_rerank_model_options(client) -> list[str]:
    """Return rerank model IDs from the configured route."""
    return _discover_model_options(client, model_filter=_is_allowed_rerank_model)


def cache_prompt_model_options(options, *, client=None, session=None):
    """Persist discovered prompt models for the active prompt route."""
    store = _session_store(session)
    store[PROMPT_MODELS_SESSION_KEY] = list(options)
    store[PROMPT_MODELS_CACHE_SESSION_KEY] = _prompt_model_cache_key(
        client,
        session=session,
    )
    # Preserve the older key names for any remaining callers.
    store["available_models"] = list(options)
    store["available_models_cache_key"] = store[PROMPT_MODELS_CACHE_SESSION_KEY]


def cache_embedding_model_options(
    options,
    *,
    client=None,
    base_url: str | None = None,
    session=None,
):
    """Persist discovered embedding models for the active embedding route."""
    store = _session_store(session)
    store[EMBEDDING_MODELS_SESSION_KEY] = list(options)
    store[EMBEDDING_MODELS_CACHE_SESSION_KEY] = _embedding_model_cache_key(
        client,
        base_url=base_url,
        session=session,
    )


def cache_rerank_model_options(
    options,
    *,
    client=None,
    base_url: str | None = None,
    session=None,
):
    """Persist discovered rerank models for the active rerank route."""
    store = _session_store(session)
    store[RERANK_MODELS_SESSION_KEY] = list(options)
    store[RERANK_MODELS_CACHE_SESSION_KEY] = _rerank_model_cache_key(
        client,
        base_url=base_url,
        session=session,
    )


def load_prompt_model_options(client, *, session=None):
    """Fetch and cache prompt model IDs for the active route."""
    store = _session_store(session)
    cache_key = _prompt_model_cache_key(client, session=session)
    cached = store.get(PROMPT_MODELS_SESSION_KEY)
    if (
        cached is not None
        and store.get(PROMPT_MODELS_CACHE_SESSION_KEY) == cache_key
    ):
        return cached

    if client is None:
        return DEFAULT_MODEL_OPTIONS

    try:
        options = discover_prompt_model_options(client)
        if options:
            cache_prompt_model_options(
                options,
                client=client,
                session=session,
            )
            return options
    except Exception:
        pass

    return DEFAULT_MODEL_OPTIONS


def load_embedding_model_options(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    client=None,
    session=None,
):
    """Fetch and cache embedding model IDs for the active route."""
    store = _session_store(session)
    cache_key = _embedding_model_cache_key(
        client,
        base_url=base_url,
        session=session,
    )
    cached = store.get(EMBEDDING_MODELS_SESSION_KEY)
    if (
        cached is not None
        and store.get(EMBEDDING_MODELS_CACHE_SESSION_KEY) == cache_key
    ):
        return cached

    route_client = client
    if route_client is None:
        if not normalize_base_url(base_url):
            return DEFAULT_EMBEDDING_MODEL_OPTIONS
        route_client = make_openai_client(
            api_key=api_key,
            base_url=base_url,
            fallback_to_env=False,
        )

    try:
        options = discover_embedding_model_options(route_client)
        if options:
            cache_embedding_model_options(
                options,
                client=route_client,
                base_url=base_url,
                session=session,
            )
            return options
    except Exception:
        pass

    return DEFAULT_EMBEDDING_MODEL_OPTIONS


def load_rerank_model_options(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    client=None,
    session=None,
):
    """Fetch and cache rerank model IDs for the active route."""
    store = _session_store(session)
    cache_key = _rerank_model_cache_key(
        client,
        base_url=base_url,
        session=session,
    )
    cached = store.get(RERANK_MODELS_SESSION_KEY)
    if (
        cached is not None
        and store.get(RERANK_MODELS_CACHE_SESSION_KEY) == cache_key
    ):
        return cached

    route_client = client
    if route_client is None:
        if not normalize_base_url(base_url):
            return []
        route_client = make_openai_client(
            api_key=api_key,
            base_url=base_url,
            fallback_to_env=False,
        )

    try:
        options = discover_rerank_model_options(route_client)
        if options:
            cache_rerank_model_options(
                options,
                client=route_client,
                base_url=base_url,
                session=session,
            )
            return options
    except Exception:
        pass

    return []


def load_model_options(client):
    """Backward-compatible wrapper for prompt-model discovery."""
    return load_prompt_model_options(client)

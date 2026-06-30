import streamlit as st

from RAG.generation.model_selector import (
    cache_embedding_model_options,
    cache_prompt_model_options,
    cache_rerank_model_options,
    discover_embedding_model_options,
    discover_prompt_model_options,
    discover_rerank_model_options,
)
from RAG.openai_compat import (
    apply_openai_env,
    make_openai_client,
    normalize_base_url,
)

PROVIDER_OPENAI = "openai"
PROVIDER_CUSTOM = "custom"
PROVIDER_OPTIONS = [PROVIDER_OPENAI, PROVIDER_CUSTOM]
PROVIDER_LABELS = {
    PROVIDER_OPENAI: "OpenAI API",
    PROVIDER_CUSTOM: "Custom base URL",
}


def _supports_manual_model_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "404",
            "405",
            "501",
            "not found",
            "method not allowed",
        )
    )


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def _provider_index(provider: str) -> int:
    if provider in PROVIDER_OPTIONS:
        return PROVIDER_OPTIONS.index(provider)
    return 0


def _infer_provider(base_url: str | None) -> str:
    return PROVIDER_CUSTOM if normalize_base_url(base_url) else PROVIDER_OPENAI


def _describe_route(provider: str, base_url: str | None) -> str:
    if provider == PROVIDER_CUSTOM and base_url:
        return f"Custom base URL -> {base_url}"
    return "OpenAI API"


class OpenAIConnectionManager:
    def __init__(self, session):
        self.s = session
        self._init_state()

    def _init_state(self):
        prompt_provider = self.s.get("prompt_provider") or _infer_provider(
            self.s.get("api_base_url")
        )
        embedding_provider = self.s.get("embedding_provider") or _infer_provider(
            self.s.get("embedding_api_base_url")
        )

        self.s.setdefault("api_key", "")
        self.s.setdefault("api_base_url", "")
        self.s.setdefault("embedding_api_base_url", "")
        self.s.setdefault("rerank_api_base_url", "")
        self.s.setdefault("prompt_provider", prompt_provider)
        self.s.setdefault("embedding_provider", embedding_provider)
        self.s.setdefault("client", None)
        self.s.setdefault("api_verified", False)
        self.s.setdefault("api_key_error", None)
        self.s.setdefault("api_connection_warning", None)
        self.s.setdefault("openai_key_input", self.s.api_key)
        self.s.setdefault("openai_base_url_input", self.s.api_base_url)
        self.s.setdefault(
            "embedding_base_url_input",
            self.s.embedding_api_base_url,
        )
        self.s.setdefault(
            "rerank_base_url_input",
            self.s.rerank_api_base_url,
        )
        self.s.setdefault("prompt_provider_input", self.s.prompt_provider)
        self.s.setdefault("embedding_provider_input", self.s.embedding_provider)
        self.s.setdefault("custom_rerank_model", self.s.get("rerank_model", ""))

    def _clear_model_cache(self):
        self.s["available_models"] = None
        self.s["available_models_cache_key"] = None
        self.s["available_prompt_models"] = None
        self.s["available_prompt_models_cache_key"] = None
        self.s["available_embedding_models"] = None
        self.s["available_embedding_models_cache_key"] = None
        self.s["available_rerank_models"] = None
        self.s["available_rerank_models_cache_key"] = None

    def clear(
        self,
        *,
        prompt_provider: str = PROVIDER_OPENAI,
        embedding_provider: str = PROVIDER_OPENAI,
    ):
        self.s.update(
            api_key="",
            api_base_url="",
            embedding_api_base_url="",
            rerank_api_base_url="",
            prompt_provider=prompt_provider,
            embedding_provider=embedding_provider,
            client=None,
            api_verified=False,
            api_key_error=None,
            api_connection_warning=None,
            custom_rerank_model="",
            rerank_model="",
        )
        self._clear_model_cache()
        apply_openai_env(None, None, include_base_url=False)

    def _store_connection(
        self,
        key: str,
        prompt_provider: str,
        base_url: str | None,
        embedding_provider: str,
        embedding_base_url: str | None,
        rerank_base_url: str | None,
        client,
        warning: str | None,
        prompt_models: list[str] | None = None,
        embedding_models: list[str] | None = None,
        rerank_models: list[str] | None = None,
    ):
        self.s.update(
            api_key=key,
            api_base_url=base_url or "",
            embedding_api_base_url=embedding_base_url or "",
            rerank_api_base_url=rerank_base_url or "",
            prompt_provider=prompt_provider,
            embedding_provider=embedding_provider,
            client=client,
            api_verified=True,
            api_key_error=None,
            api_connection_warning=warning,
        )
        apply_openai_env(key, None, include_base_url=False)
        self._apply_prompt_model_defaults(
            provider=prompt_provider,
            discovered_models=prompt_models or [],
        )
        self._apply_embedding_model_defaults(
            provider=embedding_provider,
            discovered_models=embedding_models or [],
        )
        self._apply_rerank_model_defaults(
            discovered_models=rerank_models or [],
        )

    def _apply_prompt_model_defaults(
        self,
        *,
        provider: str,
        discovered_models: list[str],
    ):
        if not discovered_models:
            return

        if provider == PROVIDER_CUSTOM and len(discovered_models) == 1:
            model_id = discovered_models[0]
            self.s["custom_gpt_model"] = model_id
            self.s["gpt_model"] = model_id
            return

        current_model = self.s.get("gpt_model")
        if current_model not in discovered_models:
            self.s["gpt_model"] = discovered_models[0]

        if provider != PROVIDER_CUSTOM and self.s.get("gpt_model") in discovered_models:
            self.s["custom_gpt_model"] = ""

    def _apply_embedding_model_defaults(
        self,
        *,
        provider: str,
        discovered_models: list[str],
    ):
        if not discovered_models:
            return

        if provider == PROVIDER_CUSTOM and len(discovered_models) == 1:
            model_id = discovered_models[0]
            self.s["custom_embedding_model"] = model_id
            self.s["embedding_model"] = model_id
            return

        current_model = self.s.get("embedding_model")
        if current_model not in discovered_models:
            self.s["embedding_model"] = discovered_models[0]

        if provider != PROVIDER_CUSTOM and self.s.get("embedding_model") in discovered_models:
            self.s["custom_embedding_model"] = ""

    def _apply_rerank_model_defaults(
        self,
        *,
        discovered_models: list[str],
    ):
        if not self.s.get("rerank_api_base_url"):
            self.s["custom_rerank_model"] = ""
            self.s["rerank_model"] = ""
            return

        if not discovered_models:
            self.s["rerank_model"] = self.s.get("custom_rerank_model", "").strip()
            return

        if len(discovered_models) == 1:
            model_id = discovered_models[0]
            self.s["custom_rerank_model"] = model_id
            self.s["rerank_model"] = model_id
            return

        current_model = self.s.get("rerank_model")
        if current_model not in discovered_models:
            self.s["rerank_model"] = discovered_models[0]

        if self.s.get("rerank_model") in discovered_models:
            self.s["custom_rerank_model"] = ""

    def _make_route_client(
        self,
        *,
        key: str,
        provider: str,
        base_url: str | None,
    ):
        return make_openai_client(
            api_key=key,
            base_url=base_url if provider == PROVIDER_CUSTOM else None,
            fallback_to_env=False,
        )

    def _discover_prompt_models(
        self,
        *,
        client,
        provider: str,
        base_url: str | None,
    ) -> tuple[list[str], str | None]:
        try:
            models = discover_prompt_model_options(client)
            cache_prompt_model_options(models, client=client, session=self.s)
            return models, None
        except Exception as exc:
            if provider == PROVIDER_CUSTOM and base_url and _supports_manual_model_fallback(exc):
                cache_prompt_model_options([], client=client, session=self.s)
                return (
                    [],
                    "The configured prompt endpoint did not expose model discovery. "
                    "You can still use it by entering model IDs manually.",
                )
            raise

    def _discover_embedding_models(
        self,
        *,
        key: str,
        provider: str,
        base_url: str | None,
        prompt_provider: str,
        prompt_base_url: str | None,
        prompt_client,
    ) -> tuple[list[str], str | None]:
        if provider != PROVIDER_CUSTOM or not base_url:
            return [], None

        route_client = prompt_client
        if not (
            prompt_provider == PROVIDER_CUSTOM
            and prompt_base_url == base_url
        ):
            route_client = self._make_route_client(
                key=key,
                provider=provider,
                base_url=base_url,
            )

        try:
            models = discover_embedding_model_options(route_client)
            cache_embedding_model_options(
                models,
                client=route_client,
                base_url=base_url,
                session=self.s,
            )
            return models, None
        except Exception as exc:
            if _supports_manual_model_fallback(exc):
                cache_embedding_model_options(
                    [],
                    client=route_client,
                    base_url=base_url,
                    session=self.s,
                )
                return (
                    [],
                    "The configured embedding endpoint did not expose model discovery. "
                    "You can still use it by entering the embedding model ID manually.",
                )
            raise

    def _resolve_rerank_model(
        self,
        *,
        key: str,
        base_url: str | None,
    ) -> tuple[list[str], str | None]:
        if not base_url:
            self.s["available_rerank_models"] = []
            self.s["available_rerank_models_cache_key"] = None
            return [], None

        route_client = self._make_route_client(
            key=key,
            provider=PROVIDER_CUSTOM,
            base_url=base_url,
        )

        try:
            discovered_models = discover_rerank_model_options(route_client)
            cache_rerank_model_options(
                discovered_models,
                client=route_client,
                base_url=base_url,
                session=self.s,
            )
        except Exception as exc:
            if _supports_manual_model_fallback(exc):
                cache_rerank_model_options(
                    [],
                    client=route_client,
                    base_url=base_url,
                    session=self.s,
                )
                return (
                    [],
                    "The configured rerank endpoint did not expose model discovery. "
                    "Enter a rerank model ID in the question panel to enable reranking.",
                )
            return (
                [],
                f"Reranker disabled because the configured rerank endpoint could not be reached: {exc}",
            )
        if not discovered_models:
            return (
                [],
                "Reranker is configured, but no rerank model was discovered. "
                "Select or enter one in the question panel to enable reranking.",
            )
        return discovered_models, None

    def _validate_settings(
        self,
        *,
        key: str,
        prompt_provider: str,
        base_url: str | None,
        embedding_provider: str,
        embedding_base_url: str | None,
    ) -> list[str]:
        errors = []

        if (
            PROVIDER_OPENAI in (prompt_provider, embedding_provider)
            and not key
        ):
            errors.append(
                "An OpenAI API key is required whenever prompts or embeddings are routed to OpenAI."
            )

        if prompt_provider == PROVIDER_CUSTOM and not base_url:
            errors.append(
                "Prompt / LLM routing is set to Custom base URL, so Model Base URL is required."
            )

        if embedding_provider == PROVIDER_CUSTOM and not embedding_base_url:
            errors.append(
                "Embedding routing is set to Custom base URL, so Embedding Base URL is required."
            )

        return errors

    def set_connection(
        self,
        raw_key: str,
        prompt_provider: str,
        raw_base_url: str,
        embedding_provider: str,
        raw_embedding_base_url: str,
        raw_rerank_base_url: str,
    ):
        key = (raw_key or "").strip()
        prompt_provider = prompt_provider or PROVIDER_OPENAI
        embedding_provider = embedding_provider or PROVIDER_OPENAI
        base_url = (
            normalize_base_url(raw_base_url)
            if prompt_provider == PROVIDER_CUSTOM
            else None
        )
        embedding_base_url = (
            normalize_base_url(raw_embedding_base_url)
            if embedding_provider == PROVIDER_CUSTOM
            else None
        )
        rerank_base_url = normalize_base_url(raw_rerank_base_url)

        if (
            prompt_provider == PROVIDER_OPENAI
            and embedding_provider == PROVIDER_OPENAI
            and not key
        ):
            self.clear(
                prompt_provider=prompt_provider,
                embedding_provider=embedding_provider,
            )
            return

        validation_errors = self._validate_settings(
            key=key,
            prompt_provider=prompt_provider,
            base_url=base_url,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
        )
        if validation_errors:
            self.s.update(
                api_key="",
                api_base_url="",
                embedding_api_base_url="",
                rerank_api_base_url="",
                prompt_provider=prompt_provider,
                embedding_provider=embedding_provider,
                client=None,
                api_verified=False,
                api_key_error="\n".join(validation_errors),
                api_connection_warning=None,
                custom_rerank_model="",
                rerank_model="",
            )
            self._clear_model_cache()
            apply_openai_env(None, None, include_base_url=False)
            return

        try:
            self._clear_model_cache()
            client = self._make_route_client(
                key=key,
                provider=prompt_provider,
                base_url=base_url,
            )
            prompt_models, prompt_warning = self._discover_prompt_models(
                client=client,
                provider=prompt_provider,
                base_url=base_url,
            )
            embedding_models, embedding_warning = self._discover_embedding_models(
                key=key,
                provider=embedding_provider,
                base_url=embedding_base_url,
                prompt_provider=prompt_provider,
                prompt_base_url=base_url,
                prompt_client=client,
            )
            rerank_models, rerank_warning = self._resolve_rerank_model(
                key=key,
                base_url=rerank_base_url,
            )
            warnings = [
                warning
                for warning in (
                    prompt_warning,
                    embedding_warning,
                    rerank_warning,
                )
                if warning
            ]
            self._store_connection(
                key,
                prompt_provider,
                base_url,
                embedding_provider,
                embedding_base_url,
                rerank_base_url,
                client,
                "\n".join(warnings) if warnings else None,
                prompt_models=prompt_models,
                embedding_models=embedding_models,
                rerank_models=rerank_models,
            )
        except Exception as exc:
            self.s.update(
                api_key="",
                api_base_url="",
                embedding_api_base_url="",
                rerank_api_base_url="",
                prompt_provider=prompt_provider,
                embedding_provider=embedding_provider,
                client=None,
                api_verified=False,
                api_key_error=str(exc),
                api_connection_warning=None,
                custom_rerank_model="",
                rerank_model="",
            )
            self._clear_model_cache()
            apply_openai_env(None, None, include_base_url=False)


def API_entry():
    """Streamlit UI for configuring prompt and embedding routing."""

    km = OpenAIConnectionManager(st.session_state)

    st.header("Endpoint Routing")
    st.caption(
        "Choose whether prompts and embeddings use OpenAI or a custom OpenAI-compatible base URL, "
        "and optionally add a rerank host. "
        "These routes propagate through GraphRAG, hybrid multi-query retrieval, contextual compression, "
        "knowledge-base indexing, vector search, reranking, and upload embeddings."
    )
    col1, col2 = st.columns(2)
    with col1:
        prompt_provider = st.selectbox(
            "Prompt / LLM routing",
            PROVIDER_OPTIONS,
            index=_provider_index(st.session_state.prompt_provider_input),
            format_func=_provider_label,
            key="prompt_provider_input",
        )
    with col2:
        embedding_provider = st.selectbox(
            "Embedding routing",
            PROVIDER_OPTIONS,
            index=_provider_index(st.session_state.embedding_provider_input),
            format_func=_provider_label,
            key="embedding_provider_input",
        )

    key_label = (
        "OpenAI API key"
        if PROVIDER_OPENAI in (prompt_provider, embedding_provider)
        else "Shared API key (optional)"
    )
    key_help = (
        "Required for any route that still uses OpenAI. "
        "If you also use custom endpoints, this same key will be sent to compatible servers."
        if PROVIDER_OPENAI in (prompt_provider, embedding_provider)
        else "Leave blank for fully local routing, or provide a token if your custom endpoints require one."
    )
    st.text_input(
        key_label,
        type="password",
        key="openai_key_input",
        help=key_help,
    )

    if prompt_provider == PROVIDER_CUSTOM:
        st.text_input(
            "Model Base URL",
            key="openai_base_url_input",
            placeholder="http://nvidiaspark:8000/v1",
            help=(
                "Used for answer generation plus LLM-driven retrieval steps such as hybrid multi-query, "
                "context compression, and GraphRAG chat."
            ),
        )
    else:
        st.caption(
            "Prompts, answer generation, GraphRAG chat, and LLM-based retrieval helpers will use OpenAI."
        )

    if embedding_provider == PROVIDER_CUSTOM:
        st.text_input(
            "Embedding Base URL",
            key="embedding_base_url_input",
            placeholder="http://host:8000/v1",
            help=(
                "Used for knowledge-base build/append, upload embeddings, FAISS/MMR retrieval, and GraphRAG embeddings."
                " You can point this at any OpenAI-compatible embedding host."
                " If you paste a full endpoint such as /v1/embeddings, the app will normalize it to the shared base URL."
            ),
        )
    else:
        st.caption(
            "Embeddings, FAISS/vector retrieval, uploads, and GraphRAG embeddings will use OpenAI."
        )

    st.text_input(
        "Rerank Base URL (optional)",
        key="rerank_base_url_input",
        placeholder="http://host:8002/v1",
        help=(
            "Optional post-retrieval reranker for FAISS, BM25, and hybrid text results."
            " Leave blank to keep the current retrieval order."
            " If you paste a full endpoint such as /v1/rerank, the app will normalize it to the shared base URL."
        ),
    )
    if st.button("Connect", type="primary"):
        km.set_connection(
            st.session_state.openai_key_input,
            st.session_state.prompt_provider_input,
            st.session_state.openai_base_url_input,
            st.session_state.embedding_provider_input,
            st.session_state.embedding_base_url_input,
            st.session_state.rerank_base_url_input,
        )

    if st.session_state.api_verified:
        st.success(
            "Prompts / LLMs: "
            + _describe_route(
                st.session_state.prompt_provider,
                st.session_state.api_base_url,
            )
        )
        st.info(
            "Embeddings: "
            + _describe_route(
                st.session_state.embedding_provider,
                st.session_state.embedding_api_base_url,
            )
        )
        if st.session_state.rerank_api_base_url:
            st.info(
                "Reranker: Custom base URL -> "
                + st.session_state.rerank_api_base_url
            )
            if st.session_state.rerank_model:
                st.caption(
                    f"Rerank model: {st.session_state.rerank_model}"
                )
        else:
            st.caption("Reranker: disabled.")

        if st.session_state.api_connection_warning:
            st.warning(st.session_state.api_connection_warning)
    elif st.session_state.api_key_error:
        st.error("Could not apply the configured routing.")
        print("Endpoint routing validation error:\n", st.session_state.api_key_error)
    else:
        st.info(
            "Select whether prompts and embeddings should use OpenAI or a custom base URL, then click Connect."
        )

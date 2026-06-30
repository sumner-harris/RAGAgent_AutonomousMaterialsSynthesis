from dataclasses import dataclass
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from RAG.openai_compat import build_openai_client_kwargs, normalize_base_url
from state.config import (
    DEFAULT_EMBEDDING_PROFILE as DEFAULT_EMBEDDING_PROFILE_NAME,
    SFR_EMBEDDING_MODEL,
    SFR_EMBEDDING_PROFILE,
)


SFR_QUERY_TEMPLATE = (
    "Instruct: Given a user question, retrieve relevant passages that answer "
    "the question\nQuery: {text}"
)


@dataclass(frozen=True)
class EmbeddingProfile:
    """Describe how a retrieval embedding model formats text."""

    query_template: str = "{text}"
    document_template: str = "{text}"

    def format_query(self, text: str) -> str:
        """Format a query string for the embedding model."""
        return self.query_template.format(text=text)

    def format_document(self, text: str) -> str:
        """Format a document chunk for the embedding model."""
        return self.document_template.format(text=text)


DEFAULT_EMBEDDING_RULES = EmbeddingProfile()
PROFILE_REGISTRY = {
    DEFAULT_EMBEDDING_PROFILE_NAME: DEFAULT_EMBEDDING_RULES,
    SFR_EMBEDDING_PROFILE: EmbeddingProfile(
        query_template=SFR_QUERY_TEMPLATE,
    ),
}
MODEL_PROFILE_OVERRIDES = {
    SFR_EMBEDDING_MODEL.casefold(): SFR_EMBEDDING_PROFILE,
}


def uses_custom_embedding_route(*, base_url: str | None) -> bool:
    """Return whether embeddings are routed through a custom OpenAI-compatible host."""
    return bool(normalize_base_url(base_url))


def build_embedding_delegate_kwargs(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_to_env: bool = True,
) -> dict:
    """Build OpenAIEmbeddings kwargs for the configured route.

    For custom OpenAI-compatible embedding servers such as vLLM, disable
    LangChain's client-side token splitting so the server receives raw text
    instead of OpenAI tiktoken token IDs.
    """
    kwargs = build_openai_client_kwargs(
        api_key=api_key,
        base_url=base_url,
        fallback_to_env=fallback_to_env,
    )
    if uses_custom_embedding_route(base_url=base_url):
        kwargs["check_embedding_ctx_length"] = False
    return kwargs


def resolve_embedding_profile_name(
    *,
    profile_name: str | None = None,
    model_name: str | None = None,
) -> str:
    """Resolve the canonical embedding profile name."""
    if profile_name in PROFILE_REGISTRY:
        return profile_name
    if model_name:
        return MODEL_PROFILE_OVERRIDES.get(
            model_name.casefold(),
            DEFAULT_EMBEDDING_PROFILE_NAME,
        )
    return DEFAULT_EMBEDDING_PROFILE_NAME


def get_embedding_profile(
    *,
    profile_name: str | None = None,
    model_name: str | None = None,
) -> EmbeddingProfile:
    """Return the configured embedding profile."""
    resolved_name = resolve_embedding_profile_name(
        profile_name=profile_name,
        model_name=model_name,
    )
    return PROFILE_REGISTRY[resolved_name]


def format_embedding_query(
    *,
    model_name: str | None,
    text: str,
    profile_name: str | None = None,
) -> str:
    """Format one retrieval query for the target embedding model."""
    return get_embedding_profile(
        profile_name=profile_name,
        model_name=model_name,
    ).format_query(text)


def format_embedding_document(
    *,
    model_name: str | None,
    text: str,
    profile_name: str | None = None,
) -> str:
    """Format one document chunk for the target embedding model."""
    return get_embedding_profile(
        profile_name=profile_name,
        model_name=model_name,
    ).format_document(text)


def format_embedding_documents(
    *,
    model_name: str | None,
    texts: Iterable[str],
    profile_name: str | None = None,
) -> list[str]:
    """Format document chunks for batch embedding."""
    profile = get_embedding_profile(
        profile_name=profile_name,
        model_name=model_name,
    )
    return [profile.format_document(text) for text in texts]


class ProfiledOpenAIEmbeddings(Embeddings):
    """Apply model-specific query/document formatting around OpenAI embeddings."""

    def __init__(
        self,
        *,
        model: str,
        profile_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_to_env: bool = True,
    ):
        self.model = model
        self.profile_name = resolve_embedding_profile_name(
            profile_name=profile_name,
            model_name=model,
        )
        self._profile = get_embedding_profile(
            profile_name=self.profile_name,
            model_name=model,
        )
        self._delegate = OpenAIEmbeddings(
            model=model,
            **build_embedding_delegate_kwargs(
                api_key=api_key,
                base_url=base_url,
                fallback_to_env=fallback_to_env,
            ),
        )

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        formatted = [self._profile.format_document(text) for text in texts]
        return self._delegate.embed_documents(formatted)

    def embed_query(self, text: str) -> list[float]:
        formatted = self._profile.format_query(text)
        return self._delegate.embed_query(formatted)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        formatted = [self._profile.format_document(text) for text in texts]
        return await self._delegate.aembed_documents(formatted)

    async def aembed_query(self, text: str) -> list[float]:
        formatted = self._profile.format_query(text)
        return await self._delegate.aembed_query(formatted)


def create_embedding_function(
    *,
    model: str,
    profile_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_to_env: bool = True,
) -> ProfiledOpenAIEmbeddings:
    """Build an embedding client with model-specific formatting rules."""
    return ProfiledOpenAIEmbeddings(
        model=model,
        profile_name=profile_name,
        api_key=api_key,
        base_url=base_url,
        fallback_to_env=fallback_to_env,
    )

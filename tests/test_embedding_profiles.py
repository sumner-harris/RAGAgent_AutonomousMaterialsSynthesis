from ingestion.embedding_profiles import (
    SFR_QUERY_TEMPLATE,
    build_embedding_delegate_kwargs,
    create_embedding_function,
    format_embedding_document,
    format_embedding_query,
)
from RAG.openai_compat import normalize_base_url
from state.config import (
    DEFAULT_EMBEDDING_MODEL_OPTIONS,
    SFR_EMBEDDING_MODEL,
    SFR_EMBEDDING_PROFILE,
)


def test_sfr_is_available_as_builtin_embedding_option():
    assert SFR_EMBEDDING_MODEL in DEFAULT_EMBEDDING_MODEL_OPTIONS


def test_normalize_base_url_trims_full_embedding_endpoint():
    assert (
        normalize_base_url("http://localhost:8001/v1/embeddings")
        == "http://localhost:8001/v1"
    )


def test_sfr_query_formatting_uses_instruction_wrapper():
    query = "Find passages about catalysts."
    assert format_embedding_query(model_name=SFR_EMBEDDING_MODEL, text=query) == (
        SFR_QUERY_TEMPLATE.format(text=query)
    )


def test_explicit_sfr_profile_can_be_used_with_custom_served_model_name():
    query = "Find passages about catalysts."
    assert format_embedding_query(
        model_name="vllm-sfr",
        profile_name=SFR_EMBEDDING_PROFILE,
        text=query,
    ) == SFR_QUERY_TEMPLATE.format(text=query)


def test_default_document_formatting_is_identity():
    chunk = "A raw chunk of knowledge-base text."
    assert (
        format_embedding_document(
            model_name="text-embedding-3-small",
            text=chunk,
        )
        == chunk
    )


def test_profiled_embeddings_apply_sfr_formatting(monkeypatch):
    class FakeOpenAIEmbeddings:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.last_query = None
            self.last_documents = None

        def embed_query(self, text):
            self.last_query = text
            return [1.0]

        def embed_documents(self, texts):
            self.last_documents = texts
            return [[1.0] for _ in texts]

        async def aembed_query(self, text):
            self.last_query = text
            return [1.0]

        async def aembed_documents(self, texts):
            self.last_documents = texts
            return [[1.0] for _ in texts]

    monkeypatch.setattr(
        "ingestion.embedding_profiles.OpenAIEmbeddings",
        FakeOpenAIEmbeddings,
    )

    embedding_fn = create_embedding_function(
        model="vllm-sfr",
        profile_name=SFR_EMBEDDING_PROFILE,
        base_url="http://remote-box:8001/v1/embeddings",
        fallback_to_env=False,
    )

    embedding_fn.embed_query("What supports local embeddings?")
    embedding_fn.embed_documents(["Document chunk"])

    assert embedding_fn.last_query == SFR_QUERY_TEMPLATE.format(
        text="What supports local embeddings?"
    )
    assert embedding_fn.last_documents == ["Document chunk"]
    assert embedding_fn.kwargs["base_url"] == "http://remote-box:8001/v1"
    assert embedding_fn.kwargs["check_embedding_ctx_length"] is False


def test_custom_embedding_route_disables_client_side_token_splitting():
    kwargs = build_embedding_delegate_kwargs(
        api_key="EMPTY",
        base_url="http://remote-box:8001/v1/embeddings",
        fallback_to_env=False,
    )

    assert kwargs["base_url"] == "http://remote-box:8001/v1"
    assert kwargs["check_embedding_ctx_length"] is False


def test_openai_embedding_route_keeps_default_length_checking():
    kwargs = build_embedding_delegate_kwargs(
        api_key="sk-test",
        base_url=None,
        fallback_to_env=False,
    )

    assert "check_embedding_ctx_length" not in kwargs

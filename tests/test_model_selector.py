from RAG.generation.model_selector import (
    EMBEDDING_MODELS_CACHE_SESSION_KEY,
    PROMPT_MODELS_CACHE_SESSION_KEY,
    RERANK_MODELS_CACHE_SESSION_KEY,
    DEFAULT_MODEL_OPTIONS,
    cache_prompt_model_options,
    discover_embedding_model_options,
    discover_prompt_model_options,
    discover_rerank_model_options,
    load_embedding_model_options,
    load_prompt_model_options,
    load_rerank_model_options,
)


class _FakeModel:
    def __init__(self, model_id: str):
        self.id = model_id


class _FakeModelsEndpoint:
    def __init__(self, model_ids: list[str]):
        self._model_ids = model_ids

    def list(self):
        return type(
            "Response",
            (),
            {"data": [_FakeModel(model_id) for model_id in self._model_ids]},
        )()


class _FakeClient:
    def __init__(self, model_ids: list[str], *, base_url: str):
        self.base_url = base_url
        self.models = _FakeModelsEndpoint(model_ids)


def test_discover_prompt_model_options_filters_embedding_ids():
    client = _FakeClient(
        [
            "text-embedding-3-small",
            "vllm-sfr-embedding-mistral",
            "codex-mini",
            "local-gemma4-chat",
        ],
        base_url="http://nvidiaspark:8000/v1",
    )

    assert discover_prompt_model_options(client) == ["local-gemma4-chat"]


def test_discover_embedding_model_options_keeps_custom_vllm_name():
    client = _FakeClient(
        ["vllm-sfr-embedding-mistral"],
        base_url="http://nvidiaspark:8001/v1",
    )

    assert discover_embedding_model_options(client) == [
        "vllm-sfr-embedding-mistral"
    ]


def test_discover_rerank_model_options_keeps_custom_vllm_name():
    client = _FakeClient(
        ["nemotron-rerank-1b-v2"],
        base_url="http://nvidiaspark:8002/v1",
    )

    assert discover_rerank_model_options(client) == ["nemotron-rerank-1b-v2"]


def test_load_prompt_model_options_returns_cached_empty_list():
    client = _FakeClient([], base_url="http://nvidiaspark:8000/v1")
    session = {}

    cache_prompt_model_options([], client=client, session=session)

    assert load_prompt_model_options(client, session=session) == []
    assert session[PROMPT_MODELS_CACHE_SESSION_KEY] == "http://nvidiaspark:8000/v1"


def test_load_embedding_model_options_discovers_model_from_base_url(monkeypatch):
    session = {}
    expected_client = _FakeClient(
        ["vllm-sfr-embedding-mistral"],
        base_url="http://nvidiaspark:8001/v1",
    )

    def _fake_make_openai_client(*, api_key, base_url, fallback_to_env):
        assert api_key == "EMPTY"
        assert base_url == "http://nvidiaspark:8001/v1/embeddings"
        assert fallback_to_env is False
        return expected_client

    monkeypatch.setattr(
        "RAG.generation.model_selector.make_openai_client",
        _fake_make_openai_client,
    )

    discovered = load_embedding_model_options(
        api_key="EMPTY",
        base_url="http://nvidiaspark:8001/v1/embeddings",
        session=session,
    )

    assert discovered == ["vllm-sfr-embedding-mistral"]
    assert (
        session[EMBEDDING_MODELS_CACHE_SESSION_KEY]
        == "http://nvidiaspark:8001/v1"
    )


def test_load_prompt_model_options_uses_defaults_without_client():
    assert load_prompt_model_options(None, session={}) == DEFAULT_MODEL_OPTIONS


def test_load_rerank_model_options_discovers_model_from_base_url(monkeypatch):
    session = {}
    expected_client = _FakeClient(
        ["nemotron-rerank-1b-v2"],
        base_url="http://nvidiaspark:8002/v1",
    )

    def _fake_make_openai_client(*, api_key, base_url, fallback_to_env):
        assert api_key == "EMPTY"
        assert base_url == "http://nvidiaspark:8002/v1/rerank"
        assert fallback_to_env is False
        return expected_client

    monkeypatch.setattr(
        "RAG.generation.model_selector.make_openai_client",
        _fake_make_openai_client,
    )

    discovered = load_rerank_model_options(
        api_key="EMPTY",
        base_url="http://nvidiaspark:8002/v1/rerank",
        session=session,
    )

    assert discovered == ["nemotron-rerank-1b-v2"]
    assert (
        session[RERANK_MODELS_CACHE_SESSION_KEY]
        == "http://nvidiaspark:8002/v1"
    )

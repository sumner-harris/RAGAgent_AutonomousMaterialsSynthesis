import os

from openai import OpenAI


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_BASE_ENV = "OPENAI_API_BASE"
LOCAL_API_KEY_PLACEHOLDER = "EMPTY"
OPENAI_ENDPOINT_SUFFIXES = (
    "/chat/completions",
    "/responses",
    "/embeddings",
    "/completions",
    "/rerank",
    "/score",
)


def normalize_base_url(base_url: str | None) -> str | None:
    """Normalize an OpenAI-compatible base URL.

    Users often paste a full endpoint such as `/v1/embeddings`,
    `/v1/responses`, or `/v1/rerank`. The OpenAI client expects the shared
    API base instead, so we trim known endpoint suffixes down to the parent
    `/v1` path.
    """
    value = (base_url or "").strip()
    if not value:
        return None

    value = value.rstrip("/")
    while True:
        lowered = value.lower()
        for suffix in OPENAI_ENDPOINT_SUFFIXES:
            if lowered.endswith(suffix):
                value = value[: -len(suffix)].rstrip("/")
                break
        else:
            break

    return value or None


def resolve_base_url(
    base_url: str | None = None,
    *,
    fallback_to_env: bool = True,
) -> str | None:
    direct_value = normalize_base_url(base_url)
    if direct_value:
        return direct_value
    if fallback_to_env:
        return normalize_base_url(os.environ.get(OPENAI_API_BASE_ENV))
    return None


def resolve_api_key(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    fallback_to_env: bool = True,
) -> str | None:
    value = (api_key or "").strip()
    if not value and fallback_to_env:
        value = (os.environ.get(OPENAI_API_KEY_ENV) or "").strip()
    if value:
        return value
    if resolve_base_url(base_url, fallback_to_env=fallback_to_env):
        return LOCAL_API_KEY_PLACEHOLDER
    return None


def connection_is_configured(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    fallback_to_env: bool = True,
) -> bool:
    return bool(
        resolve_api_key(api_key, base_url, fallback_to_env=fallback_to_env)
        or resolve_base_url(base_url, fallback_to_env=fallback_to_env)
    )


def build_openai_client_kwargs(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    fallback_to_env: bool = True,
):
    resolved_base_url = resolve_base_url(
        base_url,
        fallback_to_env=fallback_to_env,
    )
    resolved_api_key = resolve_api_key(
        api_key,
        resolved_base_url,
        fallback_to_env=fallback_to_env,
    )

    kwargs = {}
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return kwargs


def make_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    fallback_to_env: bool = True,
) -> OpenAI:
    return OpenAI(
        **build_openai_client_kwargs(
            api_key=api_key,
            base_url=base_url,
            fallback_to_env=fallback_to_env,
        )
    )


def apply_openai_env(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    include_base_url: bool = True,
):
    resolved_base_url = resolve_base_url(base_url, fallback_to_env=False)
    resolved_api_key = resolve_api_key(
        api_key,
        resolved_base_url,
        fallback_to_env=False,
    )

    if resolved_api_key:
        os.environ[OPENAI_API_KEY_ENV] = resolved_api_key
    else:
        os.environ.pop(OPENAI_API_KEY_ENV, None)

    if include_base_url and resolved_base_url:
        os.environ[OPENAI_API_BASE_ENV] = resolved_base_url
    else:
        os.environ.pop(OPENAI_API_BASE_ENV, None)

    return resolved_api_key, resolved_base_url

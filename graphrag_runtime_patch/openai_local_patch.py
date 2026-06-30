"""Patch local GraphRAG fnllm calls to disable vLLM thinking.

This module is loaded only inside the GraphRAG indexing subprocess via a
temporary PYTHONPATH + sitecustomize hook. It leaves the main Streamlit app and
non-GraphRAG model routing untouched.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


PATCH_ENV_VAR = "RAG_GRAPHRAG_DISABLE_THINKING"


def is_patch_enabled() -> bool:
    """Return whether the GraphRAG local-thinking patch is enabled."""
    return os.environ.get(PATCH_ENV_VAR, "").strip() == "1"


def is_openai_hosted_base_url(base_url: str | None) -> bool:
    """Return whether the base URL points at an OpenAI-hosted endpoint."""
    if not base_url:
        return True

    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "api.openai.com" or hostname.endswith(".openai.com")


def should_patch_chat_config(config: dict) -> bool:
    """Return whether a model config should receive local no-thinking params."""
    if not is_patch_enabled():
        return False

    model_type = str(config.get("type") or "").lower()
    if "chat" not in model_type:
        return False

    return not is_openai_hosted_base_url(config.get("api_base"))


def inject_disable_thinking_params(
    *,
    config: dict,
    params: dict,
) -> dict:
    """Return chat parameters with local vLLM thinking disabled when needed."""
    result = dict(params)
    if not should_patch_chat_config(config):
        return result

    extra_body = dict(result.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    result["extra_body"] = extra_body
    return result


def install() -> bool:
    """Install the fnllm parameter patch when enabled for GraphRAG indexing."""
    if not is_patch_enabled():
        return False

    try:
        from graphrag.language_model.providers.fnllm import utils as fnllm_utils
    except Exception:
        return False

    if getattr(fnllm_utils, "_rag_local_thinking_patch_installed", False):
        return True

    original = fnllm_utils.get_openai_model_parameters_from_dict

    def patched(config: dict) -> dict:
        params = original(config)
        return inject_disable_thinking_params(config=config, params=params)

    fnllm_utils.get_openai_model_parameters_from_dict = patched
    fnllm_utils._rag_local_thinking_patch_installed = True
    fnllm_utils._rag_original_get_openai_model_parameters_from_dict = original
    return True

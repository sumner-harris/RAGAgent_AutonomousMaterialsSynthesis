import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
import subprocess
from urllib.parse import urlparse

import yaml
from langchain_core.retrievers import BaseRetriever

from RAG.openai_compat import (
    LOCAL_API_KEY_PLACEHOLDER,
    resolve_api_key,
    resolve_base_url,
)
from state.config import (
    GRAPHRAG_CONCURRENT_REQUESTS_LOCAL,
    GRAPHRAG_CONCURRENT_REQUESTS_OPENAI,
    TOKENIZER_NAME,
)


DEFAULT_CHAT_MODEL_ID = "default_chat_model"
DEFAULT_EMBEDDING_MODEL_ID = "default_embedding_model"
GRAPHRAG_LOCAL_THINKING_PATCH_ENV = "RAG_GRAPHRAG_DISABLE_THINKING"
GRAPHRAG_WORKFLOW_PREFIXES = (
    "Starting workflow:",
    "Workflow started:",
)
GRAPHRAG_WORKFLOW_COMPLETE_PREFIXES = (
    "Workflow complete:",
    "Workflow completed:",
)
GRAPHRAG_PIPELINE_PREFIXES = (
    "Starting pipeline with workflows:",
    "Pipeline started with workflows:",
)
GRAPHRAG_PIPELINE_COMPLETE = "Pipeline complete"
GRAPHRAG_PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)\b")

StatusCallback = Callable[[str], None]
GRAPHRAG_INDEX_MODE = "index"
GRAPHRAG_UPDATE_MODE = "update"
GRAPHRAG_SUPPORTED_MODES = {
    GRAPHRAG_INDEX_MODE,
    GRAPHRAG_UPDATE_MODE,
}


def _update_model_entry(model_cfg, *, model_type, model_name, base_url):
    model_cfg["type"] = model_type
    model_cfg["auth_type"] = "api_key"
    model_cfg["api_key"] = "${GRAPHRAG_API_KEY}"
    model_cfg["model"] = model_name
    model_cfg["encoding_model"] = TOKENIZER_NAME
    if base_url:
        model_cfg["api_base"] = base_url
    else:
        model_cfg.pop("api_base", None)


def _is_openai_hosted_base_url(base_url: str | None) -> bool:
    """Return whether the base URL points at an OpenAI-hosted endpoint."""
    if not base_url:
        return True

    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "api.openai.com" or hostname.endswith(".openai.com")


def _resolve_graphrag_concurrent_requests(base_url: str | None) -> int:
    """Choose a safe GraphRAG chat concurrency for the configured endpoint."""
    if _is_openai_hosted_base_url(base_url):
        return GRAPHRAG_CONCURRENT_REQUESTS_OPENAI

    return GRAPHRAG_CONCURRENT_REQUESTS_LOCAL


def _apply_graphrag_rate_limit_defaults(
    model_cfg: dict,
    *,
    base_url: str | None,
    hosted_default: str | None,
) -> None:
    """Apply endpoint-aware RPM/TPM defaults for GraphRAG model settings."""
    if _is_openai_hosted_base_url(base_url):
        model_cfg.setdefault("requests_per_minute", hosted_default)
        model_cfg.setdefault("tokens_per_minute", hosted_default)
        return

    model_cfg["requests_per_minute"] = None
    model_cfg["tokens_per_minute"] = None


def _build_graphrag_subprocess_env(base_url: str | None) -> dict[str, str]:
    """Build subprocess environment for GraphRAG CLI runs.

    For local OpenAI-compatible chat endpoints, GraphRAG indexing uses a small
    sitecustomize patch that injects `chat_template_kwargs.enable_thinking=false`
    into chat-completion requests. OpenAI-hosted endpoints are left unchanged.
    """
    env = os.environ.copy()
    if _is_openai_hosted_base_url(base_url):
        env.pop(GRAPHRAG_LOCAL_THINKING_PATCH_ENV, None)
        return env

    patch_dir = Path(__file__).resolve().parent.parent / "graphrag_runtime_patch"
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath_parts = [str(patch_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env[GRAPHRAG_LOCAL_THINKING_PATCH_ENV] = "1"
    return env


def _update_graphrag_settings(
    settings_path: Path,
    *,
    base_url: str | None,
    embedding_base_url: str | None,
    chat_model: str | None,
    embedding_model: str | None,
):
    config = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    models = config.setdefault("models", {})

    chat_cfg = models.setdefault(DEFAULT_CHAT_MODEL_ID, {})
    embedding_cfg = models.setdefault(DEFAULT_EMBEDDING_MODEL_ID, {})

    _update_model_entry(
        chat_cfg,
        model_type="openai_chat",
        model_name=chat_model or chat_cfg.get("model") or "gpt-4.1-2025-04-14",
        base_url=base_url,
    )
    chat_cfg["concurrent_requests"] = _resolve_graphrag_concurrent_requests(
        base_url
    )
    _apply_graphrag_rate_limit_defaults(
        chat_cfg,
        base_url=base_url,
        hosted_default="auto",
    )
    _update_model_entry(
        embedding_cfg,
        model_type="openai_embedding",
        model_name=embedding_model
        or embedding_cfg.get("model")
        or "text-embedding-3-small",
        base_url=embedding_base_url,
    )
    _apply_graphrag_rate_limit_defaults(
        embedding_cfg,
        base_url=embedding_base_url,
        hosted_default=None,
    )

    settings_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _emit_status(status_callback: StatusCallback | None, message: str) -> None:
    """Emit one GraphRAG status update if a callback was provided."""
    if status_callback and message:
        status_callback(message)


def _normalize_graphrag_mode(mode: str) -> str:
    """Validate and normalize a GraphRAG CLI pipeline mode."""
    normalized = str(mode or GRAPHRAG_INDEX_MODE).strip().lower()
    if normalized not in GRAPHRAG_SUPPORTED_MODES:
        supported = ", ".join(sorted(GRAPHRAG_SUPPORTED_MODES))
        raise ValueError(f"Unsupported GraphRAG mode '{mode}'. Supported: {supported}")
    return normalized


def _graphrag_mode_label(mode: str) -> str:
    """Return a user-facing label for a GraphRAG CLI pipeline mode."""
    return (
        "incremental update"
        if mode == GRAPHRAG_UPDATE_MODE
        else "indexing"
    )


def _iter_stream_messages(stream) -> Iterable[str]:
    """Yield newline- or carriage-return-delimited messages from a text stream."""
    buffer = []
    while True:
        char = stream.read(1)
        if char == "":
            break
        if char in {"\r", "\n"}:
            message = "".join(buffer).strip()
            buffer.clear()
            if message:
                yield message
            continue
        buffer.append(char)

    if buffer:
        message = "".join(buffer).strip()
        if message:
            yield message


def _format_status_message(
    raw_line: str,
    current_workflow: str | None,
) -> tuple[str | None, str | None]:
    """Convert one GraphRAG console line into a user-facing status message."""
    line = raw_line.strip()
    if not line:
        return current_workflow, None

    if any(line.startswith(prefix) for prefix in GRAPHRAG_PIPELINE_PREFIXES):
        return current_workflow, "GraphRAG pipeline started."

    if any(line.startswith(prefix) for prefix in GRAPHRAG_WORKFLOW_PREFIXES):
        current_workflow = line.split(":", 1)[1].strip() or current_workflow
        if current_workflow:
            return current_workflow, f"GraphRAG step: {current_workflow}"
        return current_workflow, "GraphRAG started a workflow."

    if any(
        line.startswith(prefix) for prefix in GRAPHRAG_WORKFLOW_COMPLETE_PREFIXES
    ):
        completed_workflow = line.split(":", 1)[1].strip() or current_workflow
        if completed_workflow:
            return completed_workflow, f"GraphRAG completed: {completed_workflow}"
        return current_workflow, "GraphRAG completed a workflow."

    if line == GRAPHRAG_PIPELINE_COMPLETE:
        return current_workflow, "GraphRAG pipeline complete."

    progress_match = GRAPHRAG_PROGRESS_PATTERN.search(line)
    if progress_match:
        completed, total = progress_match.groups()
        if current_workflow:
            return (
                current_workflow,
                f"GraphRAG progress ({current_workflow}): {completed}/{total}",
            )
        return current_workflow, f"GraphRAG progress: {completed}/{total}"

    return current_workflow, f"GraphRAG: {line}"


def _run_graphrag_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    status_callback: StatusCallback | None = None,
) -> tuple[int, str]:
    """Run one GraphRAG CLI command and stream user-facing status updates."""
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        return process.wait(), ""

    current_workflow = None
    output_lines = []
    last_message = None
    for raw_line in _iter_stream_messages(process.stdout):
        output_lines.append(raw_line)
        current_workflow, message = _format_status_message(
            raw_line,
            current_workflow,
        )
        if message and message != last_message:
            _emit_status(status_callback, message)
            last_message = message

    return process.wait(), "\n".join(output_lines)


def run_graphrag_cli(
    root_dir: str,
    input_dir: str,
    output_dir: str,
    *,
    api_key: str | None,
    base_url: str | None = None,
    embedding_base_url: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    mode: str = GRAPHRAG_INDEX_MODE,
    status_callback: StatusCallback | None = None,
):
    """Run a GraphRAG indexing or incremental-update CLI workflow.

    Args:
        root_dir: GraphRAG workspace root containing settings and IO folders.
        input_dir: Workspace input directory. Reserved for API compatibility.
        output_dir: Workspace output directory. Reserved for API compatibility.
        api_key: API key or placeholder for the configured LLM provider.
        base_url: Optional chat-model OpenAI-compatible base URL.
        embedding_base_url: Optional embedding-model OpenAI-compatible base URL.
        chat_model: Chat model ID to write into the GraphRAG settings file.
        embedding_model: Embedding model ID to write into the settings file.
        mode: One of `index` or `update`.
        status_callback: Optional callback for streaming user-facing status text.

    Returns:
        True when the CLI run exits successfully, else False.
    """
    settings_path = Path(root_dir) / "settings.yaml"
    env_path = Path(root_dir) / ".env"
    normalized_mode = _normalize_graphrag_mode(mode)
    mode_label = _graphrag_mode_label(normalized_mode)

    resolved_base_url = resolve_base_url(base_url)
    resolved_api_key = resolve_api_key(api_key, resolved_base_url)
    graphrag_env = _build_graphrag_subprocess_env(resolved_base_url)

    _emit_status(status_callback, "Preparing GraphRAG configuration...")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"GRAPHRAG_API_KEY={resolved_api_key or LOCAL_API_KEY_PLACEHOLDER}\n")

    if not settings_path.exists():
        _emit_status(status_callback, "Initializing GraphRAG workspace...")
        subprocess.run(
            ["graphrag", "init", "--root", str(root_dir)],
            check=True,
            env=graphrag_env,
        )

    _update_graphrag_settings(
        settings_path,
        base_url=resolved_base_url,
        embedding_base_url=resolve_base_url(
            embedding_base_url,
            fallback_to_env=False,
        ),
        chat_model=chat_model,
        embedding_model=embedding_model,
    )

    _emit_status(status_callback, f"Starting GraphRAG {mode_label} pipeline...")
    returncode, _output = _run_graphrag_command(
        ["graphrag", normalized_mode, "--root", str(root_dir)],
        env=graphrag_env,
        status_callback=status_callback,
    )
    if returncode == 0:
        _emit_status(
            status_callback,
            f"GraphRAG {mode_label} finished successfully.",
        )
    else:
        _emit_status(
            status_callback,
            f"GraphRAG {mode_label} failed. Check GraphRAG logs for details.",
        )
    return returncode == 0


class StaticGraphRetriever(BaseRetriever):
    def __init__(self, graph_docs):
        super().__init__()
        self.graph_docs = graph_docs

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.graph_docs

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self.graph_docs

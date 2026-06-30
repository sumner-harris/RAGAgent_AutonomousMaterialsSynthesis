import io
import os

import pytest
import yaml

from graphrag_runtime_patch.openai_local_patch import (
    PATCH_ENV_VAR,
    inject_disable_thinking_params,
)
from ingestion.graphrag import (
    GRAPHRAG_LOCAL_THINKING_PATCH_ENV,
    _build_graphrag_subprocess_env,
    _format_status_message,
    _is_openai_hosted_base_url,
    _iter_stream_messages,
    _run_graphrag_command,
    _update_graphrag_settings,
    run_graphrag_cli,
)
from state.config import (
    GRAPHRAG_CONCURRENT_REQUESTS_LOCAL,
    GRAPHRAG_CONCURRENT_REQUESTS_OPENAI,
)


def test_iter_stream_messages_splits_on_newlines_and_carriage_returns():
    stream = io.StringIO(
        "Starting workflow: extract\r"
        "  1 / 5 ....\r"
        "Workflow complete: extract\n"
        "Pipeline complete\n"
    )

    assert list(_iter_stream_messages(stream)) == [
        "Starting workflow: extract",
        "1 / 5 ....",
        "Workflow complete: extract",
        "Pipeline complete",
    ]


def test_format_status_message_tracks_workflow_progress():
    workflow, message = _format_status_message(
        "Starting workflow: create_base_text_units",
        None,
    )
    assert workflow == "create_base_text_units"
    assert message == "GraphRAG step: create_base_text_units"

    workflow, message = _format_status_message("1 / 5 ....", workflow)
    assert workflow == "create_base_text_units"
    assert message == "GraphRAG progress (create_base_text_units): 1/5"

    workflow, message = _format_status_message(
        "Workflow complete: create_base_text_units",
        workflow,
    )
    assert workflow == "create_base_text_units"
    assert message == "GraphRAG completed: create_base_text_units"


def test_format_status_message_supports_installed_graphrag_log_variants():
    workflow, message = _format_status_message(
        "Workflow started: extract_graph",
        None,
    )
    assert workflow == "extract_graph"
    assert message == "GraphRAG step: extract_graph"

    workflow, message = _format_status_message(
        "chunker progress:  2/60",
        workflow,
    )
    assert workflow == "extract_graph"
    assert message == "GraphRAG progress (extract_graph): 2/60"

    workflow, message = _format_status_message(
        "Workflow completed: extract_graph",
        workflow,
    )
    assert workflow == "extract_graph"
    assert message == "GraphRAG completed: extract_graph"


def test_run_graphrag_command_streams_status_updates(monkeypatch):
    class FakeProcess:
        def __init__(self, text: str, returncode: int):
            self.stdout = io.StringIO(text)
            self._returncode = returncode

        def wait(self):
            return self._returncode

    output = (
        "Starting pipeline with workflows: create_base_text_units\n"
        "Starting workflow: create_base_text_units\n"
        "  1 / 5 ....\r"
        "  5 / 5 .....\r"
        "Workflow complete: create_base_text_units\n"
        "Pipeline complete\n"
    )

    monkeypatch.setattr(
        "ingestion.graphrag.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(output, 0),
    )

    statuses = []
    returncode, raw_output = _run_graphrag_command(
        ["graphrag", "index", "--root", "workspace"],
        status_callback=statuses.append,
    )

    assert returncode == 0
    assert "Starting workflow: create_base_text_units" in raw_output
    assert statuses == [
        "GraphRAG pipeline started.",
        "GraphRAG step: create_base_text_units",
        "GraphRAG progress (create_base_text_units): 1/5",
        "GraphRAG progress (create_base_text_units): 5/5",
        "GraphRAG completed: create_base_text_units",
        "GraphRAG pipeline complete.",
    ]


def test_run_graphrag_command_streams_installed_graphrag_log_messages(monkeypatch):
    class FakeProcess:
        def __init__(self, text: str, returncode: int):
            self.stdout = io.StringIO(text)
            self._returncode = returncode

        def wait(self):
            return self._returncode

    output = (
        "Workflow started: create_base_text_units\n"
        "chunker progress:  1/3\n"
        "chunker progress:  3/3\n"
        "Workflow completed: create_base_text_units\n"
        "Workflow started: extract_graph\n"
    )

    monkeypatch.setattr(
        "ingestion.graphrag.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(output, 0),
    )

    statuses = []
    returncode, raw_output = _run_graphrag_command(
        ["graphrag", "index", "--root", "workspace"],
        status_callback=statuses.append,
    )

    assert returncode == 0
    assert "Workflow started: extract_graph" in raw_output
    assert statuses == [
        "GraphRAG step: create_base_text_units",
        "GraphRAG progress (create_base_text_units): 1/3",
        "GraphRAG progress (create_base_text_units): 3/3",
        "GraphRAG completed: create_base_text_units",
        "GraphRAG step: extract_graph",
    ]


def test_update_graphrag_settings_uses_openai_concurrency_by_default(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"models": {}}, sort_keys=False),
        encoding="utf-8",
    )

    _update_graphrag_settings(
        settings_path,
        base_url=None,
        embedding_base_url=None,
        chat_model="gpt-4.1",
        embedding_model="text-embedding-3-small",
    )

    config = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert (
        config["models"]["default_chat_model"]["concurrent_requests"]
        == GRAPHRAG_CONCURRENT_REQUESTS_OPENAI
    )
    assert config["models"]["default_chat_model"]["requests_per_minute"] == "auto"
    assert config["models"]["default_chat_model"]["tokens_per_minute"] == "auto"


def test_update_graphrag_settings_uses_local_concurrency_for_custom_base_url(
    tmp_path,
):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"models": {}}, sort_keys=False),
        encoding="utf-8",
    )

    _update_graphrag_settings(
        settings_path,
        base_url="http://nvidiaspark:8000/v1",
        embedding_base_url="http://nvidiaspark:8001/v1",
        chat_model="gemma-4-31b-it",
        embedding_model="vllm-sfr-embedding-mistral",
    )

    config = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert (
        config["models"]["default_chat_model"]["concurrent_requests"]
        == GRAPHRAG_CONCURRENT_REQUESTS_LOCAL
    )
    assert config["models"]["default_chat_model"]["requests_per_minute"] is None
    assert config["models"]["default_chat_model"]["tokens_per_minute"] is None
    assert config["models"]["default_embedding_model"]["requests_per_minute"] is None
    assert config["models"]["default_embedding_model"]["tokens_per_minute"] is None


def test_update_graphrag_settings_keeps_openai_concurrency_for_openai_base_url(
    tmp_path,
):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"models": {}}, sort_keys=False),
        encoding="utf-8",
    )

    _update_graphrag_settings(
        settings_path,
        base_url="https://api.openai.com/v1",
        embedding_base_url=None,
        chat_model="gpt-4.1",
        embedding_model="text-embedding-3-small",
    )

    config = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert (
        config["models"]["default_chat_model"]["concurrent_requests"]
        == GRAPHRAG_CONCURRENT_REQUESTS_OPENAI
    )
    assert config["models"]["default_chat_model"]["requests_per_minute"] == "auto"
    assert config["models"]["default_chat_model"]["tokens_per_minute"] == "auto"


def test_run_graphrag_cli_uses_update_mode_and_emits_update_statuses(
    tmp_path,
    monkeypatch,
):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"models": {}}, sort_keys=False),
        encoding="utf-8",
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    captured = {}

    def fake_run(command, *, env=None, status_callback=None):
        captured["command"] = command
        captured["env"] = env
        if status_callback:
            status_callback("GraphRAG pipeline complete.")
        return 0, "Pipeline complete"

    monkeypatch.setattr("ingestion.graphrag._run_graphrag_command", fake_run)

    statuses = []
    ok = run_graphrag_cli(
        str(tmp_path),
        str(input_dir),
        str(output_dir),
        api_key="test-key",
        base_url="http://nvidiaspark:8000/v1",
        embedding_base_url="http://nvidiaspark:8001/v1",
        chat_model="gemma-4-31b-it",
        embedding_model="vllm-sfr-embedding-mistral",
        mode="update",
        status_callback=statuses.append,
    )

    assert ok is True
    assert captured["command"] == ["graphrag", "update", "--root", str(tmp_path)]
    assert statuses == [
        "Preparing GraphRAG configuration...",
        "Starting GraphRAG incremental update pipeline...",
        "GraphRAG pipeline complete.",
        "GraphRAG incremental update finished successfully.",
    ]


def test_run_graphrag_cli_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Unsupported GraphRAG mode"):
        run_graphrag_cli(
            str(tmp_path),
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            api_key="test-key",
            mode="not-a-real-mode",
        )


def test_is_openai_hosted_base_url_recognizes_openai_and_local_hosts():
    assert _is_openai_hosted_base_url(None) is True
    assert _is_openai_hosted_base_url("https://api.openai.com/v1") is True
    assert _is_openai_hosted_base_url("https://example.openai.com/v1") is True
    assert _is_openai_hosted_base_url("http://nvidiaspark:8000/v1") is False


def test_build_graphrag_subprocess_env_enables_local_patch(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _build_graphrag_subprocess_env("http://nvidiaspark:8000/v1")

    assert env[GRAPHRAG_LOCAL_THINKING_PATCH_ENV] == "1"
    assert env["PYTHONPATH"].startswith(
        "C:\\Users\\fqh\\Documents\\rag\\graphrag_runtime_patch"
    )
    assert env["PYTHONPATH"].endswith("existing-path")


def test_build_graphrag_subprocess_env_leaves_openai_hosted_runs_unpatched(
    monkeypatch,
):
    monkeypatch.setenv(GRAPHRAG_LOCAL_THINKING_PATCH_ENV, "1")
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _build_graphrag_subprocess_env("https://api.openai.com/v1")

    assert GRAPHRAG_LOCAL_THINKING_PATCH_ENV not in env
    assert env["PYTHONPATH"] == "existing-path"


def test_inject_disable_thinking_params_for_local_chat(monkeypatch):
    monkeypatch.setenv(PATCH_ENV_VAR, "1")

    result = inject_disable_thinking_params(
        config={
            "type": "openai_chat",
            "api_base": "http://nvidiaspark:8000/v1",
        },
        params={"temperature": 0.1},
    )

    assert result["temperature"] == 0.1
    assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_inject_disable_thinking_params_skips_openai_and_embeddings(monkeypatch):
    monkeypatch.setenv(PATCH_ENV_VAR, "1")

    openai_result = inject_disable_thinking_params(
        config={
            "type": "openai_chat",
            "api_base": "https://api.openai.com/v1",
        },
        params={"temperature": 0.1},
    )
    embedding_result = inject_disable_thinking_params(
        config={
            "type": "openai_embedding",
            "api_base": "http://nvidiaspark:8001/v1",
        },
        params={},
    )

    assert "extra_body" not in openai_result
    assert "extra_body" not in embedding_result

"""
Hosted mode's refusals.

Every test here is about something the server must *not* do. The local tool is
allowed to run commands on your machine because you asked it to; a server is
not, and the difference is entirely in this file and in `codepilot/server.py`.

What is not tested here, and is said plainly in DEPLOYING.md: no container ever
starts in this suite. The machine these were written on has no Docker daemon,
so the container path is covered as far as "the right arguments are built" and
no further.
"""

from __future__ import annotations

import pytest

from codepilot.permissions import Decision
from codepilot.server import ConfigError, ServerConfig, build_app

GOOD_TOKEN = "x" * 32


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "app.py").write_text("x = 1\n", encoding="utf-8")
    for name in (
        "CODEPILOT_SERVER_TOKEN", "CODEPILOT_WORKSPACE_ROOT", "CODEPILOT_SANDBOX",
        "CODEPILOT_ALLOW_HOST_EXECUTION", "CODEPILOT_ALLOW_SERVER_KEY",
        "CODEPILOT_MAX_USD", "CODEPILOT_EXTRA_ALLOWED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODEPILOT_SERVER_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("CODEPILOT_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# It refuses to start rather than starting insecurely
# ---------------------------------------------------------------------------


def test_no_token_is_refused(workspace, monkeypatch):
    monkeypatch.delenv("CODEPILOT_SERVER_TOKEN")
    with pytest.raises(ConfigError, match="CODEPILOT_SERVER_TOKEN"):
        ServerConfig.from_env()


def test_a_short_token_is_refused(workspace, monkeypatch):
    """A four-character token is worse than none: it looks configured."""
    monkeypatch.setenv("CODEPILOT_SERVER_TOKEN", "abcd")
    with pytest.raises(ConfigError, match="16 characters"):
        ServerConfig.from_env()


def test_a_missing_workspace_root_is_refused(workspace, monkeypatch):
    monkeypatch.setenv("CODEPILOT_WORKSPACE_ROOT", str(workspace / "nope"))
    with pytest.raises(ConfigError, match="does not exist"):
        ServerConfig.from_env()


def test_host_execution_is_never_reached_by_omission(workspace, monkeypatch):
    """The whole point of hosted mode is that commands leave the host."""
    monkeypatch.setenv("CODEPILOT_SANDBOX", "local")
    with pytest.raises(ConfigError, match="CODEPILOT_ALLOW_HOST_EXECUTION"):
        ServerConfig.from_env()


def test_host_execution_is_available_when_asked_for_by_name(workspace, monkeypatch):
    monkeypatch.setenv("CODEPILOT_SANDBOX", "local")
    monkeypatch.setenv("CODEPILOT_ALLOW_HOST_EXECUTION", "1")
    assert ServerConfig.from_env().sandbox == "local"


def test_the_default_sandbox_is_the_container(workspace):
    assert ServerConfig.from_env().sandbox == "docker"


def test_an_unknown_sandbox_is_refused(workspace, monkeypatch):
    monkeypatch.setenv("CODEPILOT_SANDBOX", "chroot")
    with pytest.raises(ConfigError, match="must be 'docker' or 'local'"):
        ServerConfig.from_env()


# ---------------------------------------------------------------------------
# A request cannot choose the policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["../secrets", "/etc", "..", "demo/../../elsewhere", "", "a\\b"]
)
def test_a_repository_name_cannot_escape_the_root(workspace, name):
    with pytest.raises(ValueError):
        ServerConfig.from_env().resolve_repo(name)


def test_a_real_repository_resolves(workspace):
    assert ServerConfig.from_env().resolve_repo("demo") == (workspace / "demo").resolve()


def test_an_unknown_repository_is_not_found(workspace):
    with pytest.raises(ValueError, match="no repository"):
        ServerConfig.from_env().resolve_repo("missing")


def test_the_gate_cannot_ask_anyone_so_it_denies(workspace):
    """Locally an unlisted command prompts. There is nobody to prompt here.

    `PermissionGate` already turns ASK into DENY when it has no callback; this
    asserts hosted mode never hands it one, because a server that blocks on an
    unanswerable prompt is a hung request, and one that auto-approves is a
    shell for strangers.
    """
    gate = ServerConfig.from_env().gate()
    assert gate.prompt is None
    assert not gate.auto_approve
    assert gate.classify("curl http://example.com")[0] is Decision.ASK
    allowed, reason = gate.check_command("curl http://example.com")
    assert not allowed
    assert "cannot ask" in reason


def test_the_operator_can_widen_the_allowlist_deliberately(workspace, monkeypatch):
    monkeypatch.setenv("CODEPILOT_EXTRA_ALLOWED", "ruff, mypy")
    gate = ServerConfig.from_env().gate()
    assert gate.check_command("ruff check .")[0]
    assert not gate.check_command("curl http://example.com")[0]


def test_the_budget_comes_from_the_deployment(workspace, monkeypatch):
    monkeypatch.setenv("CODEPILOT_MAX_USD", "0.25")
    assert ServerConfig.from_env().max_usd == 0.25


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def client(workspace):
    from fastapi.testclient import TestClient

    return TestClient(build_app(ServerConfig.from_env()))


AUTH = {"Authorization": f"Bearer {GOOD_TOKEN}"}


def test_health_needs_no_token(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_config_needs_a_token(client):
    assert client.get("/v1/config").status_code == 401
    assert client.get("/v1/config", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_config_reports_what_was_actually_deployed(client):
    body = client.get("/v1/config", headers=AUTH).json()
    assert body["sandbox"] == "docker"
    assert body["host_execution"] is False
    assert body["byok_required"] is True
    assert body["repositories"] == ["demo"]


def test_a_run_without_a_caller_key_is_refused(client):
    r = client.post("/v1/runs", headers=AUTH, json={"prompt": "hi", "repo": "demo"})
    assert r.status_code == 400
    assert "X-Anthropic-Key" in r.json()["detail"]


def test_a_run_without_a_bearer_token_is_refused(client):
    r = client.post("/v1/runs", json={"prompt": "hi", "repo": "demo"})
    assert r.status_code == 401


def test_a_run_against_an_unknown_repository_is_not_found(client):
    r = client.post(
        "/v1/runs",
        headers={**AUTH, "X-Anthropic-Key": "sk-ant-test"},
        json={"prompt": "hi", "repo": "../elsewhere"},
    )
    assert r.status_code == 404


def test_a_server_key_is_lent_out_only_when_configured(workspace, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CODEPILOT_ALLOW_SERVER_KEY", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server")
    cfg = ServerConfig.from_env()
    assert cfg.describe()["byok_required"] is False
    with TestClient(build_app(cfg)) as c:
        # Reaches the run path rather than being turned away for want of a key.
        assert c.post(
            "/v1/runs", headers=AUTH, json={"prompt": "hi", "repo": "nope"}
        ).status_code == 404


# ---------------------------------------------------------------------------
# The container's arguments, which is as far as a machine without Docker goes
# ---------------------------------------------------------------------------


def _sandbox(mount):
    """Build the argument-maker without touching a daemon.

    The constructor pings Docker, and there is none here. This asserts the
    thing that would be wrong in production if it were wrong — which files the
    container sees, and as whom.
    """
    from codepilot.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._mount = mount
    return sandbox


def test_the_eval_container_keeps_nothing():
    options = _sandbox(None)._filesystem_options()
    assert "tmpfs" in options
    assert "volumes" not in options
    assert options["user"] == "nobody"


def test_the_hosted_container_sees_the_session_workspace(tmp_path):
    options = _sandbox(tmp_path)._filesystem_options()
    assert "tmpfs" not in options
    assert options["volumes"] == {str(tmp_path): {"bind": "/workspace", "mode": "rw"}}


def test_a_mounted_container_runs_as_the_host_user_on_posix(tmp_path):
    """A bind mount keeps host ownership, so `nobody` cannot write to it.

    Getting this wrong does not fail loudly: the agent reports that it cannot
    edit the repository it was pointed at.
    """
    import os

    options = _sandbox(tmp_path)._filesystem_options()
    if hasattr(os, "getuid"):
        assert options["user"] == f"{os.getuid()}:{os.getgid()}"
    else:
        assert "user" not in options


# ---------------------------------------------------------------------------
# The run hands its work back
# ---------------------------------------------------------------------------


def test_a_patch_shows_what_changed(tmp_path):
    """Without this the service is an expensive no-op: it works on a copy."""
    from codepilot.server import build_patch

    before, after = tmp_path / "a", tmp_path / "b"
    for d in (before, after):
        d.mkdir()
    (before / "app.py").write_text("x = 1\n", encoding="utf-8")
    (after / "app.py").write_text("x = 2\n", encoding="utf-8")

    patch = build_patch(before, after, ["app.py"])
    assert "-x = 1" in patch
    assert "+x = 2" in patch


def test_a_new_file_appears_as_an_addition(tmp_path):
    from codepilot.server import build_patch

    before, after = tmp_path / "a", tmp_path / "b"
    for d in (before, after):
        d.mkdir()
    (after / "new.py").write_text("y = 3\n", encoding="utf-8")

    patch = build_patch(before, after, ["new.py"])
    assert "/dev/null" in patch
    assert "+y = 3" in patch


def test_an_unchanged_file_produces_nothing(tmp_path):
    from codepilot.server import build_patch

    before, after = tmp_path / "a", tmp_path / "b"
    for d in (before, after):
        d.mkdir()
        (d / "app.py").write_text("x = 1\n", encoding="utf-8")
    assert build_patch(before, after, ["app.py"]) == ""


def test_a_binary_file_is_named_rather_than_mangled(tmp_path):
    from codepilot.server import build_patch

    before, after = tmp_path / "a", tmp_path / "b"
    for d in (before, after):
        d.mkdir()
    (before / "logo.png").write_bytes(b"\x89PNG\x00\xff")
    (after / "logo.png").write_bytes(b"\x89PNG\x00\xfe")
    assert "not a text file" in build_patch(before, after, ["logo.png"])

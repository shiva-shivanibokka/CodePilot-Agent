"""
Hosted mode.

The local CLI edits your repository, on your machine, at your request, and
isolating your code from your own machine would be theatre. A server runs
someone else's prompt against a repository you host, on hardware you pay for,
with a key someone has to supply — and every assumption behind the local mode
is now false.

So this module is not a wrapper around the CLI. It is the list of assumptions
that stop holding, turned into refusals:

1. **There is nobody to ask.** Locally, a command that is not on the allowlist
   prompts you. A server has no one at the keyboard, so `PermissionGate` is
   built with no prompt callback and every ASK becomes a DENY with a reason.
   The gate already behaves this way; hosted mode simply never overrides it.
2. **Commands must not run on the host.** `LocalSandbox` runs the model's
   commands as the server process. Hosted mode requires the container sandbox
   and refuses to start otherwise, unless an operator sets a variable whose
   name says exactly what they are turning off.
3. **The key is not yours.** By default a caller must send their own
   `X-Anthropic-Key`, so the operator is not paying for strangers' tokens and
   is not holding their credentials. A server key is opt-in and named.
4. **A runaway loop costs the operator, not the caller.** Every run gets a
   dollar ceiling from configuration, not from the request, so a caller cannot
   raise their own limit.
5. **The workspace is per-run and inside one root.** A repository name from a
   request is resolved against the configured root and rejected if it escapes.

What this is not: multi-tenant. There is no per-caller quota, no persistence,
no audit log, and one shared bearer token rather than accounts. It is a correct
single-tenant deployment — the shape you would put behind your own gateway —
and `DEPLOYING.md` says so in the same words.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from codepilot.agent.loop import AgentLoop, new_conversation
from codepilot.events import Event, EventStream
from codepilot.llm import STRONG_MODEL, LLMClient
from codepilot.permissions import Budget, PermissionGate
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace


class ConfigError(RuntimeError):
    """The deployment is not safe to start. Raised before anything binds."""


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ServerConfig:
    """Everything the deployment decides, so a request can decide none of it."""

    token: str
    workspace_root: Path
    sandbox: str = "docker"
    sandbox_image: str = "codepilot-sandbox:latest"
    allow_server_key: bool = False
    max_usd: float = 1.00
    max_turns: int = 40
    model: str = STRONG_MODEL
    effort: str = "low"
    #: Extra executables the operator is willing to allow beyond the read-only
    #: defaults. Empty by default: a hosted allowlist is the operator's call.
    extra_allowed: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> ServerConfig:
        token = os.getenv("CODEPILOT_SERVER_TOKEN", "").strip()
        if len(token) < 16:
            raise ConfigError(
                "CODEPILOT_SERVER_TOKEN must be set to at least 16 characters. "
                "There is no default: a server with a guessable token runs "
                "shell commands for whoever guesses it."
            )

        root = os.getenv("CODEPILOT_WORKSPACE_ROOT", "").strip()
        if not root:
            raise ConfigError(
                "CODEPILOT_WORKSPACE_ROOT must name the directory holding the "
                "repositories this server may work on."
            )
        workspace_root = Path(root).expanduser().resolve()
        if not workspace_root.is_dir():
            raise ConfigError(f"CODEPILOT_WORKSPACE_ROOT does not exist: {workspace_root}")

        sandbox = os.getenv("CODEPILOT_SANDBOX", "docker").strip().lower()
        if sandbox not in {"docker", "local"}:
            raise ConfigError(f"CODEPILOT_SANDBOX must be 'docker' or 'local', not {sandbox!r}")
        if sandbox == "local" and not _flag("CODEPILOT_ALLOW_HOST_EXECUTION"):
            raise ConfigError(
                "CODEPILOT_SANDBOX=local runs model-chosen shell commands as this "
                "server process, on the host. If that is genuinely what you want "
                "— a single-user box you already trust the agent with — set "
                "CODEPILOT_ALLOW_HOST_EXECUTION=1 as well. It has no default "
                "because nobody should reach it by omission."
            )

        return cls(
            token=token,
            workspace_root=workspace_root,
            sandbox=sandbox,
            sandbox_image=os.getenv("CODEPILOT_SANDBOX_IMAGE", "codepilot-sandbox:latest"),
            allow_server_key=_flag("CODEPILOT_ALLOW_SERVER_KEY"),
            max_usd=float(os.getenv("CODEPILOT_MAX_USD", "1.00")),
            max_turns=int(os.getenv("CODEPILOT_MAX_TURNS", "40")),
            model=os.getenv("CODEPILOT_MODEL", STRONG_MODEL),
            effort=os.getenv("CODEPILOT_EFFORT", "low"),
            extra_allowed={
                c for c in os.getenv("CODEPILOT_EXTRA_ALLOWED", "").replace(",", " ").split() if c
            },
        )

    # -- policy ----------------------------------------------------------
    def resolve_repo(self, name: str) -> Path:
        """A repository name from a request, resolved inside the root.

        Same containment rule the workspace uses on file paths, applied a level
        up: a request naming `../../etc` must not reach a directory the
        operator never offered.
        """
        if not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts:
            raise ValueError(f"not a repository name: {name!r}")
        target = (self.workspace_root / name).resolve()
        if target != self.workspace_root and self.workspace_root not in target.parents:
            raise ValueError(f"{name!r} is outside the workspace root")
        if not target.is_dir():
            raise ValueError(f"no repository {name!r}")
        return target

    def gate(self) -> PermissionGate:
        """No prompt callback, so every ASK is a DENY with a reason attached."""
        gate = PermissionGate(auto_approve=False, prompt=None)
        gate.allowed |= self.extra_allowed
        return gate

    def describe(self) -> dict:
        """What an operator needs to verify they deployed what they meant to."""
        return {
            "sandbox": self.sandbox,
            "host_execution": self.sandbox == "local",
            "byok_required": not self.allow_server_key,
            "max_usd_per_run": self.max_usd,
            "max_turns_per_run": self.max_turns,
            "model": self.model,
            "effort": self.effort,
            "workspace_root": str(self.workspace_root),
            "repositories": sorted(
                p.name for p in self.workspace_root.iterdir() if p.is_dir()
            ),
            "extra_allowed_commands": sorted(self.extra_allowed),
        }


def build_patch(before: Path, after: Path, paths: list[str]) -> str:
    """A unified diff of what the run changed, against what it started from.

    A hosted run works on a copy and throws the copy away, so without this the
    service is an expensive no-op: it reports that it fixed something and
    leaves no fix. Returning a patch rather than writing to the caller's
    repository is the deliberate half of that — the server does not own their
    branch, their history or their review, and a patch is the smallest thing
    that hands the work back without taking any of those.

    Computed with difflib rather than git so it works on a directory that was
    never a repository.
    """
    chunks: list[str] = []
    for rel in sorted(set(paths)):
        old_path, new_path = before / rel, after / rel
        try:
            old = old_path.read_text(encoding="utf-8").splitlines(keepends=True) if old_path.is_file() else []
            new = new_path.read_text(encoding="utf-8").splitlines(keepends=True) if new_path.is_file() else []
        except (OSError, UnicodeDecodeError):
            # Binary or unreadable: say so rather than emitting a broken hunk.
            chunks.append(f"# {rel}: changed, but not a text file\n")
            continue
        if old == new:
            continue
        diff = difflib.unified_diff(
            old, new,
            fromfile=f"a/{rel}" if old else "/dev/null",
            tofile=f"b/{rel}" if new else "/dev/null",
        )
        chunks.append("".join(diff))
    return "".join(chunks)


async def _make_sandbox(config: ServerConfig, root: Path):
    if config.sandbox == "local":
        from codepilot.sandbox.local import LocalSandbox

        return LocalSandbox(root=root)

    from codepilot.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox(image=config.sandbox_image, mount=root)
    await sandbox.start()
    return sandbox


async def run_stream(config: ServerConfig, *, prompt: str, repo: str, api_key: str):
    """Run one turn, yielding events as they happen.

    Each run gets a copy of the repository rather than the repository itself.
    Two concurrent runs against one checkout would interleave their edits, and
    the undo checkpoints of one would roll back the other.
    """
    source = config.resolve_repo(repo)
    run_id = uuid.uuid4().hex[:12]
    root = source.parent / f".codepilot-run-{run_id}"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns(".codepilot"))

    stream = EventStream(session_id=run_id)
    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    stream.subscribe(queue.put_nowait)

    sandbox = None
    try:
        sandbox = await _make_sandbox(config, root)
        ctx = ToolContext(
            workspace=Workspace(root=root, session_id=run_id),
            sandbox=sandbox,
            permissions=config.gate(),
            events=stream,
        )
        loop = AgentLoop(
            LLMClient(model=config.model, api_key=api_key),
            ctx,
            new_conversation(),
            Budget(max_usd=config.max_usd, max_turns=config.max_turns),
            effort=config.effort,
            model=config.model,
        )
        task = asyncio.create_task(loop.run(prompt))
        task.add_done_callback(lambda _: queue.put_nowait(None))

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event.model_dump(mode="json")

        result = await task
        yield {
            "type": "run_end",
            "run_id": run_id,
            "finished": result.finished,
            "summary": result.summary,
            "stopped_by": result.stopped_by,
            "edited": result.edited,
            "patch": build_patch(source, root, result.edited),
        }
    finally:
        if sandbox is not None:
            await sandbox.close()
        shutil.rmtree(root, ignore_errors=True)


def build_app(config: ServerConfig | None = None):
    """The HTTP surface, imported lazily so policy has no web dependency."""
    from codepilot.http_api import build_app as _build

    return _build(config or ServerConfig.from_env())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run CodePilot as a service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check", action="store_true", help="validate the configuration and exit"
    )
    args = parser.parse_args(argv)

    try:
        cfg = ServerConfig.from_env()
    except ConfigError as exc:
        print(f"refusing to start: {exc}")
        return 2

    if args.check:
        print(json.dumps(cfg.describe(), indent=2))
        return 0

    import uvicorn

    uvicorn.run(build_app(cfg), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

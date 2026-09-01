"""
The command line. This is the tool.

    codepilot run "add retry logic to fetch.py"
    codepilot chat                 # a conversation; Ctrl-C steers, twice quits
    codepilot doctor [--live]      # is everything wired up
    codepilot undo                 # revert the last turn's edits
    codepilot sessions             # what has been run here

Not a web UI: a coding agent belongs where the code is. The Gradio app remains
for recording demos, but it is not where the work happens.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from codepilot.agent.interrupt import InterruptChannel
from codepilot.agent.loop import AgentLoop, new_conversation
from codepilot.events import Event, EventStream, EventType, supports_unicode
from codepilot.llm import STRONG_MODEL, LLMClient, LLMError
from codepilot.permissions import Budget, PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.session import SessionStore, list_sessions
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace, WorkspaceError

PROJECT_FILE = "CODEPILOT.md"

#: Events too chatty for the default view. --verbose shows everything.
QUIET_EVENTS = {EventType.TURN_START, EventType.TURN_END, EventType.USER_MESSAGE}


def _load_env(root: Path) -> None:
    """Read .env from the repository being worked on, then the user's home.

    Explicit paths, because find_dotenv() searches upward from the *caller*,
    which for an installed tool is site-packages.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (root / ".env", Path.home() / ".codepilot.env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


class Renderer:
    """Prints events as they happen."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.unicode = supports_unicode()

    def __call__(self, event: Event) -> None:
        if not self.verbose and event.type in QUIET_EVENTS:
            return
        if event.type is EventType.DIFF and not self.verbose:
            print(f"  {event.render(self.unicode)}")
            return
        text = event.render(self.unicode)
        if event.type in (EventType.ASSISTANT_TEXT, EventType.PLAN):
            print()
        print(text, flush=True)


def _confirm(command: str, reason: str) -> bool:
    """Ask before running something off the allowlist."""
    print(f"\n  Run this? ({reason})\n    $ {command}")
    try:
        answer = input("  [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _build(args, root: Path, session: SessionStore):
    stream = EventStream(session_id=session.session_id)
    stream.subscribe(Renderer(verbose=args.verbose))
    session.attach(stream)

    workspace = Workspace(root=root, session_id=session.session_id)
    gate = PermissionGate(
        auto_approve=args.yes,
        prompt=None if args.yes or not sys.stdin.isatty() else _confirm,
    )
    ctx = ToolContext(
        workspace=workspace,
        sandbox=LocalSandbox(root=root),
        permissions=gate,
        events=stream,
    )
    instructions = None
    project_file = root / PROJECT_FILE
    if project_file.is_file():
        instructions = project_file.read_text(encoding="utf-8", errors="replace")

    convo = new_conversation(project_instructions=instructions)
    convo.messages = session.messages()
    budget = Budget(max_usd=args.max_cost, max_turns=args.max_calls)
    client = LLMClient(model=args.model)
    return stream, ctx, convo, budget, client


async def _prepare(args, root: Path, ctx: ToolContext, client: LLMClient) -> None:
    await client.validate(args.model)
    if args.checkpoint:
        try:
            cp = ctx.workspace.checkpoint("before turn")
            ctx.events.emit(
                EventType.CHECKPOINT,
                f"checkpointed — `codepilot undo` reverts to here ({cp.commit[:8]})",
                ref=cp.ref,
            )
        except WorkspaceError as exc:
            print(f"\n  ! {exc}\n")
            raise SystemExit(2) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_run(args) -> int:
    root = Path(args.directory).resolve()
    _load_env(root)
    session = SessionStore.create(root, args.resume)
    stream, ctx, convo, budget, client = _build(args, root, session)

    try:
        await _prepare(args, root, ctx, client)
    except LLMError as exc:
        print(f"\n  ! {exc}\n")
        return 2

    if args.arm == "pipeline":
        from codepilot.agent.pipeline import Runtime, run_pipeline

        rt = Runtime(
            client=client, ctx=ctx, convo=convo, budget=budget,
            effort=args.effort, model=args.model,
        )
        state = await run_pipeline(rt, args.task)
        finished, summary, edited = state.approved, state.summary, state.edited
    else:
        loop = AgentLoop(client, ctx, convo, budget, effort=args.effort, model=args.model)
        result = await loop.run(args.task)
        finished, summary, edited = result.finished, result.summary, result.edited
        session.record_messages(convo.messages)
    session.record_turn(args.task, summary, budget.summary())

    print(f"\n  session {session.session_id} · arm={args.arm} · {budget.summary()}")
    print(f"  {convo.cache_report()}")
    if edited:
        print(f"  edited: {', '.join(edited)}")
    if args.checkpoint:
        print("  `codepilot undo` reverts this turn")
    return 0 if finished else 1


async def cmd_chat(args) -> int:
    root = Path(args.directory).resolve()
    _load_env(root)
    session = SessionStore.create(root, args.resume)
    stream, ctx, convo, budget, client = _build(args, root, session)

    try:
        await _prepare(args, root, ctx, client)
    except LLMError as exc:
        print(f"\n  ! {exc}\n")
        return 2

    interrupts = InterruptChannel()
    loop = AgentLoop(
        client, ctx, convo, budget, interrupts=interrupts, effort=args.effort, model=args.model
    )

    print(f"\ncodepilot · {root.name} · session {session.session_id}")
    print("  type a task, or /help. Ctrl-C interrupts a running turn.\n")

    while True:
        try:
            task = input("❯ " if supports_unicode() else "> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task in ("/quit", "/exit", "/q"):
            break
        if task == "/help":
            print(
                "  /undo     revert the last turn\n"
                "  /cost     spend so far\n"
                "  /session  this session's id and file\n"
                "  /quit     leave\n"
                "  Ctrl-C during a turn steers it; again stops it."
            )
            continue
        if task == "/cost":
            print(f"  {budget.summary()}\n  {convo.cache_report()}")
            continue
        if task == "/session":
            print(f"  {session.session_id} — {session.path}")
            continue
        if task == "/undo":
            cp = ctx.workspace.undo()
            print(f"  reverted to {cp.label!r}" if cp else "  nothing to undo")
            continue

        if args.checkpoint:
            try:
                ctx.workspace.checkpoint(task[:60])
            except WorkspaceError as exc:
                print(f"  ! {exc}")
                return 2

        # Ctrl-C during a turn steers rather than kills. A second one stops it.
        stopper = _install_interrupt_handler(interrupts)
        try:
            result = await loop.run(task)
        finally:
            stopper()

        session.record_messages(convo.messages)
        session.record_turn(task, result.summary, budget.summary())
        print(f"\n  {budget.summary()}\n")

    print(f"session saved: {session.path}")
    return 0


def _install_interrupt_handler(interrupts: InterruptChannel):
    """Ctrl-C asks the loop to stop at its next checkpoint, not mid-write."""
    import signal

    hits = {"n": 0}
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        hits["n"] += 1
        if hits["n"] == 1:
            interrupts.stop()
            print("\n  ! stopping after the current tool call — Ctrl-C again to force")
        else:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not on the main thread; leave the default handler alone.
        return lambda: None
    return lambda: signal.signal(signal.SIGINT, previous)


async def cmd_undo(args) -> int:
    root = Path(args.directory).resolve()
    sessions = list_sessions(root)
    if not sessions:
        print("  no sessions in this repository")
        return 1
    session_id = args.resume or sessions[0][0]
    ws = Workspace(root=root, session_id=session_id)
    # Recover the checkpoints from git and use Workspace.undo(). Reimplementing
    # the restore here is how this command ended up leaving the agent's new
    # files in place and staging the restored ones.
    if not ws.load_checkpoints():
        print(f"  no checkpoints for session {session_id}")
        return 1
    checkpoint = ws.undo()
    print(f"  restored {checkpoint.ref}" if checkpoint else "  nothing to undo")
    return 0


async def cmd_sessions(args) -> int:
    root = Path(args.directory).resolve()
    rows = list_sessions(root)
    if not rows:
        print("  no sessions yet")
        return 0
    for session_id, modified, turns in rows:
        print(f"  {session_id}  {modified:%Y-%m-%d %H:%M}  {turns} turn(s)")
    return 0


async def cmd_doctor(args) -> int:
    from codepilot.doctor import run

    _load_env(Path(args.directory).resolve())
    return await run(live=args.live)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codepilot", description=__doc__)
    parser.add_argument("-C", "--directory", default=".", help="repository to work in")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--model", default=STRONG_MODEL)
        p.add_argument(
            "--effort",
            default="high",
            choices=["low", "medium", "high", "xhigh", "max"],
            help="cost/quality dial",
        )
        p.add_argument("--max-cost", type=float, default=1.00, dest="max_cost")
        p.add_argument("--max-calls", type=int, default=40, dest="max_calls")
        p.add_argument("--resume", metavar="SESSION_ID")
        p.add_argument("--yes", action="store_true", help="approve every command")
        p.add_argument(
            "--arm",
            default="loop",
            choices=["loop", "pipeline"],
            help="loop: the model chooses each step. pipeline: a fixed plan/code/test/debug/review graph.",
        )
        p.add_argument(
            "--no-checkpoint",
            dest="checkpoint",
            action="store_false",
            help="skip the git snapshot (undo will be unavailable)",
        )
        p.set_defaults(checkpoint=True)

    run_p = sub.add_parser("run", help="run one task and exit")
    run_p.add_argument("task")
    shared(run_p)
    run_p.set_defaults(fn=cmd_run)

    chat_p = sub.add_parser("chat", help="conversational session")
    shared(chat_p)
    chat_p.set_defaults(fn=cmd_chat)

    undo_p = sub.add_parser("undo", help="revert to the last checkpoint")
    undo_p.add_argument("--resume", metavar="SESSION_ID")
    undo_p.set_defaults(fn=cmd_undo)

    sessions_p = sub.add_parser("sessions", help="list sessions in this repository")
    sessions_p.set_defaults(fn=cmd_sessions)

    doctor_p = sub.add_parser("doctor", help="check the installation end to end")
    doctor_p.add_argument("--live", action="store_true", help="include billed API checks")
    doctor_p.set_defaults(fn=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    # A cp1252 console cannot encode the icon set, and a run that dies while
    # printing its own output is a bad look for a tool that edits your code.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if not os.getenv("ANTHROPIC_API_KEY"):
        _load_env(Path(args.directory).resolve())
    try:
        return asyncio.run(args.fn(args))
    except KeyboardInterrupt:
        print("\n  interrupted")
        return 130
    except (WorkspaceError, LLMError) as exc:
        print(f"\n  ! {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

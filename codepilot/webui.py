"""
A local browser UI over the same agent the CLI runs.

Bring your own key: paste it in, or leave it blank to use ANTHROPIC_API_KEY
from the environment. The key is never rendered into the page — a hosted
instance with a key in its environment would otherwise ship that key to every
visitor.

This binds to 127.0.0.1 only. The agent runs shell commands and edits files;
exposing it on a network interface would be handing that to whoever can reach
the port.

    python -m codepilot.webui --directory /path/to/repo
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import gradio as gr

from codepilot.agent.loop import AgentLoop, new_conversation
from codepilot.events import Event, EventStream, EventType
from codepilot.llm import STRONG_MODEL, LLMClient, LLMError, load_env
from codepilot.permissions import Budget, PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.session import SessionStore
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace, WorkspaceError

EMPTY_COSTS = "_No calls yet._"


def cost_table(events: list[Event]) -> str:
    """Per call, not just a total.

    A running total hides which agent step is expensive, and that is the only
    number that changes how you configure the thing.
    """
    rows = [e for e in events if e.type is EventType.COST]
    if not rows:
        return EMPTY_COSTS

    out = [
        "| # | model | in | out | cached | cost |",
        "|--:|---|--:|--:|--:|--:|",
    ]
    total = 0.0
    unpriced = 0
    for i, e in enumerate(rows, 1):
        cost = e.data.get("cost_usd")
        model = str(e.data.get("model", "")).replace("claude-", "")
        if cost is None:
            unpriced += 1
            shown = "—"
        else:
            total += float(cost)
            shown = f"${float(cost):.5f}"
        out.append(
            f"| {i} | `{model}` | {int(e.data.get('input_tokens') or 0):,} | "
            f"{int(e.data.get('output_tokens') or 0):,} | "
            f"{int(e.data.get('cache_read') or 0):,} | {shown} |"
        )
    out.append(f"| | **total** | | | | **${total:.4f}** |")
    if unpriced:
        out.append("")
        out.append(
            f"> ⚠️ {unpriced} call(s) used a model with no published price. They "
            "are counted as $0, so the total is a lower bound."
        )
    return "\n".join(out)


def run_task(task: str, api_key: str, directory: str, max_cost: float, effort: str, history: list):
    """Gradio generator: yields after each event, so the trace appears live."""
    history = list(history) + [
        {"role": "user", "content": task},
        {"role": "assistant", "content": ""},
    ]
    key = api_key.strip() or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        history[-1]["content"] = "Provide an Anthropic API key, or set ANTHROPIC_API_KEY."
        yield history, EMPTY_COSTS, ""
        return

    root = Path(directory).expanduser().resolve()
    lines: list[str] = []
    stream = EventStream()
    # Rendering happens on the generator's thread; the loop only appends.
    stream.subscribe(lambda e: lines.append(e.render(unicode_ok=True)))

    try:
        session = SessionStore.create(root)
        workspace = Workspace(root=root, session_id=session.session_id)
        session.attach(stream)
        checkpoint = workspace.checkpoint(task[:60])
    except (WorkspaceError, OSError) as exc:
        history[-1]["content"] = f"❌ {exc}"
        yield history, EMPTY_COSTS, ""
        return

    ctx = ToolContext(
        workspace=workspace,
        sandbox=LocalSandbox(root=root),
        # No terminal to prompt on, so anything off the allowlist is refused
        # rather than silently allowed.
        permissions=PermissionGate(auto_approve=False, prompt=None),
        events=stream,
    )
    convo = new_conversation()
    convo.messages = session.messages()
    budget = Budget(max_usd=max_cost)

    async def _run():
        client = LLMClient(api_key=key, model=STRONG_MODEL)
        await client.validate(STRONG_MODEL)
        loop = AgentLoop(client, ctx, convo, budget, effort=effort)
        return await loop.run(task)

    loop_ = asyncio.new_event_loop()
    try:
        result = loop_.run_until_complete(_run())
        status = (
            f"{budget.summary()} · {convo.cache_report()}\n\n"
            f"session `{session.session_id}` · checkpoint `{checkpoint.commit[:8]}` — "
            "`codepilot undo` reverts this turn"
        )
        history[-1]["content"] = "\n".join(lines) + f"\n\n---\n{result.summary}"
        session.record_messages(convo.messages)
        session.record_turn(task, result.summary, budget.summary())
    except LLMError as exc:
        history[-1]["content"] = f"❌ {exc}"
        status = ""
    except Exception as exc:  # noqa: BLE001 - surface it rather than blanking the UI
        history[-1]["content"] = "\n".join(lines) + f"\n\n❌ {type(exc).__name__}: {exc}"
        status = budget.summary()
    finally:
        loop_.close()

    yield history, cost_table(stream.events), status


def build_ui(default_directory: str) -> gr.Blocks:
    with gr.Blocks(title="CodePilot") as demo:
        gr.Markdown(
            "# CodePilot\n"
            "A coding agent that edits **this repository, for real**. Every turn is "
            "checkpointed to a git ref first, so `codepilot undo` puts it back.\n\n"
            "Runs locally on your machine with your own key. Commands outside the "
            "allowlist are refused here rather than prompted, because a browser tab "
            "is a poor place to approve a shell command — use the CLI for those."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Trace", height=560)
                task = gr.Textbox(
                    label="", lines=2,
                    placeholder="e.g. make parse_config raise on an unknown key, and add a test",
                )
                go = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                key = gr.Textbox(
                    label="Anthropic API key",
                    type="password",
                    value="",  # never prefilled: it would render into the page
                    placeholder="sk-ant-…  (blank uses ANTHROPIC_API_KEY)",
                )
                directory = gr.Textbox(label="Repository", value=default_directory)
                max_cost = gr.Slider(0.05, 5.0, value=1.0, step=0.05, label="Cost ceiling (USD)")
                effort = gr.Radio(
                    ["low", "medium", "high"], value="high",
                    label="Effort (the cost/quality dial)",
                )
                status = gr.Markdown()
                gr.Markdown("### Cost per call")
                costs = gr.Markdown(EMPTY_COSTS)

        inputs = [task, key, directory, max_cost, effort, chat]
        outputs = [chat, costs, status]
        go.click(run_task, inputs=inputs, outputs=outputs)
        task.submit(run_task, inputs=inputs, outputs=outputs)
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-C", "--directory", default=".")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    load_env(Path(args.directory).resolve())

    build_ui(str(Path(args.directory).resolve())).launch(
        server_name="127.0.0.1",  # local only: this runs commands
        server_port=args.port,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

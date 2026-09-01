"""
CodePilot-Agent — Gradio UI

Runs the full agent pipeline directly (no separate FastAPI server needed for local use).
The UI streams AgentEvents in real time using Gradio's generator-based streaming.

Layout:
  Left panel  — Chat interface (user input + live agent trace)
  Right panel — Workspace viewer (files written, test results, cost tracker)
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # the README's `cp .env.example .env` step only works if we read it

from agent.graph import build_graph
from models.schemas import (
    AgentMode,
    AgentState,
)
from router.model_router import ModelRouter
from sandbox.factory import create_sandbox

# ---------------------------------------------------------------------------
# Build graph once at startup
# ---------------------------------------------------------------------------
_graph = build_graph()


# ---------------------------------------------------------------------------
# Core runner — yields chat messages as the agent works
# ---------------------------------------------------------------------------


def run_agent(
    user_request: str,
    api_key: str,
    sandbox_type: str,
    mode_str: str,
    uploaded_files: list,
    history: list,
) -> Generator:
    """
    Gradio generator function. Yields (history, workspace_text, cost_text) tuples
    after every AgentEvent so the UI updates in real time.
    """
    if not api_key.strip():
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        history = history + [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": "❌ Please enter your Anthropic API key."},
        ]
        yield history, "", "$0.00 | 0 tokens", "_No calls yet._"
        return

    if not user_request.strip():
        yield history, "", "$0.00 | 0 tokens", "_No calls yet._"
        return

    # Parse uploaded files into workspace dict
    workspace: dict[str, str] = {}
    if uploaded_files:
        for f in uploaded_files:
            try:
                with open(f.name, "r", encoding="utf-8", errors="replace") as fh:
                    rel = os.path.basename(f.name)
                    workspace[rel] = fh.read()
            except Exception:
                pass

    mode = AgentMode.QA if mode_str == "Q&A" else AgentMode.TASK

    state = AgentState(
        user_request=user_request,
        mode=mode,
        sandbox_type=sandbox_type.lower(),
        workspace_files=workspace,
        original_files=dict(workspace),
    )

    router = ModelRouter(api_key=api_key)

    # Add user message to history
    history = history + [
        {"role": "user", "content": user_request},
        {"role": "assistant", "content": ""},
    ]
    yield history, _format_workspace(workspace), "$0.00 | 0 tokens", "_No calls yet._"

    # Run graph synchronously via asyncio.run (Gradio runs in a thread)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agent_lines: list[str] = []
        final_workspace: dict[str, str] = {}

        async def _run():
            nonlocal final_workspace
            sandbox = create_sandbox(state.sandbox_type)
            await sandbox.setup(state.workspace_files)

            try:
                result = await _graph.ainvoke(
                    state.model_dump(mode="json"),
                    config={"configurable": {"router": router, "sandbox": sandbox}},
                )
                final_state = AgentState.model_validate(
                    {k: v for k, v in result.items() if not k.startswith("_")}
                )
                return final_state
            finally:
                await sandbox.teardown()

        final_state = loop.run_until_complete(_run())

        # Stream events into the chat
        for event in final_state.events:
            line = event.to_ui_line()
            agent_lines.append(line)
            trace = "\n".join(agent_lines)
            current_history = history[:-1] + [{"role": "assistant", "content": trace}]
            cost_text = (
                f"${final_state.total_cost_usd:.4f} | {final_state.total_tokens} tokens"
            )
            yield (
                current_history,
                _format_workspace(final_state.workspace_files),
                cost_text,
                _format_costs(final_state),
            )

        # Final reply
        final_reply = final_state.final_output or "Task complete."
        if final_state.error_message:
            final_reply = f"❌ Error: {final_state.error_message}"

        summary = "\n\n---\n" + final_reply
        full_trace = "\n".join(agent_lines) + summary
        history = history[:-1] + [{"role": "assistant", "content": full_trace}]
        cost_text = (
            f"${final_state.total_cost_usd:.4f} | {final_state.total_tokens} tokens"
        )
        yield (
            history,
            _format_workspace(final_state.workspace_files),
            cost_text,
            _format_costs(final_state),
        )

    except Exception as exc:
        err_msg = f"❌ Unexpected error: {exc}"
        history = history[:-1] + [{"role": "assistant", "content": err_msg}]
        yield history, _format_workspace(workspace), "$0.00 | 0 tokens", "_No calls yet._"
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _format_costs(state: AgentState) -> str:
    """Per-call spend, newest last. A running total alone hides which agent is
    expensive, which is the only number that changes how you configure it."""
    if not state.llm_calls:
        return "_No calls yet._"
    rows = [
        "| # | Agent | Model | In | Out | Latency | Cost |",
        "|--:|---|---|--:|--:|--:|--:|",
    ]
    for i, c in enumerate(state.llm_calls, 1):
        model = c.model.replace("claude-", "").replace("-20251001", "")
        rows.append(
            f"| {i} | {c.agent.value} | `{model}` | {c.input_tokens:,} | "
            f"{c.output_tokens:,} | {c.latency_ms / 1000:.1f}s | ${c.cost_usd:.5f} |"
        )
    rows.append(
        f"| | **total** | | **{sum(c.input_tokens for c in state.llm_calls):,}** | "
        f"**{sum(c.output_tokens for c in state.llm_calls):,}** | | "
        f"**${state.total_cost_usd:.4f}** |"
    )
    unpriced = {c.model for c in state.llm_calls if c.cost_usd == 0.0}
    if unpriced:
        rows.append("")
        rows.append(
            f"> ⚠️ No published price for {', '.join(sorted(unpriced))} — "
            "those calls are counted as $0 and the total is an underestimate."
        )
    return "\n".join(rows)


def _format_workspace(files: dict[str, str]) -> str:
    if not files:
        return "No files in workspace yet."
    lines = [f"**Workspace** ({len(files)} files)\n"]
    for fname, content in sorted(files.items()):
        loc = content.count("\n") + 1
        preview = content[:300].replace("\n", " ")
        if len(content) > 300:
            preview += "..."
        lines.append(f"**{fname}** ({loc} lines)\n```\n{preview}\n```\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = textwrap.dedent("""
# CodePilot-Agent
**Autonomous coding agent** — describe what you want, watch it plan, write, test, and fix.

Powered by LangGraph multi-agent orchestration + Claude Sonnet/Haiku model routing.
""").strip()

EXAMPLES = [
    [
        "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes. Include docstring and type hints.",
        "subprocess",
        "Task",
    ],
    [
        "Add input validation to this function so it raises ValueError for non-integer input.",
        "subprocess",
        "Task",
    ],
    ["What does this codebase do? Summarise the main functions.", "subprocess", "Q&A"],
]

with gr.Blocks(title="CodePilot-Agent") as demo:  # Gradio 6: theme moved to launch()
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left panel: Chat ──────────────────────────────────────────
        with gr.Column(scale=3):
            # Gradio 6 removed `type`: messages format is the only format, so history
            # entries are {"role", "content"} dicts throughout this file.
            chatbot = gr.Chatbot(label="Agent Trace", height=600)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Describe what you want to build or fix...",
                    label="",
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Run Agent ▶", variant="primary", scale=1)

            with gr.Accordion("Upload existing files (optional)", open=False):
                file_upload = gr.File(
                    label="Upload .py files to give the agent context",
                    file_types=[".py", ".txt", ".md", ".json", ".yaml"],
                    file_count="multiple",
                )

        # ── Right panel: Config + Workspace ──────────────────────────
        with gr.Column(scale=2):
            api_key_input = gr.Textbox(
                label="Anthropic API Key",
                placeholder="sk-ant-…  (blank uses the server's key)",
                type="password",
                # Never prefill: `value=` renders into the page every visitor loads,
                # so a hosted instance would ship its own key to everyone (F-8).
                value="",
            )
            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["Task", "Q&A"],
                    value="Task",
                    label="Mode",
                )
                sandbox_radio = gr.Radio(
                    choices=["Subprocess", "Docker"],
                    value=os.getenv("SANDBOX_TYPE", "subprocess").capitalize(),
                    label="Sandbox",
                )
            cost_display = gr.Textbox(
                label="Session Cost",
                value="$0.00 | 0 tokens",
                interactive=False,
            )
            with gr.Accordion("Cost per call", open=True):
                cost_breakdown = gr.Markdown(value="_No calls yet._")
            workspace_display = gr.Markdown(
                value="No files yet.",
                label="Workspace",
            )
            clear_btn = gr.Button("Clear Session", variant="secondary")

    gr.Examples(
        examples=[[e[0], e[1], e[2]] for e in EXAMPLES],
        inputs=[msg_input, sandbox_radio, mode_radio],
        label="Try these examples",
    )

    # ── Event wiring ────────────────────────────────────────────────
    send_btn.click(
        fn=run_agent,
        inputs=[
            msg_input,
            api_key_input,
            sandbox_radio,
            mode_radio,
            file_upload,
            chatbot,
        ],
        outputs=[chatbot, workspace_display, cost_display, cost_breakdown],
    )
    msg_input.submit(
        fn=run_agent,
        inputs=[
            msg_input,
            api_key_input,
            sandbox_radio,
            mode_radio,
            file_upload,
            chatbot,
        ],
        outputs=[chatbot, workspace_display, cost_display, cost_breakdown],
    )
    clear_btn.click(
        fn=lambda: ([], "No files yet.", "$0.00 | 0 tokens", "_No calls yet._"),
        outputs=[chatbot, workspace_display, cost_display, cost_breakdown],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # local tool: do not bind to every interface
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )

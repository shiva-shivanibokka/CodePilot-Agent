# CodePilot-Agent — Design

**Date:** 2026-09-01
**Status:** Approved, pending implementation plan
**Supersedes:** the architecture described in the current `README.md`

---

## 1. Context

CodePilot-Agent currently presents as a six-agent LangGraph coding agent with sandboxed
execution, AST-based context management, model routing, real-time streaming and Redis session
persistence. A line-by-line audit (`AUDIT.md`, 2026-09-01) established that **the application
has never been run**: it raises `KeyError: '_router'` on its first node transition, its plan
pointer never advances, and its Coder↔Reviewer loop is unbounded. Beneath those, the "six
agents" have no tools, no conversation memory, and no way to see each other's work.

The parts that are sound — the Pydantic schemas, the six system prompts, the AST walker, the
subprocess sandbox's file I/O — are the periphery. The core is the problem.

### Goal

A small-scale but architecturally honest replica of a real coding agent (Claude Code shape),
that is **simultaneously a tool the author uses on their own repositories and a measurable
artifact** with published numbers.

### Success criteria

1. `codepilot` runs in a real repository, edits real files, runs real tests, and every action
   is revertible.
2. A conversation: follow-up turns, mid-run interruption, plan-then-confirm.
3. An eval set of 15–20 tasks with held-out tests, producing pass rate / cost / tokens per
   configuration.
4. Four published comparisons (§8) with raw results committed alongside the claims.
5. A hosted replay demo showing several recorded sessions, costing nothing to run.

### Non-goals

- **Hosting the agent.** BYOK is local: clone, add a key, run. Nothing that executes code is
  exposed to the internet. This deletes the current repo's unauthenticated-RCE endpoint and
  its `docker.sock` mount rather than patching them.
- **Multi-provider support.** See ADR-2.
- **GitHub issue → PR (ISSUE mode).** The five orphan state fields supporting it are removed.
- **A web UI as the primary interface.** The web surface is the replay demo only.

---

## 2. Decisions

### ADR-1 — Two arms, sharing everything except control flow

Two agents are built: a free tool loop (`agent/loop.py`) and the repaired LangGraph state
machine (`agent/pipeline.py`). They share the LLM client, the tool set, the sandbox, the
context manager and the event stream. **Only the control flow differs.**

*Rationale:* the fixed `plan → code → test → debug → review` graph is structurally not what a
coding agent is — it is why today's Tester cannot write a missing test file, since the graph
has no edge for it. Rather than delete that finding, it becomes the control condition for the
headline experiment. If the arms differed in tools or memory, the comparison would measure the
wrong variable.

*Consequence:* LangGraph stays a dependency and stays on the résumé, but earns its place.

### ADR-2 — Anthropic only, behind one seam

One `LLMClient` protocol, one implementation, and it is the only module that imports
`anthropic`.

*Rationale:* the engineering that makes this project interesting — the tool loop, context
compaction, and prompt caching — is where providers diverge most. Prompt caching in particular
is the dominant cost lever for a coding agent (the system prompt, project instructions and file
context are re-sent every turn), and it is Anthropic-shaped. An abstraction that must satisfy
every provider drops it. The multi-provider story is already told by
`Churn-Intelligence-Platform`; repeating it here adds surface area and no new signal.

*Consequence:* a second provider later is one file. If breadth is ever wanted cheaply, the
OpenAI-compatible-adapter approach used in `Churn-Intelligence-Platform` covers many providers
at once and is the fallback — not six hand-written clients.

### ADR-3 — Edits land in the real repository, protected by git

The agent operates on the current working directory. Before each turn it writes the working
tree to a scratch ref; `codepilot undo` restores it.

*Rationale:* an agent that writes to a temp directory that is then deleted is a demo, not a
tool. The git checkpoint is what makes editing real files safe enough to do.

### ADR-4 — The sandbox splits by purpose

- **Local tool:** commands run in the user's `cwd` behind a permission gate. Isolating the
  user's own code from the user's own machine is theatre.
- **Eval harness:** Docker, hermetic, one container per task. The eval executes
  model-generated code against fixture repositories; that genuinely is untrusted.

### ADR-5 — The event stream is the public interface

Both arms emit the same `AgentEvent` sequence. Three consumers: the CLI renders it live, the
recorder serialises it to JSONL, the replay site plays back that JSONL. Recording the demo is
therefore free, and streaming must be real rather than a post-hoc replay of a finished run.

### ADR-6 — Default model and the cost dial

Default `claude-opus-5`, with `output_config.effort` as the cost lever, and
`claude-haiku-4-5` for the cheap search subagent.

*Rationale:* the current constants are `claude-sonnet-4-5` and `claude-haiku-3-5`; the latter
has never been a valid Anthropic model ID, so three of six agents 404 on every call. Beyond the
repair, the premise of the old routing table — send cheap work to a weaker model — is worth
testing rather than assuming, because lower effort on a stronger model frequently beats higher
effort on a weaker one. That question becomes experiment 4 (§8).

*Consequence:* model IDs are validated against `client.models.list()` at startup, so a
retirement fails fast instead of at request time. This is the fix that resolved the retired
Groq model in `Churn-Intelligence-Platform`.

---

## 3. Architecture

```
codepilot/
├── llm.py            LLMClient — the only module importing `anthropic`
├── context.py        Conversation: history, compaction, cache breakpoints, token accounting
├── events.py         AgentEvent + the emitter both arms write to
├── workspace.py      Repo root, gitignore-aware walk, read-ledger, git checkpoint/undo
├── permissions.py    Command allowlist, approval prompts, budget ceilings
├── tools/
│   ├── base.py       Tool protocol: name, schema, async run(ctx, **kwargs) -> str
│   ├── files.py      read_file, write_file, edit_file, list_files
│   ├── search.py     search (ripgrep-style), find_symbol
│   ├── shell.py      run_command, run_tests
│   └── control.py    propose_plan, finish
├── sandbox/
│   ├── base.py       SandboxBackend protocol — now actually used as a type
│   ├── local.py      Runs in cwd behind the permission gate (the tool)
│   ├── docker.py     Hermetic container (the eval harness)
│   └── pytest.py     Shared output parser
├── indexer/          AST index: symbols, call graph keyed by file::qualname, retrieval
├── agent/
│   ├── loop.py       ARM A — while stop_reason == "tool_use"
│   ├── pipeline.py   ARM B — the repaired LangGraph graph
│   ├── prompts.py    System prompts (kept; rewritten to match reality)
│   └── interrupt.py  InterruptChannel
├── session.py        Persistence to .codepilot/sessions/<id>.jsonl, resume
└── cli.py            codepilot run | chat | undo | replay-export

evals/
├── tasks/            *.yaml: prompt, fixture repo, held-out tests
├── runner.py         task × arm × config → results JSON
└── results/          Committed. The claims link here.

web/                  Next.js replay site, reads committed JSON. No backend.
```

### Data flow (one turn)

```
user message
  → Context.append(user)
  → Arm (loop | pipeline)
      → LLMClient.chat(messages, tools, cache_breakpoints)
      → for each tool_use block:
            PermissionGate.check ──reject──→ tool_result(is_error)
                   │ allow
                   ▼
            Tool.run(ctx) → Workspace / Sandbox
                   │
                   ▼
            tool_result → Context.append
      → repeat until `finish` or budget exhausted
  → events streamed throughout to CLI + recorder
```

### Component contracts

| Module | Purpose | Depends on |
|---|---|---|
| `llm.py` | Messages in, message out. Tool schemas, streaming, caching, usage. | `anthropic` |
| `context.py` | The conversation. Compacts when over threshold; owns cache breakpoint placement. | `llm.py` (token counting) |
| `workspace.py` | Repo root, file walk, read-ledger, checkpoint/undo. | `git` (plumbing only) |
| `permissions.py` | Allowlist, prompts, budget. Single choke point for safety. | — |
| `tools/*` | One action each. Sandbox and workspace arrive via `ctx`. | `workspace`, `sandbox` |
| `sandbox/*` | Execute a command, return structured output. | — |
| `agent/loop.py` | Arm A control flow. | all of the above |
| `agent/pipeline.py` | Arm B control flow. | all of the above + `langgraph` |
| `events.py` | The stream. | — |

Every tool's sandbox comes from `ctx`, never from its JSON schema. The current
`tools/registry.py` takes `sandbox: Any` as a tool *parameter*, which would require the model
to serialise a live object — a structural defect, not an oversight.

---

## 4. Safety model

**Checkpointing.** Before each turn, write the working tree to `refs/codepilot/<session>/<n>`
using a separate `GIT_INDEX_FILE`, so the user's index and `HEAD` are never touched.
`codepilot undo` restores the previous ref's tree. If the directory is not a git repository,
editing is refused unless `--no-checkpoint` is passed explicitly.

**Read-before-write.** `Workspace` records `path → content hash` at read time. `edit_file` and
`write_file` refuse a path that was not read this session, or whose hash has changed on disk
since. The refusal is returned as a tool error so the model re-reads and recovers, rather than
raising and killing the turn.

**Path containment.** All writes resolve and assert containment within the repo root. Absolute
paths and `..` segments are rejected. (Today's `setup()` and `write_file()` join caller-supplied
keys onto the workdir unnormalised.)

**Permission gate.** An allowlist of read-only and test commands runs without prompting;
anything else prompts with the exact command. `--yes` bypasses it and is used only by the eval
runner.

**Budget.** Tokens, dollars and turns are each capped per session and checked in the executor,
so both arms inherit the ceiling. Exceeding one emits a `BUDGET` event naming the limit and
ends the turn — a visible stop, not a silent one.

**Test-file protection.** During a debug turn, edits to files matching the project's test glob
require explicit confirmation. The prompt already says not to change tests to make them pass;
this makes it structural.

**Error handling.** Tool failures become `tool_result` blocks with `is_error: true` — never
exceptions. API failures are handled by class (`RateLimitError`, `APIStatusError` ≥ 500,
`APIConnectionError`, `NotFoundError`) with backoff on the retryable ones; a `refusal` stop
reason is surfaced, not swallowed.

---

## 5. Interaction model

A REPL. One turn is: user message → arm runs until `finish` or budget → control returns.

**Interruption.** `InterruptChannel` runs a reader on a separate thread; it sets a flag and
queues text. The arm checks the flag **between tool calls only** — never mid-tool — then
finishes the in-flight tool, appends the user's message to the conversation, and continues.
The worst case is that an interrupt lands one tool call late; state is never observed
half-written.

**Plan-then-confirm.** For a task the model judges non-trivial, its first action is
`propose_plan`. The CLI renders the plan and waits for approve / edit / reject. Disabled by
`--yes` and by config.

**Diff preview.** `edit_file` renders a unified diff. Small diffs auto-apply; large ones prompt.

**Persistence.** Every session appends to `.codepilot/sessions/<id>.jsonl` — the same records
the recorder consumes. `codepilot chat --resume <id>` rebuilds the context and continues.

---

## 6. Context management

The current code has no conversation at all: every node is a stateless one-shot call, which is
why the Debugger cannot see what the Coder just tried.

- **History** is a real message list, carried across turns and across `--resume`.
- **Prefix order is fixed** — `tools → system → CODEPILOT.md → history` — with the cache
  breakpoint after the stable segment. Nothing volatile (timestamps, session ids, costs) goes
  before that breakpoint. Cache hits are verified via `usage.cache_read_input_tokens`; zero
  across repeated turns means a silent invalidator, and that is a test.
- **Compaction** triggers at a token threshold: older tool results are summarised while the
  system prefix and the last N turns are preserved verbatim.
- **`CODEPILOT.md`**, if present at the repo root, is injected after the system prompt —
  project conventions, the test command, things to avoid.
- **Retrieval** feeds the agent file candidates from the AST index rather than whole files.
  The index keys the call graph by `file::qualname` (today it uses the bare function name, so
  same-named methods in different classes overwrite each other) and exposes a called-by
  neighbourhood, which is the input to experiment 3.

---

## 7. What survives from the current code

| Kept | Rewritten | Deleted |
|---|---|---|
| `models/schemas.py` (data model) | `agent/graph.py` → `agent/pipeline.py` | `tools/registry.py` (0 call sites, unusable schema) |
| `agent/prompts.py` (prose) | `tools/` (real, ctx-bound) | `chat_with_tools` (returns inside its first loop iteration) |
| `indexer/` AST walker | `router/model_router.py` → `llm.py` | `langchain-core` dependency |
| `sandbox/` file I/O + timeout handling | `session/manager.py` → `session.py` | `app.py` Gradio UI as primary interface |
| | | `api/main.py` — the whole FastAPI surface (§1 non-goals: nothing that executes code is hosted) |
| | | `AgentMode.ISSUE` + 4 orphan state fields |
| | | The `"e2b"` sandbox literal (file does not exist) |

The package is restructured under `codepilot/`. This makes the diff large and the history less
linear, but it is the shape something installable needs.

---

## 8. Evaluation

`evals/tasks/*.yaml` — each task has a prompt, a small fixture repository, and a **held-out
test file the agent never sees**. Pass means the held-out tests pass. 15–20 tasks across three
tiers (single-function, multi-file, debug-an-existing-failure).

The runner executes task × arm × config in a fresh Docker container, non-interactive, and
records pass/fail, input/output tokens, cost, wall time and turn count to
`evals/results/<date>.json`, which is committed.

**Experiment 1 — free loop vs. fixed pipeline.** Pass rate and cost per completed task. The
headline result either way.

**Experiment 2 — whole-file rewrite vs. `str_replace` edits.** Tokens per task, plus a
regression check: how often did code that was not part of the task disappear? Whole-file
rewrite is the current behaviour and its failure mode is silent deletion.

**Experiment 3 — keyword retrieval vs. call-graph retrieval.** The AST indexer's README claim,
finally measured. Tokens sent and pass rate.

**Experiment 4 — model routing vs. effort tuning.** Three configurations: routed
(strong model for planning/debugging, cheap model for the rest), single strong model at low
effort, single strong model at high effort. Cost per completed task, not per request.

**Honesty rules**, carried from `CAG-vs-RAG-Showdown`: report run-to-run variance before
claiming a gap; state N, date and model ID beside every number; never average a failed judge or
errored run as if it were a result; link the raw results file from every published claim.

---

## 9. Recording and replay

Every session writes JSONL. `codepilot replay-export <session>` converts one to
`web/public/replays/<name>.json`. The Next.js site on Vercel reads the committed files —
several recorded sessions, selectable from a list, played back at their original pacing with
the same event renderer the CLI uses. No backend, no key, no cost.

Recorded sessions include user turns and interruptions, which makes them a better demo than a
single one-shot run: the viewer sees the agent being corrected and adapting.

The site states plainly that it is a recording of a real local session, gives the date and
model, and links both the raw trace and the eval results. There is no live hosted agent and the
page says so.

---

## 10. Testing

**Flow tests** (stub `LLMClient`, no key, no network):
- both arms terminate on an always-rejecting reviewer
- budget ceiling ends a run and emits `BUDGET`
- a tool that raises becomes `tool_result(is_error=True)` and the loop continues
- an interrupt is observed between tool calls, never mid-tool
- `edit_file` refuses an unread path, and a path changed on disk

**Unit tests:** path containment, pytest-parser fixtures (pass / fail / collection error / no
tests / a traceback containing the word "passed"), git checkpoint→edit→undo round-trip,
gitignore-aware walk, compaction preserving the system prefix and last N turns, cache
breakpoint placement producing non-zero `cache_read_input_tokens` on the second turn.

**Integration:** the eval set, run manually with a key. Never in CI.

**CI:** pinned `ruff` (with `S110`/`S112` enabled), `pyright` with the five currently-disabled
rules restored, `pytest`, and a `docker build` of the eval sandbox image.

---

## 11. Milestones

| # | Milestone | Outcome |
|---|---|---|
| 0 | Repair | The three criticals fixed, model IDs corrected, `.env` loaded, flow tests green. The current code runs. |
| 1 | Foundation | `llm.py`, `context.py`, `events.py`, `workspace.py`, `permissions.py` + tests. |
| 2 | Tools & sandbox | Real ctx-bound tools, local + docker backends, shared pytest parser. |
| 3 | Arm A | The loop. `codepilot run "task"` works end to end on a real repo with checkpoint/undo. |
| 4 | Chat | REPL, interrupt, plan-then-confirm, persistence, resume. |
| 5 | Arm B | The repaired pipeline on the shared substrate. |
| 6 | Eval | Task set, runner, the four experiments, committed results. |
| 7 | Replay & docs | Recorded sessions, the Vercel site, README rewritten around what exists. |

Milestone 0 is worth having on its own: it converts the repository from "does not run" to
"runs honestly," which is the state it should be in regardless of how far the rest goes.

This spec is too large for a single implementation plan. It is executed as three:
**0–2** (repair, foundation, tools), **3–5** (the two arms and chat), **6–7** (eval and
replay). Each gets its own plan, written when the previous one lands, so later plans are
informed by what the earlier ones actually cost.

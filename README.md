# CodePilot

A small coding agent that edits real git repositories, built to find out how
coding agents actually work — and to measure the design choices instead of
asserting them.

It reads your code, changes it by exact-string replacement, runs the tests, and
checkpoints every turn to a git ref so `codepilot undo` puts things back. It
runs on your machine, on your repositories, with your own API key.

Built by Shivani Bokka.

```
$ codepilot run "mean() crashes on an empty list — make it raise ValueError, add
                 type hints, and write the tests"

⎘ checkpointed — `codepilot undo` reverts to here (24763a35)

· I'll find the code first.
🔧 list_files()
🔧 search(pattern=def (mean|median), glob=*.py)
📋 stats.py:4: def mean(values):
🔧 read_file(path=stats.py)
🔧 edit_file(path=stats.py, old="""Simple statistics hel…)
  ± stats.py  +8 -2
🔧 write_file(path=tests/test_stats.py, …)
  ± tests/test_stats.py  +29 -0
🔧 run_tests()
🧪 6 passed (2983ms)
✅ stats.py now raises ValueError with an explicit message on an empty
   sequence, and both functions are annotated. 6 tests added, all passing.

   One thing I did not change, since it was outside what you asked:
   median() on an even-length list returns the upper of the two middle
   elements rather than their average. Say the word and I'll fix it.

  session efc97e9d6607 · $0.0574 / $0.60 · 6 calls
  prompt cache: 10,090 tokens read (67% of input served from cache)
  edited: stats.py, tests/test_stats.py
  `codepilot undo` reverts this turn
```

That transcript is from a real run, lightly trimmed for width.

---

## Why this exists

It is easy to build something that *looks* like a coding agent. This repository
began as one: six named agents, a state graph, a sandbox, a live cost ticker, a
README describing all of it. [`docs/`](docs/superpowers/specs/) has the design;
the short version is that a line-by-line audit found the application had never
been run. It raised `KeyError` on its first node transition, its plan pointer
never advanced, its retry loop was unbounded, and one of the two model IDs it
routed to had never existed.

So the rebuild has one rule: **every claim in this README is either checkable
from the code or backed by a committed results file.** Where something is
untested, it says so.

---

## Install

```bash
git clone https://github.com/shiva-shivanibokka/CodePilot-Agent
cd CodePilot-Agent
pip install -r requirements.txt
cp .env.example .env        # add your Anthropic API key
codepilot doctor            # 16 checks that the install actually works
```

`doctor` verifies the things that break silently: that git checkpointing works,
that a process can be spawned and killed on your OS, that no tool schema leaks
an object the model cannot serialise, and — with `--live` — that the configured
models exist and the model really calls a tool.

## Use

```bash
codepilot run "add retry logic to fetch.py"   # one task
codepilot chat                                 # a conversation
codepilot undo                                 # revert the last turn
codepilot sessions                             # what has run here
codepilot export                               # turn a session into a replay
python -m codepilot.webui                      # a local browser UI, if you prefer
```

In `chat`, Ctrl-C interrupts a running turn and hands control back rather than
killing it; a second Ctrl-C forces the issue. `/undo`, `/cost` and `/session`
work mid-conversation, and `--resume <id>` picks a session back up later.

A `CODEPILOT.md` at your repository root is injected into the system prompt —
conventions, the test command, things to avoid.

---

## How it works

```
codepilot/
├── llm.py         the only module that imports `anthropic`
├── context.py     the conversation: history, compaction, cache breakpoints
├── workspace.py   real files, containment, read-before-write, checkpoint/undo
├── permissions.py one gate: command allowlist, budgets, test-file protection
├── tools.py       10 tools, and the one place a tool call is executed
├── indexer.py     AST index: symbols and a call graph keyed by file::qualname
├── sandbox/       local (the tool) and docker (the eval)
├── agent/
│   ├── loop.py      ARM A — the model chooses each step
│   └── pipeline.py  ARM B — a fixed plan/code/test/debug/review graph
├── session.py     append-only JSONL, resumable
├── replay.py      a session → a recording the web page plays
└── cli.py
```

**Two arms, one substrate.** The interesting question about a coding agent is
whether the model should choose what to do next, or whether a fixed pipeline
should choose for it. Both are built here, and they share the client, tools,
sandbox, workspace, permissions, context and event stream — the control flow is
the only difference, which is what makes comparing them mean anything.

**Safety is one layer, not scattered.** Every tool call goes through the same
gate: path containment, read-before-write, the command allowlist, the budget,
and protection of test files during a debugging turn (editing the test until it
passes is the most common way an agent reports success without fixing
anything).

**Nothing raises out of a tool.** An unknown tool, bad arguments, a refused
command, a path escaping the repository — all come back as `tool_result` blocks
the model reads and recovers from. A traceback ends a turn; an error message
does not.

**Undo is real.** Before each turn the working tree is written to
`refs/codepilot/<session>/<n>` using a throwaway git index, so your staged work
is never touched and `HEAD` never moves. `undo` restores the tree *and* removes
files created since, because a restore that leaves them behind is not an undo.

---

## What is measured

Fifteen tasks, each a small repository plus a prompt plus a **held-out test file
the agent never sees**. Pass means those tests pass afterwards. Asking the agent
whether it succeeded measures its self-report; running tests it could not have
written to measures the work.

```bash
python -m evals.runner --dry-run                      # free: do the fixtures measure anything?
python -m evals.runner --arms loop pipeline           # experiment 1
python -m evals.runner --experiment edit-style        # experiment 2
python -m evals.runner --experiment retrieval         # experiment 3
python -m evals.runner --experiment effort            # experiment 4
```

Results land in [`evals/results/`](evals/results/) and are committed. Every
number below links back to one.

<!-- RESULTS -->

---

## Things this does not do

- **It is not hosted, and will not be.** It runs shell commands and edits files;
  exposing that publicly means running commands for strangers. The public
  artefact is a [replay](web/) of real recorded sessions.
- **Anthropic only.** Prompt caching and tool-use semantics are where providers
  differ most, and an abstraction that satisfies all of them drops the caching —
  which is the difference between an agent that is affordable to iterate with
  and one that is not. The seam is one file if that changes.
- **Python tasks only** in the eval, because the sandbox runs pytest. The agent
  itself is not language-specific.
- **No GitHub issue → PR mode.** The original modelled it in five state fields
  and implemented none of it; those fields are gone rather than left as an API
  that silently does something else.

## Development

```bash
pytest -q                    # 123 tests, no API key needed
ruff check .
python -m codepilot.doctor   # wiring checks
```

The tests use a stubbed model, so the whole suite runs offline in about twenty
seconds. `doctor` covers what unit tests structurally cannot — it found a pytest
parser that read every `-q` run as zero passed, and an `undo` that staged what it
restored, both while the suite was green.

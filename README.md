# CodePilot

A small coding agent that edits real git repositories, built to find out how
coding agents actually work — and to measure the design choices instead of
asserting them.

It reads your code, edits it, runs the tests, and
checkpoints every turn to a git ref so `codepilot undo` puts things back. It
runs on your machine, on your repositories, with your own API key — and it also
runs as a service if your team needs one, on the terms in
[DEPLOYING.md](DEPLOYING.md).

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
codepilot doctor            # 17 checks that the install actually works
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
codepilot serve --check                        # hosted mode: print the policy, start nothing
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

Twenty tasks, each a repository plus a prompt plus a **held-out test file the
agent never sees**. Pass means those tests pass afterwards. Asking the agent
whether it succeeded measures its self-report; running tests it could not have
written to measures the work.

They come in two sizes, and the size turned out to matter more than anything
else measured here:

| | tasks | repository | biggest file |
|---|---|---|---|
| small | 15 | 1–4 files | 2–15 lines |
| large | 5 | [`evals/fixtures/shop`](evals/fixtures/shop) — 14 files, 2,007 lines, its own 54-test suite | 437 lines |

The large fixture is a working order-and-pricing package: money as integer
minor units, stacking discount rules with a cap, a tax table, weight-banded
shipping, stock reservations. It has thirteen different `apply` methods and
thirteen different `validate` methods, so a regex on a method name returns the
wrong definition — which is the whole reason a symbol index is supposed to
exist.

```bash
python -m evals.runner --dry-run                      # free: do the fixtures measure anything?
python -m evals.runner --arms loop pipeline           # experiment 1
python -m evals.runner --experiment edit-style        # experiment 2
python -m evals.runner --experiment retrieval         # experiment 3
python -m evals.runner --experiment effort            # experiment 4: opus low/high vs sonnet
python -m evals.runner --tier large --experiment edit-style   # the same questions, large
```

Results land in [`evals/results/`](evals/results/) and are committed. Every
number below links back to one.

### Experiment 1 — does the model choosing beat a fixed pipeline?

Fifteen tasks, both arms, `claude-opus-5` at `effort=low`, 1 September 2026.
Raw: [`2026-09-01-2210-arms.json`](evals/results/).

| | pass | cost per completed task | median calls | median wall | input from cache |
|---|---|---|---|---|---|
| **loop** — the model chooses each step | 15/15 | **$0.042** | **6** | **22s** | **75%** |
| **pipeline** — fixed plan/code/test/debug/review | 15/15 | $0.184 | 23 | 88s | 56% |

**The pipeline bought nothing and cost 4.2× more.** Both arms solved every task,
so the structure did not buy reliability at this difficulty; it bought a 4.19×
median cost multiple (range 1.70×–8.27×) and four times the wall clock. The
loop was cheaper on **15 of 15 tasks** — not on average, on every one.

The gap widens as tasks get harder, which is the opposite of the usual argument
for a pipeline:

| tier | loop | pipeline | ratio |
|---|---|---|---|
| single-file | $0.041 | $0.125 | 3.0× |
| multi-file | $0.048 | $0.201 | 4.2× |
| debug an existing failure | $0.037 | $0.225 | **6.0×** |

Two mechanisms explain most of it, and both are visible in the data. The
pipeline re-derives its context every phase, so it makes ~4× the model calls.
And each phase builds its own short system prompt, which falls below the
model's minimum cacheable prefix — so it served 56% of its input from cache
against the loop's 75%, on 68% more input tokens overall.

**Is that gap real, or noise?** The loop arm was run twice on the same fifteen
tasks: **$0.6331 and $0.6396 total (1.0% apart), 15/15 both times**, median
per-task cost varying 3.4%. A 4.19× difference is roughly two orders of
magnitude larger than the run-to-run variation, so it is not a sampling
artifact. The pass rates, being identical, say nothing either way — fifteen
tasks cannot distinguish two arms that both solve all of them, and this
comparison should not be read as evidence that the loop is *more reliable*.

**What this does not show.** Fifteen small Python tasks, one model, one effort
setting, one day. Nothing here says a pipeline is a bad idea at a difficulty
where the loop starts failing; it says that at this difficulty the structure is
pure overhead.

### Experiment 2 — exact-string edits, or rewrite the file?

Same fifteen tasks, loop arm, one tool removed from each configuration.
Raw: [`2026-09-01-2231-edit-style.json`](evals/results/).

| | pass | cost per completed task | median calls | input tokens | input from cache |
|---|---|---|---|---|---|
| `edit_file` only | 15/15 | $0.087 | 8 | 147,422 | 58% |
| `write_file` only | 15/15 | **$0.044** | **6** | **62,520** | **69%** |

**This contradicted the design, and the design was changed.** The agent's prompt
used to say "prefer `edit_file`" flatly, on the reasoning that an exact-string
replacement costs tokens proportional to the change rather than to the file.
That reasoning is sound and the measurement says it does not apply here:
rewriting was **1.83× cheaper** at the median and cheaper on 11 of 15 tasks.

The mechanism is file size. These fixtures are 2–15 lines. Editing one costs a
`read_file` plus an exact-match string that must reproduce whitespace byte for
byte, and a failed match costs a retry — more than simply retyping ten lines.
The token-proportionality argument only starts paying when the file is bigger
than the round trip needed to edit it.

The safety argument did not show up either: **zero of thirty runs deleted code
that was supposed to survive**, checked by name against the fixture. Whole-file
rewriting is supposed to drop siblings the model forgot to reproduce; at ten
lines, there is nothing to forget.

**So the eval does not test the regime the advice was written for.** Nothing
above measures a 400-line file, which is where the argument for `edit_file`
lives and where a dropped function would actually hurt. That was the honest
limit of the experiment when it was written, and it is the reason the large
fixture exists.

#### The same question on a 2,007-line repository

Five tasks against `evals/fixtures/shop`, four of them changes inside the
437-line pricing module.
Raw: [`2026-09-02-0456-large-edit-style.json`](evals/results/).

| | pass | cost per completed task | median calls | input tokens | input from cache |
|---|---|---|---|---|---|
| `edit_file` only | 5/5 | **$0.369** | **7** | **324,914** | 19% |
| `write_file` only | 5/5 | $0.613 | 10 | 545,468 | 14% |

**The result reverses.** Exact-string editing is **1.66× cheaper** here and
cheaper on four of the five tasks, against being 1.83× *more* expensive on the
small ones. The mechanism is the one the original argument named: rewriting a
437-line file costs about 5,000 output tokens whether the change is one line or
fifty, and the model pays that on every attempt.

So neither result was wrong and neither generalises. The prompt's rule is
conditional on file size because the measurement is too:

> `edit_file` costs tokens proportional to the change […] so it is the right
> tool for anything of real size. On a file of a few lines, `write_file` is
> cheaper — matching an exact string costs more than retyping it.

**The safety argument still has not shown up.** Zero of fifty runs across both
sizes deleted code that was supposed to survive, checked by name against the
fixture — including the whole-file rewrites of a 437-line module with sixteen
functions in it. Whole-file rewriting is supposed to drop siblings the model
forgot to reproduce. At 437 lines it still does not, so the case for
`edit_file` rests on cost alone, which is what the numbers actually support.

### Experiment 3 — does an AST index beat plain text search?

Same fifteen tasks, loop arm, one retrieval tool each.
Raw: [`2026-09-01-2306-retrieval.json`](evals/results/).

| | pass | cost per completed task | median calls | input from cache |
|---|---|---|---|---|
| `search` — regex over file contents | 15/15 | $0.0398 | 6 | 73% |
| `find_symbol` — AST index: definition, callers, callees | 14/14 | $0.0432 | 6 | 73% |

*One `find_symbol` run is excluded: the provider returned 529 before the agent
took a turn. An outage is not a result.*

**No difference worth claiming.** Both configurations solved every task they
were allowed to attempt, and the 8.5% cost gap sits close enough to the 1–3.4%
run-to-run variation measured in experiment 1 that a single sweep of fifteen
tasks cannot separate them. The honest reading is that at this scale **an AST
index buys nothing over grep** — which is worth saying, because indexing the
codebase was this project's original headline feature.

That earlier version was worse than useless: its call graph was keyed by bare
function name, so two classes with a `setup` method silently overwrote each
other, and nothing in the codebase read the graph at all. It has been repaired
here — keyed by `file::qualname`, nested functions included, callers and
callees returned together — and the repaired version still does not beat a
regex on a four-file repository.

**Where it should pay, and why this cannot show it.** A symbol index earns its
keep when grep returns two hundred hits across a thousand files and you need
the one definition plus its call sites. These fixtures have one to four files.
The experiment establishes that the index is not free and not automatically
better; it says nothing about a real codebase.

#### The same question on a 2,007-line repository

Same five large tasks, run twice with the same inputs.
Raw: [`2026-09-02-0502-large-retrieval.json`](evals/results/) and
[`2026-09-02-0509-large-retrieval-repeat.json`](evals/results/).

| | run 1 | run 2 | change |
|---|---|---|---|
| `search` — regex over file contents | $0.3783 | $0.3753 | −0.8% |
| `find_symbol` — AST index | $0.3526 | $0.4158 | **+17.9%** |

Both configurations passed 5/5 in both runs.

**Still no difference worth claiming, and now the reason is measured rather
than assumed.** Run 1 has the index 6.8% cheaper. Run 2 has it 10.8% dearer.
The gap between the two tools is smaller than one tool's disagreement with
itself, and it changed sign on a repeat — which is what noise looks like.

That is the point of running the repeat. Reporting run 1 alone would have given
a 6.8% win for the feature this project is named after, and it would not have
survived being run again. The contrast with experiment 2 is the useful part:
there the gap was 66% on the same fixture, comfortably outside this 18% band,
which is why that one is reported as a result and this one is not.

**What is still untested.** Two thousand lines is a large fixture, not a large
codebase. The regime where an index should be decisive — thousands of files,
where `search` cannot return its hits inside a context window at all — is
beyond what this harness runs, and the honest summary is that the index has now
failed to beat grep at two sizes rather than one.

### Experiment 4 — routing the work down, or turning the dial down

There are two ways to spend less on the same task: leave the strong model in
place and lower `effort`, or send the work to a weaker model. Same five large
tasks, three configurations.
Raw: [`2026-09-02-1436-large-effort.json`](evals/results/).

| | pass | cost per completed task | median calls | input tokens | wall |
|---|---|---|---|---|---|
| `claude-opus-5`, effort low | 5/5 | $0.526 | 8 | 473,854 | 261s |
| `claude-opus-5`, effort high | 5/5 | $0.577 | 9 | 491,589 | 346s |
| `claude-sonnet-5`, effort low | 5/5 | **$0.087** | **6** | **182,692** | **178s** |

**Routing wins, and it is not close.** Sonnet 5 cost **6.0×** less than Opus 5
at the same effort, was cheaper on all five tasks — the range is 5.2× to 8.5× —
used 61% fewer input tokens and 33% fewer model calls, finished in two thirds
of the wall clock, and passed everything Opus passed.

**The effort dial did nothing measurable.** Low to high moved cost by 9.7%,
which is *inside* the 18% run-to-run band measured in experiment 3, and changed
no outcome. It is not that effort does nothing; it is that on tasks this size
there was nothing for the extra thinking to buy. One task even came out cheaper
at high effort than at low, which is the noise showing through again.

**The reason this cannot recommend Sonnet outright.** Every one of the fifteen
runs passed. When all three configurations score 5/5, the pass-rate column has
stopped measuring anything, and the honest reading is narrow:

> On work the weaker model can already do, paying six times more for the
> stronger one buys nothing this eval can see.

That is a real result and a common one in practice, but it is not "Sonnet is as
good as Opus" — it is "these tasks do not separate them." Separating them needs
a task the weaker model fails, and there is not one in this suite. Building one
is the next honest piece of work, and until it exists the default stays
`claude-opus-5` rather than being changed on the strength of an eval that
cannot see the difference it would be trading away.

What the number is good for is the bill: if your work looks like these tasks,

```bash
codepilot run --model claude-sonnet-5 "..."
```

is the same result for a sixth of the cost.

---

## Everything measured, in one place

| claim | evidence |
|---|---|
| A fixed pipeline cost 4.2× the loop and passed no more tasks | experiment 1, 15/15 both arms, variance 1.0% |
| Which edit tool is cheaper depends on file size, and reverses | experiment 2: `write_file` 1.83× cheaper at 2–15 lines, `edit_file` 1.66× cheaper at 437 |
| An AST index beat grep at neither size | experiment 3: gap within noise on small repos; on the large one it changed sign on a repeat |
| Routing to a weaker model beat turning the effort dial down | experiment 4: Sonnet 5 was 6.0× cheaper than Opus 5 and passed the same 5/5; low→high effort moved cost 9.7%, inside the noise |
| The suite can no longer separate the models it compares | experiment 4: 15 of 15 runs passed, so the pass-rate column measures nothing |
| Run-to-run variance is 1% on small tasks and up to 18% on large ones | experiment 1 repeat, experiment 3 repeat — the noise floor every other claim is judged against |
| A prompt prefix under ~4k tokens silently does not cache | measured directly: 2,119 tokens caches on neither model, 7,239 caches fully |
| Prompt caching serves ~75% of the loop's input on small repos, ~19% on large ones | every run in experiments 1–3; tool results grow faster than the cached prefix |
| Whole-file rewrites did not drop code, even at 437 lines | 0 of 50 runs, checked by name |
| Every task passed on the large fixture, in every configuration | 30 large runs, 30 passes |

Total spend to produce every number above: roughly $24 of API usage.

---

## Things this does not do

- **It is not hosted by me, and the public artefact is a
  [replay](web/) of real recorded sessions.** Hosting something that runs shell
  commands means paying for strangers' compute, so what ships instead is the
  ability for someone else to host it — see below, and
  [DEPLOYING.md](DEPLOYING.md) for the terms.
- **Hosted mode is single-tenant.** One shared bearer token, no accounts, no
  per-caller quota, no audit log, no persistence. The shape you put behind your
  own gateway, not one you expose to the public.
- **The container sandbox is not verified end to end.** The machine this was
  built on has no Docker daemon. The refusals, the policy, the HTTP surface and
  a complete run were tested — the last against the live API — but "a container
  actually started" was not, and is flagged rather than implied.
- **Anthropic only.** Prompt caching and tool-use semantics are where providers
  differ most, and an abstraction that satisfies all of them drops the caching —
  which is the difference between an agent that is affordable to iterate with
  and one that is not. The seam is one file if that changes.
- **Python tasks only** in the eval, because the sandbox runs pytest. The agent
  itself is not language-specific.
- **No GitHub issue → PR mode.** The original modelled it in five state fields
  and implemented none of it; those fields are gone rather than left as an API
  that silently does something else.

## Running it for other people

Local mode edits your repository on your machine because you asked it to;
isolating your code from your own computer would be theatre. A server runs
someone else's prompt on hardware you pay for, and every one of those
assumptions stops holding at once. `codepilot serve` is not a wrapper around
the CLI — it is that list of assumptions turned into refusals:

| | local | hosted |
|---|---|---|
| commands run | on your machine, in your repository | in a container, on a copy |
| a command outside the allowlist | asks you | refused, with the reason |
| your repository | edited in place, behind git checkpoints | untouched; you get a patch back |
| the API key | yours, from `.env` | the caller's, per request |
| the spending ceiling | yours | the deployment's; a request cannot raise it |

It refuses to start without a token of at least 16 characters. It refuses to run
commands on the host unless two separate variables say so, the second named
`CODEPILOT_ALLOW_HOST_EXECUTION`. It never hands the permission gate a prompt
callback, because there is nobody at the keyboard — so every command that would
have asked you is denied with its reason instead.

```bash
docker build -f deploy/sandbox.Dockerfile -t codepilot-sandbox:latest .
export CODEPILOT_SERVER_TOKEN=$(openssl rand -hex 24)
export CODEPILOT_WORKSPACE_ROOT=/srv/repos
codepilot serve --check     # prints the effective policy and starts nothing
codepilot serve
```

`--check` first, every time: it prints what you actually deployed — sandbox,
whether host execution is on, whether callers must bring their own key, the
ceiling, which repositories are visible — so you can compare it against what
you meant. The full guide, including why mounting the Docker socket into the
service container undoes most of the point, is in
[DEPLOYING.md](DEPLOYING.md).

## Development

```bash
pytest -q                    # 188 tests, no API key needed
ruff check .
python -m codepilot.doctor   # wiring checks
```

The tests use a stubbed model, so the whole suite runs offline in about twenty
seconds. `doctor` covers what unit tests structurally cannot — it found a pytest
parser that read every `-q` run as zero passed, and an `undo` that staged what it
restored, both while the suite was green.

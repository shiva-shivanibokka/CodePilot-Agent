"""
The eval runner.

    python -m evals.runner --arms loop pipeline --tier single
    python -m evals.runner --experiment edit-style
    python -m evals.runner --dry-run           # no API calls, checks the wiring

Each run gets a fresh temporary git repository, so no task can see another's
leftovers. The agent works with the same tools and sandbox it uses in anger;
only the held-out tests are withheld, written after the agent has finished and
run then. Pass means those tests pass.

Every number the write-up quotes comes out of `evals/results/<date>.json`,
which is committed. A claim whose evidence is not in the repository is not a
claim, it is a memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from codepilot.agent.loop import AgentLoop, new_conversation
from codepilot.context import Conversation
from codepilot.events import EventStream
from codepilot.llm import STRONG_MODEL, LLMClient, load_env
from codepilot.permissions import Budget, PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.tools import ToolContext
from codepilot.workspace import Workspace
from evals.tasks import HELD_OUT, Task, by_ids, by_tier

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class RunResult:
    task: str
    tier: str
    arm: str
    config: str
    passed: bool
    held_out_summary: str
    cost_usd: float
    unpriced_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    model_calls: int
    wall_seconds: float
    stopped_by: str
    files_edited: list[str] = field(default_factory=list)
    survivors_lost: list[str] = field(default_factory=list)
    error: str = ""
    #: True when the run died to the provider being unavailable rather than to
    #: anything the agent did. Counting a 529 as a failed task is the same
    #: mistake as averaging an unreachable judge's zero in as a verdict.
    infra_error: bool = False


def _init_repo(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "eval@example.com"],
        ["git", "config", "user.name", "eval"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "task fixture"],
    ):
        subprocess.run(cmd, cwd=root, capture_output=True, check=False)


#: Provider-side failures. The agent never got a turn, so the run says nothing
#: about the agent.
INFRASTRUCTURE = (
    "overloaded_error", "OverloadedError", "RateLimitError",
    "APIConnectionError", "APITimeoutError", "InternalServerError",
    "Error code: 429", "Error code: 5",
)


#: Failures that are the harness's own fault. Scoring these as agent failures
#: is worse than an outage, because the number looks plausible: a sweep with no
#: key ran to completion and reported 0/2 passed at a cost of $0.00.
HARNESS = (
    "Could not resolve authentication method",
    "ANTHROPIC_API_KEY",
)


def _is_harness_fault(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in HARNESS)


def _is_infrastructure(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in INFRASTRUCTURE)


def _check_survivors(root: Path, task: Task) -> list[str]:
    """Names that were in the fixture and must still be there.

    This is how a whole-file rewrite that drops unrelated code is detected: the
    task passes its own tests while having quietly deleted a sibling function.
    """
    lost = []
    for rel, names in task.must_survive.items():
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        lost += [f"{rel}:{n}" for n in names if f"def {n}" not in text]
    return lost


async def run_one(
    task: Task,
    arm: str,
    *,
    model: str,
    effort: str | None,
    config_name: str,
    tool_names: list[str] | None = None,
    max_cost: float = 0.50,
) -> RunResult:
    started = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
    try:
        _init_repo(tmp, task.repo())
        stream = EventStream(session_id=f"{task.id}:{arm}")
        ctx = ToolContext(
            workspace=Workspace(root=tmp, session_id="eval"),
            sandbox=LocalSandbox(root=tmp),
            permissions=PermissionGate(auto_approve=True),
            events=stream,
        )
        budget = Budget(max_usd=max_cost, max_turns=60)
        client = LLMClient(model=model)
        error, stopped_by, edited = "", "", []
        infra = False

        try:
            if arm == "pipeline":
                from codepilot.agent.pipeline import Runtime, run_pipeline

                rt = Runtime(
                    client=client,
                    ctx=ctx,
                    convo=Conversation(system_prompt="unused"),
                    budget=budget,
                    effort=effort,
                    model=model,
                )
                state = await run_pipeline(rt, task.prompt)
                stopped_by, edited = state.stopped_by or "finished", state.edited
            else:
                convo = new_conversation()
                loop = AgentLoop(
                    client, ctx, convo, budget,
                    tool_names=tool_names, effort=effort, model=model,
                )
                result = await loop.run(task.prompt)
                stopped_by, edited = result.stopped_by, result.edited
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the sweep
            if _is_harness_fault(exc):
                # Not a result. Ending the sweep is the honest response — the
                # alternative is a results file full of zeros the harness
                # earned rather than the agent.
                raise
            error, stopped_by = f"{type(exc).__name__}: {exc}", "error"
            infra = _is_infrastructure(exc)

        # Only now do the held-out tests exist.
        for rel, content in task.held_out.items():
            path = tmp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        outcome = await LocalSandbox(root=tmp).run_tests(f"pytest {HELD_OUT} -q")

        cost_events = [e for e in stream.events if e.type.value == "cost"]
        return RunResult(
            task=task.id,
            tier=task.tier,
            arm=arm,
            config=config_name,
            passed=outcome.success,
            held_out_summary=outcome.summary,
            cost_usd=round(budget.spent_usd, 6),
            unpriced_calls=budget.unpriced_calls,
            input_tokens=sum(int(e.data.get("input_tokens") or 0) for e in cost_events),
            output_tokens=sum(int(e.data.get("output_tokens") or 0) for e in cost_events),
            cache_read_tokens=sum(int(e.data.get("cache_read") or 0) for e in cost_events),
            model_calls=budget.turns,
            wall_seconds=round(time.monotonic() - started, 1),
            stopped_by=stopped_by,
            files_edited=edited,
            survivors_lost=_check_survivors(tmp, task),
            error=error,
            infra_error=infra,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarise(results: list[RunResult]) -> dict:
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.arm, r.config), []).append(r)

    summary = {}
    for (arm, config), rows in sorted(groups.items()):
        # A run the provider refused to serve is excluded rather than scored.
        # Keeping it would make an outage look like a worse agent.
        infra = [r for r in rows if r.infra_error]
        rows = [r for r in rows if not r.infra_error]
        if not rows:
            summary[f"{arm}/{config}"] = {"tasks": 0, "excluded_api_unavailable": len(infra)}
            continue
        passed = [r for r in rows if r.passed]
        costs = [r.cost_usd for r in rows]
        summary[f"{arm}/{config}"] = {
            "tasks": len(rows),
            "excluded_api_unavailable": len(infra),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(rows), 3) if rows else 0.0,
            "total_cost_usd": round(sum(costs), 4),
            "median_cost_usd": round(statistics.median(costs), 4) if costs else 0.0,
            "cost_per_pass_usd": (
                round(sum(costs) / len(passed), 4) if passed else None
            ),
            "median_model_calls": (
                statistics.median([r.model_calls for r in rows]) if rows else 0
            ),
            "cache_read_tokens": sum(r.cache_read_tokens for r in rows),
            "input_tokens": sum(r.input_tokens for r in rows),
            "runs_with_lost_code": sum(1 for r in rows if r.survivors_lost),
            "errors": sum(1 for r in rows if r.error),
            "unpriced_calls": sum(r.unpriced_calls for r in rows),
        }
    return summary


def render(results: list[RunResult], summary: dict) -> str:
    lines = [
        "",
        f"{'task':<22} {'tier':<7} {'arm':<9} {'config':<12} {'pass':<5} "
        f"{'cost':<9} {'calls':<6} {'sec':<6} stopped_by",
        "-" * 104,
    ]
    for r in sorted(results, key=lambda x: (x.task, x.arm, x.config)):
        mark = "n/a" if r.infra_error else ("yes" if r.passed else "no")
        flag = (
            "  !api-unavailable" if r.infra_error
            else "  !lost-code" if r.survivors_lost
            else "  !error" if r.error
            else ""
        )
        lines.append(
            f"{r.task:<22} {r.tier:<7} {r.arm:<9} {r.config:<12} {mark:<5} "
            f"${r.cost_usd:<8.4f} {r.model_calls:<6} {r.wall_seconds:<6.1f} {r.stopped_by}{flag}"
        )
    lines += ["", "summary", "-" * 104]
    for key, s in summary.items():
        # Guard first: a configuration where every run hit an outage carries
        # none of the keys below.
        if not s["tasks"]:
            lines.append(
                f"{key:<24} no scorable runs "
                f"({s['excluded_api_unavailable']} excluded: API unavailable)"
            )
            continue
        cache_pct = (
            100 * s["cache_read_tokens"] / (s["input_tokens"] + s["cache_read_tokens"])
            if (s["input_tokens"] + s["cache_read_tokens"])
            else 0
        )
        per_pass = (
            f"${s['cost_per_pass_usd']:.4f}" if s["cost_per_pass_usd"] is not None else "n/a"
        )
        lines.append(
            f"{key:<24} {s['passed']}/{s['tasks']} passed  "
            f"total ${s['total_cost_usd']:.4f}  per pass {per_pass}  "
            f"median {s['median_model_calls']:.0f} calls  cache {cache_pct:.0f}%"
        )
        if s.get("excluded_api_unavailable"):
            lines.append(
                f"{'':<24} note: {s['excluded_api_unavailable']} run(s) excluded — the "
                "provider was unavailable, which says nothing about the agent"
            )
        if s["unpriced_calls"]:
            lines.append(
                f"{'':<24} note: {s['unpriced_calls']} call(s) had no published "
                "price, so the cost above is a lower bound"
            )
        if s["runs_with_lost_code"]:
            lines.append(
                f"{'':<24} note: {s['runs_with_lost_code']} run(s) deleted code "
                "that was supposed to survive"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _preflight(dry_run: bool) -> None:
    """Fail before spending anything, rather than after scoring nothing."""
    load_env(Path.cwd())
    if not dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Put it in .env at the repository "
            "root, or run with --dry-run to check the fixtures for free."
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["loop"], choices=["loop", "pipeline"])
    parser.add_argument(
        "--tier", choices=["single", "multi", "debug", "large", "large-debug"]
    )
    parser.add_argument("--tasks", nargs="+", help="specific task ids")
    parser.add_argument("--model", default=STRONG_MODEL)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-cost", type=float, default=0.50, dest="max_cost")
    parser.add_argument(
        "--experiment",
        choices=["arms", "edit-style", "retrieval", "effort"],
        help="a prepared comparison instead of a plain sweep",
    )
    parser.add_argument("--dry-run", action="store_true", help="no API calls")
    parser.add_argument("--label", default="", help="tag for the results file")
    args = parser.parse_args()

    _preflight(args.dry_run)

    tasks = by_ids(args.tasks) if args.tasks else by_tier(args.tier)
    if args.dry_run:
        return await _dry_run(tasks)

    # Each configuration is (arm, config_name, tools, effort). Restricting the
    # tool set is how the edit-style experiment isolates one variable.
    configs: list[tuple[str, str, list[str] | None, str | None]] = []
    if args.experiment == "edit-style":
        base = ["read_file", "list_files", "search", "run_tests", "run_command", "finish"]
        configs = [
            ("loop", "edit_file", [*base, "edit_file"], args.effort),
            ("loop", "write_file", [*base, "write_file"], args.effort),
        ]
    elif args.experiment == "retrieval":
        # Experiment 3: does an AST index beat plain text search? The tool sets
        # differ by exactly one tool; everything else is identical.
        base = ["read_file", "list_files", "edit_file", "write_file", "run_tests", "finish"]
        configs = [
            ("loop", "search", [*base, "search"], args.effort),
            ("loop", "find_symbol", [*base, "find_symbol"], args.effort),
        ]
    elif args.experiment == "effort":
        configs = [("loop", f"effort={e}", None, e) for e in ("low", "high")]
    else:
        configs = [(arm, args.effort or "default", None, args.effort) for arm in args.arms]

    total = len(tasks) * len(configs)
    print(f"{total} runs: {len(tasks)} tasks x {len(configs)} configs ({args.model})")

    results: list[RunResult] = []
    for task in tasks:
        for arm, config_name, tool_names, effort in configs:
            print(f"  [{len(results) + 1}/{total}] {task.id} · {arm}/{config_name} … ", end="", flush=True)
            result = await run_one(
                task, arm,
                model=args.model, effort=effort, config_name=config_name,
                tool_names=tool_names, max_cost=args.max_cost,
            )
            results.append(result)
            print(
                f"{'pass' if result.passed else 'FAIL'} "
                f"${result.cost_usd:.4f} {result.wall_seconds:.0f}s"
            )

    summary = summarise(results)
    print(render(results, summary))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M")
    name = f"{stamp}{'-' + args.label if args.label else ''}.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "experiment": args.experiment or "sweep",
        "note": (
            "Pass means the held-out tests passed. The agent never saw them; "
            "they are written after it finishes."
        ),
        "summary": summary,
        "runs": [asdict(r) for r in results],
    }
    (RESULTS_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: evals/results/{name}")
    return 0


async def _dry_run(tasks: list[Task]) -> int:
    """Check the fixtures without spending anything.

    Two things must hold for a task to measure what it claims: the held-out
    tests must FAIL on the untouched fixture (otherwise the task is already
    done and every arm 'passes'), and the fixture's own suite, if it has one,
    must fail for the debug tier (otherwise there is nothing to debug).
    """
    print(f"checking {len(tasks)} task fixtures (no API calls)\n")
    bad = 0
    for task in tasks:
        tmp = Path(tempfile.mkdtemp(prefix=f"dry_{task.id}_"))
        try:
            repo = task.repo()
            _init_repo(tmp, repo)
            for rel, content in task.held_out.items():
                path = tmp / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            sandbox = LocalSandbox(root=tmp)
            outcome = await sandbox.run_tests(f"pytest {HELD_OUT} -q")
            problems = []
            if outcome.success:
                problems.append("held-out tests already pass on the untouched fixture")

            # A debug task whose visible suite is already green is telling the
            # agent something untrue. One such task cost a run: the agent read
            # the code, found nothing failing, said so, and was scored a
            # failure for being right.
            if task.tier in ("debug", "large-debug") and any(
                rel.startswith("tests/") and rel != HELD_OUT for rel in repo
            ):
                visible = [
                    rel for rel in repo if rel.startswith("tests/") and rel != HELD_OUT
                ]
                own = await sandbox.run_tests(f"pytest {' '.join(visible)} -q")
                if own.success:
                    problems.append(
                        "a debug task whose own suite already passes — the prompt "
                        "says it fails, so the premise is false"
                    )

            if problems:
                for problem in problems:
                    print(f"  [BAD ] {task.id}: {problem}")
                bad += 1
            else:
                print(f"  [ok  ] {task.id}: {outcome.summary} before the agent runs")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tasks) - bad}/{len(tasks)} fixtures are measuring something")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

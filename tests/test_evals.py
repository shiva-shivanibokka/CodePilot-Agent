"""The eval's own correctness.

A task that measures nothing is the worst failure this harness has, because it
reports a number rather than an error. Both checks below exist because a task
slipped through: one whose held-out tests already passed on the untouched
fixture, and one whose prompt said the suite was failing when it was green —
the agent read the code, said so, and was scored a failure for being right.
"""

from __future__ import annotations

import pytest

from evals.runner import RunResult, _check_survivors, summarise
from evals.tasks import HELD_OUT, TASKS, Task, by_ids, by_tier, load_fixture

TIERS = ("single", "multi", "debug", "large", "large-debug")


def test_every_task_has_held_out_tests_and_a_prompt():
    for task in TASKS:
        assert task.prompt.strip(), task.id
        assert HELD_OUT in task.held_out, f"{task.id} has no held-out tests"
        assert task.tier in TIERS, task.id


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_held_out_tests_are_never_part_of_the_starting_repository():
    """If the agent can read them, the eval measures nothing.

    Checked against the assembled repository rather than the inline `files`,
    because a fixture directory can ship a `tests/` tree of its own and one of
    those files landing on the held-out path would hand the agent the answer.
    """
    for task in TASKS:
        repo = task.repo()
        for path in task.held_out:
            assert path not in repo, f"{task.id} ships its own held-out tests"


def test_every_debug_task_ships_a_visible_failing_suite():
    """A debug task with no visible tests has nothing to debug."""
    for task in by_tier("debug") + by_tier("large-debug"):
        repo = task.repo()
        assert any(rel.startswith("tests/") for rel in repo), (
            f"{task.id} is a debug task but has no visible test file"
        )


# ---------------------------------------------------------------------------
# The large fixture
# ---------------------------------------------------------------------------


def test_the_large_fixture_is_actually_large():
    """The point of it is the regime the small tasks cannot reach.

    Experiments 2 and 3 both ended in "no difference worth claiming" because
    every fixture was under fifteen lines — too small for rewriting a file to
    cost anything, and too small for a regex to return the wrong definition.
    """
    repo = load_fixture("shop")
    lines = sum(len(text.splitlines()) for text in repo.values())
    assert lines > 1500, f"only {lines} lines"
    biggest = max(len(text.splitlines()) for text in repo.values())
    assert biggest > 400, f"biggest file is only {biggest} lines"


def test_the_large_fixture_is_ambiguous_to_grep():
    """A symbol index can only beat search where the name is not unique."""
    source = "\n".join(
        text for rel, text in load_fixture("shop").items() if rel.endswith(".py")
    )
    assert source.count("def apply") >= 10
    assert source.count("def validate") >= 10


def test_loading_a_fixture_skips_build_artefacts():
    assert not any("__pycache__" in rel for rel in load_fixture("shop"))
    assert not any(rel.endswith(".pyc") for rel in load_fixture("shop"))


def test_loading_an_unknown_fixture_says_so():
    with pytest.raises(FileNotFoundError, match="no fixture"):
        load_fixture("not-a-fixture")


def test_inline_files_are_laid_over_the_fixture():
    task = Task(
        id="t", tier="large", prompt="p", fixture="shop",
        files={"shop/errors.py": "replaced\n"},
    )
    repo = task.repo()
    assert repo["shop/errors.py"] == "replaced\n"
    assert "def rate_bps" in repo["shop/pricing.py"]


def test_a_mutation_changes_exactly_what_it_names():
    task = Task(
        id="t", tier="large-debug", prompt="p", fixture="shop",
        mutate=(("shop/errors.py", "class ShopError", "class Broken"),),
    )
    repo = task.repo()
    assert "class Broken" in repo["shop/errors.py"]
    assert "class ShopError" not in repo["shop/errors.py"]


def test_a_mutation_that_no_longer_matches_is_an_error():
    """Otherwise editing the fixture silently disarms the bug a task hunts."""
    task = Task(
        id="t", tier="large-debug", prompt="p", fixture="shop",
        mutate=(("shop/errors.py", "text that is not there", "x"),),
    )
    with pytest.raises(ValueError, match="mutation text not found"):
        task.repo()

    missing = Task(
        id="t", tier="large-debug", prompt="p", fixture="shop",
        mutate=(("shop/nope.py", "a", "b"),),
    )
    with pytest.raises(KeyError, match="missing file"):
        missing.repo()


def test_selecting_tasks_by_id():
    assert [t.id for t in by_ids(["empty-guard"])] == ["empty-guard"]
    with pytest.raises(KeyError, match="unknown task ids"):
        by_ids(["not-a-task"])


def test_survivor_check_notices_deleted_code(tmp_path):
    """How a whole-file rewrite that drops unrelated code gets caught: the task
    passes its own tests while a sibling function has quietly vanished."""
    task = Task(
        id="t", tier="multi", prompt="p",
        must_survive={"report.py": ["line_total", "subtotal"]},
    )
    (tmp_path / "report.py").write_text("def line_total(n):\n    return n\n", encoding="utf-8")
    assert _check_survivors(tmp_path, task) == ["report.py:subtotal"]


def test_survivor_check_is_quiet_when_everything_survives(tmp_path):
    task = Task(id="t", tier="multi", prompt="p", must_survive={"a.py": ["keep"]})
    (tmp_path / "a.py").write_text("def keep():\n    pass\n", encoding="utf-8")
    assert _check_survivors(tmp_path, task) == []


def _result(**kw) -> RunResult:
    base = dict(
        task="t", tier="single", arm="loop", config="low", passed=True,
        held_out_summary="1 passed", cost_usd=0.05, unpriced_calls=0,
        input_tokens=100, output_tokens=50, cache_read_tokens=0,
        model_calls=5, wall_seconds=10.0, stopped_by="finished",
    )
    return RunResult(**(base | kw))


def test_summary_reports_cost_per_pass_not_just_total():
    """Total cost rewards an arm that fails cheaply. Cost per completed task is
    the number that compares two agents fairly."""
    results = [
        _result(task="a", passed=True, cost_usd=0.10),
        _result(task="b", passed=False, cost_usd=0.10),
    ]
    s = summarise(results)["loop/low"]
    assert s["total_cost_usd"] == pytest.approx(0.20)
    assert s["cost_per_pass_usd"] == pytest.approx(0.20), "0.20 spent, 1 task done"
    assert s["pass_rate"] == 0.5


def test_summary_reports_no_cost_per_pass_when_nothing_passed():
    """Not zero, and not a division by zero — an absent number."""
    s = summarise([_result(passed=False)])["loop/low"]
    assert s["cost_per_pass_usd"] is None


def test_summary_surfaces_unpriced_calls_so_a_total_is_not_read_as_exact():
    s = summarise([_result(unpriced_calls=2)])["loop/low"]
    assert s["unpriced_calls"] == 2


def test_a_provider_outage_is_excluded_not_scored_as_a_failure():
    """A 529 means the agent never got a turn. Counting it as a failed task is
    the same error as averaging an unreachable judge's zero in as a verdict."""
    results = [
        _result(task="a", passed=True),
        _result(task="b", passed=False, infra_error=True, stopped_by="error",
                error="OverloadedError: Error code: 529"),
    ]
    s = summarise(results)["loop/low"]
    assert s["tasks"] == 1, "the outage was counted in the denominator"
    assert s["pass_rate"] == 1.0
    assert s["excluded_api_unavailable"] == 1


def test_the_exclusion_is_reported_not_hidden():
    from evals.runner import render

    results = [_result(passed=False, infra_error=True, error="Error code: 529")]
    text = render(results, summarise(results))
    assert "API unavailable" in text
    assert "api-unavailable" in text


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("OverloadedError: Error code: 529 - overloaded_error"),
        RuntimeError("Error code: 429 - rate_limit_error"),
        RuntimeError("APIConnectionError: connection reset"),
    ],
)
def test_provider_failures_are_recognised(exc):
    from evals.runner import _is_infrastructure

    assert _is_infrastructure(exc)


def test_an_agent_bug_is_not_mistaken_for_an_outage():
    from evals.runner import _is_infrastructure

    assert not _is_infrastructure(ValueError("that text does not appear in the file"))

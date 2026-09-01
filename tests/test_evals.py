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
from evals.tasks import HELD_OUT, TASKS, Task, by_ids, by_tier


def test_every_task_has_held_out_tests_and_a_prompt():
    for task in TASKS:
        assert task.prompt.strip(), task.id
        assert HELD_OUT in task.held_out, f"{task.id} has no held-out tests"
        assert task.tier in ("single", "multi", "debug"), task.id


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_held_out_tests_are_never_part_of_the_starting_repository():
    """If the agent can read them, the eval measures nothing."""
    for task in TASKS:
        for path in task.held_out:
            assert path not in task.files, f"{task.id} ships its own held-out tests"


def test_every_debug_task_ships_a_visible_failing_suite():
    """A debug task with no visible tests has nothing to debug."""
    for task in by_tier("debug"):
        assert any(rel.startswith("tests/") for rel in task.files), (
            f"{task.id} is tier=debug but has no visible test file"
        )


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

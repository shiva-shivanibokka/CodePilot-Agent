"""The tool layer: schemas, execution, and the guarantee that nothing raises."""

from __future__ import annotations

import pytest

from codepilot.events import EventStream, EventType
from codepilot.llm import ToolCall
from codepilot.permissions import PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.tools import REGISTRY, ToolContext, execute, schemas
from codepilot.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "fib.py").write_text("def fib(n):\n    return n\n", encoding="utf-8")
    return ToolContext(
        workspace=Workspace(root=tmp_path),
        sandbox=LocalSandbox(root=tmp_path),
        permissions=PermissionGate(auto_approve=True),
        events=EventStream(session_id="t"),
    )


async def call(ctx, name, **args):
    return await execute(ctx, ToolCall(id="tu_1", name=name, arguments=args))


# ------------------------------------------------------------------- schemas


def test_no_tool_takes_the_sandbox_as_a_parameter():
    """The old registry declared `sandbox: Any` as a tool argument, which would
    require the model to serialise a live object into JSON."""
    for spec in schemas():
        assert "sandbox" not in spec["input_schema"]["properties"], spec["name"]
        assert "workspace" not in spec["input_schema"]["properties"], spec["name"]


def test_every_tool_has_a_description_and_valid_schema():
    for spec in schemas():
        assert spec["description"].strip()
        assert spec["input_schema"]["type"] == "object"
        for name in spec["input_schema"]["required"]:
            assert name in spec["input_schema"]["properties"], (
                f"{spec['name']} requires {name} but does not declare it"
            )


def test_schema_order_is_stable_across_calls():
    """Tools sit in front of the cache breakpoint; a reordering invalidates it."""
    assert [s["name"] for s in schemas()] == [s["name"] for s in schemas()]


# ----------------------------------------------------------------- execution


@pytest.mark.asyncio
async def test_read_then_edit_round_trip(ctx):
    await call(ctx, "read_file", path="fib.py")
    result = await call(ctx, "edit_file", path="fib.py", old="return n", new="return n + 1")
    assert result.get("is_error") is not True
    assert "return n + 1" in (ctx.workspace.root / "fib.py").read_text()


@pytest.mark.asyncio
async def test_editing_without_reading_is_an_error_result_not_an_exception(ctx):
    result = await call(ctx, "edit_file", path="fib.py", old="return n", new="return 0")
    assert result["is_error"] is True
    assert "has not been read" in result["content"]


@pytest.mark.asyncio
async def test_an_unknown_tool_is_reported_not_raised(ctx):
    result = await call(ctx, "definitely_not_a_tool")
    assert result["is_error"] is True
    assert "No such tool" in result["content"]


@pytest.mark.asyncio
async def test_bad_arguments_are_reported_precisely(ctx):
    result = await call(ctx, "read_file", wrong_argument="x")
    assert result["is_error"] is True
    assert "Bad arguments" in result["content"]


@pytest.mark.asyncio
async def test_a_tool_that_raises_unexpectedly_still_returns_a_result(ctx, monkeypatch):
    async def boom(context, **kw):
        raise RuntimeError("internal explosion")

    monkeypatch.setattr(REGISTRY["list_files"], "run", boom)
    result = await call(ctx, "list_files")
    assert result["is_error"] is True
    assert "internal explosion" in result["content"]


@pytest.mark.asyncio
async def test_escaping_the_repo_root_is_refused(ctx):
    result = await call(ctx, "read_file", path="../../../etc/passwd")
    assert result["is_error"] is True
    assert "outside the workspace" in result["content"]


@pytest.mark.asyncio
async def test_a_refused_command_never_runs(ctx):
    ctx.permissions = PermissionGate(auto_approve=False, prompt=None)
    result = await call(ctx, "run_command", command="terraform apply")
    assert result["is_error"] is True
    assert "was not run" in result["content"]


@pytest.mark.asyncio
async def test_a_denied_command_is_refused_even_with_auto_approve(ctx):
    result = await call(ctx, "run_command", command="rm -rf /")
    assert result["is_error"] is True


# --------------------------------------------------------------------- events


@pytest.mark.asyncio
async def test_an_edit_emits_a_diff_event_with_line_counts(ctx):
    await call(ctx, "read_file", path="fib.py")
    await call(ctx, "edit_file", path="fib.py", old="return n", new="return n * 2")
    diffs = [e for e in ctx.events.events if e.type is EventType.DIFF]
    assert len(diffs) == 1
    assert diffs[0].data["added"] == 1 and diffs[0].data["removed"] == 1


@pytest.mark.asyncio
async def test_every_call_emits_a_call_and_a_result_event(ctx):
    await call(ctx, "list_files")
    types = [e.type for e in ctx.events.events]
    assert types.count(EventType.TOOL_CALL) == 1
    assert types.count(EventType.TOOL_RESULT) == 1


# ------------------------------------------------------------------- control


@pytest.mark.asyncio
async def test_finish_sets_the_flag_the_loop_reads(ctx):
    await call(ctx, "finish", summary="all done")
    assert ctx.finished is True
    assert ctx.final_message == "all done"


@pytest.mark.asyncio
async def test_propose_plan_records_the_steps_and_emits_them(ctx):
    await call(ctx, "propose_plan", steps=["read", "edit", "test"])
    assert ctx.plan == ["read", "edit", "test"]
    assert any(e.type is EventType.PLAN for e in ctx.events.events)


# ------------------------------------------------------------ real execution


@pytest.mark.asyncio
async def test_run_tests_reports_a_missing_suite_as_missing_not_failing(ctx):
    result = await call(ctx, "run_tests", command="pytest tests/ -q")
    assert result.get("is_error") is not True
    assert "missing suite" in result["content"]
    assert ctx.debugging is False, "a missing suite must not start a debugging turn"


@pytest.mark.asyncio
async def test_run_tests_on_a_real_passing_suite(ctx):
    (ctx.workspace.root / "test_ok.py").write_text(
        "def test_one():\n    assert 1 == 1\n", encoding="utf-8"
    )
    result = await call(ctx, "run_tests", command="pytest -q")
    # Assert on the PARSED summary, not on the raw output: "1 passed" appears
    # in pytest's own text even when the parser failed to read it, so the
    # substring check passed while the parser was returning zero.
    assert "1 passed (exit 0)" in result["content"], result["content"][:300]
    assert ctx.debugging is False


@pytest.mark.asyncio
async def test_a_failing_suite_starts_a_debugging_turn(ctx):
    (ctx.workspace.root / "test_bad.py").write_text(
        "def test_one():\n    assert 1 == 2\n", encoding="utf-8"
    )
    await call(ctx, "run_tests", command="pytest -q")
    assert ctx.debugging is True, "a real failure should protect the test files"

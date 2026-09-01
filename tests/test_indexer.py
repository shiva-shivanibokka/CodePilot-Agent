"""The AST index and symbol lookup."""

from __future__ import annotations

import pytest

from codepilot.events import EventStream
from codepilot.indexer import build_index
from codepilot.llm import ToolCall
from codepilot.permissions import PermissionGate
from codepilot.sandbox.local import LocalSandbox
from codepilot.tools import ToolContext, execute
from codepilot.workspace import Workspace

SOURCE = '''
def helper(x):
    """A helper."""
    return x + 1


class Alpha:
    def setup(self):
        return helper(1)

    def run(self):
        return self.setup()


class Beta:
    def setup(self):
        return 2


def outer():
    def inner():
        return helper(2)
    return inner()
'''


def _index(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    return build_index(tmp_path, ["mod.py"])


def test_same_named_methods_in_different_classes_both_survive(tmp_path):
    """The original keyed the graph on the bare name, so one setup silently
    overwrote the other."""
    index = _index(tmp_path)
    setups = index.lookup("setup")
    assert len(setups) == 2
    assert {s.qualname for s in setups} == {"Alpha.setup", "Beta.setup"}


def test_nested_functions_are_indexed(tmp_path):
    """The original skipped recursion while claiming nested functions were
    'picked up as top-level'. They were not picked up at all."""
    index = _index(tmp_path)
    assert index.lookup("inner"), "a nested function was not indexed"
    assert index.lookup("inner")[0].qualname == "outer.inner"


def test_callers_are_found_across_scopes(tmp_path):
    index = _index(tmp_path)
    callers = {s.qualname for s in index.callers_of("helper")}
    assert "Alpha.setup" in callers
    assert "outer.inner" in callers


def test_classes_and_docstrings_are_captured(tmp_path):
    index = _index(tmp_path)
    assert index.lookup("Alpha")[0].kind == "class"
    assert index.lookup("helper")[0].docstring == "A helper."


def test_an_unparseable_file_is_reported_not_silently_dropped(tmp_path):
    """A file the agent just broke should be visible as broken."""
    (tmp_path / "broken.py").write_text("def oops(\n", encoding="utf-8")
    index = build_index(tmp_path, ["broken.py"])
    assert index.unparseable == ["broken.py"]


@pytest.mark.asyncio
async def test_find_symbol_reports_definition_callers_and_callees(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        sandbox=LocalSandbox(root=tmp_path),
        permissions=PermissionGate(auto_approve=True),
        events=EventStream(session_id="t"),
    )
    result = await execute(
        ctx, ToolCall(id="1", name="find_symbol", arguments={"name": "helper"})
    )
    assert result.get("is_error") is not True
    assert "def helper(x)" in result["content"]
    assert "called by" in result["content"]


@pytest.mark.asyncio
async def test_an_unknown_symbol_lists_what_does_exist(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    ctx = ToolContext(
        workspace=Workspace(root=tmp_path),
        sandbox=LocalSandbox(root=tmp_path),
        permissions=PermissionGate(auto_approve=True),
        events=EventStream(session_id="t"),
    )
    result = await execute(
        ctx, ToolCall(id="1", name="find_symbol", arguments={"name": "nope"})
    )
    assert result["is_error"] is True
    assert "helper" in result["content"], "the error should say what does exist"

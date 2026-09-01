"""Workspace safety: containment, staleness, and the checkpoint/undo round trip."""

from __future__ import annotations

import subprocess

import pytest

from codepilot.workspace import Workspace, WorkspaceError


def _git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "keep.py").write_text("original = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return Workspace(root=tmp_path, session_id="test")


# ---------------------------------------------------------------- containment


@pytest.mark.parametrize(
    "path",
    ["../escape.py", "../../escape.py", "a/../../escape.py", "sub/../../out.py"],
)
def test_refuses_paths_that_escape_the_root(tmp_path, path):
    ws = Workspace(root=tmp_path)
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve(path)


def test_allows_nested_paths_inside_the_root(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.resolve("a/b/c.py").parent.parent.name == "a"


# ------------------------------------------------------------ read-before-write


def test_writing_an_unread_existing_file_is_refused(tmp_path):
    (tmp_path / "existing.py").write_text("precious = 1\n", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    with pytest.raises(WorkspaceError, match="has not been read"):
        ws.write("existing.py", "clobbered = 1\n")
    assert (tmp_path / "existing.py").read_text() == "precious = 1\n"


def test_creating_a_new_file_needs_no_prior_read(tmp_path):
    ws = Workspace(root=tmp_path)
    assert "Created" in ws.write("brand_new.py", "x = 1\n")


def test_write_is_refused_if_the_file_changed_since_it_was_read(tmp_path):
    target = tmp_path / "f.py"
    target.write_text("v = 1\n", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    ws.read("f.py")
    target.write_text("v = 2  # edited in your editor\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="changed on disk"):
        ws.write("f.py", "v = 3\n")
    assert "edited in your editor" in target.read_text()


# ------------------------------------------------------------------------ edit


def test_edit_replaces_an_exact_string(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    ws.read("f.py")
    ws.edit("f.py", "b = 2", "b = 99")
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 99\n"


def test_edit_refuses_an_ambiguous_match(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    ws.read("f.py")
    with pytest.raises(WorkspaceError, match="appears 2 times"):
        ws.edit("f.py", "x = 1", "x = 2")


def test_edit_reports_a_missing_match_usefully(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    ws = Workspace(root=tmp_path)
    ws.read("f.py")
    with pytest.raises(WorkspaceError, match="does not appear"):
        ws.edit("f.py", "y = 2", "y = 3")


# ------------------------------------------------------------------ checkpoints


def test_checkpoint_then_undo_restores_edits_and_removes_new_files(tmp_path):
    ws = _git_repo(tmp_path)
    ws.checkpoint("before")

    ws.read("keep.py")
    ws.write("keep.py", "original = 999\n")
    ws.write("added_by_agent.py", "junk = True\n")
    assert (tmp_path / "added_by_agent.py").exists()

    ws.undo()
    assert (tmp_path / "keep.py").read_text() == "original = 1\n"
    assert not (tmp_path / "added_by_agent.py").exists(), (
        "undo left the agent's new file behind, which is not an undo"
    )


def test_checkpoint_does_not_disturb_the_users_staged_index(tmp_path):
    ws = _git_repo(tmp_path)
    (tmp_path / "mine.py").write_text("mine = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "mine.py"], cwd=tmp_path, check=True)
    before = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout

    ws.checkpoint("mid-work")

    after = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout
    assert before == after == "mine.py\n"


def test_checkpoint_outside_a_git_repo_refuses_clearly(tmp_path):
    ws = Workspace(root=tmp_path)
    with pytest.raises(WorkspaceError, match="not a git repository"):
        ws.checkpoint("nope")


# ------------------------------------------------------------------- listing


def test_list_files_honours_gitignore(tmp_path):
    ws = _git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("shh", encoding="utf-8")
    (tmp_path / "visible.py").write_text("x = 1\n", encoding="utf-8")
    listed = ws.list_files()
    assert "visible.py" in listed
    assert "secret.txt" not in listed


def test_undo_does_not_stage_anything(tmp_path):
    """`git checkout <commit> -- .` writes to the index too, staging work the
    user never staged. Undo must touch the worktree only."""
    ws = _git_repo(tmp_path)
    ws.checkpoint("before")
    ws.read("keep.py")
    ws.write("keep.py", "original = 2\n")
    ws.undo()

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert staged == "", f"undo staged: {staged!r}"


def test_undo_removes_directories_it_emptied(tmp_path):
    ws = _git_repo(tmp_path)
    ws.checkpoint("before")
    ws.write("tests/test_new.py", "def test_x():\n    assert True\n")
    assert (tmp_path / "tests").is_dir()
    ws.undo()
    assert not (tmp_path / "tests").exists(), "an emptied directory was left behind"


def test_checkpoints_are_recoverable_in_a_fresh_process(tmp_path):
    """`codepilot undo` runs later, with no in-memory checkpoint list."""
    ws = _git_repo(tmp_path)
    ws.checkpoint("first")
    ws.read("keep.py")
    ws.write("keep.py", "original = 5\n")

    reopened = Workspace(root=tmp_path, session_id="test")
    assert reopened.load_checkpoints(), "checkpoints were not recoverable from refs"
    reopened.undo()
    assert (tmp_path / "keep.py").read_text() == "original = 1\n"


def test_undo_prunes_a_directory_left_holding_only_build_artifacts(tmp_path):
    """`tests/` surviving an undo reads as a failed undo, even when the test
    file itself is gone and only __pycache__ remains."""
    ws = _git_repo(tmp_path)
    ws.checkpoint("before")
    ws.write("tests/test_new.py", "def test_x():\n    assert True\n")
    cache = tmp_path / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_new.cpython-312.pyc").write_bytes(b"\x00")

    ws.undo()
    assert not (tmp_path / "tests").exists()


def test_undo_never_deletes_the_session_log(tmp_path):
    """The log survived by accident of ordering; it must survive by design."""
    ws = _git_repo(tmp_path)
    ws.checkpoint("before")
    log = tmp_path / ".codepilot" / "sessions" / "abc.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"kind":"meta"}\n', encoding="utf-8")

    ws.undo()
    assert log.exists(), "undo erased the record of what it undid"

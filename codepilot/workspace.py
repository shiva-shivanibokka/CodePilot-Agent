"""
The real working tree, and the safety net that makes editing it survivable.

An agent that writes into a temp directory which is then deleted is a demo. To
be a tool it has to edit the files you actually care about — which is only
reasonable if every turn is revertible.

Three guarantees:

* **Containment.** Nothing is written outside the repo root, whatever the model
  or the caller asks for.
* **Read-before-write.** A file must be read this session, and be unchanged
  since that read, before it can be edited. Otherwise an edit silently discards
  whatever you changed in your editor while the agent was thinking.
* **Checkpoints.** Before each turn the working tree is committed to a scratch
  ref, using a throwaway index so your real index and HEAD are never touched.

The gitignore walk and the checkpoint both delegate to git plumbing. Writing a
gitignore matcher or an undo stack by hand would be more code and less correct.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Directories never worth showing an agent, even if git would list them.
ALWAYS_SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}


class WorkspaceError(RuntimeError):
    """A refused operation. Always safe to hand back to the model as a tool error."""


@dataclass
class Checkpoint:
    ref: str
    commit: str
    label: str


@dataclass
class Workspace:
    root: Path
    session_id: str = "session"
    #: path -> sha256 of the content when the agent last read it.
    _read_ledger: dict[str, str] = field(default_factory=dict)
    _checkpoints: list[Checkpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Not a directory: {self.root}")

    # ------------------------------------------------------------------
    # Git
    # ------------------------------------------------------------------

    def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    @property
    def is_git_repo(self) -> bool:
        try:
            return self._git("rev-parse", "--is-inside-work-tree").strip() == "true"
        except WorkspaceError:
            return False

    def checkpoint(self, label: str) -> Checkpoint:
        """Snapshot the working tree to a scratch ref.

        Uses a temporary GIT_INDEX_FILE so staging anything here cannot disturb
        what the user has staged, and commit-tree rather than commit so HEAD
        does not move.
        """
        if not self.is_git_repo:
            raise WorkspaceError(
                f"{self.root} is not a git repository, so edits cannot be checkpointed. "
                "Run `git init`, or pass --no-checkpoint to accept that undo is unavailable."
            )
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            self._git("add", "-A", env=env)
            tree = self._git("write-tree", env=env).strip()

        parent = self._checkpoints[-1].commit if self._checkpoints else None
        args = ["commit-tree", tree, "-m", f"codepilot checkpoint: {label}"]
        if parent:
            args += ["-p", parent]
        commit = self._git(*args).strip()

        ref = f"refs/codepilot/{self.session_id}/{len(self._checkpoints)}"
        self._git("update-ref", ref, commit)
        cp = Checkpoint(ref=ref, commit=commit, label=label)
        self._checkpoints.append(cp)
        return cp

    def undo(self) -> Checkpoint | None:
        """Restore the most recent checkpoint's tree.

        Restores tracked content and removes files created since — a restore
        that leaves the agent's new files behind is not an undo.
        """
        if not self._checkpoints:
            return None
        cp = self._checkpoints.pop()
        in_tree = {
            line.strip()
            for line in self._git("ls-tree", "-r", "--name-only", cp.commit).splitlines()
            if line.strip()
        }
        for rel in self.list_files():
            if rel not in in_tree:
                (self.root / rel).unlink(missing_ok=True)
        # `-- .` restricts the checkout to the worktree, leaving HEAD alone.
        self._git("checkout", cp.commit, "--", ".")
        self._read_ledger.clear()
        return cp

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        """Resolve inside the root, or refuse.

        Rejects absolute paths and any `..` that escapes, including via a
        symlink — `resolve()` follows links before the containment check.
        """
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"Refusing to touch {path!r}: it resolves outside the workspace root."
            )
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def list_files(self) -> list[str]:
        """Every file the agent should be able to see, honouring .gitignore."""
        if self.is_git_repo:
            out = self._git("ls-files", "--cached", "--others", "--exclude-standard")
            names = [line.strip() for line in out.splitlines() if line.strip()]
        else:
            names = [
                (p.relative_to(self.root)).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
            ]
        return sorted(
            n for n in names if not any(part in ALWAYS_SKIP for part in Path(n).parts)
        )

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"No such file: {path}")
        content = target.read_text(encoding="utf-8", errors="replace")
        self._read_ledger[self.relative(target)] = self._hash(content)
        return content

    def _assert_fresh(self, path: str, target: Path) -> None:
        """Refuse to write over a file the agent has not seen the current state of."""
        rel = self.relative(target)
        if not target.exists():
            return  # creating a new file needs no prior read
        seen = self._read_ledger.get(rel)
        if seen is None:
            raise WorkspaceError(
                f"{rel} exists but has not been read in this session. Read it first — "
                "writing blind would discard whatever is currently in it."
            )
        current = self._hash(target.read_text(encoding="utf-8", errors="replace"))
        if current != seen:
            raise WorkspaceError(
                f"{rel} changed on disk since it was read. Read it again before editing; "
                "the version in context is stale."
            )

    def write(self, path: str, content: str) -> str:
        target = self.resolve(path)
        self._assert_fresh(path, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        self._read_ledger[self.relative(target)] = self._hash(content)
        verb = "Updated" if existed else "Created"
        return f"{verb} {self.relative(target)} ({content.count(chr(10)) + 1} lines)"

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace an exact string.

        The default edit primitive, in preference to rewriting whole files: a
        whole-file rewrite costs tokens proportional to the file rather than the
        change, and silently drops anything the model forgot to reproduce.
        """
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"No such file: {path}")
        self._assert_fresh(path, target)
        content = target.read_text(encoding="utf-8", errors="replace")

        count = content.count(old)
        if count == 0:
            raise WorkspaceError(
                f"That exact text does not appear in {self.relative(target)}. "
                "Read the file again and match it byte for byte, whitespace included."
            )
        if count > 1 and not replace_all:
            raise WorkspaceError(
                f"That text appears {count} times in {self.relative(target)}. "
                "Include more surrounding context to make it unique, or pass "
                "replace_all=true if every occurrence should change."
            )

        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        self._read_ledger[self.relative(target)] = self._hash(updated)
        where = f"{count} occurrences" if replace_all and count > 1 else "1 occurrence"
        return f"Edited {self.relative(target)} ({where})"

    def mark_read(self, path: str, content: str) -> None:
        """Record a read the workspace did not perform itself (replay, tests)."""
        self._read_ledger[self.relative(self.resolve(path))] = self._hash(content)

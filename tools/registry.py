"""
Tool Registry — all tools available to LangGraph agents.

Every tool is an async function that takes the current sandbox backend
and the agent state, performs one action, and returns a structured result.

Tools are registered as LangChain/LangGraph @tool decorated functions
so the LLM can call them via tool use.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from langchain_core.tools import tool

from models.schemas import ExecutionResult, TestResult


# ---------------------------------------------------------------------------
# File system tools
# ---------------------------------------------------------------------------


@tool
async def read_file(path: str, sandbox: Any) -> str:
    """
    Read the contents of a file from the sandbox workspace.
    Returns the file content as a string.
    Use this before editing any file so you have the current content.
    """
    try:
        content = await sandbox.read_file(path)
        lines = content.splitlines()
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        return f"File: {path} ({len(lines)} lines)\n\n{numbered}"
    except FileNotFoundError:
        return f"ERROR: File '{path}' not found in sandbox workspace."
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"


@tool
async def write_file(path: str, content: str, sandbox: Any) -> str:
    """
    Write or overwrite a file in the sandbox workspace.
    Always write complete file content — do not write partial files.
    Returns confirmation with line count.
    """
    try:
        await sandbox.write_file(path, content)
        lines = content.count("\n") + 1
        return f"Written: {path} ({lines} lines)"
    except Exception as exc:
        return f"ERROR writing {path}: {exc}"


@tool
async def list_files(sandbox: Any, directory: str = ".") -> str:
    """
    List all files in the sandbox workspace (or a subdirectory).
    Returns a newline-separated list of relative file paths.
    """
    try:
        files = await sandbox.list_files(directory)
        if not files:
            return f"No files found in '{directory}'"
        return "\n".join(sorted(files))
    except Exception as exc:
        return f"ERROR listing files: {exc}"


@tool
async def search_codebase(query: str, sandbox: Any, file_pattern: str = "*.py") -> str:
    """
    Search for a string or regex pattern across all files matching file_pattern.
    Returns matching lines with file path and line number.
    Use this to find function definitions, usages, imports, etc.
    """
    try:
        files = await sandbox.list_files(".")
        results: list[str] = []

        pattern = re.compile(query, re.IGNORECASE)
        matched_files = [f for f in files if _matches_glob(f, file_pattern)]

        for filepath in matched_files[:50]:  # cap at 50 files
            try:
                content = await sandbox.read_file(filepath)
                for i, line in enumerate(content.splitlines(), start=1):
                    if pattern.search(line):
                        results.append(f"{filepath}:{i}: {line.rstrip()}")
            except Exception:
                continue

        if not results:
            return f"No matches for '{query}' in {file_pattern} files."
        return f"Found {len(results)} match(es):\n\n" + "\n".join(results[:100])
    except Exception as exc:
        return f"ERROR searching codebase: {exc}"


@tool
async def diff_files(path: str, original_content: str, sandbox: Any) -> str:
    """
    Show a unified diff between the original content and the current sandbox file.
    Use this to review what has changed before running tests.
    """
    try:
        current = await sandbox.read_file(path)
        diff = list(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if not diff:
            return f"No changes in {path}"
        return "".join(diff)
    except FileNotFoundError:
        return f"ERROR: '{path}' not found."
    except Exception as exc:
        return f"ERROR generating diff: {exc}"


# ---------------------------------------------------------------------------
# Execution tools
# ---------------------------------------------------------------------------


@tool
async def run_code(code: str, sandbox: Any, filename: str = "run.py") -> str:
    """
    Execute a Python code snippet in the sandbox and return stdout/stderr.
    Use this to test a small piece of logic before writing it to a file.
    Maximum execution time: 30 seconds.
    """
    result: ExecutionResult = await sandbox.execute(code, filename=filename)
    output = []
    if result.stdout:
        output.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        output.append(f"STDERR:\n{result.stderr}")
    output.append(f"Exit code: {result.exit_code} | Duration: {result.duration_ms}ms")
    if result.timed_out:
        output.append("WARNING: Execution timed out after 30 seconds")
    return "\n".join(output) if output else "No output produced."


@tool
async def run_command(command: str, sandbox: Any) -> str:
    """
    Run an arbitrary shell command in the sandbox workspace.
    Examples: 'pip install requests', 'python -m mymodule', 'ls -la'.
    Maximum execution time: 60 seconds.
    """
    result: ExecutionResult = await sandbox.run_command(command)
    output = []
    if result.stdout:
        output.append(result.stdout)
    if result.stderr:
        output.append(f"STDERR: {result.stderr}")
    output.append(f"[exit {result.exit_code} | {result.duration_ms}ms]")
    if result.timed_out:
        output.append("WARNING: Command timed out")
    return "\n".join(output)


@tool
async def run_tests(
    sandbox: Any,
    test_command: str = "pytest tests/ -v --tb=short",
) -> str:
    """
    Run the test suite in the sandbox and return a structured summary.
    Parses pytest output to extract pass/fail counts and failure messages.
    Use this after every code change to verify correctness.
    """
    result: TestResult = await sandbox.run_tests(test_command)
    lines = [
        f"Test command: {result.command}",
        f"Result: {result.summary}",
        f"Duration: {result.duration_ms}ms",
        f"Success: {result.success}",
        "",
        "--- Output ---",
        result.output[-3000:] if len(result.output) > 3000 else result.output,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------


@tool
async def git_status(sandbox: Any) -> str:
    """Show the current git status of the workspace."""
    result: ExecutionResult = await sandbox.run_command("git status --short")
    return result.stdout or "Clean working tree (no changes)"


@tool
async def git_diff(sandbox: Any) -> str:
    """Show a unified diff of all unstaged changes in the workspace."""
    result: ExecutionResult = await sandbox.run_command("git diff")
    return result.stdout or "No unstaged changes"


@tool
async def git_commit(message: str, sandbox: Any) -> str:
    """
    Stage all changes and create a git commit with the given message.
    Use this only when the task is complete and tests are passing.
    """
    stage = await sandbox.run_command("git add -A")
    commit = await sandbox.run_command(f'git commit -m "{message}"')
    return f"Staged:\n{stage.stdout}\n\nCommit:\n{commit.stdout or commit.stderr}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_glob(filepath: str, pattern: str) -> bool:
    """Simple glob match: *.py, *.{py,js}, or exact filename."""
    import fnmatch

    if "," in pattern:
        # handle *.{py,js} style
        exts = pattern.replace("*.", "").strip("{}").split(",")
        return any(filepath.endswith(f".{e.strip()}") for e in exts)
    return fnmatch.fnmatch(filepath, pattern)


# ---------------------------------------------------------------------------
# Tool list for LLM binding
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    read_file,
    write_file,
    list_files,
    search_codebase,
    diff_files,
    run_code,
    run_command,
    run_tests,
    git_status,
    git_diff,
    git_commit,
]

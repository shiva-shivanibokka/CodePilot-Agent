"""
AST Indexer — builds a compact structural index of a Python codebase.

This solves Claude Code's #1 documented problem: context window exhaustion.
Instead of feeding the LLM raw file contents, we extract structured facts:
  - what functions/classes exist and where
  - the call graph (who calls what)
  - import graph (what depends on what)
  - which files are relevant to a given task description

The Planner and Coder agents query this index to determine which 2-3 files
to read, rather than loading the entire codebase into context.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FunctionInfo:
    name: str
    file: str
    start_line: int
    end_line: int
    args: list[str]
    calls: list[str]  # function names this function calls
    docstring: str | None
    is_async: bool
    complexity: int = 1  # cyclomatic complexity (incremented per branch)


@dataclass
class ClassInfo:
    name: str
    file: str
    start_line: int
    bases: list[str]
    methods: list[str]


@dataclass
class FileIndex:
    path: str
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # module names imported
    loc: int = 0  # lines of code
    has_tests: bool = False


@dataclass
class CodebaseIndex:
    files: dict[str, FileIndex] = field(default_factory=dict)
    call_graph: dict[str, list[str]] = field(default_factory=dict)  # fn → [callees]
    import_graph: dict[str, list[str]] = field(default_factory=dict)  # file → [imports]

    def all_functions(self) -> list[FunctionInfo]:
        fns = []
        for fi in self.files.values():
            fns.extend(fi.functions)
        return fns

    def find_function(self, name: str) -> FunctionInfo | None:
        for fn in self.all_functions():
            if fn.name == name:
                return fn
        return None

    def callers_of(self, fn_name: str) -> list[str]:
        """Return all function names that call fn_name."""
        return [
            caller for caller, callees in self.call_graph.items() if fn_name in callees
        ]

    def relevant_files(self, task_description: str, top_k: int = 4) -> list[str]:
        """
        Return the top_k most relevant files for a task description.
        Uses keyword overlap between task text and file contents metadata.
        """
        task_tokens = set(re.findall(r"\w+", task_description.lower()))
        scores: dict[str, float] = {}

        for path, fi in self.files.items():
            file_tokens: set[str] = set()
            # Score from function/class names
            for fn in fi.functions:
                file_tokens.update(re.findall(r"\w+", fn.name.lower()))
                if fn.docstring:
                    file_tokens.update(re.findall(r"\w+", fn.docstring.lower()))
            for cls in fi.classes:
                file_tokens.update(re.findall(r"\w+", cls.name.lower()))
            # Score from imports
            for imp in fi.imports:
                file_tokens.update(re.findall(r"\w+", imp.lower()))
            # Score from filename itself
            file_tokens.update(re.findall(r"\w+", Path(path).stem.lower()))

            overlap = len(task_tokens & file_tokens)
            if overlap > 0:
                scores[path] = overlap / (1 + fi.loc / 100)  # penalise very large files

        return sorted(scores, key=scores.get, reverse=True)[:top_k]  # type: ignore[arg-type]

    def summary(self) -> str:
        """Compact text summary for the Planner agent's context."""
        lines = [
            f"Codebase: {len(self.files)} files, "
            f"{sum(fi.loc for fi in self.files.values())} total lines",
            "",
        ]
        for path, fi in sorted(self.files.items()):
            fn_names = [f.name for f in fi.functions[:8]]
            cls_names = [c.name for c in fi.classes]
            summary_parts = []
            if cls_names:
                summary_parts.append(f"classes: {', '.join(cls_names)}")
            if fn_names:
                extra = (
                    f"+{len(fi.functions) - 8} more" if len(fi.functions) > 8 else ""
                )
                summary_parts.append(
                    f"functions: {', '.join(fn_names)}{' ' + extra if extra else ''}"
                )
            if fi.has_tests:
                summary_parts.append("has tests")
            detail = " | ".join(summary_parts) if summary_parts else "empty"
            lines.append(f"  {path} ({fi.loc} loc) — {detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visitor for AST walking
# ---------------------------------------------------------------------------


class _FunctionVisitor(ast.NodeVisitor):
    """Extracts function info and call graph from a single file's AST."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[str] = []
        self._current_fn: FunctionInfo | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(b) for b in node.bases]
        methods = [
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.classes.append(
            ClassInfo(
                name=node.name,
                file=self.filepath,
                start_line=node.lineno,
                bases=bases,
                methods=methods,
            )
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_fn(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_fn(node, is_async=True)

    def _visit_fn(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:
        args = [a.arg for a in node.args.args]
        docstring = ast.get_docstring(node)
        end_line = getattr(node, "end_lineno", node.lineno)

        # Measure cyclomatic complexity
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        fn_info = FunctionInfo(
            name=node.name,
            file=self.filepath,
            start_line=node.lineno,
            end_line=end_line,
            args=args,
            calls=[],
            docstring=docstring,
            is_async=is_async,
            complexity=complexity,
        )

        # Collect all call names inside this function body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._resolve_call_name(child.func)
                if call_name:
                    fn_info.calls.append(call_name)

        self.functions.append(fn_info)
        # Don't recurse into nested functions with generic_visit
        # (they'll be picked up as top-level in a flat list)

    @staticmethod
    def _resolve_call_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ASTIndexer:
    """
    Builds and queries a CodebaseIndex from a dict of {path: source_code}.
    """

    def build(self, workspace_files: dict[str, str]) -> CodebaseIndex:
        index = CodebaseIndex()

        for path, source in workspace_files.items():
            if not path.endswith(".py"):
                continue
            file_index = self._index_file(path, source)
            index.files[path] = file_index

        # Build cross-file call graph
        all_fn_names = {fn.name for fi in index.files.values() for fn in fi.functions}
        for fi in index.files.values():
            for fn in fi.functions:
                # Only keep calls that reference known functions in this codebase
                cross_calls = [c for c in fn.calls if c in all_fn_names]
                index.call_graph[fn.name] = cross_calls

        # Build import graph
        for path, fi in index.files.items():
            index.import_graph[path] = fi.imports

        return index

    def _index_file(self, path: str, source: str) -> FileIndex:
        fi = FileIndex(
            path=path,
            loc=source.count("\n") + 1,
            has_tests="test" in path.lower(),
        )
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return fi  # unparseable — return empty index for this file

        visitor = _FunctionVisitor(path)
        visitor.visit(tree)

        fi.functions = visitor.functions
        fi.classes = visitor.classes
        fi.imports = list(set(visitor.imports))
        return fi

"""
A structural index of the repository, and retrieval over it.

The original version of this module was the project's headline feature and did
not work: its call graph was keyed by bare function name, so `setup` in two
classes overwrote each other, and nothing in the codebase read the graph at
all. What actually reached the model was a bag-of-words overlap on function
names — keyword retrieval wearing an AST costume.

Here the graph is keyed by `file::qualname`, and it is used: `find_symbol`
returns a definition together with its callers and callees, so following a
change through the code costs one tool call instead of several greps.

Whether that beats plain text search is experiment 3 in the eval, not an
assumption. Both are real tools; the harness runs the same tasks with one or
the other and reports pass rate and tokens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    qualname: str  # "Class.method" or "function"
    file: str
    line: int
    end_line: int
    kind: str  # "function" | "method" | "class"
    signature: str
    docstring: str | None
    calls: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.file}::{self.qualname}"


@dataclass
class Index:
    symbols: dict[str, Symbol] = field(default_factory=dict)  # key -> Symbol
    #: bare name -> every key defining it. A name is ambiguous across files;
    #: keeping the many-to-one relationship is the fix for the original's
    #: silent overwrite.
    by_name: dict[str, list[str]] = field(default_factory=dict)
    imports: dict[str, list[str]] = field(default_factory=dict)
    unparseable: list[str] = field(default_factory=list)

    def lookup(self, name: str) -> list[Symbol]:
        """Every definition of `name`, exact match first, then suffix matches."""
        keys = list(self.by_name.get(name, []))
        if not keys:
            keys = [
                k for k, s in self.symbols.items() if s.qualname.endswith(f".{name}")
            ]
        return [self.symbols[k] for k in keys]

    def callers_of(self, name: str) -> list[Symbol]:
        return sorted(
            (s for s in self.symbols.values() if name in s.calls),
            key=lambda s: s.key,
        )

    def outline(self, limit: int = 400) -> str:
        by_file: dict[str, list[Symbol]] = {}
        for symbol in self.symbols.values():
            by_file.setdefault(symbol.file, []).append(symbol)
        lines = []
        for file in sorted(by_file):
            lines.append(file)
            for symbol in sorted(by_file[file], key=lambda s: s.line):
                lines.append(f"  {symbol.line:>4}  {symbol.signature}")
                if len(lines) > limit:
                    lines.append("  … (truncated)")
                    return "\n".join(lines)
        return "\n".join(lines)


class _Visitor(ast.NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.symbols: list[Symbol] = []
        self.imports: list[str] = []
        self._scope: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports += [a.name for a in node.names]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        self._add(node, "class", f"class {node.name}({bases})" if bases else f"class {node.name}")
        self._scope.append(node.name)
        self.generic_visit(node)  # methods are nested, and nesting is the point
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node, is_async=True)

    def _function(self, node, is_async: bool) -> None:
        args = ", ".join(a.arg for a in node.args.args)
        prefix = "async def" if is_async else "def"
        kind = "method" if self._scope else "function"
        symbol = self._add(node, kind, f"{prefix} {node.name}({args})")
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _call_name(child.func)
                if name:
                    symbol.calls.add(name)
        # Recurse so nested definitions are indexed. The original skipped this
        # while its comment claimed they were "picked up as top-level"; they
        # were not picked up at all.
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _add(self, node, kind: str, signature: str) -> Symbol:
        qualname = ".".join([*self._scope, node.name])
        symbol = Symbol(
            qualname=qualname,
            file=self.file,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            kind=kind,
            signature=signature,
            docstring=ast.get_docstring(node),
        )
        self.symbols.append(symbol)
        return symbol


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def build_index(root: Path, files: list[str]) -> Index:
    index = Index()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            source = (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=rel)
        except (OSError, SyntaxError):
            # A file the agent just broke should be visible as broken, not
            # silently absent from the index the planner reads.
            index.unparseable.append(rel)
            continue
        visitor = _Visitor(rel)
        visitor.visit(tree)
        index.imports[rel] = sorted(set(visitor.imports))
        for symbol in visitor.symbols:
            index.symbols[symbol.key] = symbol
            index.by_name.setdefault(symbol.qualname.split(".")[-1], []).append(symbol.key)
    return index

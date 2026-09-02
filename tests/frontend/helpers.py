"""Fixture builders for the new-system (frontend) tests.

Every phase function runs on synthetic input with no executor and no live net
(COMPILER_REFACTOR.md §7); these helpers keep that one-liner shape.
"""

import ast
import inspect
import itertools
import linecache
import textwrap
from typing import Any

from natsune.frontend.diagnostics import DiagnosticSink, Position, SourceMap
from natsune.frontend.ir import IrFunction, build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import Signature, analyze_signature
from natsune.frontend.source import FunctionSource, extract_source
from natsune.frontend.symbols import SymbolTable, collect_symbols

_snippet_counter = itertools.count(1)


def parse_function(snippet: str) -> ast.FunctionDef:
    """Parse a (possibly indented) snippet and return its first FunctionDef."""
    module = ast.parse(textwrap.dedent(snippet))
    func = module.body[0]
    assert isinstance(func, ast.FunctionDef)
    return func


def source_map_for(
    func: Any, filename: str = "prog.py"
) -> tuple[SourceMap, ast.FunctionDef]:
    """Build a `SourceMap` for a real function via the phase-1 extractor."""
    source = extract_source(func, globals=func.__globals__, filename=filename)
    return source.source_map, source.func_def


def make_source(
    snippet: str,
    *,
    namespace: dict[str, Any] | None = None,
    name: str | None = None,
    filename: str | None = None,
) -> FunctionSource:
    """Extract a `FunctionSource` from a snippet, no real file needed.

    Executes the (dedented) snippet and registers the text with `linecache`
    under a unique filename (unless `filename=` pins one, e.g. for snapshot
    stability), so `inspect.getsourcelines` inside `extract_source` works
    naturally (CPython 3.14's `getsourcefile` accepts linecache-only
    filenames). The snippet's `__globals__` is the returned
    `FunctionSource.globals`, seeded with `namespace` — pass annotation
    targets there (e.g. `{"Par": Par}`).

    With several function definitions in the snippet, pass `name=`;
    otherwise the snippet must define exactly one. Leading newlines are
    stripped so the definition sits at snippet line 1 (`base_lineno == 1`).
    """
    text = textwrap.dedent(snippet).lstrip("\n")
    if filename is None:
        filename = f"snippet_{next(_snippet_counter)}.py"

    ns: dict[str, Any] = dict(namespace) if namespace else {}
    before = set(ns)
    exec(compile(text, filename, "exec"), ns)
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )

    if name is not None:
        func = ns[name]
        assert callable(func), f"{name!r} is not callable"
    else:
        defined = [
            value
            for key, value in ns.items()
            if key not in before and inspect.isfunction(value)
        ]
        if len(defined) != 1:
            raise ValueError(
                "snippet must define exactly one function"
                f" (got {len(defined)}); pass name= to disambiguate"
            )
        func = defined[0]

    return extract_source(func, globals=ns, filename=filename)


def analyze_for(
    snippet: str, **kwargs
) -> tuple[FunctionSource, Signature, SymbolTable]:
    """Run phases 1–4: source, signature, and symbol table for a snippet."""
    source = make_source(snippet, **kwargs)
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())
    return source, signature, symbols


def build_ir_for(snippet: str, **kwargs) -> tuple[IrFunction, DiagnosticSink]:
    """Run the full new-system pipeline (§9): source → signature → symbols →
    links → IR. Returns the IR and the builder's sink (whose dedup absorbs
    the collector's overlapping findings)."""
    source, signature, symbols = analyze_for(snippet, **kwargs)
    links = collect_call_links(source.func_def.body, source.globals)
    sink = DiagnosticSink()
    ir = build_ir(source, signature, symbols, links, sink)
    return ir, sink


def make_position(lineno: int, col_offset: int) -> Position:
    return Position(lineno=lineno, col_offset=col_offset)

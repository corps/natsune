"""Fixture builders for the new-system (frontend) tests.

Every phase function runs on synthetic input with no executor and no live net
(COMPILER_REFACTOR.md §7); these helpers keep that one-liner shape. The module
grows as phases land (`make_source`, `make_symbols`, `build_ir_for`, ...).
"""

import ast
import inspect
import textwrap

from natsune.frontend.diagnostics import Position, SourceMap


def parse_function(snippet: str) -> ast.FunctionDef:
    """Parse a (possibly indented) snippet and return its first FunctionDef."""
    module = ast.parse(textwrap.dedent(snippet))
    func = module.body[0]
    assert isinstance(func, ast.FunctionDef)
    return func


def source_map_for(
    func, filename: str = "prog.py"
) -> tuple[SourceMap, ast.FunctionDef]:
    """Build a `SourceMap` for a real function, mirroring the old `func_def`.

    Reproduces `InetFunctionCompiler.func_def`: `inspect.getsourcelines` for
    the dedented snippet and base lineno, `ast.parse` of the snippet, and the
    snippet-root `FunctionDef` as the fallback position.
    """
    lines, base_lineno = inspect.getsourcelines(func)
    func_def = parse_function("".join(lines))
    source = SourceMap(
        filename=filename,
        base_lineno=base_lineno,
        fallback=Position(func_def.lineno, func_def.col_offset),
    )
    return source, func_def

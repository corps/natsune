"""Phase 0 tests — diagnostics.

These pin the old `InetFunctionCompiler.syntax_error` behavior (position
arithmetic and `SyntaxError` shape) exactly as it is today, per
COMPILER_REFACTOR.md §11 ("base-lineno offset arithmetic appears correct but
untested; pin current behavior in Phase 0 tests, then decide if it's right").
"""

import ast
import inspect

import pytest

from natsune.frontend.diagnostics import (
    CompileDiagnostic,
    DiagnosticSink,
    Position,
    Severity,
    SourceMap,
)
from tests.frontend.helpers import parse_function, source_map_for


def _module_level(a: int) -> int:
    return a


class _Container:
    def method(self, a: int) -> int:
        return a


def _located_stmt() -> ast.stmt:
    """A statement with real location info, as every `ast.parse` node has."""
    return ast.parse("x = 1").body[0]


# --- SourceMap.resolve: the pinned old arithmetic ---------------------------


def test_resolve_applies_base_lineno_offset():
    source = SourceMap("prog.py", base_lineno=10)
    func_def = parse_function("""
        def f():
            return 1
        """)
    ret = func_def.body[0]

    # The snippet's leading blank line keeps 1-based numbering: the return
    # sits at lineno 3, col 4.
    assert ret.lineno == 3
    assert source.resolve(ret) == Position(ret.lineno + 10 - 1, 4)


def test_resolve_returns_none_for_nodes_without_location():
    # `ast.Load` carries no lineno/col_offset, and SourceMap deliberately has
    # no fallback: resolving is the caller's problem.
    source = SourceMap("prog.py", base_lineno=10)
    assert source.resolve(ast.Load()) is None


def test_resolve_passes_column_through_untouched():
    # Column offsets get no base adjustment (pinning old behavior).
    source = SourceMap("prog.py", base_lineno=5)
    func_def = parse_function("""
        def f():
            return 1
        """)
    ret = func_def.body[0]
    assert ret.col_offset == 4
    assert source.resolve(ret) == Position(ret.lineno + 5 - 1, 4)


# --- Base-lineno offsets for real, non-top-of-module definitions ------------


def test_base_lineno_for_function_below_top_of_module():
    source, func_def = source_map_for(_module_level)

    assert source.base_lineno == _module_level.__code__.co_firstlineno
    assert source.base_lineno > 1
    assert func_def.lineno == 1  # extracted snippet is numbered from 1

    ret = func_def.body[0]
    resolved = source.resolve(ret)
    assert resolved is not None
    assert resolved.lineno == source.base_lineno + 1


def test_base_lineno_for_indented_function():
    source, func_def = source_map_for(_Container.method)

    # `inspect.getsourcelines` dedents: the snippet's def sits at column 0,
    # but base_lineno still points at the real file line.
    assert source.base_lineno == _Container.method.__code__.co_firstlineno
    assert func_def.col_offset == 0

    ret = func_def.body[0]
    resolved = source.resolve(ret)
    assert resolved is not None
    assert resolved.lineno == source.base_lineno + 1


def test_locationless_nodes_return_none_and_callers_own_fallback():
    source, func_def = source_map_for(_module_level)
    assert source.resolve(ast.Load()) is None
    # The callsite owns the fallback — here, the FunctionDef context that
    # `source_map_for` happens to have on hand.
    position = source.resolve(ast.Load()) or source.resolve(func_def)
    assert position == Position(source.base_lineno, func_def.col_offset)


# --- CompileDiagnostic ------------------------------------------------------


def test_str_formatting():
    error = CompileDiagnostic("boom", "prog.py", 12, 4)
    assert str(error) == "prog.py:12:4: error: boom"
    warning = CompileDiagnostic("hmm", "prog.py", 12, 4, Severity.WARNING)
    assert str(warning) == "prog.py:12:4: warning: hmm"


def test_at_resolves_node_position():
    source = SourceMap("prog.py", base_lineno=10)
    node = parse_function("""
        def f():
            return 1
        """).body[0]

    diagnostic = CompileDiagnostic.at("boom", node, source)
    assert diagnostic == CompileDiagnostic("boom", "prog.py", 12, 4, Severity.ERROR)


def test_at_defaults_to_error_severity():
    source = SourceMap("prog.py", 1)
    diagnostic = CompileDiagnostic.at("x", _located_stmt(), source)
    assert diagnostic is not None
    assert diagnostic.severity is Severity.ERROR


def test_at_returns_none_for_locationless_nodes():
    source = SourceMap("prog.py", 1)
    assert CompileDiagnostic.at("x", ast.Load(), source) is None


def test_at_position_uses_caller_resolved_position():
    source = SourceMap("prog.py", base_lineno=10)
    diagnostic = CompileDiagnostic.at_position("where", source, Position(3, 4))
    assert diagnostic == CompileDiagnostic("where", "prog.py", 3, 4, Severity.ERROR)


def test_raised_matches_old_syntax_error_shape():
    source, func_def = source_map_for(_module_level)
    target = func_def.body[0]
    diagnostic = CompileDiagnostic.at("boom", target, source)
    assert diagnostic is not None

    # The old inline expression, verbatim from `InetFunctionCompiler.syntax_error`.
    old = SyntaxError(
        "boom",
        (
            source.filename,
            target.lineno + source.base_lineno - 1,
            target.col_offset,
            None,
        ),
    )

    exc = diagnostic.raised()
    assert type(exc) is SyntaxError
    assert exc.msg == old.msg
    assert exc.filename == old.filename
    assert exc.lineno == old.lineno
    assert exc.offset == old.offset
    assert exc.text is None and old.text is None


# --- DiagnosticSink ---------------------------------------------------------


def test_sink_accumulates_in_order():
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()
    module = ast.parse("x = 1\ny = 2")

    first = sink.error("first", module.body[0], source)
    second = sink.warning("second", module.body[1], source)

    assert first is not None and second is not None
    assert list(sink.diagnostics) == [first, second]
    assert list(sink.errors) == [first]
    assert sink.has_errors


def test_error_records_nothing_for_locationless_nodes():
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()

    assert sink.error("lost", ast.Load(), source) is None
    assert len(sink.diagnostics) == 0

    # ...whereas a caller-resolved position can never be lost:
    kept = sink.add_at("kept", source, Position(1, 0))
    assert list(sink.diagnostics) == [kept]


def test_add_at_records_at_explicit_position():
    source = SourceMap("prog.py", base_lineno=10)
    sink = DiagnosticSink()

    diagnostic = sink.add_at("explicit", source, Position(12, 4))
    assert diagnostic == CompileDiagnostic("explicit", "prog.py", 12, 4, Severity.ERROR)
    assert list(sink.diagnostics) == [diagnostic]


def test_raise_if_errors_ignores_warnings():
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()
    sink.warning("only a warning", _located_stmt(), source)

    sink.raise_if_errors()  # does not raise


def test_raise_if_errors_raises_first_error():
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()
    sink.warning("a warning", _located_stmt(), source)
    first = sink.error("first error", _located_stmt(), source)
    sink.error("second error", _located_stmt(), source)
    assert first is not None

    with pytest.raises(SyntaxError) as excinfo:
        sink.raise_if_errors()
    assert excinfo.value.msg == "first error"
    assert excinfo.value.filename == source.filename
    assert excinfo.value.lineno == first.lineno


def test_fail_raises_immediately_and_records_nothing():
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()

    with pytest.raises(SyntaxError, match="unreachable"):
        sink.fail("unreachable state", _located_stmt(), source)
    assert len(sink.diagnostics) == 0


def test_fail_uses_positionless_legacy_shape_without_location():
    # Bare `raise SyntaxError(msg)` — the old compiler's other shape (e.g. the
    # unsupported-args raise) — when even the defense path has no position.
    source = SourceMap("prog.py", 1)
    sink = DiagnosticSink()

    with pytest.raises(SyntaxError, match="unreachable") as excinfo:
        sink.fail("unreachable state", ast.Load(), source)
    assert excinfo.value.msg == "unreachable state"
    assert excinfo.value.filename is None
    assert excinfo.value.lineno is None


def test_inspect_sanity_for_helpers():
    # Guards the fixtures themselves: co_firstlineno and getsourcelines agree.
    lines, base = inspect.getsourcelines(_module_level)
    assert base == _module_level.__code__.co_firstlineno
    assert lines[0].startswith("def _module_level")

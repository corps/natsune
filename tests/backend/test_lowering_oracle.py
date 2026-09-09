"""The golden-net oracle (§6): the new lowering's recorded nets must be
graph-for-graph identical to the legacy compiler's for the supported
straight-line subset. Equality is exact — emission order included — since
the lowering mirrors parse_statement_body statement-for-statement."""

import ast
import linecache
import textwrap

import pytest

from natsune.backend import PythonBackend
from natsune.backend.lowering import lower_function
from natsune.backend.types import NetTemplate
from natsune.compiler import InetFunctionCompiler
from natsune.connector import serialize_active_pairs
from tests.frontend.helpers import build_ir_for

_CASES = [
    # dynamic return expression
    "def f(a: int, b: int) -> int:\n    return a + b",
    # assign dynamic to a local, read it back
    "def g(a: int) -> int:\n    x = a + 1\n    return x * 2",
    # augassign: fresh target, repeated rebinding
    "def h(a: int) -> int:\n    total = 0\n    total += a\n    total += 5\n    return total",
    # constants stay in the source; only the var is captured
    "def k(a: int) -> int:\n    return -a",
    # NOTE: `return -1` is deliberately NOT here — the IR constant-folds it
    # to IrConst where legacy emitted a dynamic; see the divergence test below.
    # expression statement evaluated for effect, then discarded
    "def p(a: int) -> None:\n    print(a)\n    print(a + 1)",
    # implicit None return when the body falls through
    "def q(a: int) -> None:\n    print(a)",
    # bare return
    "def r(a: int) -> None:\n    print(a)\n    return",
    # chained assignment (legacy target-to-target chain)
    "def s(a: int) -> int:\n    b = a + 1\n    c = b + 1\n    return c",
    # augassign with nested dynamic value: legacy unparse parenthesizes the
    # right operand — text comes from the synthesized node, not an f-string
    "def w(a: int) -> int:\n    x = 0\n    x += a + 1\n    return x",
    # multiple statements, interleaved reads and writes
    "def t(a: int, b: int) -> int:\n    x = a + b\n    y = x + a\n    return y + b",
]


def _legacy_flow(source: str):
    filename = "case.py"
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)
    # The legacy compiler reads the function back through
    # inspect.getsourcelines; register the exec'd source (cf. helpers.make_source).
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    func = next(v for v in ns.values() if callable(v))
    compiler = InetFunctionCompiler(func, {}, filename)
    compiler.compile()
    return compiler.compiled


def _new_serialization(source: str) -> list[str]:
    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    unit = lower_function(ir, PythonBackend())
    template = unit.body_def.impl
    assert isinstance(template, NetTemplate)
    return serialize_active_pairs(list(template.pairs), {})


@pytest.mark.parametrize("source", _CASES)
def test_golden_net(source: str) -> None:
    old = serialize_active_pairs(list(_legacy_flow(source).active_pairs), {})
    new = _new_serialization(source)
    assert new == old


def test_folded_unary_const_is_a_deliberate_divergence():
    # First recorded golden-net divergence. The IR folds `-1` to IrConst
    # (mirroring CPython's own optimizer), so the new lowering wires a bare
    # constant; legacy emitted a dynamic eval net for the same source. The
    # IR is the spec — old behavior is reference, not law (§6).
    source = "def m(a: int) -> int:\n    return -1"
    old = serialize_active_pairs(list(_legacy_flow(source).active_pairs), {})
    new = _new_serialization(source)
    assert old and "-1 = " in old[0]
    assert new == []


def test_augassign_synthesizes_binop_node():
    # node is always a real ast.expr: augassign dynamics carry a synthesized
    # BinOp — the same construction as old compiler.py:833.
    captured: list[ast.expr] = []

    class SpyBackend(PythonBackend):
        def materialize_dynamic(self, node, source_text, captures, adapter, connector):
            captured.append(node)
            return super().materialize_dynamic(
                node, source_text, captures, adapter, connector
            )

    ir, sink = build_ir_for("def u(a: int) -> int:\n    x = 0\n    x += a + 1\n    return x")
    assert not sink.diagnostics
    lower_function(ir, SpyBackend())

    assert len(captured) == 1
    node = captured[0]
    assert isinstance(node, ast.BinOp)
    assert isinstance(node.left, ast.Name) and node.left.id == "x"
    assert isinstance(node.right, ast.BinOp)  # the value's original ast_node
    assert ast.unparse(node) == "x + (a + 1)"  # legacy unparse parenthesizes

"""The golden-net oracle (§6): the new lowering's recorded nets must be
graph-for-graph identical to the legacy compiler's for the supported
straight-line subset. Equality is exact — emission order included — since
the lowering mirrors parse_statement_body statement-for-statement."""

import ast
import dataclasses
import inspect
import linecache
import textwrap

import pytest

from natsune.adapters import VA
from natsune.backend import PythonBackend
from natsune.backend.lowering import _FunctionLowering, lower_function
from natsune.backend.types import NetTemplate
from natsune.compiler import InetFunctionCompiler
from natsune.connector import serialize_active_pairs
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.ports import Graft
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


def _untag(pairs) -> list[tuple]:
    """Strip declaration-layer agent tags so call grafts serialize exactly
    like legacy's untagged ``graft`` — the tag is registry identity, not
    net structure (§5.1 step 3), and the oracle compares structure."""

    def scrub(p):
        if isinstance(p, Graft) and p.agent is not None:
            return dataclasses.replace(p, agent=None)
        return p

    return [tuple(scrub(p) for p in pair) for pair in pairs]


_CALL_CASES = [
    # single call, plain arg. NOTE on naming: legacy FlowVariableMap.update
    # (control_flow.py:483) requires every callee variable name to already
    # exist in the caller's map — a legacy constraint that only holds when
    # names coincide (the legacy suite's call sites all do). The new
    # lowering skips that side effect by design (§7.2a), so it doesn't
    # inherit the constraint; these cases name variables to keep the
    # legacy side compilable, which the oracle needs for comparison.
    (
        "f",
        "def f(a: int) -> int:\n"
        "    b = inc(a)\n"
        "    return b\n"
        "\n"
        "def inc(a: int) -> int:\n"
        "    return a + 1",
    ),
    # call result into a local; two call sites share one declaration
    (
        "f",
        "def f(a: int, b: int) -> int:\n"
        "    c = add(a, b)\n"
        "    return add(c, a)\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b",
    ),
    # callee with its own local (exercises the extras-closing branch of
    # the register sort on both sides)
    (
        "f",
        "def f(a: int) -> int:\n"
        "    b = step(a)\n"
        "    return b\n"
        "\n"
        "def step(a: int) -> int:\n"
        "    b = a + 1\n"
        "    return b * 2",
    ),
]


def _call_case_serialization(caller: str, source: str) -> tuple[list[str], list[str]]:
    """Compile a multi-function snippet both ways. The callee is legacy-
    compiled and __inet__-attached first (exactly what @inet does), so the
    caller links against a legacy callee — the prototype's only linkable
    kind (new-lowered callees linking to each other are cutover territory,
    §7.4)."""
    filename = "call_case.py"
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )

    callee_name = next(
        n for n, v in ns.items() if n != caller and inspect.isfunction(v)
    )
    callee_compiler = InetFunctionCompiler(ns[callee_name], ns, filename)
    callee_compiler.compile()
    setattr(ns[callee_name], "__inet__", callee_compiler)

    legacy = InetFunctionCompiler(ns[caller], ns, filename)
    legacy.compile()
    old = serialize_active_pairs(list(legacy.compiled.active_pairs), {})

    src = extract_source(ns[caller], globals=ns, filename=filename)
    signature = analyze_signature(src, DiagnosticSink())
    symbols = collect_symbols(src, signature, DiagnosticSink())
    links = collect_call_links(src.func_def.body, ns)
    sink = DiagnosticSink()
    ir = build_ir(src, signature, symbols, links, sink)
    assert not sink.diagnostics
    unit = lower_function(ir, PythonBackend())
    template = unit.body_def.impl
    assert isinstance(template, NetTemplate)
    return old, serialize_active_pairs(_untag(template.pairs), {})


@pytest.mark.parametrize("caller,source", _CALL_CASES)
def test_golden_net_calls(caller: str, source: str) -> None:
    old, new = _call_case_serialization(caller, source)
    assert new == old


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


_COLLECTION_CASES = [
    # plain locals in statement order
    "def f(a: int) -> int:\n    x = a\n    y = x\n    return y",
    # locals declared inside branches only: the collection walk must
    # descend into nested bodies (legacy's visitor always did)
    "def g(a: int) -> int:\n    if a > 0:\n        y = a + 1\n    else:\n        y = a - 1\n    return y",
    # tuple targets: element names join the bundle, left-to-right (the
    # assignment itself lowers with the composite prototype — only the
    # collection pass is under test here)
    "def h(a: int) -> int:\n    x, y = a, a\n    return a",
    "def k(a: int) -> int:\n    (x, (y, z)) = a, (a, a)\n    return a",
    # fresh augassign target: legacy marks it VA unconditionally
    "def m(a: int) -> int:\n    total = 0\n    total += a\n    return total",
    # attribute/subscript lvalues are dynamics: they declare nothing
    "def p(a) -> None:\n    a.b = 1\n    a[0] = 2",
]


@pytest.mark.parametrize("source", _COLLECTION_CASES)
def test_variable_collection_matches_legacy(source: str) -> None:
    """The bundle is the interface: names, adapters, AND order must equal
    the legacy collection pass (interface position is order-bearing, §6).
    The new walk is recursive over targets and nested bodies, so it collects
    exactly the names legacy's recursive visitor does."""
    filename = "collection.py"
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    func = next(v for v in ns.values() if callable(v))
    legacy = InetFunctionCompiler(func, {}, filename)
    legacy.compile()

    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    lowering = _FunctionLowering(ir, PythonBackend())
    assert list(lowering._collect_variables()) == list(legacy.variables)

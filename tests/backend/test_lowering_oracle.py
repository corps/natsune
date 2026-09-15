"""The oracle (§6), re-formed for the restructured lowering: the legacy
compiler and the new lowering must agree with the source's own Python
semantics on the supported straight-line subset.

Exact net-shape equality against legacy is retired BY DESIGN: the new
lowering sequences one VariablesFlow per statement list and catalogs
agents after lowering (change 1 of the restructure), so recorded nets
intentionally differ from legacy's single-flow shape. Behavioral
equivalence with legacy — and through it, with the source — is the
contract the oracle now checks."""

import ast
import inspect
import linecache
import textwrap

import pytest

from natsune.backend.python_backend import PythonBackend
from natsune.backend.lowering import _FunctionLowering, lower_function
from natsune.compiler import InetFunctionCompiler
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.ports import ExtMergeFuncPort
from tests.frontend.helpers import build_ir_for

_CASES = [
    # (source, args) — the expected value comes from running the source's
    # own Python function; legacy and the new lowering must both match it.
    # dynamic return expression
    ("def f(a: int, b: int) -> int:\n    return a + b", (3, 4)),
    # assign dynamic to a local, read it back
    ("def g(a: int) -> int:\n    x = a + 1\n    return x * 2", (5,)),
    # augassign: fresh target, repeated rebinding
    (
        "def h(a: int) -> int:\n    total = 0\n    total += a\n    total += 5\n    return total",
        (7,),
    ),
    # constants stay in the source; only the var is captured
    ("def k(a: int) -> int:\n    return -a", (9,)),
    # NOTE: `return -1` is deliberately NOT here — the IR constant-folds it
    # to IrConst where legacy emitted a dynamic; see the divergence test below.
    # expression statement evaluated for effect, then discarded
    ("def p(a: int) -> None:\n    print(a)\n    print(a + 1)", (2,)),
    # implicit None return when the body falls through
    ("def q(a: int) -> None:\n    print(a)", (2,)),
    # bare return
    ("def r(a: int) -> None:\n    print(a)\n    return", (2,)),
    # chained assignment (legacy target-to-target chain)
    ("def s(a: int) -> int:\n    b = a + 1\n    c = b + 1\n    return c", (1,)),
    # augassign with nested dynamic value: legacy unparse parenthesizes the
    # right operand — text comes from the synthesized node, not an f-string
    ("def w(a: int) -> int:\n    x = 0\n    x += a + 1\n    return x", (4,)),
    # multiple statements, interleaved reads and writes
    (
        "def t(a: int, b: int) -> int:\n    x = a + b\n    y = x + a\n    return y + b",
        (2, 3),
    ),
]


def _exec_source(source: str, filename: str):
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    # The legacy compiler reads the function back through
    # inspect.getsourcelines; register the exec'd source (cf. helpers.make_source).
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    return ns


def _python_fn(source: str):
    ns = _exec_source(source, "case.py")
    return next(v for v in ns.values() if callable(v))


def _legacy_flow(source: str):
    func = _python_fn(source)
    compiler = InetFunctionCompiler(func, {}, "case.py")
    compiler.compile()
    return compiler.compiled


def _new_fn(source: str, name: str | None = None):
    """The new lowering, through the backend: lower_function returns the
    finished artifact (a callable for the Python backend). Multi-function
    snippets need ``name`` to select the entry function."""
    ir, sink = build_ir_for(source, name=name)
    assert not sink.diagnostics
    return lower_function(ir, PythonBackend())


class _CapturingBackend(PythonBackend):
    """finish-capturing backend: the LoweredUnit is internal to
    lower_function, so introspection tests recover it here — conforming
    to the protocol, no src hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.artifact = None

    def finish(self, artifact):
        self.artifact = artifact
        return artifact


def _capturing_lower(source: str):
    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    backend = _CapturingBackend()
    lower_function(ir, backend)
    assert backend.artifact is not None
    return backend.artifact


@pytest.mark.parametrize("source,args", _CASES)
def test_matches_legacy_execution(source: str, args: tuple) -> None:
    expected = _python_fn(source)(*args)
    assert _run_legacy(_legacy_flow(source), *args) == expected
    assert _new_fn(source)(*args) == expected


def _run_legacy(expansion, *args):
    """Drive a legacy-compiled flow with concrete args through a
    deterministic executor (the inet decorator's runtime)."""
    from natsune.control_flow_generated import FlowControlInto, FlowInputInto
    from natsune.executor import DeterministicSerialExecutor
    from natsune.invocations import expansion_invocation, filter_invocation
    from natsune.registers import as_constant_register, send_value

    import threading

    exec = DeterministicSerialExecutor()

    with expansion_invocation(
        expansion, exec, FlowInputInto, FlowControlInto
    ) as invocation:
        variable_inputs = invocation.port.variables.readin().split()
        variable_inputs[0].close()
        for extra in variable_inputs[len(args) + 1 :]:
            extra.close()
        for register, arg in zip(variable_inputs[1 : len(args) + 1], args, strict=True):
            send_value(as_constant_register(arg, exec), register)

        outputs: list = []
        end_event = threading.Event()

        def output_callback(x):
            outputs.append(x)
            end_event.set()

        to_register, from_register = filter_invocation(output_callback, exec)
        from_register.close()
        send_value(invocation.wire.return_value.readout(), to_register)

    exec.run(end_event)
    if not outputs:
        raise ValueError("No output produced by the function")
    return outputs[0]


_CALL_CASES = [
    # (caller, source, args). NOTE on naming: legacy FlowVariableMap.update
    # requires every callee variable name to already exist in the caller's
    # map — a legacy constraint that only holds when names coincide (the
    # legacy suite's call sites all do). These cases name variables to keep
    # the legacy side compilable, which the comparison needs.
    (
        "f",
        "def f(a: int) -> int:\n"
        "    b = inc(a)\n"
        "    return b\n"
        "\n"
        "def inc(a: int) -> int:\n"
        "    return a + 1",
        (2,),
    ),
    # call result into a local; two call sites share one callee
    (
        "f",
        "def f(a: int, b: int) -> int:\n"
        "    c = add(a, b)\n"
        "    return add(c, a)\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b",
        (2, 3),
    ),
    # callee with its own local
    (
        "f",
        "def f(a: int) -> int:\n"
        "    b = step(a)\n"
        "    return b\n"
        "\n"
        "def step(a: int) -> int:\n"
        "    b = a + 1\n"
        "    return b * 2",
        (3,),
    ),
]


def _legacy_caller_callable(caller: str, source: str):
    """Compile a multi-function snippet the legacy way: the callee is
    legacy-compiled and __inet__-attached first (exactly what @inet does),
    so the caller links against a legacy callee."""
    filename = "call_case.py"
    ns = _exec_source(source, filename)
    callee_name = next(
        n for n, v in ns.items() if n != caller and inspect.isfunction(v)
    )
    callee_compiler = InetFunctionCompiler(ns[callee_name], ns, filename)
    callee_compiler.compile()
    setattr(ns[callee_name], "__inet__", callee_compiler)

    legacy = InetFunctionCompiler(ns[caller], ns, filename)
    legacy.compile()
    return legacy.compiled


def _linked_ir(source: str, caller: str, filename: str = "call_case.py"):
    """Build the IR for a multi-function snippet with the callee LINKED:
    the callee is legacy-compiled and __inet__-attached first (exactly
    what @inet does), so collect_call_links resolves the call to
    IrCallInet instead of the eval fallback. build_ir_for execs a fresh
    namespace and cannot see the attachment, so the pipeline runs
    manually here."""
    ns = _exec_source(source, filename)
    callee_name = next(
        n for n, v in ns.items() if n != caller and inspect.isfunction(v)
    )
    callee_compiler = InetFunctionCompiler(ns[callee_name], ns, filename)
    callee_compiler.compile()
    setattr(ns[callee_name], "__inet__", callee_compiler)

    src = extract_source(ns[caller], globals=ns, filename=filename)
    signature = analyze_signature(src, DiagnosticSink())
    symbols = collect_symbols(src, signature, DiagnosticSink())
    links = collect_call_links(src.func_def.body, ns)
    sink = DiagnosticSink()
    ir = build_ir(src, signature, symbols, links, sink)
    assert not sink.diagnostics
    return ir


@pytest.mark.parametrize("caller,source,args", _CALL_CASES)
def test_call_cases_match_legacy_execution(
    caller: str, source: str, args: tuple
) -> None:
    expected = _exec_source(source, "call_case.py")[caller](*args)
    assert _run_legacy(_legacy_caller_callable(caller, source), *args) == expected

    ir = _linked_ir(source, caller)
    from natsune.frontend.ir.nodes import IrCallInet

    assert isinstance(ir.body.statements[0].value, IrCallInet), (
        "the call must link to the legacy callee, not fall back to eval"
    )
    assert lower_function(ir, PythonBackend())(*args) == expected


def test_call_case_one() -> None:
    """Same as the first parametrized call case, single-function-entry
    form: the callee must be linked (see _linked_ir) or the call degrades
    to the eval fallback and the runtime NameErrors through the catch."""
    caller, source, args = _CALL_CASES[0]
    fn = lower_function(_linked_ir(source, caller), PythonBackend())
    assert fn(*args) == 3  # f(2) = inc(2) = 3 — the parametrized twin above derives this


def test_folded_unary_const_lowers_without_eval_machinery():
    # Recorded golden-net divergence, re-formed for the restructure. The
    # IR folds `-1` to IrConst (mirroring CPython's own optimizer), so the
    # recorded graph wires a bare constant: no dynamic eval machinery
    # (ExtMergeFuncPort, the eval-expression merge) appears anywhere in
    # the main flow or the post-hoc agent catalog's recorded flows. The
    # IR is the spec — old behavior is reference, not law (§6).
    source = "def m(a: int) -> int:\n    return -1"
    unit = _capturing_lower(source)
    flows = [unit.main] + [a for a in unit.agents if hasattr(a, "active_pairs")]
    for flow in flows:
        for pair in flow.active_pairs:
            for port in pair:
                assert not isinstance(port, ExtMergeFuncPort)
    assert _new_fn(source)(5) == -1


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

    ir, sink = build_ir_for(
        "def u(a: int) -> int:\n    x = 0\n    x += a + 1\n    return x"
    )
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
    ns = _exec_source(source, filename)
    func = next(v for v in ns.values() if callable(v))
    legacy = InetFunctionCompiler(func, {}, filename)
    legacy.compile()

    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    lowering = _FunctionLowering(ir, PythonBackend())
    assert list(lowering.collect_variables()) == list(legacy.variables)

"""§7.2b slice 1: IrIf lowering under the per-branch scheme.

The union is never formed: the parent wires the full variables bundle
into the composite context (every cell extended), each branch body is a
self-contained flow whose fall-through emits every variable's final
state, and the taken branch's bundle feeds the parent's extended cells.
Consequences (§6/§7.2b):
- composite-level pair-equality with legacy is abandoned by design —
  legacy classified its context from the union; we don't classify at all;
- branch BODIES are still plain flows lowered by the same statement
  machinery as the straight-line slice, so they remain pair-comparable
  with the legacy branch flows (asserted here; the legacy reference is
  reconstructed via InetBranchCompiler because the compiled parent net's
  optimize pass inlines branch bodies and dissolves the composite);
- the decisive check is differential execution: the same program run
  through the legacy compiler and through the new lowering must produce
  identical results. The driver below is the §8.1 runtime in miniature
  (graft a flow into an executor, feed args, collect the return) —
  executing from the data-only NetTemplate lands with §8.1.
"""

import ast
import linecache
import textwrap
import threading

import pytest

from natsune.adapters import Variables, adapter_from_type
from natsune.backend import PythonBackend
from natsune.backend.lowering import _FunctionLowering, lower_function
from natsune.backend.types import InetCallable
from natsune.compiler import InetBranchCompiler, InetFunctionCompiler
from natsune.connector import serialize_active_pairs
from natsune.control_flow import (
    FlowControlInto,
    FlowInputInto,
    IfThenElseStatement,
    VariablesFlow,
)
from natsune.executor import DeterministicSerialExecutor
from natsune.invocations import expansion_invocation, filter_invocation
from natsune.registers import as_constant_register, send_value
from tests.frontend.helpers import build_ir_for

IF_ELSE = """
def f(a: int) -> int:
    x = 0
    if a > 0:
        x = a + 1
    else:
        x = a - 1
    return x
"""

IF_NO_ELSE = """
def g(a: int) -> int:
    x = 0
    if a > 0:
        x = a + 1
    return x
"""

NESTED_IF = """
def h(a: int, b: int) -> int:
    x = 0
    if a > 0:
        if b > 0:
            x = a + b
        else:
            x = a - b
    return x
"""

# The variable is declared inside the branches only — the collection pass
# must walk nested bodies (legacy's visitor did), or the branch flow has no
# register for it.
BRANCH_LOCAL = """
def k(a: int) -> int:
    if a > 0:
        y = a + 1
    else:
        y = a - 1
    return y
"""

# Slice 2: every branch returns — the if is the body's closer, the
# composite's return is the only return source, and the implicit-None
# tail must not fire on top of it.
BOTH_RETURN = """
def r(a: int) -> int:
    if a > 0:
        return 1
    else:
        return 2
"""

# Slice 2b: MIXED ifs — the rest of the list is absorbed into the
# fall-through branches, so the trailing return lowers inside the
# dispatch (it is is_it_even's exact shape).
MIXED_TAIL = """
def r2(a: int) -> int:
    if a > 0:
        return 1
    return 2
"""

# The doc's §7.2b headline program, verbatim: dynamic test, bool return.
IS_IT_EVEN = """
def is_it_even(input: int) -> bool:
    if input % 2 == 0:
        return True
    return False
"""

# Mixed with NO trailing return: the implicit-None tail is what gets
# absorbed into the fall-through branch.
MIXED_IMPLICIT_NONE = """
def t(a: int) -> int:
    if a > 0:
        return 1
"""

# Mixed where the fall-through branch keeps computing after the if; the
# trailing region is absorbed past it.
MIXED_THEN_TAIL = """
def s(a: int) -> int:
    x = 10
    if a > 0:
        return 1
    x = x + a
    return x
"""

NESTED_BOTH_RETURN = """
def n(a: int, b: int) -> int:
    if a > 0:
        if b > 0:
            return 1
        else:
            return 2
    else:
        return 3
"""

# Mixed via nesting: the inner if returns through one branch and falls
# through the other; the outer falls through its then-branch into the
# tail return. The absorbed copy of the tail lands inside the outer's
# then-branch, after the inner composite.
MIXED_NESTED = """
def m(a: int, b: int) -> int:
    x = 0
    if a > 0:
        if b > 0:
            return 1
        else:
            return 2
    else:
        x = a - b
    return x
"""


def _exec_source(source: str, filename: str):
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    return ns


def _legacy_flow(source: str):
    ns = _exec_source(source, "if_case.py")
    func = next(v for v in ns.values() if callable(v))
    compiler = InetFunctionCompiler(func, {}, "if_case.py")
    compiler.compile()
    return compiler.compiled


def _legacy_branches(source: str):
    """Reconstruct the legacy branch flows exactly as the If branch does
    (compiler.py:884-885)."""
    ns = _exec_source(source, "if_case.py")
    func = next(v for v in ns.values() if callable(v))
    compiler = InetFunctionCompiler(func, {}, "if_case.py")
    compiler.compile()

    module = ast.parse(textwrap.dedent(source))
    if_stmt = next(n for n in ast.walk(module) if isinstance(n, ast.If))
    branches = []
    for body in (if_stmt.body, if_stmt.orelse):
        flow = VariablesFlow(
            variables=Variables(compiler.variables),
            return_adapter=adapter_from_type(compiler.return_annot),
        )
        InetBranchCompiler(compiler, flow, False).parse_statement_body(body)
        branches.append(flow)
    return branches


def _new_lowering(source: str) -> _FunctionLowering:
    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    lowering = _FunctionLowering(ir, PythonBackend())
    lowering.run(ir.name)
    return lowering


def _our_composite(lowering: _FunctionLowering) -> IfThenElseStatement:
    assert isinstance(lowering.backend, PythonBackend)
    impl = lowering.backend.agents["if_0"].impl
    assert isinstance(impl, InetCallable)
    return impl.ref


def _serialize(flow) -> list[str]:
    return serialize_active_pairs(list(flow.active_pairs), {})


def test_branch_bodies_match_legacy():
    """Branch bodies are plain flows: their recorded nets must be
    graph-for-graph identical to the legacy branch flows."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(IF_ELSE))
    composite = _our_composite(_new_lowering(textwrap.dedent(IF_ELSE)))

    assert _serialize(composite.true_case) == _serialize(legacy_then)
    assert _serialize(composite.false_case) == _serialize(legacy_else)


def test_branch_local_variable_matches_legacy():
    """Same, for a variable declared inside the branches only: the bundle
    comes from the recursive collection walk, so both sides carry y."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(BRANCH_LOCAL))
    composite = _our_composite(_new_lowering(textwrap.dedent(BRANCH_LOCAL)))

    assert _serialize(composite.true_case) == _serialize(legacy_then)
    assert _serialize(composite.false_case) == _serialize(legacy_else)


def test_parent_net_carries_tagged_composite():
    lowering = _new_lowering(textwrap.dedent(IF_ELSE))
    rendered = "\n".join(_serialize(lowering.flow))
    assert "graft:if_0" in rendered


def _run(expansion, *args):
    """Drive a compiled flow with concrete args through a deterministic
    executor (mirrors the inet decorator's runtime). Executing from the
    data-only NetTemplate lands with the §8.1 runtime — the driver needs
    the live flow's interface wires."""
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


@pytest.mark.parametrize(
    "source,args,expected",
    [
        (IF_ELSE, (5,), 6),
        (IF_ELSE, (-5,), -6),
        (IF_ELSE, (0,), -1),  # else taken: a - 1
        (IF_NO_ELSE, (5,), 6),
        (IF_NO_ELSE, (-3,), 0),  # branch skipped: the initial value stands
        (NESTED_IF, (1, 1), 2),
        (NESTED_IF, (1, -1), 2),  # inner else: a - b
        (NESTED_IF, (-1, 5), 0),  # outer branch skipped
        (BRANCH_LOCAL, (5,), 6),
        (BRANCH_LOCAL, (-5,), -6),
        (BOTH_RETURN, (5,), 1),
        (BOTH_RETURN, (-5,), 2),
        (MIXED_TAIL, (5,), 1),
        (MIXED_TAIL, (-5,), 2),
        (MIXED_THEN_TAIL, (5,), 1),
        (MIXED_THEN_TAIL, (-1,), 9),
        (MIXED_NESTED, (1, 1), 1),
        (MIXED_NESTED, (1, -1), 2),
        (MIXED_NESTED, (-1, 5), -6),
        (IS_IT_EVEN, (10,), True),
        (IS_IT_EVEN, (11,), False),
    ],
)


def test_differential_execution(source, args, expected):
    """The same program through the legacy compiler and through the new
    lowering must produce identical results."""
    legacy = _run(_legacy_flow(source), *args)
    lowering = _new_lowering(source)
    ours = _run(lowering.flow, *args)

    assert legacy == expected
    assert ours == expected


def test_mixed_implicit_none_tail():
    """A mixed if with NO trailing return: the implicit-None tail is
    absorbed into the fall-through branch.

    Deliberate oracle divergence (§6: the IR is the spec): legacy produces
    NO output on the fall-through path here — its wire_continuation
    starves when the continuation is the implicit-None tail — so the
    expected values are asserted directly."""
    lowering = _new_lowering(textwrap.dedent(MIXED_IMPLICIT_NONE))
    assert _run(lowering.flow, 5) == 1
    assert _run(lowering.flow, -5) is None


def test_nested_both_return_differential():
    """Slice 2 handles nested all-branches-return ifs on every path.

    Deliberate oracle divergence (the §6 rule: the IR is the spec): legacy
    produces NO output for the paths that return through the inner if —
    its wire_continuation merge starves when a returning branch's own
    composite sits between it and the branch return — so legacy cannot
    serve as the oracle here and the expected values are asserted
    directly."""
    lowering = _new_lowering(textwrap.dedent(NESTED_BOTH_RETURN))
    assert _run(lowering.flow, 1, 1) == 1
    assert _run(lowering.flow, 1, -1) == 2
    assert _run(lowering.flow, -1, 5) == 3


def test_branch_bodies_match_legacy_both_return():
    """Slice 2: returning branch bodies are pair-identical to legacy's
    (both skip the finish tail when the body closed)."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(BOTH_RETURN))
    composite = _our_composite(_new_lowering(textwrap.dedent(BOTH_RETURN)))

    assert _serialize(composite.true_case) == _serialize(legacy_then)
    assert _serialize(composite.false_case) == _serialize(legacy_else)


# break/continue outside loops never reach lowering: the frontend
# rejects them at build time (IrStructureError), so there is no
# lowering-level refusal left to test after slice 2b.

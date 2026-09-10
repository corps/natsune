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


def test_exiting_branches_are_still_out_of_scope():
    source = "def r(a: int) -> int:\n    if a > 0:\n        return 1\n    return 2"
    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    with pytest.raises(NotImplementedError, match="closer machinery"):
        lower_function(ir, PythonBackend())

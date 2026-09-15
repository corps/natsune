"""§7.2b: IrIf lowering under the per-branch scheme, reconciled to the
restructured lowering (statement-per-flow sequencing, ControlBranchFlow,
post-hoc agent catalog).

- Legacy remains the behavioral oracle wherever it works: the same
  program through the legacy compiler and through the new lowering must
  produce identical results (two legacy divergences are pinned below and
  asserted directly against expected values, §6: the IR is the spec).
- Composite introspection recovers the LoweredUnit through a
  finish-capturing backend (the unit is internal to lower_function);
  branch bodies remain plain flows, so they stay pair-comparable with
  the legacy branch flows (reconstructed via InetBranchCompiler, since
  the compiled parent net inlines branch bodies).
- The old tagged-graft serialization test is retired: agent labeling is
  post-hoc now (the catalog is walked from the recorded graph), so there
  are no intermediate graft tags to assert.
"""

import ast
import linecache
import textwrap
import threading

import pytest

from natsune.adapters import Variables, adapter_from_type
from natsune.backend.python_backend import PythonBackend
from natsune.backend.lowering import lower_function
from natsune.compiler import InetBranchCompiler, InetFunctionCompiler
from natsune.connector import Graft, serialize_active_pairs
from natsune.control_flow import IfThenElseStatement, VariablesFlow
from natsune.control_flow_generated import FlowControlInto, FlowInputInto
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

# Slice 2: every branch returns — the if closes the list, the composite's
# return is the only return source, and the implicit-None tail must not
# fire on top of it.
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

# Mixed with NO trailing return: the implicit-None tail is what gets
# absorbed into the fall-through branch.
MIXED_IMPLICIT_NONE = """
def t(a: int) -> int:
    if a > 0:
        return 1
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
# tail return.
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


def _new_fn(source: str):
    """The new lowering, through the backend: lower_function returns the
    finished artifact (a callable for the Python backend)."""
    ir, sink = build_ir_for(source)
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


def _our_composite(unit) -> IfThenElseStatement:
    """The lowered if composite, from the post-hoc agent catalog."""
    return next(a for a in unit.agents if isinstance(a, IfThenElseStatement))


def _branch_children(unit) -> list[VariablesFlow]:
    """The statement flows behind the composite's branches, in then/else
    order. Under the restructured lowering a branch body is a containing
    ControlBranchFlow whose statements lower into a sequenced child flow;
    the child is the statement-for-statement counterpart of legacy's
    branch flow, and the pair-equality oracle lives at that level."""
    composite = _our_composite(unit)
    children = []
    for case in (composite.true_case, composite.false_case):
        for pair in case.active_pairs:
            for port in pair:
                if isinstance(port, Graft) and isinstance(
                    port.execute, VariablesFlow
                ):
                    children.append(port.execute)
    return children


def _serialize(flow) -> list[str]:
    return serialize_active_pairs(list(flow.active_pairs), {})


def _run_legacy(expansion, *args):
    """Drive a legacy-compiled flow with concrete args through a
    deterministic executor (mirrors the inet decorator's runtime)."""
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


def test_branch_bodies_match_legacy():
    """Branch bodies are plain flows: their recorded nets must be
    graph-for-graph identical to the legacy branch flows. Under the
    restructured lowering the comparison lives at the child statement
    flow (the containing ControlBranchFlow's sequencing machinery is
    new by design and carries no legacy counterpart)."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(IF_ELSE))
    children = _branch_children(_capturing_lower(textwrap.dedent(IF_ELSE)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then)
    assert _serialize(children[1]) == _serialize(legacy_else)


def test_branch_local_variable_matches_legacy():
    """Same, for a variable declared inside the branches only: the bundle
    comes from the recursive collection walk, so both sides carry y."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(BRANCH_LOCAL))
    children = _branch_children(_capturing_lower(textwrap.dedent(BRANCH_LOCAL)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then)
    assert _serialize(children[1]) == _serialize(legacy_else)


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
    legacy = _run_legacy(_legacy_flow(source), *args)
    ours = _new_fn(source)(*args)

    assert legacy == expected
    assert ours == expected


def test_nested_both_return_differential():
    """Slice 2 handles nested all-branches-return ifs on every path.

    Deliberate oracle divergence (the §6 rule: the IR is the spec): legacy
    produces NO output for the paths that return through the inner if —
    its wire_continuation merge starves when a returning branch's own
    composite sits between it and the branch return — so legacy cannot
    serve as the oracle here and the expected values are asserted
    directly."""
    ours = _new_fn(textwrap.dedent(NESTED_BOTH_RETURN))
    assert ours(1, 1) == 1
    assert ours(1, -1) == 2
    assert ours(-1, 5) == 3


def test_mixed_implicit_none_tail():
    """A mixed if with NO trailing return: the implicit-None tail is
    absorbed into the fall-through branch.

    Deliberate oracle divergence (§6: the IR is the spec): legacy produces
    NO output on the fall-through path here — its wire_continuation
    starves when the continuation is the implicit-None tail — so the
    expected values are asserted directly."""
    ours = _new_fn(textwrap.dedent(MIXED_IMPLICIT_NONE))
    assert ours(5) == 1
    assert ours(-5) is None


def test_branch_bodies_match_legacy_both_return():
    """Slice 2: closing branches pair-match legacy too — legacy's
    optimizer dissolves a returning branch flow to an EMPTY net, and the
    restructured lowering now dissolves the same way (the non-fall-
    through tail's None-return write optimizes away). Closing branches
    are additionally covered by differential execution."""
    legacy_then, legacy_else = _legacy_branches(textwrap.dedent(BOTH_RETURN))
    children = _branch_children(_capturing_lower(textwrap.dedent(BOTH_RETURN)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then) == []
    assert _serialize(children[1]) == _serialize(legacy_else) == []

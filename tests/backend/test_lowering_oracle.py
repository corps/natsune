"""The oracle (§6): the lowering must agree with the source's own Python
semantics on the supported straight-line subset.

Cases are real module-level functions (see tests/backend/helpers.py):
the expected value comes from calling the case's own function, and the
lowering must match it. Multi-function cases pair a caller with its
callees in a `Program`, which attaches the callees before the caller is
compiled or linked.

Exact net-shape equality against the pre-cutover legacy compiler is
retired BY DESIGN (and the differential harness went with it at the
cutover): the lowering sequences one VariablesFlow per statement list
and catalogs agents after lowering, so recorded nets intentionally
differ from legacy's single-flow shape. Agreement with the source is
the contract the oracle checks."""

import ast

import pytest

from natsune.backend.lowering import _FunctionLowering, lower_function
from natsune.backend.python_backend import PythonBackend
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir.nodes import IrAssign, IrCallInet
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.first_order.ports import ExtMergeFuncPort
from tests.backend.helpers import (
    Program,
    build_ir_for_function,
    capturing_lower,
    program_ids,
)

# --- straight-line execution cases ---------------------------------------


# dynamic return expression
def return_sum(a: int, b: int) -> int:
    return a + b


# assign dynamic to a local, read it back
def local_roundtrip(a: int) -> int:
    x = a + 1
    return x * 2


# augassign: fresh target, repeated rebinding
def augassign_rebind(a: int) -> int:
    total = 0
    total += a
    total += 5
    return total


# constants stay in the source; only the var is captured
def unary_negate(a: int) -> int:
    return -a


# NOTE: `return -1` is deliberately NOT a parametrized case — the IR
# constant-folds it to IrConst where legacy emitted a dynamic; see
# folded_unary_const and its divergence test below.


# expression statement evaluated for effect, then discarded
def print_twice(a: int) -> None:
    print(a)
    print(a + 1)


# implicit None return when the body falls through
def fallthrough_none(a: int) -> None:
    print(a)


# bare return
def bare_return(a: int) -> None:
    print(a)
    return


# chained assignment (legacy target-to-target chain)
def chained_locals(a: int) -> int:
    b = a + 1
    c = b + 1
    return c


# augassign with nested dynamic value: legacy unparse parenthesizes the
# right operand — text comes from the synthesized node, not an f-string
# (also the program behind test_augassign_synthesizes_binop_node)
def augassign_nested_value(a: int) -> int:
    x = 0
    x += a + 1
    return x


# multiple statements, interleaved reads and writes
def interleaved_locals(a: int, b: int) -> int:
    x = a + b
    y = x + a
    return y + b


# the old suite's basic: constant locals, augassign on a constant, and
# effect statements over both dynamics and bare constants
def basic(b: int) -> int:
    a = 10
    a = a + b
    print(a)
    print(123)
    return a


# --- call cases (callee linked through __inet__) --------------------------


def call_inc(a: int) -> int:
    b = inc(a)
    return b


def inc(a: int) -> int:
    return a + 1


# call result into a local; two call sites share one callee
def call_add_twice(a: int, b: int) -> int:
    c = add(a, b)
    return add(c, a)


def add(a: int, b: int) -> int:
    return a + b


# callee with its own local
def call_step(a: int) -> int:
    b = step(a)
    return b


def step(a: int) -> int:
    b = a + 1
    return b * 2


# --- collection cases (variable-bundle interface; never executed) ---------


# plain locals in statement order
def collect_locals(a: int) -> int:
    x = a
    y = x
    return y


# locals declared inside branches only: the collection walk must descend
# into nested bodies (legacy's visitor always did)
def collect_branch_locals(a: int) -> int:
    if a > 0:
        y = a + 1
    else:
        y = a - 1
    return y


# tuple targets: element names join the bundle, left-to-right (the
# assignment itself lowers with the composite prototype — only the
# collection pass is under test here)
def collect_tuple_targets(a: int) -> int:
    x, y = a, a
    return a


def collect_nested_tuple_targets(a: int) -> int:
    x, (y, z) = a, (a, a)
    return a


# fresh augassign target: legacy marks it VA unconditionally
def collect_augassign_target(a: int) -> int:
    total = 0
    total += a
    return total


# attribute/subscript lvalues are dynamics: they declare nothing
def collect_attr_subscript_targets(a) -> None:
    a.b = 1
    a[0] = 2


# --- the case lists --------------------------------------------------------

_CASES = [
    # (program, args) — the expected value comes from calling the case's
    # own Python function; the lowering must match it.
    (Program(return_sum), (3, 4)),
    (Program(local_roundtrip), (5,)),
    (Program(augassign_rebind), (7,)),
    (Program(unary_negate), (9,)),
    (Program(print_twice), (2,)),
    (Program(fallthrough_none), (2,)),
    (Program(bare_return), (2,)),
    (Program(chained_locals), (1,)),
    (Program(augassign_nested_value), (4,)),
    (Program(interleaved_locals), (2, 3)),
    (Program(basic), (29,)),
]

_CALL_PROGRAMS = [
    # (program, args). Cases pair a caller with its callees so the link
    # resolves to the marked callee instead of falling back to eval.
    (Program(call_inc, inc), (2,)),
    (Program(call_add_twice, add), (2, 3)),
    (Program(call_step, step), (3,)),
]

_COLLECTION_CASES = [
    collect_locals,
    collect_branch_locals,
    collect_tuple_targets,
    collect_nested_tuple_targets,
    collect_augassign_target,
    collect_attr_subscript_targets,
]


# --- tests -----------------------------------------------------------------


@pytest.mark.parametrize("program,args", _CASES, ids=program_ids)
def test_execution_matches_source(program: Program, args: tuple) -> None:
    expected = program.call(*args)
    assert program.lower()(*args) == expected


@pytest.mark.parametrize("program,args", _CALL_PROGRAMS, ids=program_ids)
def test_call_cases_execute(program: Program, args: tuple) -> None:
    expected = program.call(*args)

    ir = program.build_ir()
    statement = ir.body.statements[0]
    assert isinstance(statement, IrAssign)
    assert isinstance(
        statement.value, IrCallInet
    ), "the call must link to the marked callee, not fall back to eval"
    assert program.lower()(*args) == expected


def test_call_case_one() -> None:
    """Same as the first parametrized call case, single-function-entry
    form: the callee must be linked (see Program) or the call degrades
    to the eval fallback and the runtime NameErrors through the catch."""
    program = _CALL_PROGRAMS[0][0]
    ir = program.build_ir()
    statement = ir.body.statements[0]
    assert isinstance(statement, IrAssign)
    assert isinstance(statement.value, IrCallInet)
    assert (
        program.lower()(2) == 3
    )  # call_inc(2) = inc(2) = 3 — the parametrized twin above derives this


# `return -1` — the IR folds it to IrConst (mirroring CPython's own
# optimizer), so this is a divergence test, not a parametrized case.
def folded_unary_const(a: int) -> int:
    return -1


def test_folded_unary_const_lowers_without_eval_machinery():
    # Recorded golden-net divergence, re-formed for the restructure. The
    # IR folds `-1` to IrConst (mirroring CPython's own optimizer), so the
    # recorded graph wires a bare constant: no dynamic eval machinery
    # (ExtMergeFuncPort, the eval-expression merge) appears anywhere in
    # the main flow or the post-hoc agent catalog's recorded flows. The
    # IR is the spec — old behavior is reference, not law (§6).
    unit = capturing_lower(Program(folded_unary_const))
    flows = [unit.main] + [a for a in unit.agents if hasattr(a, "active_pairs")]
    for flow in flows:
        for pair in flow.active_pairs:
            for port in pair:
                assert not isinstance(port, ExtMergeFuncPort)
    assert Program(folded_unary_const).lower()(5) == -1


def test_augassign_synthesizes_binop_node():
    # node is always a real ast.expr: augassign dynamics carry a synthesized
    # BinOp — the same construction as old compiler.py:833.
    captured: list[ast.expr] = []

    class SpyBackend(PythonBackend):
        def materialize_dynamic(
            self, node, source_text, captures, function_globals, adapter, connector
        ):
            captured.append(node)
            return super().materialize_dynamic(
                node, source_text, captures, function_globals, adapter, connector
            )

    lower_function(build_ir_for_function(augassign_nested_value), SpyBackend())

    assert len(captured) == 1
    node = captured[0]
    assert isinstance(node, ast.BinOp)
    assert isinstance(node.left, ast.Name) and node.left.id == "x"
    assert isinstance(node.right, ast.BinOp)  # the value's original ast_node
    assert ast.unparse(node) == "x + (a + 1)"  # legacy unparse parenthesizes


@pytest.mark.parametrize("fn", _COLLECTION_CASES)
def test_variable_collection_matches_symbols(fn) -> None:
    """The bundle is the interface: names, adapters, AND order must equal
    the frontend symbols table (interface position is order-bearing, §6;
    this is also the invariant the recursive-callee promise's adapters
    rest on, CUTOVER.md §2). The lowering walk is recursive over targets
    and nested bodies, so it collects exactly the names the symbols walk
    does."""
    program = Program(fn)
    lowering = _FunctionLowering(program.build_ir(), PythonBackend())

    source = extract_source(
        program.entry,
        globals=program.entry.__globals__,
        filename=program.entry.__code__.co_filename,
    )
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())

    assert list(lowering.collect_variables()) == list(symbols.variables)

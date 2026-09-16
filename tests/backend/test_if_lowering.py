"""§7.2b: IrIf lowering under the per-branch scheme, reconciled to the
restructured lowering (statement-per-flow sequencing, ControlBranchFlow,
post-hoc agent catalog).

Cases are real module-level functions (see tests/backend/helpers.py):
the same object drives the plain program, the legacy compiler, and the
new lowering.

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
import inspect

import pytest

from natsune.adapters import Variables, adapter_from_type
from natsune.compiler import InetBranchCompiler
from natsune.connector import Graft, serialize_active_pairs
from natsune.control_flow import IfThenElseStatement, VariablesFlow
from tests.backend.helpers import Program, capturing_lower, program_ids, run_legacy

# --- case programs -------------------------------------------------------
# Each function below is a program under test; the comment above it says
# what the case pins.


def if_else(a: int) -> int:
    x = 0
    if a > 0:
        x = a + 1
    else:
        x = a - 1
    return x


def if_no_else(a: int) -> int:
    x = 0
    if a > 0:
        x = a + 1
    return x


def nested_if(a: int, b: int) -> int:
    x = 0
    if a > 0:
        if b > 0:
            x = a + b
        else:
            x = a - b
    return x


# The variable is declared inside the branches only — the collection pass
# must walk nested bodies (legacy's visitor did), or the branch flow has no
# register for it.
def branch_local(a: int) -> int:
    if a > 0:
        y = a + 1
    else:
        y = a - 1
    return y


# Slice 2: every branch returns — the if closes the list, the composite's
# return is the only return source, and the implicit-None tail must not
# fire on top of it.
def both_return(a: int) -> int:
    if a > 0:
        return 1
    else:
        return 2


# Slice 2b: MIXED ifs — the rest of the list is absorbed into the
# fall-through branches, so the trailing return lowers inside the
# dispatch (it is is_it_even's exact shape).
def mixed_tail(a: int) -> int:
    if a > 0:
        return 1
    return 2


# The doc's §7.2b headline program, verbatim: dynamic test, bool return.
def is_it_even(input: int) -> bool:
    if input % 2 == 0:
        return True
    return False


# Mixed where the fall-through branch keeps computing after the if; the
# trailing region is absorbed past it.
def mixed_then_tail(a: int) -> int:
    x = 10
    if a > 0:
        return 1
    x = x + a
    return x


# Mixed with NO trailing return: the implicit-None tail is what gets
# absorbed into the fall-through branch.
def mixed_implicit_none(a: int) -> int:  # ty:ignore[invalid-return-type]
    if a > 0:
        return 1


def nested_both_return(a: int, b: int) -> int:
    if a > 0:
        if b > 0:
            return 1
        else:
            return 2
    else:
        return 3


# Mixed via nesting: the inner if returns through one branch and falls
# through the other; the outer falls through its then-branch into the
# tail return.
def mixed_nested(a: int, b: int) -> int:
    x = 0
    if a > 0:
        if b > 0:
            return 1
        else:
            return 2
    else:
        x = a - b
    return x


# --- plumbing ------------------------------------------------------------


def _legacy_branches(fn):
    """Reconstruct the legacy branch flows exactly as the If branch does
    (compiler.py:884-885)."""
    compiler = Program(fn).compile_legacy()

    module = ast.parse(inspect.getsource(fn))
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
                if isinstance(port, Graft) and isinstance(port.execute, VariablesFlow):
                    children.append(port.execute)
    return children


def _serialize(flow) -> list[str]:
    return serialize_active_pairs(list(flow.active_pairs), {})


# --- tests ---------------------------------------------------------------


def test_branch_bodies_match_legacy():
    """Branch bodies are plain flows: their recorded nets must be
    graph-for-graph identical to the legacy branch flows. Under the
    restructured lowering the comparison lives at the child statement
    flow (the containing ControlBranchFlow's sequencing machinery is
    new by design and carries no legacy counterpart)."""
    legacy_then, legacy_else = _legacy_branches(if_else)
    children = _branch_children(capturing_lower(Program(if_else)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then)
    assert _serialize(children[1]) == _serialize(legacy_else)


def test_branch_local_variable_matches_legacy():
    """Same, for a variable declared inside the branches only: the bundle
    comes from the recursive collection walk, so both sides carry y."""
    legacy_then, legacy_else = _legacy_branches(branch_local)
    children = _branch_children(capturing_lower(Program(branch_local)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then)
    assert _serialize(children[1]) == _serialize(legacy_else)


@pytest.mark.parametrize(
    ("program", "args", "expected"),
    [
        (Program(if_else), (5,), 6),
        (Program(if_else), (-5,), -6),
        (Program(if_else), (0,), -1),  # else taken: a - 1
        (Program(if_no_else), (5,), 6),
        (Program(if_no_else), (-3,), 0),  # branch skipped: the initial value stands
        (Program(nested_if), (1, 1), 2),
        (Program(nested_if), (1, -1), 2),  # inner else: a - b
        (Program(nested_if), (-1, 5), 0),  # outer branch skipped
        (Program(branch_local), (5,), 6),
        (Program(branch_local), (-5,), -6),
        (Program(both_return), (5,), 1),
        (Program(both_return), (-5,), 2),
        (Program(mixed_tail), (5,), 1),
        (Program(mixed_tail), (-5,), 2),
        (Program(mixed_then_tail), (5,), 1),
        (Program(mixed_then_tail), (-1,), 9),
        (Program(mixed_nested), (1, 1), 1),
        (Program(mixed_nested), (1, -1), 2),
        (Program(mixed_nested), (-1, 5), -6),
        (Program(is_it_even), (10,), True),
        (Program(is_it_even), (11,), False),
    ],
    ids=program_ids,
)
def test_differential_execution(program, args, expected):
    """The same program through the legacy compiler and through the new
    lowering must produce identical results."""
    legacy = run_legacy(program.compile_legacy().compiled, *args)
    ours = program.lower()(*args)

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
    ours = Program(nested_both_return).lower()
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
    ours = Program(mixed_implicit_none).lower()
    assert ours(5) == 1
    assert ours(-5) is None


def test_branch_bodies_match_legacy_both_return():
    """Slice 2: closing branches pair-match legacy too — legacy's
    optimizer dissolves a returning branch flow to an EMPTY net, and the
    restructured lowering now dissolves the same way (the non-fall-
    through tail's None-return write optimizes away). Closing branches
    are additionally covered by differential execution."""
    legacy_then, legacy_else = _legacy_branches(both_return)
    children = _branch_children(capturing_lower(Program(both_return)))

    assert len(children) == 2
    assert _serialize(children[0]) == _serialize(legacy_then) == []
    assert _serialize(children[1]) == _serialize(legacy_else) == []

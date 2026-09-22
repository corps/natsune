"""§7.2b: IrIf lowering under the per-branch scheme (statement-per-flow
sequencing, ControlBranchFlow, post-hoc agent catalog).

Cases are real module-level functions (see tests/backend/helpers.py):
the same object drives the plain program and the lowering.

- Execution is asserted against pinned expected values — historically
  the legacy compiler's output where it agreed with the source, and the
  source's own semantics where it diverged (§6: the IR is the spec). The
  pre-cutover differential harness is gone; the pins remain.
- Composite introspection recovers the LoweredUnit through a
  finish-capturing backend (the unit is internal to lower_function);
  branch statement flows are recovered from the composite for the
  empty-net dissolution pin.
- The old tagged-graft serialization test is retired: agent labeling is
  post-hoc now (the catalog is walked from the recorded graph), so there
  are no intermediate graft tags to assert.
"""

import pytest

from natsune.connector import Graft, serialize_active_pairs
from natsune.control_flow import IfThenElseStatement, VariablesFlow
from natsune.ports import Port, Wire
from tests.backend.helpers import Program, capturing_lower, program_ids

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


def _our_composite(unit) -> IfThenElseStatement:
    """The lowered if composite, from the post-hoc agent catalog."""
    return next(a for a in unit.agents if isinstance(a, IfThenElseStatement))


def _branch_children(unit) -> list[VariablesFlow]:
    """The statement flows behind the composite's branches, in then/else
    order. Under the restructured lowering a branch body is a containing
    ControlBranchFlow whose statements lower into a sequenced child flow;
    the child is the statement-for-statement counterpart of legacy's
    branch flow, and the pair-equality oracle lives at that level.

    The child graft is recovered by a transitive walk over the case's
    wires (the sequencing machinery sits between the case interface and
    the layer invocation, so the graft is not a top-level pair)."""
    composite = _our_composite(unit)
    children = []
    for case in (composite.true_case, composite.false_case):
        targets: list[Port | Wire] = [
            port for pair in case.active_pairs for port in pair
        ]
        child = None
        while targets:
            target = targets.pop()
            if isinstance(target, Graft):
                if isinstance(target.execute, VariablesFlow):
                    child = target.execute
                    break
            if isinstance(target, Port):
                targets.extend(target.wires)
            elif isinstance(target, Wire) and target.target is not None:
                targets.append(target.target)
        assert child is not None, "no statement flow behind the branch"
        children.append(child)
    return children


def _serialize(flow) -> list[str]:
    return serialize_active_pairs(list(flow.active_pairs), {})


# --- tests ---------------------------------------------------------------


def test_closing_branches_dissolve_to_empty_nets():
    """Slice 2: closing (all-branches-return) branches dissolve to EMPTY
    nets — the non-fall-through tail's None-return write optimizes away.
    Covered behaviorally by the execution matrix below; this pins the
    shape."""
    children = _branch_children(capturing_lower(Program(both_return)))

    assert len(children) == 2
    assert _serialize(children[0]) == []
    assert _serialize(children[1]) == []


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
    """Execution against pinned expected values — historically the legacy
    output where it agreed with the source, now simply the spec."""
    ours = program.lower()(*args)

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

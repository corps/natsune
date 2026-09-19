"""§: Par — composite values at the function boundary, consolidated from
the old suite's Par cases (test_compiler.py's other_basic family).

Cases are real module-level functions (see tests/backend/helpers.py).
Par today works where it stays a WHOLE value — a Par-typed parameter
flows through the bundle interface and can be returned as-is — but the
two composite decomposition features are not in scope yet:

- value side: an IrTuple literal (`return b + 2, b * 4`) raises
  NotImplementedError("IrTuple lowering is not in scope") at lowering;
- target side: unpacking a linked call's Par return into tuple targets
  (`a, b = other_basic(10)`) raises NotImplementedError("composite
  targets land with composites") at lowering — the frontend accepts it
  (targets must MATCH the Par size) but the backend has no landing.

Every decomposition case from the old suite is copied below under
pytest.mark.skip with its exact blocker, differential body ready; the
passing boundary-passthrough case stays unskipped to pin the surface
that DOES work.
"""

import pytest

from natsune.special_forms import Inverse, Par
from tests.backend.helpers import Program, program_ids, run_legacy

# --- case programs -------------------------------------------------------
# Each function below is a program under test; the comment above it says
# what the case pins.


# Par as a whole value: param flows in, flows back out untouched
def pair_identity(p: Par[int, int]) -> Par[int, int]:
    return p


# value-side gap: IrTuple return literal
def other_basic(b: int) -> Par[int, int]:
    return b + 2, b * 4


# target-side gap: the linked callee's Par return unpacks into tuple
# targets (old suite: invoke_an_inet() == 12)
def invoke_an_inet() -> int:
    a, b = other_basic(10)
    return a


# value-side gap, Par-of-Inverse flavor: the tuple literal again — this
# is the old suite's delayed-value construction (a = b + 10 snapshots the
# inverse read; a later write to b retro-updates a through the Par)
def delayed_inverse() -> Par[int, Inverse[int]]:
    b: Inverse[int] = -1
    a = b + 10
    return a, b


# target-side gap, Par-of-Inverse flavor: the unpack again, with the
# caller's `b = 30` retro-propagating into `a` (old suite:
# use_delayed_inverse() == 40, NOT the plain-Python 9)
def use_delayed_inverse() -> int:
    a, b = delayed_inverse()
    b = 30

    return a


# --- tests -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("program", "args", "expected"),
    [(Program(pair_identity), ((4, 5),), (4, 5))],
    ids=program_ids,
)
def test_par_passthrough_matches_legacy(program: Program, args: tuple, expected):
    """The supported Par surface: a composite parameter crosses the
    bundle interface and returns whole, identically through legacy and
    the new lowering."""
    assert run_legacy(program.compile_legacy().compiled, *args) == expected
    assert program.lower()(*args) == expected


# --- consolidation skips (old suite's Par cases, unblocked together) -------
# Bodies are the differentials they will run the day the blockers land:
# legacy already computes every expected value today (except
# delayed_inverse's direct call — legacy produces NO output for it, the
# §6 divergence convention applies and the source's own semantics are
# the spec).


@pytest.mark.skip(reason="IrTuple lowering is not in scope (value side)")
def test_par_return_tuple_literal():
    program = Program(other_basic)
    expected = (12, 40)
    assert run_legacy(program.compile_legacy().compiled, 10) == expected
    assert program.lower()(10) == expected


@pytest.mark.skip(reason="composite targets land with composites (target side)")
def test_par_unpack_call():
    program = Program(invoke_an_inet, other_basic)
    assert run_legacy(program.compile_legacy().compiled) == 12
    assert program.lower()() == 12


@pytest.mark.skip(
    reason="IrTuple lowering is not in scope; legacy also produces no "
    "output for the direct call — source semantics are the spec (§6)"
)
def test_par_of_inverse_construction():
    program = Program(delayed_inverse)
    assert program.lower()() == (9, -1)


@pytest.mark.skip(reason="composite targets land with composites (target side)")
def test_par_of_inverse_write_back():
    """The old suite's inverse-through-Par payoff: `b = 30` after the
    call must retro-update `a` to 40 through the callee's Par."""
    program = Program(use_delayed_inverse, delayed_inverse)
    assert run_legacy(program.compile_legacy().compiled) == 40
    assert program.lower()() == 40

"""§: Par — composite values at the function boundary, consolidated from
the old suite's Par cases (test_compiler.py's other_basic family).

Cases are real module-level functions (see tests/backend/helpers.py).
Par works where it stays a WHOLE value (a Par-typed parameter flows
through the bundle interface and can be returned as-is) and, since
IrTuple/IrTargetTuple lowering landed, across both decomposition
directions:

- value side: an IrTuple literal (`return b + 2, b * 4`) joins the
  element registers into a Par-typed register (old
  evaluate_from_expression shape — join_from_registers);
- target side: unpacking into tuple targets (`a, b = other_basic(10)`)
  joins the element readins into a Par-typed ToRegister that send_value
  fans the incoming Par out through (old evaluate_to_expression shape —
  join_to_registers). The frontend marks unpack targets with the value
  Par's concurrent-item adapters and rejects arity mismatches
  ("Tuple assignment targets do not match the value's Par size").

The oracle is differential wherever legacy works. Two pinned
divergences, both asserted directly (§6: the IR is the spec):

- nested unpacking (`a, (b, c) = p`) starves legacy's output entirely;
- a function whose Inverse read is never completed by a write is an
  INCOMPLETE computation: neither implementation produces output for the
  direct call (legacy starves; ours surfaces the neutral control cell).
  The old suite only ever ran its Par-of-Inverse construction through a
  consumer that writes the inverse (test_par_of_inverse_write_back),
  which retro-propagates through the Par — the time-travel payoff.
"""

import pytest

from natsune.frontend.ir.nodes import IrAssign, IrCallInet
from natsune.special_forms import Inverse, Par
from tests.backend.helpers import Program, program_ids, run_legacy

# --- case programs -------------------------------------------------------
# Each function below is a program under test; the comment above it says
# what the case pins.


# Par as a whole value: param flows in, flows back out untouched
def pair_identity(p: Par[int, int]) -> Par[int, int]:
    return p


# value side: IrTuple return literal (old suite: other_basic(10))
def other_basic(b: int) -> Par[int, int]:
    return b + 2, b * 4


# target side: the linked callee's Par return unpacks into tuple targets
# (old suite: invoke_an_inet() == 12)
def invoke_an_inet() -> int:
    a, b = other_basic(10)
    return a


# unpack then REBIND one element: the bundle registers must stand
# independently after the fan-out
def unpack_rebind(p: Par[int, int]) -> int:
    a, b = p
    b = b + 10
    return a + b


# nested tuple targets: the join recurses (legacy starves — divergence
# test below)
def nested_unpack(p: Par[int, Par[int, int]]) -> int:
    a, (b, c) = p
    return a + b + c


# Par-of-Inverse construction: `a = b + 10` snapshots the inverse read;
# the return carries the still-live inverse cell out through the Par.
# Only meaningful through a consumer that writes the inverse (below) —
# the direct call is an incomplete computation (divergence test below).
def delayed_inverse() -> Par[int, Inverse[int]]:
    b: Inverse[int] = -1
    a = b + 10
    return a, b


# target side, Par-of-Inverse flavor: the caller's `b = 30` retro-
# propagates into `a` through the callee's Par (old suite:
# use_delayed_inverse() == 40, NOT the plain-Python 9)
def use_delayed_inverse() -> int:
    a, b = delayed_inverse()
    b = 30

    return a


# --- tests -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("program", "args", "expected"),
    [
        (Program(pair_identity), ((4, 5),), (4, 5)),
        (Program(other_basic), (10,), (12, 40)),
        (Program(invoke_an_inet, other_basic), (), 12),
        (Program(unpack_rebind), ((4, 5),), 19),
        (Program(use_delayed_inverse, delayed_inverse), (), 40),
    ],
    ids=program_ids,
)
def test_par_matches_legacy_execution(program: Program, args: tuple, expected):
    """The Par surface: whole-value passthrough, tuple-literal returns,
    call-result unpacking, unpack-then-rebind, and the inverse write-back
    through a Par element — identically through legacy and the new
    lowering."""
    assert run_legacy(program.compile_legacy().compiled, *args) == expected
    assert program.lower()(*args) == expected


def test_par_unpack_call_links_the_callee():
    """The unpacked call must link to the legacy callee, not fall back to
    eval (the oracle file's call-case invariant, on a tuple target)."""
    program = Program(invoke_an_inet, other_basic)
    ir = program.build_ir()
    statement = ir.body.statements[0]
    assert isinstance(statement, IrAssign)
    assert isinstance(statement.value, IrCallInet)


def test_nested_unpack():
    """Nested tuple targets recurse through the join.

    Deliberate oracle divergence (§6: the IR is the spec): legacy
    produces NO output for the nested shape — its wire_continuation
    starves — so the source's own semantics are asserted directly."""
    program = Program(nested_unpack)
    assert program.call((1, (2, 3))) == 6
    assert program.lower()((1, (2, 3))) == 6


def test_par_of_inverse_construction_is_incomplete_alone():
    """`delayed_inverse` reads the inverse `b` and returns the live cell
    without ever writing it: the read has no completing write anywhere in
    the net, so the computation never finishes. Neither implementation
    produces output for the direct call — legacy starves (no output
    event), ours resolves the return slot to its neutral control cell
    (RuntimeError). The construction is meaningful only through a
    consumer that writes the inverse, which test_par_matches_legacy
    execution pins at 40 via use_delayed_inverse."""
    program = Program(delayed_inverse)
    with pytest.raises(ValueError):
        run_legacy(program.compile_legacy().compiled)
    with pytest.raises(RuntimeError):
        program.lower()()

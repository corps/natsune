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

The oracle is the source's own Python semantics. Two pinned divergences
from the pre-cutover legacy compiler, asserted directly (§6: the IR is
the spec):

- nested unpacking (`a, (b, c) = p`) starves legacy's output entirely;
- a function whose Inverse read is never completed by a write is an
  INCOMPLETE computation: neither implementation produces output for the
  direct call (legacy starves; ours surfaces the neutral control cell).
  The old suite only ever ran its Par-of-Inverse construction through a
  consumer that writes the inverse (test_par_of_inverse_write_back),
  which retro-propagates through the Par — the time-travel payoff.

Since IrParIndex lowering landed, element reads work too: `p[i]` keeps
the indexed element of the base's Par (old evaluate_from_expression
ast.Subscript shape).

That shape reads pass-through now: a bare Par VARIABLE is read one
element at a time — the indexed element goes out through its own
adapter's egression (a copy for VALUE wiring, a linear alias for a Ref
element) and every other element flows verbatim back into the
variable's retained cell, so no sibling is ever consumed. Values under
Ref wrappers index too (Ref[Par[...]]): the reference cell is
linearized WHOLE — only the wrapper moves; the wrapped Par passes
through untouched — and the indexed element is read from the exposed
temporary, whose leftover elements are closed. The usage cross-check's
element-wise classification pins ride along (per-indexed-element
discipline, recursively over nesting). The frontend builds IrParIndex
for constant in-range integer subscripts of Par-carrying bases (tuple
literals, linked-call results, and Ref-wrapped pars included).
"""

import pytest

from natsune.backend.connector import Connector
from natsune.backend.lowering import lower_function
from natsune.frontend.ir.nodes import IrAssign, IrCallInet
from natsune.special_forms import Inverse, Par, Ref
from tests.backend.helpers import CapturingBackend, Program, program_ids

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


# element reads: index 1 and index 0 (the usage cross-check's `second` is
# this exact first shape — usage-classified a write, now lowered)
def second(p: Par[int, int]) -> int:
    return p[1]


def first(p: Par[int, int]) -> int:
    return p[0]


# two subscripts on ONE base: each is a separate linearizing read of the
# variable's cell
def both_elements(p: Par[int, int]) -> int:
    return p[0] + p[1]


# subscript of a tuple LITERAL: the base lowers through the IrTuple join
# before splitting
def literal_element(a: int) -> int:
    return (a, a + 1)[1]


# subscript of a nested tuple literal: split, then split again
def nested_literal(a: int) -> int:
    return (a, (a + 1, a + 2))[1][0]


# subscript of a LINKED-CALL result: the base is the callee's output
# register
def pair_call(a: int) -> Par[int, int]:
    return a + 1, a * 2


def call_element(a: int) -> int:
    return pair_call(a)[0]


# subscript inside a dynamic capture: the eval context receives the
# ELEMENT, not the whole Par
def printed_element(p: Par[int, int]) -> int:
    print(p[0])
    return p[1]


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


# --- par-index sibling handling: pass-through reads -------------------------


# The ref element reads first AND the value sibling survives the value
# element read — pass-through keeps every sibling in the variable's
# retained cell, in both orders.
def ref_element_then_value_element(p: Par[Ref[int], int]) -> int:
    y = p[0]
    return y + p[1]


def value_element_then_ref_element(p: Par[Ref[int], int]) -> int:
    x = p[1]
    y = p[0]
    return x + y


# The variable's whole cell survives an element read: `return p` after
# `x = p[1]` flows the retained Par out intact.
def value_element_then_whole(p: Par[Ref[int], int]) -> Par[Ref[int], int]:
    x = p[1]
    return p


# The same pass-through through a linked callee's Par result.
def make_ref_pair(a: int) -> Par[Ref[int], int]:
    r: Ref[int] = a
    return r, a + 1


def call_value_then_ref(a: int) -> int:
    pair = make_ref_pair(a)
    x = pair[1]
    y = pair[0]
    return x + y


# Ref[Par[...]]: indexing linearizes only the reference cell — it moves
# out whole and the wrapped Par passes through untouched; the indexed
# element is then read from the exposed temporary (whose leftover
# elements close with the temporary).
def ref_par_string_element(s: str, n: int) -> str:
    pair: Par[Ref[str], int] = (s, n)
    r: Ref[Par[Ref[str], int]] = pair
    return r[0]


def ref_par_int_element(s: str, n: int) -> int:
    pair: Par[Ref[str], int] = (s, n)
    r: Ref[Par[Ref[str], int]] = pair
    return r[1]


# The Ref element of a Ref[Par] stays a usable cell: write through it
# after reading it out of the wrapper.
def ref_par_ref_element_write(s: str, n: int) -> str:
    pair: Par[Ref[str], int] = (s, n)
    r: Ref[Par[Ref[str], int]] = pair
    inner: Ref[str] = r[0]
    inner += "!"
    return inner


# Nested indexing: p[0] is itself a Par — the pass-through read hands
# the element to the next index step, both directly and through a typed
# local.
def nested_index(p: Par[Par[int, int], int]) -> int:
    return p[0][1]


def nested_index_local(p: Par[Par[int, int], int]) -> int:
    inner: Par[int, int] = p[0]
    return inner[1]


# Double reference wrapper: indexing linearizes wrapper cells one at a
# time until the Par is exposed.
def double_ref(s: int) -> int:
    pair: Par[int, int] = (s, s + 1)
    inner: Ref[Par[int, int]] = pair
    outer: Ref[Ref[Par[int, int]]] = inner
    return outer[1]


# Ref-par whose indexed element is itself a Par: two index steps, the
# first through the wrapper.
def ref_par_nested(s: int) -> int:
    inner_pair: Par[int, int] = (s, s + 1)
    r: Ref[Par[Par[int, int], int]] = (inner_pair, s + 2)
    return r[0][1]


# Three-element pars: reads are index-independent — every non-indexed
# element passes through, so any index reads the same machinery (the
# differential pin below).
def pick_first_of_three(p: Par[int, int, int]) -> int:
    return p[0]


def pick_second_of_three(p: Par[int, int, int]) -> int:
    return p[1]


def pick_third_of_three(p: Par[int, int, int]) -> int:
    return p[2]


# --- tests -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("program", "args", "expected"),
    [
        (Program(pair_identity), ((4, 5),), (4, 5)),
        (Program(other_basic), (10,), (12, 40)),
        (Program(invoke_an_inet, other_basic), (), 12),
        (Program(unpack_rebind), ((4, 5),), 19),
        (Program(second), ((4, 5),), 5),
        (Program(first), ((4, 5),), 4),
        (Program(both_elements), ((4, 5),), 9),
        (Program(literal_element), (10,), 11),
        (Program(nested_literal), (1,), 2),
        (Program(call_element, pair_call), (3,), 4),
        (Program(printed_element), ((4, 5),), 5),
        (Program(use_delayed_inverse, delayed_inverse), (), 40),
        # pass-through regression guards: ref element first, value
        # element first, and three-element indexes
        (Program(ref_element_then_value_element), ((3, 4),), 7),
        (Program(pick_first_of_three), ((1, 2, 3),), 1),
        (Program(pick_second_of_three), ((1, 2, 3),), 2),
        (Program(pick_third_of_three), ((1, 2, 3),), 3),
        (Program(nested_index), (((1, 2), 3),), 2),
        (Program(nested_index_local), (((1, 2), 3),), 2),
        (Program(double_ref), (1,), 2),
        (Program(ref_par_nested), (1,), 2),
    ],
    ids=program_ids,
)
def test_par_execution(program: Program, args: tuple, expected):
    """The Par surface: whole-value passthrough, tuple-literal returns,
    call-result unpacking, unpack-then-rebind, element reads (of vars,
    tuple literals, nested literals, and linked-call results, repeated
    reads included), and the inverse write-back through a Par element."""
    assert program.lower()(*args) == expected


def test_par_unpack_call_links_the_callee():
    """The unpacked call must link to the marked callee, not fall back to
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
    the net, so the computation never finishes. The direct call produces
    no usable output — the lowering resolves the return slot to its
    neutral control cell (RuntimeError). The construction is meaningful
    only through a consumer that writes the inverse, which
    test_par_execution pins at 40 via use_delayed_inverse."""
    program = Program(delayed_inverse)
    with pytest.raises(RuntimeError):
        program.lower()()


# --- par-index sibling handling: pass-through pins --------------------------


def test_par_index_reads_are_index_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass-through reads are index-independent: every non-indexed element
    passes through into the retained cell, so the lowering performs no
    sibling closes at all — and no index does more or less work than
    another. (The pre-pass-through lowering returned the indexed element
    from the middle of its close-loop: p[0] leaked two trailing siblings
    on a three-element Par, p[1] one, p[2] none — 180/181/182
    annihilations.) The annihilate count is an implementation-agnostic
    stand-in: any fix style must keep the counts equal."""
    calls = 0
    original = Connector.annihilate

    def counting(self, target, erasure=None):
        nonlocal calls
        calls += 1
        return original(self, target, erasure)

    monkeypatch.setattr(Connector, "annihilate", counting)

    def lowered_annihilations(prog: Program) -> int:
        nonlocal calls
        calls = 0
        lower_function(prog.build_ir(), CapturingBackend())
        return calls

    first = lowered_annihilations(Program(pick_first_of_three))
    second = lowered_annihilations(Program(pick_second_of_three))
    third = lowered_annihilations(Program(pick_third_of_three))

    assert first == second == third


def test_par_value_read_keeps_ref_sibling_usable() -> None:
    """Reading the VALUE element of Par[Ref[int], int] leaves the Ref
    sibling live in the retained cell: the subsequent `p[0]` read moves
    the ref out and its content joins the sum. (Pre-pass-through, the
    whole-Par readout aliased the ref into a readout that the sibling
    close then annihilated — the ref cell was destroyed and the second
    read starved.)"""
    program = Program(value_element_then_ref_element)
    assert program.lower()((3, 4)) == 7


def test_par_value_read_keeps_whole_par_usable() -> None:
    """After an element read the variable's retained cell is intact: the
    whole Par still flows out. (Pre-pass-through, the destroyed sibling
    slot starved the reassembly.)"""
    program = Program(value_element_then_whole)
    assert program.lower()((3, 4)) == (3, 4)


def test_call_value_read_keeps_ref_sibling_usable() -> None:
    """The same pass-through through a LINKED callee's Par result."""
    program = Program(call_value_then_ref, make_ref_pair)
    assert program.lower()(3) == 7


def test_ref_par_element_reads() -> None:
    """Ref[Par[Ref[str], int]]: indexing linearizes only the reference
    cell — it moves out whole and the wrapped Par passes through
    untouched — so both elements read correctly out of the wrapper, each
    in its own discipline (the Ref element's content reads as the str;
    the int element reads as a copy)."""
    assert Program(ref_par_string_element).lower()("hello", 4) == "hello"
    assert Program(ref_par_int_element).lower()("hello", 4) == 4


def test_ref_par_ref_element_write() -> None:
    """The Ref element of a Ref[Par] is a live cell after the read: the
    wrapper moved out whole, the element moved into `inner`, and writes
    through it are visible on the way back out."""
    program = Program(ref_par_ref_element_write)
    assert program.lower()("hello", 4) == "hello!"

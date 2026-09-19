"""Linear variables — Ref/Inverse lowering, consolidated from the old
suite's inverse cases (test_compiler.py's simple_inverse_example family).

Cases are real module-level functions (see tests/backend/helpers.py).
The oracle is differential: the same program through the legacy compiler
and through the new lowering must produce identical results, and both
must match the old suite's asserted values — these programs are the
README's time-travel showcase (an Inverse written AFTER the loop
propagates backwards through it), so the expected values are the point.

- `Inverse[A]` variables carry InverseAdapter: writing the variable
  retro-propagates through every dynamic that read it
  (simple_inverse_example's `[5, 5]`, shift_list_by_smallest's
  backwards shift).
- `Ref[A]` variables carry ReferenceAdapter: aliasing cell whose
  mutations (`a.append(...)`, cross-program `a += 10`) are observed by
  every reader.
- Both wirings are linear (adapters.read_independently refuses), which
  the usage cross-check pins as read-normalizes-to-write.

The old suite's Par-of-Inverse cases (delayed_inverse /
use_delayed_inverse) live in test_par_lowering.py — they are blocked on
Par unpacking, not on the inverse machinery.
"""

import pytest

from natsune.adapters import InverseAdapter, ReferenceAdapter, ValueAdapter
from natsune.backend.lowering import _FunctionLowering
from natsune.backend.python_backend import PythonBackend
from natsune.special_forms import Inverse, Ref
from tests.backend.helpers import Program, program_ids, run_legacy

# --- case programs -------------------------------------------------------
# Each function below is a program under test; the comment above it says
# what the case pins.


# cross-program Ref mutation: the callee's `a += 10` must land on the
# caller's cell (old suite: use_references == 30)
def take_reference(a: Ref[int]) -> None:
    a += 10


def use_references() -> int:
    a: Ref[int] = 20
    take_reference(a)
    return a


# the headline: one Inverse cell, two Ref-list captures, write-back after
# the captures — both appended elements retro-update (old suite: [5, 5])
def simple_inverse_example() -> list:
    a: Ref[list] = []
    b: Inverse[int] = 0
    a.append(b)
    a.append(b)
    b = 5
    return a


# inverse write-back through a loop: `b = 3` rewrites every `total += b`
# already sequenced (old suite: simple_inverse_loop_example(10) == 30)
def simple_inverse_loop_example(scale: int) -> int:
    total = 0
    b: Inverse[int] = -1
    for _ in range(scale):
        total += b
    b = 3
    return total


# the README's flagship: `smallest = smallest_acc` after the loop
# retro-shifts every element (old suite: [4, 9, 1, 10] -> [3, 8, 0, 9])
def shift_list_by_smallest(l: list[int]) -> list[int]:
    if len(l) == 0:
        return []

    smallest: Inverse[int] = -1
    smallest_acc: int = l[0]

    result: Ref[list[int]] = []

    for v in l:
        if v < smallest_acc:
            smallest_acc = v
        print(smallest)
        result.append(v - smallest)

    smallest = smallest_acc

    return result


# Ref mutation through method calls only — no inverse, no rebinding
# (old suite: ref_for_expressions() == [1, 2])
def ref_for_expressions() -> list:
    a: Ref[list] = []
    a.append(1)
    a.append(2)
    print(a)
    return a


# --- differential matrix ---------------------------------------------------


_CASES = [
    # (program, args) — the expected values are the old suite's asserted
    # returns; legacy and the new lowering must both reproduce them.
    (Program(use_references, take_reference), (), 30),
    (Program(simple_inverse_example), (), [5, 5]),
    (Program(simple_inverse_loop_example), (10,), 30),
    (Program(shift_list_by_smallest), ([4, 9, 1, 10],), [3, 8, 0, 9]),
    (Program(ref_for_expressions), (), [1, 2]),
]


@pytest.mark.parametrize("program,args,expected", _CASES, ids=program_ids)
def test_matches_legacy_execution(program: Program, args: tuple, expected) -> None:
    assert run_legacy(program.compile_legacy().compiled, *args) == expected
    assert program.lower()(*args) == expected


# --- introspection ---------------------------------------------------------


def test_bundle_adapters_match_legacy_by_name():
    """The linear wirings must survive collection: every variable's
    adapter TYPE equals the legacy bundle's (interface position is
    order-bearing, §6 — the types are what make the behavior above
    possible), with Inverse/Ref annotations landing on Inverse/Reference
    adapters and unannotated names staying values."""
    program = Program(shift_list_by_smallest)
    legacy = program.compile_legacy()
    ours = _FunctionLowering(program.build_ir(), PythonBackend()).collect_variables()

    assert set(ours) == set(legacy.variables)
    for name, adapter in ours.items():
        assert type(adapter) is type(legacy.variables[name]), name

    assert isinstance(ours["smallest"], InverseAdapter)
    assert isinstance(ours["result"], ReferenceAdapter)


def test_unannotated_names_stay_values():
    """Negative half of the bundle check: names without Ref/Inverse
    annotations must stay values, not linear cells."""
    program = Program(shift_list_by_smallest)
    ours = _FunctionLowering(program.build_ir(), PythonBackend()).collect_variables()
    assert type(ours["l"]) is ValueAdapter
    assert type(ours["smallest_acc"]) is ValueAdapter
    assert type(ours["v"]) is ValueAdapter

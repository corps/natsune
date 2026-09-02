"""Example programs for IR snapshots — copied from tests/test_compiler.py.

These are REAL functions (no `@inet` decoration: nothing compiles at import),
extracted through the genuine phase-1 path (`extract_source` →
`inspect.getsourcelines`), not text snippets. Add a program by writing a
module-level function and adding it to `PROGRAMS`, then run:

    make snapshots-update

which (re)writes `tests/frontend/snapshots/<name>.ir`. Snapshot tests fail
with a unified diff whenever a rendered IR drifts from its stored snapshot.

Programs that call other programs get fake `__inet__` compilers attached (the
bottom of this file), mirroring what the paused `inet` decorator will one day
attach for real — so their snapshots exercise `IrCallInet` nodes with copied
metadata instead of falling back to `IrDynamic`.
"""

from types import SimpleNamespace

from natsune.adapters import ParValueAdapter, adapter_from_type
from natsune.special_forms import Inverse, Par, Ref


def _compiler(*arg_types, return_type=None) -> SimpleNamespace:
    """An old-compiler-shaped object behind `__inet__` (copied metadata)."""
    return SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_annot=return_type,
    )


def basic(b: int) -> int:
    a = 10
    a = a + b
    print(a)
    print(123)
    return a


def other_basic(b: int) -> Par[int, int]:
    return b + 2, b * 4


def invoke_an_inet() -> int:
    a, b = other_basic(10)
    return a


def take_reference(a: Ref[int]) -> None:
    a += 10


def use_references() -> int:
    a: Ref[int] = 20
    take_reference(a)
    return a


def sum_it_up(start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        print(total)
        print(i)
        total += i
    return total


def is_it_even(input: int) -> bool:
    if input % 2 == 0:
        return True
    return False


def basic_sum_with_while(start: int, end: int) -> int:
    total = 0
    i = start
    while i < end:
        total += i
        i += 1
    return total


def simple_inverse_example() -> list:
    a: Ref[list] = []
    b: Inverse[int] = 0
    a.append(b)
    a.append(b)
    b = 5
    return a


def simple_inverse_loop_example(scale: int) -> int:
    total = 0
    b: Inverse[int] = -1
    for _ in range(scale):
        total += b
    b = 3
    return total


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


def delayed_inverse() -> Par[int, Inverse[int]]:
    b: Inverse[int] = -1
    a = b + 10
    return a, b


def test_delayed_inverse() -> int:
    a, b = delayed_inverse()
    b = 30

    return a


def ref_for_expressions() -> list:
    a: Ref[list] = []
    a.append(1)
    a.append(2)
    print(a)
    return a


def ignored_infinite_loop() -> Par[int, int]:
    b = 0
    while True:
        b += 1
    return 10, b


def drops_infinite_loop() -> int:
    a, b = ignored_infinite_loop()
    return a


def infinite_value() -> int:
    a = 0
    while True:
        a += 1
    return a


def and_or_with_finites_and_infinites() -> list:
    a = infinite_value()
    paths: Ref[list] = []

    if a < 10 or True:
        paths.append("Infinite Or")

    if a < 10 and False:
        paths.append("Infinite And")

    paths.append(10 and 0)
    paths.append(10 or 0)
    return paths


# Fake compilers for cross-program calls. The paused `inet` decorator will
# attach real ones at cutover; the shape is all the frontend ever reads.
setattr(other_basic, "__inet__", _compiler(int, return_type=Par[int, int]))
setattr(take_reference, "__inet__", _compiler(Ref[int]))
setattr(delayed_inverse, "__inet__", _compiler(return_type=Par[int, Inverse[int]]))
setattr(ignored_infinite_loop, "__inet__", _compiler(return_type=Par[int, int]))
setattr(infinite_value, "__inet__", _compiler())


PROGRAMS = [
    basic,
    other_basic,
    invoke_an_inet,
    take_reference,
    use_references,
    sum_it_up,
    is_it_even,
    basic_sum_with_while,
    simple_inverse_example,
    simple_inverse_loop_example,
    shift_list_by_smallest,
    delayed_inverse,
    test_delayed_inverse,
    ref_for_expressions,
    ignored_infinite_loop,
    drops_infinite_loop,
    infinite_value,
    and_or_with_finites_and_infinites,
]

"""Example programs for IR snapshots — copied from tests/test_compiler.py.

The `@inet` decorators are stripped: these are the *sources* fed through the
new pipeline (make_source → analyze_signature → collect_symbols →
collect_call_links → build_ir). To add a program, append a `Program` entry to
`PROGRAMS` and run:

    NATSUNE_UPDATE_SNAPSHOTS=1 uv run pytest tests/frontend/test_snapshots.py

which (re)writes `tests/frontend/snapshots/<name>.ir`. Snapshot tests fail
with a unified diff whenever a rendered IR drifts from its stored snapshot.

Programs that call other programs get `inet_stub`s shaped like the old
compiler objects, so their snapshots exercise real `IrCallInet` nodes with
copied metadata instead of falling back to `IrDynamic`.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any

from natsune.adapters import ParValueAdapter, adapter_from_type
from natsune.special_forms import Inverse, Par, Ref


def inet_stub(*arg_types: Any, return_type: Any = None):
    """A function-shaped stand-in carrying an old-compiler-shaped `__inet__`."""
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_annot=return_type,
    )

    def stub(*_args: Any) -> Any:
        raise AssertionError("program stubs are never executed")

    setattr(stub, "__inet__", compiler)
    return stub


_BUILTIN = {"Par": Par, "Ref": Ref, "Inverse": Inverse}


@dataclasses.dataclass(frozen=True)
class Program:
    name: str
    snippet: str
    namespace: dict[str, Any] = dataclasses.field(default_factory=dict)

    def full_namespace(self) -> dict[str, Any]:
        return {**_BUILTIN, **self.namespace}


PROGRAMS: list[Program] = [
    Program(
        "basic",
        """
        def basic(b: int) -> int:
            a = 10
            a = a + b
            print(a)
            print(123)
            return a
        """,
    ),
    Program(
        "other_basic",
        """
        def other_basic(b: int) -> Par[int, int]:
            return b + 2, b * 4
        """,
    ),
    Program(
        "invoke_an_inet",
        """
        def invoke_an_inet() -> int:
            a, b = other_basic(10)
            return a
        """,
        {"other_basic": inet_stub(int, return_type=Par[int, int])},
    ),
    Program(
        "take_reference",
        """
        def take_reference(a: Ref[int]) -> None:
            a += 10
        """,
    ),
    Program(
        "use_references",
        """
        def use_references() -> int:
            a: Ref[int] = 20
            take_reference(a)
            return a
        """,
        {"take_reference": inet_stub(Ref[int])},
    ),
    Program(
        "sum_it_up",
        """
        def sum_it_up(start: int, end: int) -> int:
            total = 0
            for i in range(start, end):
                print(total)
                print(i)
                total += i
            return total
        """,
    ),
    Program(
        "is_it_even",
        """
        def is_it_even(input: int) -> bool:
            if input % 2 == 0:
                return True
            return False
        """,
    ),
    Program(
        "basic_sum_with_while",
        """
        def basic_sum_with_while(start: int, end: int) -> int:
            total = 0
            i = start
            while i < end:
                total += i
                i += 1
            return total
        """,
    ),
    Program(
        "simple_inverse_example",
        """
        def simple_inverse_example() -> list:
            a: Ref[list] = []
            b: Inverse[int] = 0
            a.append(b)
            a.append(b)
            b = 5
            return a
        """,
    ),
    Program(
        "simple_inverse_loop_example",
        """
        def simple_inverse_loop_example(scale: int) -> int:
            total = 0
            b: Inverse[int] = -1
            for _ in range(scale):
                total += b
            b = 3
            return total
        """,
    ),
    Program(
        "shift_list_by_smallest",
        """
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
        """,
    ),
    Program(
        "delayed_inverse",
        """
        def delayed_inverse() -> Par[int, Inverse[int]]:
            b: Inverse[int] = -1
            a = b + 10
            return a, b
        """,
    ),
    Program(
        "test_delayed_inverse",
        """
        def test_delayed_inverse() -> int:
            a, b = delayed_inverse()
            b = 30

            return a
        """,
        {"delayed_inverse": inet_stub(return_type=Par[int, Inverse[int]])},
    ),
    Program(
        "ref_for_expressions",
        """
        def ref_for_expressions() -> list:
            a: Ref[list] = []
            a.append(1)
            a.append(2)
            print(a)
            return a
        """,
    ),
    Program(
        "ignored_infinite_loop",
        """
        def ignored_infinite_loop() -> Par[int, int]:
            b = 0
            while True:
                b += 1
            return 10, b
        """,
    ),
    Program(
        "drops_infinite_loop",
        """
        def drops_infinite_loop() -> int:
            a, b = ignored_infinite_loop()
            return a
        """,
        {"ignored_infinite_loop": inet_stub(return_type=Par[int, int])},
    ),
    Program(
        "infinite_value",
        """
        def infinite_value() -> int:
            a = 0
            while True:
                a += 1
            return a
        """,
    ),
    Program(
        "and_or_with_finites_and_infinites",
        """
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
        """,
        {"infinite_value": inet_stub()},
    ),
]


def program_by_name(name: str) -> Program:
    for program in PROGRAMS:
        if program.name == name:
            return program
    raise KeyError(f"unknown program: {name!r}")

import pytest

from natsune.first_order.adapters import InverseAdapter, ReferenceAdapter, ValueAdapter
from natsune.backend.lowering import _FunctionLowering
from natsune.backend.python_backend import PythonBackend
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.special_forms import Inverse, Ref
from tests.backend.helpers import Program, program_ids


def take_reference(a: Ref[int]) -> None:
    a += 10


def use_references() -> int:
    a: Ref[int] = 20
    take_reference(a)
    return a


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


def ref_for_expressions() -> list:
    a: Ref[list] = []
    a.append(1)
    a.append(2)
    print(a)
    return a


_CASES = [
    (Program(use_references, take_reference), (), 30),
    (Program(simple_inverse_example), (), [5, 5]),
    (Program(simple_inverse_loop_example), (10,), 30),
    (Program(shift_list_by_smallest), ([4, 9, 1, 10],), [3, 8, 0, 9]),
    (Program(ref_for_expressions), (), [1, 2]),
]


@pytest.mark.parametrize("program,args,expected", _CASES, ids=program_ids)
def test_execution(program: Program, args: tuple, expected) -> None:
    assert program.lower()(*args) == expected


def test_bundle_adapters_match_symbols_by_name():
    """The lowering's variable walk must agree with the frontend symbols
    table in names, adapters, AND order — this is the same invariant the
    recursive-callee promise's interface adapters rest on (CUTOVER.md
    §2: the promise builds its adapters from the collected symbols
    through the same constructors the net's VariablesFlow uses)."""
    program = Program(shift_list_by_smallest)
    ours = _FunctionLowering(program.build_ir(), PythonBackend()).collect_variables()

    source = extract_source(
        program.entry,
        globals=program.entry.__globals__,
        filename=program.entry.__code__.co_filename,
    )
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())

    assert list(ours) == list(symbols.variables)
    for name, adapter in ours.items():
        assert type(adapter) is type(symbols.variables[name]), name

    assert isinstance(ours["smallest"], InverseAdapter)
    assert isinstance(ours["result"], ReferenceAdapter)


def test_unannotated_names_stay_values():
    program = Program(shift_list_by_smallest)
    ours = _FunctionLowering(program.build_ir(), PythonBackend()).collect_variables()
    assert type(ours["l"]) is ValueAdapter
    assert type(ours["smallest_acc"]) is ValueAdapter
    assert type(ours["v"]) is ValueAdapter

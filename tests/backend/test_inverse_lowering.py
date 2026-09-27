import pytest

from natsune.first_order.adapters import InverseAdapter, ReferenceAdapter, ValueAdapter
from natsune.backend.lowering import _FunctionLowering
from natsune.backend.python_backend import PythonBackend
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.special_forms import Inverse, Par, Ref
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


# --- Inverse[Par[...]]: a whole Par under an inverse loan -------------------


# The retro discipline is positional: reads complete only at a LATER
# write in the same net segment. A read after the last write has no
# completing write anywhere downstream and starves (the neutral control
# cell surfaces as RuntimeError) — true for plain Inverse[int] too, not
# a Par-specific gap.
def int_read_after_write(s: int) -> int:
    b: Inverse[int] = s
    b = s + 10
    x = b + 0
    return x


def consume_int_with_inverse(inv: Inverse[int]) -> int:
    x = inv + 0
    return x


def int_loan_across_callee(s: int) -> int:
    b: Inverse[int] = s
    total = consume_int_with_inverse(b)
    b = s + 10
    return total


def consume_int_with_ref_inverse(inv: Inverse[int]) -> None:
    inv = 10


def int_loan_across_callee_with_ref(s: int) -> int:
    b: Inverse[int] = s
    s = b + 10
    consume_int_with_ref_inverse(b)
    del b
    return s


def inverse_par_time_travel(s: int) -> int:
    inv: Inverse[Par[int, int]] = (s, s + 1)
    pair: Par[int, int] = inv
    inv = (s + 10, s + 20)
    return pair[0]


def inverse_par_time_travel_whole(s: int) -> Par[int, int]:
    inv: Inverse[Par[int, int]] = (s, s + 1)
    pair: Par[int, int] = inv
    inv = (s + 10, s + 20)
    return pair


# Direct element indexing of the loan is unsupported: Inverse wrappers
# are deliberately not indexable (find_par_adapter declines them — a
# read is a retro loan, not a dereferenceable cell), so `inv[0]` falls
# to the exec fallback, whose capture of the whole inverse is refused by
# the borrow (no Inverse(Par) to-value adaptation).
def inverse_par_index(s: int) -> int:
    inv: Inverse[Par[int, int]] = (s, s + 1)
    x = inv[0]
    inv = (s + 10, s + 20)
    return x


# Unpack targets are refused at compile time: the tuple-target check
# requires a Par VALUE adapter and does not look through the Inverse
# wrapper — even though the net's IA.unpack adaptation could serve it.
def inverse_par_unpack(inv: Inverse[Par[int, int]]) -> int:
    a, b = inv
    return a + b


_CASES = [
    (Program(use_references, take_reference), (), 30),
    (Program(simple_inverse_example), (), [5, 5]),
    (Program(simple_inverse_loop_example), (10,), 30),
    (Program(shift_list_by_smallest), ([4, 9, 1, 10],), [3, 8, 0, 9]),
    (Program(ref_for_expressions), (), [1, 2]),
    (Program(inverse_par_time_travel), (1,), 11),
    (Program(inverse_par_time_travel_whole), (1,), (11, 21)),
]


@pytest.mark.parametrize("program,args,expected", _CASES, ids=program_ids)
def test_execution(program: Program, args: tuple, expected) -> None:
    assert program.lower()(*args) == expected


def test_inverse_read_needs_later_write() -> None:
    program = Program(int_read_after_write)
    with pytest.raises(RuntimeError):
        program.lower()(1)


def test_inverse_loan_does_not_cross_callees() -> None:
    program = Program(int_loan_across_callee, consume_int_with_inverse)
    with pytest.raises(RuntimeError):
        program.lower()(1)


def test_inverse_loan_with_ref() -> None:
    program = Program(int_loan_across_callee_with_ref, consume_int_with_ref_inverse)
    assert program.lower()(1) == 1


def test_inverse_par_index_unsupported() -> None:
    """Inverse[Par] element reads are pending work: the desired end state
    is an element-wise retro read (x = inv[0] completes at the write, so
    11 here). Today the subscript falls to the exec fallback and the
    capture borrow refuses the whole-inverse read at lowering time.
    Remove the xfail when loans support element taps."""
    pytest.xfail(
        "lowering: Inverse[Par] indexing falls to the exec fallback and "
        "the capture borrow has no Inverse(Par) adaptation"
    )
    program = Program(inverse_par_index)
    assert program.lower()(1) == 11


def test_inverse_par_unpack_targets_rejected() -> None:
    """Pinned compile-time refusal: the tuple-target validation requires
    a Par value adapter and does not look through the Inverse wrapper."""
    program = Program(inverse_par_unpack)
    with pytest.raises(ExceptionGroup) as info:
        program.build_ir()
    assert "Tuple assignment targets" in str(info.value.exceptions[0])


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

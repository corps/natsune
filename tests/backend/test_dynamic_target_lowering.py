from typing import Callable

import pytest

from natsune.backend.python_backend import PythonBackend
from natsune.executor import DeterministicSerialExecutor, Executor, ThreadPoolExecutor
from natsune.special_forms import Ref
from tests.backend.helpers import Program, program_ids


def ref_list_appends() -> list:
    a: Ref[list] = []
    a.append(1)
    a.append(2)
    return a


def ref_dict_target() -> dict:
    d: Ref[dict] = {}
    d["k"] = 1
    return d


def ref_repeated_targets() -> dict:
    d: Ref[dict] = {}
    d["a"] = 1
    d["b"] = 2
    return d


# The linearization probe: the read between the two captures must see
# exactly the completed first exec ([1], not [] or [1, 2]).
def ref_read_between_captures() -> list:
    a: Ref[list] = []
    a.append(1)
    print(a)
    a.append(2)
    return a


# Consecutive pure reads each consume and restore the ref cell.
def ref_read_twice() -> list:
    a: Ref[list] = []
    a.append(1)
    print(a)
    print(a)
    return a


# The borrow must leave the cell writable: a plain rebinding after a
# dynamic capture takes effect.
def ref_rebind_after_capture() -> list:
    a: Ref[list] = []
    a.append(1)
    a = [9]
    return a


# Nested Ref constituents: a ref captured into another ref's exec.
def ref_nested_capture() -> list:
    a: Ref[list] = []
    xs: Ref[list] = []
    a.append(1)
    xs.append(a)
    return xs


# Linearization across a linked-callee boundary: the callee's dynamic
# capture of its Ref parameter must complete before the caller's read.
def append_two(a: Ref[list]) -> None:
    a.append(2)


def cross_function_dynamic() -> list:
    a: Ref[list] = []
    a.append(1)
    append_two(a)
    return a


# Augassign flavor: the synthesized read-modify-write exec with a Ref
# capture, written back through the ref cell.
def ref_augassign_callee(a: Ref[int]) -> None:
    a += 10


def use_ref_augassign() -> int:
    a: Ref[int] = 20
    ref_augassign_callee(a)
    return a


# Ref-parameter flavors: the read after the capture is what carries the
# completed exec's effect out (to the return and to the caller's object).
# These take the container as a parameter, so each test drives them with
# fresh arguments per implementation.
def ref_param_append_then_read(a: Ref[list]) -> list:
    a.append(1)
    return a


def ref_param_target_then_read(d: Ref[dict]) -> dict:
    d["k"] = 1
    return d


def ref_mixed_capture(a: Ref[list], xs: list) -> list:
    a.append(len(xs))
    return a


def plain_local_dict_target() -> dict:
    d = {}
    d["k"] = 1
    return d


class copyless_dict(dict):
    def __copy__(self):
        return self


def plain_local_repeated_targets() -> dict:
    d = copyless_dict()
    d["a"] = 1
    d["b"] = 2

    while True:
        if "a" in d and "b" in d:
            break

    return d


def plain_param_dict_target(d: dict) -> None:
    d["injected"] = True


def plain_param_keyed_insert(d: dict, k: str, v: int) -> None:
    d[k] = v


def plain_local_rebind_after_target() -> dict:
    d = {}
    d["k"] = 1
    d = {"fresh": 9}
    return d


def global_producer(v: int) -> int:
    return v + 1


def ref_target_calls_global() -> dict:
    d: Ref[dict] = {}
    d["k"] = global_producer(1)
    return d


def lower_with(program: Program, executor: Executor) -> Callable:
    return program.lower(PythonBackend(executor=executor))


BOTH_EXECUTORS = [
    pytest.param(None, id="threaded"),
    pytest.param(DeterministicSerialExecutor(), id="deterministic"),
]


@pytest.mark.parametrize(
    ("program", "args", "expected"),
    [
        (Program(ref_list_appends), (), [1, 2]),
        (Program(ref_dict_target), (), {"k": 1}),
        (Program(ref_repeated_targets), (), {"a": 1, "b": 2}),
        (Program(ref_read_between_captures), (), [1, 2]),
        (Program(ref_read_twice), (), [1]),
        (Program(ref_nested_capture), (), [[1]]),
        (Program(ref_rebind_after_capture), (), [9]),
        (Program(cross_function_dynamic, append_two), (), [1, 2]),
        # sanity: flow after a dynamic target is unaffected
        (Program(plain_local_rebind_after_target), (), {"fresh": 9}),
    ],
    ids=program_ids,
)
def test_ref_dynamics_execution(program: Program, args: tuple, expected):
    assert program.call(*args) == expected
    assert program.lower()(*args) == expected


@pytest.mark.parametrize("executor", BOTH_EXECUTORS)
def test_ref_target_linearizes_under_both_executors(executor):
    program = Program(ref_repeated_targets)
    assert lower_with(program, executor)() == {"a": 1, "b": 2}


@pytest.mark.parametrize("executor", BOTH_EXECUTORS)
def test_ref_read_between_captures_observes_completed_exec(executor, capsys):
    program = Program(ref_read_between_captures)
    lowered = lower_with(program, executor)
    for _ in range(10):
        assert capsys.readouterr().out == ""
        assert lowered() == [1, 2]
        assert capsys.readouterr().out == "[1]\n"


@pytest.mark.parametrize("executor", BOTH_EXECUTORS)
def test_ref_reads_restore_the_cell(executor, capsys):
    program = Program(ref_read_twice)
    lowered = lower_with(program, executor)
    for _ in range(5):
        assert capsys.readouterr().out == ""
        assert lowered() == [1]
        assert capsys.readouterr().out == "[1]\n[1]\n"


def test_ref_mixed_capture_linearizes_ref_but_copies_value():
    program = Program(ref_mixed_capture)

    caller_list = []
    assert program.call(caller_list, [10, 20]) is caller_list
    assert caller_list == [2]

    new_list = []
    assert program.lower()(new_list, [10, 20]) is new_list
    assert new_list == [2]


def test_ref_param_read_after_sees_completed_exec():
    append_program = Program(ref_param_append_then_read)
    target_program = Program(ref_param_target_then_read)

    for program, expected in (
        (append_program, [1]),
        (target_program, {"k": 1}),
    ):
        plain_args = ([],) if expected == [1] else ({},)
        assert program.call(*plain_args) is plain_args[0]
        assert plain_args[0] == expected

        new_args = ([],) if expected == [1] else ({},)
        out = program.lower()(*new_args)
        assert out is new_args[0]
        assert new_args[0] == expected


def test_ref_augassign_write_back_diverges_from_plain_python():
    program = Program(use_ref_augassign, ref_augassign_callee)
    assert program.call() == 20
    assert program.lower()() == 30


def test_plain_local_target_mutation_is_not_linearized():
    program = Program(plain_local_dict_target)
    assert program.call() == {"k": 1}
    assert lower_with(program, DeterministicSerialExecutor())() == {}


def test_plain_local_repeated_targets():
    program = Program(plain_local_repeated_targets)
    assert program.call() == {"a": 1, "b": 2}
    assert program.lower(PythonBackend(executor=ThreadPoolExecutor()))() == {
        "a": 1,
        "b": 2,
    }


@pytest.mark.parametrize(
    "program",
    [Program(plain_param_dict_target), Program(plain_param_keyed_insert)],
    ids=program_ids,
)
def test_plain_param_target_mutation_does_not_escape(program: Program):
    d = {}
    args = (d, "k", 5) if program.name == "plain_param_keyed_insert" else (d,)
    program.lower(PythonBackend(executor=DeterministicSerialExecutor()))(*args)
    assert d == {}


def test_ref_target_calling_a_global():
    program = Program(ref_target_calls_global)
    assert program.lower()() == {"k": 2}


def test_ir_function_carries_the_globals_namespace():
    program = Program(ref_target_calls_global)
    ir = program.build_ir()
    assert ir.globals is global_producer.__globals__

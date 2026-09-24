import pytest

from natsune.backend.control_flow import Loop
from natsune.special_forms import Par, Ref
from tests.backend.helpers import (
    Program,
    capturing_lower,
    program_ids,
)


def while_sum(start: int, end: int) -> int:
    total = 0
    i = start
    while i < end:
        total += i
        i += 1
    return total


def sum_it_up(start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        print(total)
        print(i)
        total += i
    return total


def for_sum(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    return total


def while_continue(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        if i % 2 == 0:
            continue
        total += i
    return total


def for_continue(a: int) -> int:
    total = 0
    for i in range(a):
        if i % 2 == 0:
            continue
        total += i
    return total


def while_continue_closer(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        continue
    return total


def for_continue_closer(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
        continue
    return total


def while_break_bound(a: int) -> int:
    total = 0
    i = 0
    while i < 100:
        if i >= a:
            break
        total += i
        i += 1
    return total


def for_break(a: int) -> int:
    total = 0
    for i in range(a):
        if i > 2:
            break
        total += i
    return total


def while_break_closer(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        break
    return total


def while_direct_break(a: int) -> int:
    total = 0
    while a > 0:
        total = 5
        break
    return total


def for_return_first(a: int) -> int:
    for i in range(a):
        return i
    return -1


def while_return_test(a: int) -> int:
    while a >= 0:
        return 42
    return -1


def for_break_only(a: int) -> int:
    x = 5
    for i in range(a):
        break
    x = x + 1
    return x


def for_if_always_exits(a: int) -> int:
    for i in range(a):
        if i > 10:
            return 1
        else:
            return 2
    return 0


def outer_recurse_inner_no_recurse(outer: int) -> int:
    total = 0
    for i in range(outer):
        for j in range(3):
            break
        total += i
    return total


# orelse: runs on exhaustion, skipped by break
def while_orelse(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        total += i
        i += 1
    else:
        total += 100
    return total


def while_break_orelse(a: int) -> int:
    i = 0
    while i < a:
        break
    else:
        i = 99
    return i


def for_orelse(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    else:
        total += 100
    return total


def for_break_orelse(a: int) -> int:
    total = 0
    for i in range(a):
        break
    else:
        total += 100
    return total


def loop_early_return(a: int) -> int:
    for i in range(a):
        if i > 2:
            return i
    return -1


def nested_break(a: int) -> int:
    total = 0
    for i in range(a):
        for j in range(a):
            if j == 2:
                break
            total += 1
    return total


def nested_while_in_for(a: int) -> int:
    total = 0
    for i in range(a):
        j = 0
        while j < i:
            total += j
            j += 1
    return total


def loop_then_trailing(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    total = total * 2
    return total


def while_then_trailing(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        total += i
        i += 1
    total = total * 2
    return total


def loop_implicit_none(a: int) -> None:
    total = 0
    for i in range(a):
        total += i
    print(total)


def flag_break(a: int) -> int:
    total = 0
    i = 0
    flag = True
    while flag:
        if i >= a:
            flag = False
        else:
            total += i
            i += 1
    return total


def while_true_break(a: int) -> int:
    total = 0
    while True:
        total += a
        break
    return total


def while_true_if_break(a: int) -> int:
    total = 0
    i = 0
    while True:
        if i >= a:
            break
        else:
            total += i
            i += 1
    return total


_CASES = [
    (Program(while_sum), (2, 6)),
    (Program(for_sum), (5,)),
    (Program(sum_it_up), (1, 10)),
    (Program(while_continue), (6,)),
    (Program(for_continue), (6,)),
    (Program(while_continue_closer), (5,)),
    (Program(for_continue_closer), (5,)),
    (Program(while_break_closer), (5,)),
    (Program(while_direct_break), (7,)),
    (Program(for_break), (10,)),
    (Program(while_orelse), (4,)),
    (Program(while_break_orelse), (5,)),
    (Program(for_orelse), (4,)),
    (Program(for_break_orelse), (5,)),
    (Program(loop_early_return), (10,)),
    (Program(loop_early_return), (2,)),
    (Program(for_return_first), (0,)),
    (Program(for_return_first), (3,)),
    (Program(while_return_test), (5,)),
    (Program(while_return_test), (-1,)),
    (Program(for_break_only), (0,)),
    (Program(for_break_only), (7,)),
    (Program(for_if_always_exits), (0,)),
    (Program(for_if_always_exits), (5,)),
    (Program(for_if_always_exits), (20,)),
    (Program(outer_recurse_inner_no_recurse), (0,)),
    (Program(outer_recurse_inner_no_recurse), (4,)),
    (Program(nested_while_in_for), (4,)),
    (Program(loop_then_trailing), (5,)),
    (Program(while_then_trailing), (5,)),
]


@pytest.mark.parametrize("program,args", _CASES, ids=program_ids)
def test_matches_source_execution(program: Program, args: tuple) -> None:
    expected = program.call(*args)
    assert program.lower()(*args) == expected

    program = Program(while_break_bound)
    assert program.call(5) == 10
    assert program.lower()(5) == 10


def test_implicit_none_after_loop_legacy_starves():
    program = Program(loop_implicit_none)
    assert program.call(3) is None
    assert program.lower()(3) is None


def test_flag_loop_legacy_starves():
    program = Program(flag_break)
    assert program.call(5) == 10
    assert program.lower()(5) == 10


def test_while_true_if_break():
    program = Program(while_true_if_break)
    assert program.lower()(5) == 10


def test_while_true_break():
    program = Program(while_true_break)
    assert program.lower()(5) == 5


def test_loop_agent_is_cataloged():
    unit = capturing_lower(Program(for_sum))
    loops = [agent for agent in unit.agents if isinstance(agent, Loop)]
    assert len(loops) == 1
    loop = loops[0]
    assert any(flow is loop.body for flow in unit.agents)
    assert any(flow is loop.orelse for flow in unit.agents)


def infinite_value() -> int:
    a = 0
    while True:
        a += 1
    return a


def ignored_infinite_loop() -> Par[int, int]:
    b = 0
    while True:
        b += 1
    return 10, b


def drops_infinite_loop() -> int:
    a, b = ignored_infinite_loop()
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


def test_drops_infinite_loop_differential():
    program = Program(drops_infinite_loop, ignored_infinite_loop)
    assert program.lower()() == 10


def test_and_or_finites_and_infinites_differential():
    program = Program(and_or_with_finites_and_infinites, infinite_value)
    expected = ["Infinite Or", 0, 10]
    assert program.lower()() == expected

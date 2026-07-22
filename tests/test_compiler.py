from natsune.registers import send_value
import pytest

from natsune.calculus import Calculus
from natsune.compiler import inet, InetFunctionCompiler
from natsune.special_forms import Par, Ref, Inverse


@pytest.fixture(scope='function')
def c() -> Calculus:
    return Calculus()

@inet
def basic(b: int) -> int:
    a = 10
    a = a + b
    print(a)
    print(123)
    return a

@inet
def other_basic(b: int) -> Par[int, int]:
    return b + 2, b * 4

@inet
def invoke_an_inet() -> int:
    a, b = other_basic(10)
    return a

@inet
def take_reference(a: Ref[int]) -> None:
    a += 10

@inet
def use_references() -> int:
    a: Ref[int] = 20
    take_reference(a)
    return a

@inet
def sum_it_up(start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        print(total)
        print(i)
        total += i
    return total

@inet
def is_it_even(input: int) -> bool:
    if input % 2 == 0:
        return True
    return False

@inet
def basic_sum_with_while(start: int, end: int) -> int:
    total = 0
    i = start
    while i < end:
        total += i
        i += 1
    return total

@inet
def shift_list_by_smallest(l: list[int]) -> list[int]:
    if len(l) == 0:
        return []

    smallest: Inverse[int] = -1
    result: Ref[list[int]] = []
    smallest_acc: int = l[0]
    for v in l:
        if v < smallest_acc:
            smallest_acc = v
        print(smallest)
        result.append(v - smallest)

    return result


@inet
def ref_for_expressions() -> list:
    a: Ref[list] = []
    a.append(1)
    a.append(2)
    print(a)
    return a

def test_compiled_functions() -> None:
    # assert basic(29) == 39
    # assert invoke_an_inet() == 12
    # assert use_references() == 30
    # assert sum_it_up(1, 10) == 45
    # assert is_it_even(10) == True
    # assert is_it_even(11) == False
    # assert basic_sum_with_while(1, 10) == 45
    # assert ref_for_expressions() == [1, 2]
    assert shift_list_by_smallest([4, 9, 1, 10]) == []
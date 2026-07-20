from natsune.registers import send_value
import pytest

from natsune.calculus import Calculus
from natsune.compiler import inet, InetFunctionCompiler
from natsune.special_forms import Par, Ref


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

def test_compiled_functions(c: Calculus) -> None:
    # assert basic(29) == 39
    # assert invoke_an_inet() == 12
    # assert use_references() == 30
    assert sum_it_up(1, 10) == 45

import pytest

from natsune.calculus import Calculus
from natsune.compiler import inet
from natsune.special_forms import Par


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

def test_basic(c: Calculus) -> None:
    assert basic(29) == 39
    assert invoke_an_inet() == 12

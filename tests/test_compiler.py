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

def test_compiled_functions(c: Calculus) -> None:
    # assert basic(29) == 39
    # assert invoke_an_inet() == 12
    inet: InetFunctionCompiler = getattr(use_references, '__inet__')
    inputs, outputs = inet.invocation(c.executor)
    send_value(outputs, c.to_key(0))
    c.optimize()
    while any('graft' in line for line in c.serialize_active_pairs()):
        c.process_next_interaction()
    assert c.serialize_active_pairs() == []

    assert use_references() == 30

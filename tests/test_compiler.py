from typing import Any

import pytest

from natsune.calculus import Calculus
from natsune.compiler import inet


@pytest.fixture(scope='function')
def c() -> Calculus:
    return Calculus()

@inet
def basic() -> Any:
    a = 10
    a = a + 1
    print(a)
    print(123)
    return a

def test_basic(c: Calculus) -> None:
    assert basic() == 11

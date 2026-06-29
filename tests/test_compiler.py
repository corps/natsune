from typing import Any

from natsune.compiler import inet

@inet
def basic() -> Any:
    print(123)
    print(456)
    return 10

def test_basic() -> None:
    assert basic() == 10

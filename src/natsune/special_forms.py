from typing import TYPE_CHECKING, TypeVar

_A = TypeVar("_A")
if TYPE_CHECKING:
    type Ref[A] = A
    type Inverse[A] = A
    Par = tuple
else:

    class Par(tuple): ...

    class Ref(Generic[_A]): ...

    class Inverse(Generic[_A]): ...

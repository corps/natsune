"""Cross-module callee: the caller links a Name-call to a marker defined
in another module; the driver's callee pass compiles it on demand."""

from natsune.inet import inet


@inet
def double(x: int) -> int:
    return x * 2

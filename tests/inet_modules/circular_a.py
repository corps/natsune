"""Circular-import pair, side A (CUTOVER.md §4a): A imports B, B imports
A. The inet calls cross the boundary through module attribute access, so
they resolve at first call — after both modules are fully loaded — which
is only possible because compilation is deferred past import."""

from natsune.inet import inet

import tests.inet_modules.circular_b as b


@inet
def ping(n: int) -> int:
    if n <= 0:
        return 0
    return b.pong(n - 1) + 1

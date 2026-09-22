"""Circular-import pair, side B — see circular_a."""

from natsune.inet import inet

import tests.inet_modules.circular_a as a


@inet
def pong(n: int) -> int:
    if n <= 0:
        return 1
    return a.ping(n - 1) + 1

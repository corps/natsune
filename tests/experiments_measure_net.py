"""Compare static artifacts: pair counts + serialized content length."""

import sys

from natsune.backend.connector import (
    ExpansionBuilder,
    FrozenExpansion,
    serialize_active_pairs,
)
from natsune.backend.protocol import LoweredUnit
from tests.backend.helpers import CapturingBackend, Program


def describe(artifact: LoweredUnit) -> tuple[int, int, int]:
    """(total pairs, serialized chars, number of Z erasures in serialization)"""
    builders = [
        a
        for a in artifact.agents
        if isinstance(a, (ExpansionBuilder, FrozenExpansion))
    ]
    total = len(artifact.main.active_pairs) + sum(
        len(b.active_pairs) for b in builders
    )
    text = "\n".join(serialize_active_pairs(list(artifact.main.active_pairs), {}))
    for b in builders:
        text += "\n" + "\n".join(serialize_active_pairs(list(b.active_pairs), {}))
    return total, len(text), text.count("Z")


def loop_orelse(n: int) -> int:
    total = 0
    for i in range(n):
        if i % 2 == 0:
            continue
        total += i
    else:
        total = -1
    return total


def if_chain(n: int) -> int:
    if n > 0:
        return 1
    else:
        if n < -5:
            return 2
    return 3


for program in (Program(loop_orelse), Program(if_chain)):
    artifact = program.lower(CapturingBackend())
    pairs, chars, erasures = describe(artifact)
    print(
        f"{program.name}: pairs={pairs} serialized_chars={chars} erasures={erasures}",
        file=sys.stderr,
    )

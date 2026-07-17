from typing import Literal

from .connector import Connector

__all__ = ["optimize"]

from natsune.interactions import (
    execute_commute_or_anihilate,
    execute_read_wire,
    execute_erasure,
    execute_interaction,
)

from natsune.ports import CombPort, WirePort, Erasure, Port


def optimize(
    c: Connector, active_pairs: list[tuple[Port, Port]], level: Literal[1, 2, 3] = 3
) -> None:
    potential_optimization = True

    # We pre-reduce wire port reads and combinator annihilations and cascade erasures where
    # we can.  Technicaly wire ports can lead to infinite loops -- but if this is the case anywhere
    # outside of a graft, it would be a compiler bug that we should look into.
    # Otherwise, we avoid other pre-reductions that could lead to infinite loops or change memory.
    # These are generally knowably safe and demonstrably reduce the graph.
    while potential_optimization:
        potential_optimization = False
        source_pairs = [*active_pairs]
        active_pairs.clear()
        for l, r in source_pairs:
            if (
                isinstance(l, CombPort)
                and isinstance(r, CombPort)
                and l.label == r.label
                and level >= 1
            ):
                potential_optimization = True
                execute_commute_or_anihilate(c, l, r)
            elif isinstance(l, WirePort) and (level >= 2 or l.wires[0].target):
                potential_optimization = True
                execute_read_wire(c, r, l)
            elif isinstance(r, WirePort) and (level >= 2 or r.wires[0].target):
                potential_optimization = True
                execute_read_wire(c, l, r)
            elif isinstance(l, Erasure) and isinstance(r, Erasure) and level >= 3:
                potential_optimization = True
                execute_erasure(c, l, r)
            else:
                for i, wire in enumerate(l.wires):
                    if isinstance(wire.target, WirePort):
                        l.wires[i] = wire.target.wires[0]
                        potential_optimization = True
                for i, wire in enumerate(r.wires):
                    if isinstance(wire.target, WirePort):
                        r.wires[i] = wire.target.wires[0]
                        potential_optimization = True
                c.connect(l, r)

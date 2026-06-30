from typing import Literal

from .connector import BufferingConnector, Connector

__all__ = ["optimize"]

from .interactions import (
    execute_commute_or_anihilate,
    execute_read_wire,
    execute_erasure,
)

from .ports import CombPort, WirePort, Erasure, Port


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
        new_active_pairs = []
        for l, r in active_pairs:
            if (
                isinstance(l, CombPort)
                and isinstance(r, CombPort)
                and l.label_eq(r)
                and level >= 2
            ):
                potential_optimization = True
                execute_commute_or_anihilate(c, l, r)
            elif isinstance(l, WirePort):
                potential_optimization = True
                execute_read_wire(c, r, l)
            elif isinstance(r, WirePort):
                potential_optimization = True
                execute_read_wire(c, l, r)
            elif isinstance(l, Erasure) or isinstance(r, Erasure) and level >= 3:
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
                new_active_pairs.append((l, r))

        active_pairs.clear()
        active_pairs.extend(new_active_pairs)

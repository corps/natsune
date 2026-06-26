from python.inet.connector import BufferingConnector

__all__ = ["optimize"]

from .interactions import (
    execute_commute_or_anihilate,
    execute_read_wire,
    execute_erasure,
)

from .ports import CombPort, WirePort, Erasure


def optimize(c: BufferingConnector) -> None:
    new_active_pairs = []
    potential_optimization = True

    # We pre-reduce wire port reads and combinator annihilations and cascade erasures where
    # we can.  Technicaly wire ports can lead to infinite loops -- but if this is the case anywhere
    # outside of a graft, it would be a compiler bug that we should look into.
    # Otherwise, we avoid other pre-reductions that could lead to infinite loops or change memory.
    # These are generally knowably safe and demonstrably reduce the graph.
    while potential_optimization:
        potential_optimization = False
        for l, r in c.active_pairs:
            if isinstance(l, CombPort) and isinstance(r, CombPort) and l.label_eq(r):
                potential_optimization = True
                execute_commute_or_anihilate(c, l, r)
            elif isinstance(l, WirePort) and l.wires[0].target is not None:
                potential_optimization = True
                execute_read_wire(c, r, l)
            elif isinstance(r, WirePort) and r.wires[0].target is not None:
                potential_optimization = True
                execute_read_wire(c, l, r)
            elif isinstance(l, Erasure) or isinstance(r, Erasure):
                potential_optimization = True
                execute_erasure(c, l, r)
            else:
                new_active_pairs.append((l, r))

        c.active_pairs = new_active_pairs

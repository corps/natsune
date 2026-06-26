from python.inet.executor import Executor
from .connector import Connector
import copy
import dataclasses

from .ports import (
    ExtMergeFuncPort,
    ValuePort,
    Erasure,
    Wire,
    ExtSplitFuncPort,
    CombPort,
    Port,
    WirePort,
    ForkPort,
    Graft,
)

__all__ = [
    "execute_ext_merge_func",
    "execute_ext_split_func",
    "execute_commute_or_anihilate",
    "execute_interaction",
    "execute_erasure",
    "execute_read_wire",
]


def execute_ext_merge_func(
    connector: Connector, l: ExtMergeFuncPort, r: ValuePort
) -> None:
    rhs = l.wires[0].target

    if isinstance(rhs, ValuePort):
        l.wires[0].target = None
        if l.swapped:
            try:
                result = ValuePort(l.fn(rhs.value, r.value))
            except Exception as e:
                result = Erasure(e)
        else:
            try:
                result = ValuePort(l.fn(r.value, rhs.value))
            except Exception as e:
                result = Erasure(e)
        connector.connect(l.wires[1], result)
        return

    swapped = dataclasses.replace(l, swapped=not l.swapped, wires=[Wire(r), l.wires[1]])
    connector.connect(l.wires[0], swapped)


def execute_ext_split_func(
    connector: Connector, l: ExtSplitFuncPort, r: ValuePort
) -> None:
    try:
        result = l.fn(r.value)
    except Exception as e:
        execute_erasure(connector, l, Erasure(e))
        return
    connector.connect(l.wires[0], ValuePort(result[0]))
    connector.connect(l.wires[1], ValuePort(result[1]))


def execute_commute_or_anihilate(
    connector: Connector,
    l: ExtSplitFuncPort | ExtMergeFuncPort | CombPort,
    r: ExtSplitFuncPort | ExtMergeFuncPort | CombPort,
) -> None:
    if l.label_eq(r):
        connector.connect(l.wires[0], r.wires[0])
        connector.connect(l.wires[1], r.wires[1])
        return

    l1 = dataclasses.replace(l, aux1=Wire(), aux2=Wire())
    l2 = dataclasses.replace(l, aux1=Wire(), aux2=Wire())
    r1 = dataclasses.replace(r, aux1=Wire(), aux2=Wire())
    r2 = dataclasses.replace(r, aux1=Wire(), aux2=Wire())

    connector.connect(l1.wires[0], r1.wires[1])
    connector.connect(l1.wires[1], r2.wires[1])
    connector.connect(l2.wires[0], r1.wires[0])
    connector.connect(l2.wires[1], r2.wires[0])

    connector.connect(l.wires[0], l1)
    connector.connect(l.wires[1], l2)
    connector.connect(r.wires[1], r1)
    connector.connect(r.wires[0], r2)


def execute_erasure(connector: Connector, l: Port, r: Port) -> None:
    for wire in l.wires:
        connector.annihilate(wire)
    for wire in r.wires:
        connector.annihilate(wire)


def execute_clone(connector: Connector, l: CombPort, r: ValuePort) -> None:
    connector.connect(l.wires[0], r)
    connector.connect(l.wires[1], copy.copy(r))


def execute_read_wire(connector: Connector, l: Port, r: WirePort) -> None:
    connector.connect(r.wires[0], l)


def execute_fork(connector: Connector, p: Port, l: ForkPort) -> None:
    new_p = copy.copy(p)

    for i in range(len(p.wires)):
        amb = ForkPort(l.fork)
        connector.connect(p.wires[i], amb)
        p.wires[i] = Wire()
        new_p.wires[i] = Wire()

        connector.connect(p.wires[i], amb.wires[0])
        connector.connect(new_p.wires[i], amb.wires[1])

    connector.connect(l.wires[0], p)
    l.fork.connect(l.wires[1], new_p)


def execute_interaction(executor: Executor, l: Port, r: Port) -> None:
    # WirePorts execute with highest priority as the wires themselves can be shared references,
    # and we don't want to thus copy via Graft behaviors.  They are "pass through" behaviors of
    # wires.
    if isinstance(l, WirePort):
        execute_read_wire(executor, r, l)
        return

    elif isinstance(r, WirePort):
        execute_read_wire(executor, l, r)
        return

    if isinstance(l, Graft):
        try:
            l.execute(executor, r, l.wires)
        except Exception as e:
            execute_erasure(executor, l, Erasure(e))
        return
    elif isinstance(r, Graft):
        try:
            r.execute(executor, l, r.wires)
        except Exception as e:
            execute_erasure(executor, r, Erasure(e))
        return

    if isinstance(r, Erasure) or isinstance(l, Erasure):
        execute_erasure(executor, l, r)
        return

    if isinstance(l, (ExtSplitFuncPort, ExtMergeFuncPort, CombPort)) and isinstance(
        r, (ExtSplitFuncPort, ExtMergeFuncPort, CombPort)
    ):
        execute_commute_or_anihilate(executor, l, r)
        return

    if isinstance(l, ForkPort):
        execute_fork(executor, r, l)
        return

    elif isinstance(r, ForkPort):
        execute_fork(executor, l, r)
        return

    if isinstance(l, ValuePort):
        x = l
        l = r
        r = x

    assert isinstance(r, ValuePort)
    if isinstance(l, ExtSplitFuncPort):
        execute_ext_split_func(executor, l, r)
        return
    elif isinstance(l, ExtMergeFuncPort):
        execute_ext_merge_func(executor, l, r)
        return
    elif isinstance(l, CombPort):
        execute_clone(executor, l, r)
        return
    elif isinstance(l, ValuePort):
        execute_erasure(executor, l, r)
        return

    assert False, f"unreachable: {l} {r}"

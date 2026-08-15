import copy
import dataclasses
from typing import TYPE_CHECKING

from natsune.ports import (
    CombPort,
    Erasure,
    ExtMergeFuncPort,
    ExtSplitFuncPort,
    Graft,
    Port,
    ValuePort,
    Wire,
    WirePort,
)

from .connector import Connector

if TYPE_CHECKING:
    from .connector import Connector

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
    l: CombPort,
    r: CombPort,
) -> None:
    if l.label == r.label:
        connector.connect(l.wires[0], r.wires[0])
        connector.connect(l.wires[1], r.wires[1])
        return

    l1 = dataclasses.replace(l, wires=[Wire(), Wire()])
    l2 = dataclasses.replace(l, wires=[Wire(), Wire()])
    r1 = dataclasses.replace(r, wires=[Wire(), Wire()])
    r2 = dataclasses.replace(r, wires=[Wire(), Wire()])

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
    connector.connect(
        l.wires[1], copy.copy(r) if l.label == "dup" else ValuePort(r.value)
    )


def execute_read_wire(connector: Connector, l: Port, r: WirePort) -> None:
    connector.connect(r.wires[0], l)


def execute_interaction(executor: Connector, l: Port, r: Port) -> None:
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
        if isinstance(r, Graft):
            raise ValueError("Cannot execute interactions between two Graft ports")
        l.execute(executor, r, l.wires)
        return
    elif isinstance(r, Graft):
        r.execute(executor, l, r.wires)
        return

    if isinstance(r, Erasure) or isinstance(l, Erasure):
        execute_erasure(executor, l, r)
        return

    if isinstance(l, (ExtSplitFuncPort, ExtMergeFuncPort)) and isinstance(
        r, (ExtSplitFuncPort, ExtMergeFuncPort)
    ):
        raise ValueError(
            "Cannot execute interactions between two ExtSplitFuncPort or ExtMergeFuncPort"
        )

    if isinstance(l, CombPort) and isinstance(r, CombPort):
        execute_commute_or_anihilate(executor, l, r)
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
        raise ValueError(
            "Cannot execute interactions between two ValuePorts: "
            + str(l)
            + ", "
            + str(r)
        )

    assert False, f"unreachable: {l} {r}"

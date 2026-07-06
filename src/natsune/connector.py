import dataclasses
import abc
import contextlib
import copy
from contextlib import AbstractContextManager
from typing import Generator, Iterator, Callable

from .ports import Port, Wire, WirePort, Target, CombPort, Erasure, Graft

__all__ = [
    "Connector",
    "BufferingConnector",
]


def connect_wire_to_port(connector: Connector, wire: Wire, port: Port) -> None:
    old = wire.target
    wire.target = port
    if old is not None:
        connector.connect_ports(old, port)


def connect_wire_to_wire(connector: Connector, l: Wire, r: Wire) -> None:
    connect_wire_to_port(connector, l, WirePort([r]))


def connect_to_target(connector: Connector, l: Port, r: Target) -> None:
    if isinstance(r, Port):
        connector.connect_ports(l, r)
        return
    connect_wire_to_port(connector, r, l)


class Connector(abc.ABC):
    @abc.abstractmethod
    def connect_ports(self, l: Port, r: Port) -> None: ...

    def connect(self, l: Target, r: Target) -> None:
        if isinstance(l, Port):
            connect_to_target(self, l, r)
            return
        elif isinstance(r, Port):
            connect_to_target(self, r, l)
            return
        connect_wire_to_wire(self, l, r)

    def duplicate(self, target: Target) -> tuple[Wire, Wire]:
        comb = CombPort("dup")
        self.connect(comb, target)
        return comb.wires[0], comb.wires[1]

    def tuplate(self, target: Target) -> tuple[Wire, Wire]:
        comb = CombPort("x")
        self.connect(comb, target)
        return comb.wires[0], comb.wires[1]

    def as_wire(self, port: Target) -> Wire:
        if isinstance(port, Wire):
            return port
        p, w = Wire.as_tautology()
        self.connect_ports(port, p)
        return w

    def annihilate(self, target: Target, erasure: Erasure | Port | None = None) -> None:
        self.connect(
            copy.copy(erasure) if isinstance(erasure, Erasure) else Erasure(), target
        )

    def sequenced_tuplate_from(
        self, target: Target
    ) -> AbstractContextManager[Iterator[Wire]]:
        return self.sequenced_from(target, self.tuplate)

    @contextlib.contextmanager
    def sequenced_from(
        self, target: Target, factory_function: Callable[[Target], tuple[Wire, Wire]]
    ) -> Generator[Iterator[Wire]]:
        open_end = self.as_wire(target)

        def iter() -> Iterator[Wire]:
            nonlocal open_end
            while True:
                next_wire, open_end = factory_function(open_end)
                yield next_wire

        yield iter()
        self.annihilate(open_end)


@dataclasses.dataclass(slots=True)
class BufferingConnector(Connector):
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    def connect_ports(self, l: Port, r: Port) -> None:
        self.active_pairs.append((l, r))

import abc
import contextlib
import copy
import dataclasses
from contextlib import AbstractContextManager
from functools import cached_property
from typing import Generator, Iterator, Callable, Sequence, Self, TYPE_CHECKING

from natsune.ports import Port, Wire, WirePort, Target, CombPort, Erasure

if TYPE_CHECKING:
    from natsune.adapters import Adapter
    from natsune.registers import (
        InterfaceRegister,
        FromInterfaceRegister,
        ToInterfaceRegister,
    )

__all__ = [
    "Connector",
    "ExpansionBuilder",
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
class ExpansionBuilder(Connector):
    input_adapter: Adapter
    output_adapter: Adapter
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    @cached_property
    def input_interface(self) -> FromInterfaceRegister:
        from natsune.registers import FromInterfaceRegister

        return FromInterfaceRegister(self.input_adapter, self)

    @cached_property
    def output_interface(self) -> ToInterfaceRegister:
        from natsune.registers import ToInterfaceRegister

        return ToInterfaceRegister(self.output_adapter, self)

    def connect_ports(self, l: Port, r: Port) -> None:
        self.active_pairs.append((l, r))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.output_interface.close()
        self.input_interface.close()

    def __call__(self, exec: Connector, port: Port, wires: Sequence[Wire], /) -> None:
        if isinstance(port, Erasure):
            for wire in wires:
                exec.annihilate(wire, port)
            return

        new_wire_identity: dict[Wire, Wire] = {}
        q: list[Port] = []
        pairs: list[tuple[Port, Port]] = []


        for l, r in self.active_pairs:
            ll = copy.copy(l)
            rr = copy.copy(r)
            pairs.append((ll, rr))
            q.append(ll)
            q.append(rr)

        return_port = copy.copy(self.output_interface.interface)
        exec.connect(return_port, wires[0])
        q.append(return_port)

        args_port = copy.copy(self.input_interface.interface)
        exec.connect(args_port, port)
        q.append(args_port)

        while q:
            head = q.pop()

            for i, wire in enumerate(head.wires):
                if wire in new_wire_identity:
                    wire = new_wire_identity[wire]
                    if wire.target:
                        raise AssertionError("Burp")
                else:
                    old_wire = wire
                    wire = copy.copy(wire)
                    new_wire_identity[old_wire] = wire
                    if wire.target:
                        q.append(wire.target)

                head.wires[i] = wire

        for l, r in pairs:
            exec.connect_ports(l, r)

    def __copy__(self) -> Self:
        return self

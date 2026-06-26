from .registers import FromRegister
import dataclasses
from enum import IntEnum
from functools import cached_property
from typing import (
    Protocol,
    Iterator,
    Sequence,
    MutableMapping,
    get_origin,
    get_args,
    Any,
)

from .connector import Connector
from .ports import Wire, WirePort, Target, Port, Erasure, CombPort
from .special_forms import Par, Ref, Inverse


class LinearWiringType(IntEnum):
    VALUE = 0
    REFERENCE = 1
    INVERSE = 2


type AdapterWiringType = LinearWiringType | tuple[AdapterWiringType, ...]


class Adapter(Protocol):
    def initialize(
        self, connector: Connector, initial: Wire | None = None
    ) -> WirePort: ...

    def close(self, target: Target, connector: Connector) -> None: ...

    def produce_com(self, taken: Wire, given: Wire, connector: Connector) -> Port: ...

    def unpack(self, target: Port, connector: Connector) -> Port: ...

    def __iter__(self) -> Iterator[Adapter]: ...

    def adapter_wiring_type(self) -> AdapterWiringType: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ValueAdapter(Adapter):
    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        return WirePort([initial or Wire(Erasure())])

    def close(self, target: Target, connector: Connector) -> None:
        connector.annihilate(target)

    def produce_com(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        left, right = connector.duplicate(taken)
        connector.connect(given, right)
        return WirePort([left])

    def unpack(self, target: Port, connector: Connector) -> Port:
        return target

    def __iter__(self) -> Iterator[Adapter]:
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return LinearWiringType.VALUE


@dataclasses.dataclass(slots=True, frozen=True)
class ParValueAdapter(Adapter):
    concurrent_items: Sequence[Adapter]

    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        if initial is not None:
            raise ValueError("Initial value not supported for par")
        x1, x2 = Wire.as_tautology()
        with connector.sequenced_tuplate_from(x2) as packing_iter:
            for adapter, wire in zip(self.concurrent_items, packing_iter):
                connector.connect(adapter.initialize(connector), wire)
        return x1

    def close(self, target: Target, connector: Connector) -> None:
        with connector.sequenced_tuplate_from(target) as packing_iter:
            for adapter, wire in zip(self.concurrent_items, packing_iter):
                adapter.close(wire, connector)

    def produce_com(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        x1, x2 = Wire.as_tautology()
        with (
            connector.sequenced_tuplate_from(x2) as packing_iter,
            connector.sequenced_tuplate_from(given) as given_iter,
            connector.sequenced_tuplate_from(taken) as taken_iter,
        ):
            for adapter, packing_wire, given_wire, taken_wire in zip(
                self.concurrent_items, packing_iter, given_iter, taken_iter
            ):
                connector.connect(
                    packing_wire, adapter.produce_com(taken_wire, given_wire, connector)
                )
        return x1

    def unpack(self, target: Port, connector: Connector) -> Port:
        raise SyntaxError("Implicit unpack for par unsupported")

    def __iter__(self) -> Iterator[Adapter]:
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return tuple(a.adapter_wiring_type() for a in self.concurrent_items)


# A version of par value adapter indicating that only one of the results should be set, while
# the others should be annihilated (not closed).
class AlternativesAdapter(ParValueAdapter): ...


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceAdapter(Adapter):
    inner: Adapter

    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        ref = CombPort("x")
        if initial:
            connector.connect(ref.wires[0], initial)
        else:
            connector.connect(ref.wires[0], Erasure())
        connector.connect(ref.wires[1], Erasure())
        return WirePort([connector.abstract_port(ref)])

    def close(self, target: Target, connector: Connector) -> None:
        incoming, outgoing = connector.tuplate(target)
        connector.connect(incoming, outgoing)

    def produce_com(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        taken_incoming, taken_outgoing = connector.tuplate(taken)
        given_incoming, given_outgoing = connector.tuplate(given)
        readout = CombPort("x")
        connector.connect(readout.wires[0], taken_incoming)
        connector.connect(readout.wires[1], given_incoming)
        connector.connect(taken_outgoing, given_outgoing)
        return readout

    def unpack(self, target: Port, connector: Connector) -> Port:
        taken_incoming, taken_outgoing = connector.tuplate(target)
        incoming1, incoming2 = connector.duplicate(taken_incoming)
        continuation = CombPort("x")
        connector.connect(continuation.wires[0], incoming1)
        connector.connect(continuation.wires[1], taken_outgoing)
        self.close(continuation, connector)
        return WirePort([incoming2])

    def __iter__(self) -> Iterator[Adapter]:
        yield from self.inner
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return LinearWiringType.REFERENCE


@dataclasses.dataclass(frozen=True, slots=True)
class InverseAdapter(Adapter):
    inner: Adapter

    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        inv = CombPort("x")
        connector.connect(inv.wires[0], inv.wires[1])
        return WirePort([connector.abstract_port(inv)])

    def close(self, target: Target, connector: Connector) -> None:
        connector.annihilate(target)

    def produce_com(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        taken_work, taken_solution = connector.tuplate(taken)
        given_work, given_solution = connector.tuplate(given)
        readout = CombPort("x")

        sol1, a = connector.duplicate(given_solution)
        sol2, sol3 = connector.duplicate(a)
        connector.connect(sol1, given_work)
        connector.connect(sol2, taken_solution)
        connector.connect(sol3, readout.wires[1])
        connector.connect(readout.wires[0], taken_work)
        return readout

    def unpack(self, target: Port, connector: Connector) -> Port:
        taken_incoming, taken_outgoing = connector.tuplate(target)
        incoming1, incoming2 = connector.duplicate(taken_incoming)
        continuation = CombPort("x")
        connector.connect(continuation.wires[0], incoming1)
        connector.connect(continuation.wires[1], taken_outgoing)
        self.close(continuation, connector)
        return WirePort([incoming2])

    def __iter__(self) -> Iterator[Adapter]:
        yield from self.inner
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return LinearWiringType.INVERSE


@dataclasses.dataclass(slots=True)
class Variables(MutableMapping[str, Adapter]):
    variables: dict[str, Adapter]

    @cached_property
    def adapter(self) -> ParValueAdapter:
        return ParValueAdapter(list(self.variables.values()))

    def __setitem__(self, key, value, /):
        self.variables[key] = value

    def __delitem__(self, key, /):
        raise NotImplementedError

    def __getitem__(self, key, /):
        return self.variables[key]

    def __len__(self):
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)


type TypeExpression = Any


def adapter_from_type(te: TypeExpression | None) -> Adapter:
    if te is None:
        return ValueAdapter()

    container = get_origin(te) or te
    args = get_args(te)

    if container is Par:
        if len(args) > 1:
            return ParValueAdapter([adapter_from_type(arg) for arg in args])
        else:
            return ValueAdapter()
    elif container is Ref:
        if len(args) == 1:
            return ReferenceAdapter(adapter_from_type(args[0]))
        return ReferenceAdapter(ValueAdapter())
    elif container is Inverse:
        if len(args) == 1:
            return InverseAdapter(adapter_from_type(args[0]))
        return InverseAdapter(ValueAdapter())
    return ValueAdapter()

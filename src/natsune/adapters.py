import dataclasses
from enum import IntEnum
from functools import cached_property
from typing import (
    Any,
    Iterator,
    MutableMapping,
    Protocol,
    Sequence,
    get_args,
    get_origin,
)

from natsune.connector import Connector
from natsune.ports import CombPort, Erasure, Port, Target, Wire, WirePort
from natsune.special_forms import Inverse, Par, Ref


class LinearWiringType(IntEnum):
    VALUE = 0
    REFERENCE = 1


type AdapterWiringType = LinearWiringType | tuple[AdapterWiringType, ...]


class Adapter(Protocol):
    def initialize(
        self, connector: Connector, initial: Wire | None = None
    ) -> WirePort: ...

    def close(self, target: Target, connector: Connector) -> None: ...

    def produce_egression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port: ...

    def produce_ingression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port: ...

    def unpack(self, target: Target, connector: Connector) -> Target: ...

    def repack(self, target: Target, connector: Connector) -> Target: ...

    def __iter__(self) -> Iterator[Adapter]: ...

    def adapter_wiring_type(self) -> AdapterWiringType: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ValueAdapter(Adapter):
    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        wp, w = Wire.as_tautology()
        connector.connect(w, initial or Erasure())
        return wp

    def close(self, target: Target, connector: Connector) -> None:
        connector.annihilate(target)

    def produce_egression(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        left, right = connector.duplicate(taken)
        connector.connect(given, right)
        return connector.as_port(left)

    def produce_ingression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port:
        connector.annihilate(taken)
        return connector.as_port(given)

    def unpack(self, target: Target, connector: Connector) -> Target:
        return target

    def repack(self, target: Target, connector: Connector) -> Target:
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

    def produce_egression(self, taken: Wire, given: Wire, connector: Connector) -> Port:
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
                    packing_wire,
                    adapter.produce_egression(taken_wire, given_wire, connector),
                )
        return x1

    def produce_ingression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port:
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
                    packing_wire,
                    adapter.produce_ingression(taken_wire, given_wire, connector),
                )
        return x1

    def unpack(self, target: Target, connector: Connector) -> Target:
        raise SyntaxError("Implicit unpack for par unsupported")

    def repack(self, target: Target, connector: Connector) -> Target:
        raise SyntaxError("Implicit repack for par unsupported")

    def __iter__(self) -> Iterator[Adapter]:
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return tuple(a.adapter_wiring_type() for a in self.concurrent_items)


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceAdapter(Adapter):
    inner: Adapter

    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        ref = CombPort("x")
        if initial:
            connector.connect(ref.wires[0], initial)
        else:
            connector.connect(ref.wires[0], Erasure())
        self.inner.close(ref.wires[1], connector)
        wp, w = Wire.as_tautology()
        connector.connect(w, ref)
        return wp

    def close(self, target: Target, connector: Connector) -> None:
        incoming, outgoing = connector.tuplate(target)
        connector.connect(incoming, outgoing)

    def produce_egression(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        taken_incoming, taken_outgoing = connector.tuplate(taken)
        given_incoming, given_outgoing = connector.tuplate(given)
        readout = CombPort("x")
        connector.connect(readout.wires[0], taken_incoming)
        connector.connect(readout.wires[1], given_incoming)
        connector.connect(taken_outgoing, given_outgoing)
        return readout

    def produce_ingression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port:
        taken_incoming, taken_outgoing = connector.tuplate(taken)
        given_incoming, given_outgoing = connector.tuplate(given)

        given_outgoing1, given_outgoing2 = connector.duplicate(given_outgoing, "share")
        self.inner.close(taken_incoming, connector)

        readout = CombPort("x")
        connector.connect(readout.wires[0], given_incoming)
        connector.connect(readout.wires[1], given_outgoing1)
        connector.connect(taken_outgoing, given_outgoing2)
        return readout

    def unpack(self, target: Target, connector: Connector) -> Target:
        taken_incoming, taken_outgoing = connector.tuplate(target)
        incoming1, incoming2 = connector.duplicate(taken_incoming, "share")
        connector.connect(incoming1, taken_outgoing)
        return incoming2

    def repack(self, target: Target, connector: Connector) -> Target:
        return self.initialize(connector, connector.as_wire(target))

    def __iter__(self) -> Iterator[Adapter]:
        yield from self.inner
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return LinearWiringType.REFERENCE


@dataclasses.dataclass(frozen=True, slots=True)
class InverseAdapter(Adapter):
    inner: Adapter

    def initialize(self, connector: Connector, initial: Wire | None = None) -> WirePort:
        wp = WirePort()
        self.inner.close(wp, connector)
        return wp

    def close(self, target: Target, connector: Connector) -> None:
        connector.annihilate(target)

    def produce_egression(self, taken: Wire, given: Wire, connector: Connector) -> Port:
        return self.inner.produce_egression(given, taken, connector)

    def produce_ingression(
        self, taken: Wire, given: Wire, connector: Connector
    ) -> Port:
        return self.inner.produce_ingression(given, taken, connector)

    def unpack(self, target: Target, connector: Connector) -> Target:
        return self.inner.unpack(target, connector)

    def repack(self, target: Target, connector: Connector) -> Target:
        return self.inner.repack(target, connector)

    def __iter__(self) -> Iterator[Adapter]:
        parts = list(self.inner)
        yield from parts[:-1]
        yield self

    def adapter_wiring_type(self) -> AdapterWiringType:
        return self.inner.adapter_wiring_type()


@dataclasses.dataclass
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

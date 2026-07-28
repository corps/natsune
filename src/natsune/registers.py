import dataclasses
from typing import Any, Callable, Iterable, Literal, Protocol, Self, Sequence, cast

from karakuri.inference.mapping_inference import MappingInference

from natsune.adapters import Adapter, ParValueAdapter, ReferenceAdapter, ValueAdapter
from natsune.connector import Connector
from natsune.ports import ConstantValuePort, Graft, Port, Target, Wire, WirePort

__all__ = [
    "FromRegister",
    "ToRegister",
    "InterfaceRegister",
    "FlowRegister",
    "as_to_register",
    "as_from_register",
    "send_values",
    "send_value",
    "parallelize_value",
    "join_to_registers",
    "join_from_registers",
    "borrow_registers",
    "serialize_values",
    "as_constant_register",
    "FromInterfaceRegister",
    "ToInterfaceRegister",
]

register_inferences: list[MappingInference] = []


# These represent public interface wrappers around the inner register concept
# Ideally, one does not interact directly with inner register methods, but uses "send_value" and "send_values"
# alongside as_from_register and as_to_register.
class FromRegister(Protocol):
    def close(self) -> None: ...
    @property
    def t(self) -> Literal["from"]: ...
    @property
    def adapter(self) -> Adapter: ...
    @property
    def connector(self) -> Connector: ...
    def split(self) -> Sequence[FromRegister]: ...
    def duplicate(self, label: str = "dup") -> tuple[FromRegister, FromRegister]: ...
    def __or__(self, other: FromRegister) -> FromRegister: ...
    def __and__(self, other: FromRegister) -> FromRegister: ...
    def trace(self, label: str) -> FromRegister: ...


class ToRegister(Protocol):
    def close(self) -> None: ...
    @property
    def t(self) -> Literal["to"]: ...
    @property
    def adapter(self) -> Adapter: ...
    @property
    def connector(self) -> Connector: ...
    def split(self) -> Sequence[ToRegister]: ...
    def invert(self) -> FromRegister: ...


def as_constant_register(value: Any, connector: Connector) -> FromRegister:
    return as_from_register(ConstantValuePort(value), ValueAdapter(), connector)


def as_from_register(target: Target, adapter: Adapter, c: Connector) -> FromRegister:
    return _FromRegister(target, adapter, c)


def as_to_register(target: Target, adapter: Adapter, c: Connector) -> ToRegister:
    return _ToRegister(target, adapter, c)


# Registers are one-time usage targets that control a port through an adapter.
@dataclasses.dataclass(slots=True)
class _FromRegister:
    port: Target
    adapter: Adapter
    connector: Connector
    t: Literal["from"] = "from"

    def close(self) -> None:
        self.connector.annihilate(self.port)

    def split(self) -> Sequence[FromRegister]:
        if isinstance(self.adapter, ParValueAdapter):
            result: list[_FromRegister] = []
            with self.connector.sequenced_tuplate_from(self.port) as parts_iter:
                for adapter, part in zip(self.adapter.concurrent_items, parts_iter):
                    result.append(
                        _FromRegister(
                            part,
                            adapter,
                            self.connector,
                        )
                    )
            return result
        return [self]

    def duplicate(self, label: str = "dup") -> tuple[FromRegister, FromRegister]:
        x1, x2 = self.connector.duplicate(self.port, label)
        return (
            _FromRegister(x1, self.adapter, self.connector),
            _FromRegister(x2, self.adapter, self.connector),
        )

    def __or__(self, other: FromRegister) -> FromRegister:
        from natsune.control_flow import WeakeningSelection

        with WeakeningSelection(other.adapter).invocation(self.connector) as selection:
            send_value(self, selection.port.readin())
            send_value(other, selection.wire.second_value.readin())
            return selection.wire.result.readout()

    def __and__(self, other: FromRegister) -> FromRegister:
        from natsune.control_flow import GatedSelection

        with GatedSelection(self.adapter, other.adapter).invocation(
            self.connector
        ) as selection:
            send_value(self, selection.port.readin())
            send_value(other, selection.wire.second_value.readin())
            return selection.wire.result.readout()

    def trace(self, label: str) -> FromRegister:
        from natsune.control_flow import Tracer

        g = Graft(Tracer(label), [Wire()])
        self.connector.connect(self.port, g)
        return _FromRegister(g.wires[0], self.adapter, self.connector)


@dataclasses.dataclass(slots=True)
class _ToRegister:
    port: Target
    adapter: Adapter
    connector: Connector
    t: Literal["to"] = "to"

    def set(self, p: Target) -> None:
        self.connector.connect(p, self.port)

    def close(self, initial: Wire | None = None) -> None:
        self.connector.annihilate(self.port)

    def split(self) -> Sequence[ToRegister]:
        if isinstance(self.adapter, ParValueAdapter):
            result: list[ToRegister] = []
            with self.connector.sequenced_tuplate_from(self.port) as parts_iter:
                for adapter, part in zip(self.adapter.concurrent_items, parts_iter):
                    result.append(
                        _ToRegister(
                            part,
                            adapter,
                            self.connector,
                        )
                    )
            return result
        return [self]

    def invert(self) -> FromRegister:
        return _FromRegister(self.port, self.adapter, self.connector)


@dataclasses.dataclass(slots=True)
class InterfaceRegister:
    adapter: Adapter
    connector: Connector

    # The type is non optional, but we fill in a valid value if none is provided
    interface: Wire = dataclasses.field(default=cast(Any, None))
    state: Port = dataclasses.field(default=cast(Any, None))

    def __post_init__(self):
        if self.interface is None or self.state is None:
            self.state, self.interface = Wire.as_tautology()

    def extend(self) -> tuple[Wire, Wire]:
        take, give = Wire(), Wire()
        self.connector.connect(take, self.state)
        self.state = WirePort([give])
        return take, give

    def interface_readin(self) -> ToRegister:
        return _ToRegister(self.interface, self.adapter, self.connector)

    def split(self) -> Sequence[Self]:
        if isinstance(self.adapter, ParValueAdapter):
            taken, given = self.extend()
            self.connector.annihilate(given)
            result = [
                dataclasses.replace(
                    self,
                    adapter=adapter,
                    interface=cast(Any, None),
                    state=cast(Any, None),
                )
                for adapter in self.adapter.concurrent_items
            ]
            send_values(
                _FromRegister(taken, self.adapter, self.connector).split(),
                [interface.interface_readin() for interface in result],
            )
            return result
        return [self]

    def close(self) -> None:
        self.connector.annihilate(self.state)


class FromInterfaceRegister(InterfaceRegister):
    def readout(self) -> FromRegister:
        taken, given = self.extend()
        self.connector.annihilate(given)
        return _FromRegister(taken, self.adapter, self.connector)


class ToInterfaceRegister(InterfaceRegister):
    def readin(self) -> ToRegister:
        taken, given = self.extend()
        self.connector.annihilate(given)
        return _ToRegister(taken, self.adapter, self.connector)


# Unlike all other registers, a flow register supports the idea of "extension" and thus can be read out
# or readin multiple times, producing an extension (sharing) for each.
class FlowRegister(FromInterfaceRegister, ToInterfaceRegister):
    def readout(self) -> FromRegister:
        taken, given = self.extend()
        return _FromRegister(
            (self.adapter.produce_egression(taken, given, self.connector)),
            self.adapter,
            self.connector,
        )

    def readin(self) -> ToRegister:
        taken, given = self.extend()
        return _ToRegister(
            self.adapter.produce_ingression(taken, given, self.connector),
            self.adapter,
            self.connector,
        )

    def close(self) -> None:
        self.adapter.close(self.state, self.connector)


def send_values(
    from_registers: Sequence[FromRegister], to_registers: Sequence[ToRegister]
) -> None:
    if len(from_registers) != len(to_registers):
        if len(from_registers) == 1 and len(to_registers) > 1:
            source_register = from_registers[0]
            assert isinstance(source_register, _FromRegister)
            input_register, from_registers = parallelize_value(
                source_register.connector, len(to_registers)
            )
            send_value(source_register, input_register)
        elif len(to_registers) == 1 and len(from_registers) > 1:
            destination_register = to_registers[0]
            assert isinstance(destination_register, _ToRegister)
            to_registers, output_register = serialize_values(
                destination_register.connector, len(from_registers)
            )
            send_value(output_register, destination_register)
        else:
            raise SyntaxError(
                f"Cannot commute values because they have incompatible shapes."
            )

    for from_register, to_register in zip(from_registers, to_registers):
        send_value(from_register, to_register)


def send_value(from_register: FromRegister, to_register: ToRegister) -> None:
    assert isinstance(from_register, _FromRegister)
    assert isinstance(to_register, _ToRegister)
    assert (
        from_register.connector is to_register.connector
    ), f"Registers {from_register} and {to_register} are not native to the same connector"

    from_parts = list(from_register.adapter)
    to_parts = list(to_register.adapter)

    i = -1
    for i in range(min(len(from_parts), len(to_parts))):
        if from_parts[i].adapter_wiring_type() != to_parts[i].adapter_wiring_type():
            break
    else:
        i += 1

    if i == 0:
        if not isinstance(from_register.adapter, ParValueAdapter) and not isinstance(
            to_register.adapter, ParValueAdapter
        ):
            raise SyntaxError(
                f"Incompatible adapters: {to_register.adapter} and {from_register.adapter}"
            )
        send_values(from_register.split(), to_register.split())
        return

    readout = from_register.port

    j = len(from_parts)
    while j > i:
        readout = from_parts[j - 1].unpack(readout, from_register.connector)
        j -= 1

    while j < len(to_parts):
        readout = to_parts[j].repack(readout, to_register.connector)
        j += 1

    to_register.set(readout)


def unroll(v: Any) -> tuple[Any, Any]:
    return next(v), v


def join(acc: tuple, n: Any) -> tuple:
    return acc + (n,)


def parallelize_value(
    connector: Connector, count: int
) -> tuple[ToRegister, Sequence[FromRegister]]:
    from .invocations import filter_invocation

    result: list[FromRegister] = []

    value_in, cur_iter = filter_invocation(iter, connector)

    for _ in range(count):
        next_value, cur_iter = fold_split(unroll, cur_iter)
        result.append(next_value)
    cur_iter.close()
    return value_in, result


def serialize_values(
    connector: Connector, count: int
) -> tuple[Sequence[ToRegister], FromRegister]:
    result: list[ToRegister] = []

    cur_acc = _FromRegister(ConstantValuePort(()), ValueAdapter(), connector)

    for _ in range(count):
        cur_acc, next_n = fold_merge(join, cur_acc)
        result.append(next_n)

    return result, cur_acc


def join_to_registers(
    registers: Sequence[ToRegister], connector: Connector
) -> ToRegister:
    par_adapter = ParValueAdapter([register.adapter for register in registers])
    x1, x2 = Wire.as_tautology()
    from_register = as_from_register(x1, par_adapter, connector)
    send_values(from_register.split(), registers)

    return as_to_register(
        x2,
        par_adapter,
        connector,
    )


def join_from_registers(
    registers: Sequence[FromRegister], connector: Connector
) -> FromRegister:
    par_adapter = ParValueAdapter([register.adapter for register in registers])
    x1, x2 = Wire.as_tautology()
    to_register = as_to_register(x1, par_adapter, connector)
    send_values(registers, to_register.split())

    return as_from_register(
        x2,
        par_adapter,
        connector,
    )


def fold_split(
    fn: Callable[[Any], tuple[Any, Any]], acc: FromRegister
) -> tuple[FromRegister, FromRegister]:
    from .invocations import split_invocation

    assert isinstance(acc, _FromRegister)
    a, result = split_invocation(fn, acc.connector)
    assert isinstance(a, _ToRegister)
    a.set(acc.port)
    return result


def fold_merge(
    fn: Callable[[Any, Any], Any], acc: FromRegister
) -> tuple[FromRegister, ToRegister]:
    from .invocations import merge_invocation

    assert isinstance(acc, _FromRegister)
    (a, b), c = merge_invocation(fn, acc.connector)
    assert isinstance(a, _ToRegister)
    a.set(acc.port)
    return c, b


def borrow_registers(
    registers: list[FromRegister], finished_from: FromRegister
) -> list[FromRegister]:
    from natsune.control_flow import CloseAfterContingent, MergeInputTo, MergeOutputInto
    from natsune.invocations import expansion_invocation

    if not registers:
        return registers

    connector = registers[0].connector

    held_context = [
        FlowRegister(ReferenceAdapter(ValueAdapter()), connector) for _ in registers
    ]

    # "cast" values into a reference
    send_values(registers, [r.readin() for r in held_context])

    value_readout = [r.readout() for r in held_context]

    # Implicit close here as we transfer the state directly
    joined_content = join_from_registers(
        [as_from_register(r.state, r.adapter, r.connector) for r in held_context],
        connector,
    )

    with expansion_invocation(
        CloseAfterContingent(finished_from.adapter, joined_content.adapter),
        connector,
        MergeInputTo,
        MergeOutputInto,
    ) as invocation:
        send_value(finished_from, invocation.port.readin())
        send_value(joined_content, invocation.wire.second_value.readin())
        invocation.wire.result.readout().close()

    return value_readout

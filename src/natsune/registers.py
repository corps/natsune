import copy

from .executor import Executor
import dataclasses
from typing import Sequence, Protocol, Any, Literal, Callable

from .adapters import Adapter, ParValueAdapter, ValueAdapter, AlternativesAdapter
from .connector import Connector
from .ports import Port, WirePort, Target, Wire, ConstantValuePort

__all__ = [
    "FromRegister",
    "ToRegister",
    "InterfaceRegister",
    "FlowRegister",
    "as_to_register",
    "as_from_register",
    "send_values",
    "send_value",
]


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
    def invert(self) -> ToRegister: ...
    def duplicate(self) -> tuple[FromRegister, FromRegister]: ...


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


def as_from_register(port: Port, adapter: Adapter, c: Connector) -> FromRegister:
    return _FromRegister(port, adapter, c)


def as_to_register(port: Port, adapter: Adapter, c: Connector) -> ToRegister:
    return _ToRegister(port, adapter, c)


# Registers are one-time usage targets that control a port through an adapter.
@dataclasses.dataclass(slots=True, frozen=True)
class _FromRegister:
    port: Port
    adapter: Adapter
    connector: Connector
    alternative: bool = False
    t: Literal["from"] = "from"

    def close(self) -> None:
        if self.alternative:
            self.connector.annihilate(self.port)
        else:
            self.adapter.close(self.port, self.connector)

    def split(self) -> Sequence[FromRegister]:
        if isinstance(self.adapter, ParValueAdapter):
            result: list[_FromRegister] = []
            with self.connector.sequenced_tuplate_from(self.port) as parts_iter:
                for adapter, part in zip(self.adapter.concurrent_items, parts_iter):
                    result.append(
                        _FromRegister(
                            WirePort([part]),
                            adapter,
                            self.connector,
                            isinstance(self.adapter, AlternativesAdapter),
                        )
                    )
            return result
        return [self]

    def invert(self) -> ToRegister:
        return _ToRegister(self.port, self.adapter, self.connector)

    def duplicate(self) -> tuple[FromRegister, FromRegister]:
        x1, x2 = self.connector.duplicate(self.port)
        return (
            _FromRegister(WirePort([x1]), self.adapter, self.connector),
            _FromRegister(WirePort([x2]), self.adapter, self.connector),
        )


@dataclasses.dataclass(slots=True, frozen=True)
class _ToRegister:
    port: Port
    adapter: Adapter
    connector: Connector
    alternative: bool = False
    t: Literal["to"] = "to"

    def set(self, p: Target) -> None:
        self.connector.connect(p, self.port)

    def close(self, initial: Wire | None = None) -> None:
        if self.alternative:
            self.connector.annihilate(self.port)
        else:
            self.set(self.adapter.initialize(self.connector, initial))

    def split(self) -> Sequence[ToRegister]:
        if isinstance(self.adapter, ParValueAdapter):
            x1, x2 = Wire.as_tautology()
            self.set(x1)
            result: list[ToRegister] = []
            with self.connector.sequenced_tuplate_from(x2) as parts_iter:
                for adapter, part in zip(self.adapter.concurrent_items, parts_iter):
                    result.append(
                        _ToRegister(
                            WirePort([part]),
                            adapter,
                            self.connector,
                            isinstance(self.adapter, AlternativesAdapter),
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
    alternative: bool = False

    interface: WirePort = dataclasses.field(init=False)
    state: Port = dataclasses.field(init=False)

    def __post_init__(self):
        self.interface, self.state = Wire.as_interface()

    def extend(self) -> tuple[Wire, Wire]:
        take, give = Wire(), Wire()
        self.connector.connect(take, self.state)
        self.state = WirePort([give])
        return take, give

    def interface_readin(self) -> ToRegister:
        return _ToRegister(self.interface, self.adapter, self.connector)

    def readout(self, owned: bool) -> FromRegister:
        taken, given = self.extend()
        self.connector.annihilate(given)
        return _FromRegister(WirePort([taken]), self.adapter, self.connector)

    def readin(self) -> ToRegister:
        taken, given = self.extend()
        self.connector.annihilate(given)
        return _ToRegister(WirePort([taken]), self.adapter, self.connector)

    def split(self, owned: bool) -> Sequence[InterfaceRegister]:
        if isinstance(self.adapter, ParValueAdapter):
            out = self.readout(owned)
            result = [
                InterfaceRegister(
                    adapter,
                    self.connector,
                    isinstance(self.adapter, AlternativesAdapter),
                )
                for adapter in self.adapter.concurrent_items
            ]
            send_values(
                out.split(), [interface.interface_readin() for interface in result]
            )
            return result
        return [self]

    def close(self) -> None:
        if self.alternative:
            self.connector.annihilate(self.state)
        else:
            self.adapter.close(self.state, self.connector)

    @classmethod
    def from_to_register(cls, to_reg: ToRegister) -> InterfaceRegister:
        result = InterfaceRegister(to_reg.adapter, to_reg.connector)
        assert isinstance(to_reg, _ToRegister)
        to_reg.connector.connect(to_reg.port, result.interface)
        return result

    @classmethod
    def from_from_register(cls, from_reg: FromRegister) -> InterfaceRegister:
        result = InterfaceRegister(from_reg.adapter, from_reg.connector)
        assert isinstance(from_reg, _FromRegister)
        from_reg.connector.connect(from_reg.port, result.interface)
        return result


# Unlike all other registers, a flow register supports the idea of "extension" and thus can be read out
# or readin multiple times, producing an extension (sharing) for each.
class FlowRegister(InterfaceRegister):
    def readout(self, owned: bool) -> FromRegister:
        taken, given = self.extend()
        return _FromRegister(
            (
                WirePort([taken])
                if owned
                else self.adapter.produce_com(taken, given, self.connector)
            ),
            self.adapter,
            self.connector,
        )

    def readin(self) -> ToRegister:
        taken, given = self.extend()
        self.adapter.close(taken, self.connector)
        return _ToRegister(WirePort([given]), self.adapter, self.connector)

    def is_assigned(self) -> bool:
        return self.state.wires[0] is not self.interface.wires[0]


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

    if i == -1:
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
        readout = to_parts[j].initialize(
            to_register.connector, to_register.connector.abstract_port(readout)
        )
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

import dataclasses
from contextlib import AbstractContextManager
from typing import (
    Any,
    Callable,
    Iterator,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from karakuri.annotations import Annotation

from natsune.adapters import VA, Adapter
from natsune.connector import Connector
from natsune.ports import (
    Erasure,
    Expansion,
    ExtMergeFuncPort,
    ExtSplitFuncPort,
    Graft,
    Port,
    Wire,
)
from natsune.registers import (
    FromInterfaceRegister,
    FromRegister,
    InterfaceRegister,
    ToInterfaceRegister,
    ToRegister,
    as_constant_register,
    as_from_register,
    as_to_register,
    send_value,
)


class ExpansionWithAdapters(Expansion, Protocol):
    @property
    def input_adapter(self) -> Adapter: ...
    @property
    def output_adapter(self) -> Adapter: ...


def expansion_invocation[P, W](
    expansion: ExpansionWithAdapters,
    connector: Connector,
    p: type[P],
    w: type[W],
) -> closer[Invocation[P, W]]:
    graft = Graft(expansion, [Wire()])

    inputs = ToInterfaceRegister(expansion.input_adapter, connector)
    outputs = FromInterfaceRegister(expansion.output_adapter, connector)

    connector.connect(graft, inputs.interface)
    connector.connect(graft.wires[0], outputs.interface)

    return closer(Invocation(pack_into(inputs, p), pack_from(outputs, w)))


def unpack_wires[W](
    expansion: ExpansionWithAdapters, wires: Sequence[Wire], exec: Connector, w: type[W]
) -> closer[W]:
    interface = ToInterfaceRegister(expansion.output_adapter, exec)
    exec.connect(wires[0], interface.interface)
    return closer(pack_into(interface, w))


def unpack_port_and_wires[P, W](
    expansion: ExpansionWithAdapters,
    port: Port,
    wires: Sequence[Wire],
    exec: Connector,
    p: type[P],
    w: type[W],
) -> closer[Invocation[P, W]]:
    inputs = FromInterfaceRegister(expansion.input_adapter, exec)
    exec.connect(port, inputs.interface)

    outputs = ToInterfaceRegister(expansion.output_adapter, exec)
    exec.connect(wires[0], outputs.interface)
    return closer(Invocation(pack_from(inputs, p), pack_into(outputs, w)))


def merge_invocation(
    fn: Callable[[Any, Any], Any], connector: Connector
) -> tuple[tuple[ToRegister, ToRegister], FromRegister]:
    port = ExtMergeFuncPort(fn)
    return (
        as_to_register(port, VA, connector),
        as_to_register(port.wires[0], VA, connector),
    ), as_from_register(port.wires[1], VA, connector)


def split_invocation(
    fn: Callable[[Any], tuple[Any, Any]], connector: Connector
) -> tuple[ToRegister, tuple[FromRegister, FromRegister]]:
    port = ExtSplitFuncPort(fn)
    return as_to_register(port, VA, connector), (
        as_from_register(port.wires[0], VA, connector),
        as_from_register(port.wires[1], VA, connector),
    )


def filter_invocation(
    fn: Callable[[Any], Any], connector: Connector
) -> tuple[ToRegister, FromRegister]:
    (a, b), c = merge_invocation(lambda x, _: fn(x), connector)
    send_value(as_constant_register(None, connector), b)
    return a, c


@dataclasses.dataclass(frozen=True, slots=True)
class catch(Expansion):
    handler: Callable[[Erasure], None]

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            self.handler(port)


def send_parameters(
    invocation: tuple[Sequence[ToRegister], FromRegister],
    values: Sequence[FromRegister],
) -> FromRegister:
    args, result = invocation
    assert len(args) == len(values)
    for arg, value in zip(args, values):
        send_value(value, arg)
    return result


def send_parameter(
    invocation: tuple[ToRegister, FromRegister],
    values: FromRegister,
) -> FromRegister:
    args, result = invocation
    send_value(values, args)
    return result


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None:
        pass


assert issubclass(ToRegister, Closeable)
assert issubclass(FromRegister, Closeable)
assert issubclass(InterfaceRegister, Closeable)


class closer[T](AbstractContextManager[T]):
    def __init__(self, o: T) -> None:
        assert isinstance(o, Closeable) or dataclasses.is_dataclass(o)
        self.o = o

    def __enter__(self) -> T:
        return self.o

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        for closeable in self:
            closeable.close()

    def __iter__(self) -> Iterator[Closeable]:
        if isinstance(self.o, Closeable):
            yield self.o
            return

        for field in dataclasses.fields(self.o):
            if isinstance(field.type, Closeable):
                yield getattr(self.o, field.name)


@dataclasses.dataclass(slots=True, frozen=True)
class Invocation[P, W]:
    port: P
    wire: W

    def close(self) -> None:
        closer(self.port).close()
        closer(self.wire).close()


def pack_into[T](to_register: ToInterfaceRegister, struct: type[T]) -> T:
    method = getattr(struct, "pack_into", None)
    if method is not None:
        return method(to_register)
    return _pack_into(to_register, struct)


def _pack_into[T](
    to_register: ToInterfaceRegister, struct: type[T], serialize: bool = False
) -> T:
    if issubclass(struct, ToInterfaceRegister):
        return cast(T, to_register)
    elif issubclass(struct, FromInterfaceRegister):
        return cast(
            T,
            FromInterfaceRegister(
                to_register.adapter,
                to_register.connector,
                to_register.interface,
                to_register.state,
            ),
        )

    assert dataclasses.is_dataclass(struct)
    args: dict = {}

    for field, register in zip(
        dataclasses.fields(struct), to_register.split(serialize=serialize)
    ):
        if Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(ToInterfaceRegister):
            field_to: ToInterfaceRegister = register
            args[field.name] = field_to
        elif Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(FromInterfaceRegister):
            field_from: FromInterfaceRegister = FromInterfaceRegister(
                register.adapter,
                register.connector,
                register.interface,
                register.state,
            )
            args[field.name] = field_from
        else:
            args[field.name] = pack_into(
                register, Annotation.from_type_expression(field.type).source
            )

    return struct(**args)


def pack_from[T](from_register: FromInterfaceRegister, struct: type[T]) -> T:
    method = getattr(struct, "pack_from", None)
    if method is not None:
        return method(from_register)
    return _pack_from(from_register, struct)


def _pack_from[T](from_register: FromInterfaceRegister, struct: type[T]) -> T:
    if issubclass(struct, ToInterfaceRegister):
        return cast(
            T,
            ToInterfaceRegister(
                from_register.adapter,
                from_register.connector,
                from_register.interface,
                from_register.state,
            ),
        )
    elif issubclass(struct, FromInterfaceRegister):
        return cast(T, from_register)

    assert dataclasses.is_dataclass(struct)
    args: dict = {}

    for field, register in zip(dataclasses.fields(struct), from_register.split()):
        if Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(ToInterfaceRegister):
            field_to: ToInterfaceRegister = ToInterfaceRegister(
                register.adapter,
                register.connector,
                register.interface,
                register.state,
            )
            args[field.name] = field_to
        elif Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(FromInterfaceRegister):
            field_from: FromInterfaceRegister = register
            args[field.name] = field_from
        else:
            args[field.name] = pack_from(
                register, Annotation.from_type_expression(field.type).source
            )

    return struct(**args)

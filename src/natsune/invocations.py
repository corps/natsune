import dataclasses
import sys
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterator, Protocol, Sequence, cast, runtime_checkable

from karakuri.annotations import Annotation
from karakuri.call_mapping import CallMapping
from karakuri.codegen_buffer import generate
from karakuri.concrete_typing import DataclassTyping

from natsune.adapters import Adapter, ValueAdapter
from natsune.connector import Connector
from natsune.executor import Executor
from natsune.ports import (
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
    expansion: ExpansionWithAdapters, wires: Sequence[Wire], exec: Executor, w: type[W]
) -> closer[W]:
    interface = ToInterfaceRegister(expansion.output_adapter, exec)
    exec.connect(wires[0], interface.interface)
    return closer(pack_into(interface, w))


def unpack_port_and_wires[P, W](
    expansion: ExpansionWithAdapters,
    port: Port,
    wires: Sequence[Wire],
    exec: Executor,
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
        as_to_register(port, ValueAdapter(), connector),
        as_to_register(port.wires[0], ValueAdapter(), connector),
    ), as_from_register(port.wires[1], ValueAdapter(), connector)


def split_invocation(
    fn: Callable[[Any], tuple[Any, Any]], connector: Connector
) -> tuple[ToRegister, tuple[FromRegister, FromRegister]]:
    port = ExtSplitFuncPort(fn)
    return as_to_register(port, ValueAdapter(), connector), (
        as_from_register(port.wires[0], ValueAdapter(), connector),
        as_from_register(port.wires[1], ValueAdapter(), connector),
    )


def filter_invocation(
    fn: Callable[[Any], Any], connector: Connector
) -> tuple[ToRegister, FromRegister]:
    (a, b), c = merge_invocation(lambda x, _: fn(x), connector)
    send_value(as_constant_register(None, connector), b)
    return a, c


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


class LHS: ...


class RHS: ...


def _map_into(f: tuple[str, Annotation]) -> tuple[str, Annotation]:
    name, annotation = f
    if issubclass(annotation.source, LHS):
        return name, Annotation.from_type_expression(ToInterfaceRegister)
    if issubclass(annotation.source, RHS):
        return name, Annotation.from_type_expression(FromInterfaceRegister)
    raise TypeError(f"Unexpected type {annotation.source}")


def _map_from(f: tuple[str, Annotation]) -> tuple[str, Annotation]:
    name, annotation = f
    if issubclass(annotation.source, LHS):
        return name, Annotation.from_type_expression(FromInterfaceRegister)
    if issubclass(annotation.source, RHS):
        return name, Annotation.from_type_expression(ToInterfaceRegister)
    raise TypeError(f"Unexpected type {annotation.source}")


def pack_into[T](to_register: ToInterfaceRegister, struct: type[T]) -> T:
    if issubclass(struct, ToInterfaceRegister):
        return cast(T, to_register)
    elif issubclass(struct, FromInterfaceRegister):
        return cast(T, to_register.invert())

    assert dataclasses.is_dataclass(struct)
    args: dict = {}

    for field, register in zip(dataclasses.fields(struct), to_register.split()):
        if Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(ToInterfaceRegister):
            field_to: ToInterfaceRegister = register
            args[field.name] = field_to
        elif Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(FromInterfaceRegister):
            field_from: FromInterfaceRegister = register.invert()
            args[field.name] = field_from
        else:
            args[field.name] = pack_into(
                register, Annotation.from_type_expression(field.type).source
            )

    return struct(**args)


def pack_from[T](from_register: FromInterfaceRegister, struct: type[T]) -> T:
    if issubclass(struct, ToInterfaceRegister):
        return cast(T, from_register.invert())
    elif issubclass(struct, FromInterfaceRegister):
        return cast(T, from_register)

    assert dataclasses.is_dataclass(struct)
    args: dict = {}

    for field, register in zip(dataclasses.fields(struct), from_register.split()):
        if Annotation.from_type_expression(
            field.type
        ) <= Annotation.from_type_expression(ToInterfaceRegister):
            field_to: ToInterfaceRegister = register.invert()
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


def generate_register_pair_types(t: type):
    mapping = CallMapping.maybe_from_structure(t)
    assert mapping

    generate(
        f"{t.__name__}Into",
        DataclassTyping(
            parameters=mapping.parameters.non_variadic_parameters.map(_map_into)
        ),
        sys._getframe(1).f_globals,
    )

    generate(
        f"{t.__name__}From",
        DataclassTyping(
            parameters=mapping.parameters.non_variadic_parameters.map(_map_from)
        ),
        sys._getframe(1).f_globals,
    )

from .executor import Executor
from typing import Protocol, Callable, Any, Sequence, overload

from .adapters import Adapter, ValueAdapter, ParValueAdapter
from .connector import Connector
from .ports import (
    Graft,
    Wire,
    WirePort,
    Expansion,
    ExtMergeFuncPort,
    ExtSplitFuncPort,
    Port,
)
from .registers import (
    ToRegister,
    FromRegister,
    as_to_register,
    as_from_register,
    send_value,
    InterfaceRegister,
)


class ExpansionWithAdapters(Expansion, Protocol):
    @property
    def input_adapter(self) -> Adapter: ...
    @property
    def output_adapter(self) -> Adapter: ...


def expansion_invocation(
    expansion: ExpansionWithAdapters, connector: Connector
) -> tuple[InterfaceRegister, InterfaceRegister]:
    graft = Graft(expansion, [Wire()])

    inputs = InterfaceRegister(expansion.input_adapter, connector)
    outputs = InterfaceRegister(expansion.output_adapter, connector)

    send_value(
        as_from_register(graft, expansion.input_adapter, connector),
        inputs.interface_readin(),
    )
    send_value(
        as_from_register(
            WirePort([graft.wires[0]]), expansion.output_adapter, connector
        ),
        outputs.interface_readin(),
    )

    return (inputs, outputs)


def unpack_wires(
    expansion: ExpansionWithAdapters, wires: Sequence[Wire], exec: Executor
) -> InterfaceRegister:
    interface = InterfaceRegister(expansion.output_adapter, exec)
    exec.connect(wires[0], interface.interface)
    return interface


def unpack_port_and_wires(
    expansion: ExpansionWithAdapters, port: Port, wires: Sequence[Wire], exec: Executor
) -> tuple[InterfaceRegister, InterfaceRegister]:
    interface = InterfaceRegister(expansion.input_adapter, exec)
    exec.connect(port, interface.interface)
    return interface, unpack_wires(expansion, wires, exec)


def merge_invocation(
    fn: Callable[[Any, Any], Any], connector: Connector
) -> tuple[tuple[ToRegister, ToRegister], FromRegister]:
    port = ExtMergeFuncPort(fn)
    return (
        as_to_register(port, ValueAdapter(), connector),
        as_to_register(WirePort([port.wires[0]]), ValueAdapter(), connector),
    ), as_from_register(WirePort([port.wires[1]]), ValueAdapter(), connector)


def split_invocation(
    fn: Callable[[Any], tuple[Any, Any]], connector: Connector
) -> tuple[ToRegister, tuple[FromRegister, FromRegister]]:
    port = ExtSplitFuncPort(fn)
    return as_to_register(port, ValueAdapter(), connector), (
        as_from_register(WirePort([port.wires[0]]), ValueAdapter(), connector),
        as_from_register(WirePort([port.wires[1]]), ValueAdapter(), connector),
    )


def filter_invocation(
    fn: Callable[[Any], Any], connector: Connector
) -> tuple[ToRegister, FromRegister]:
    (a, b), c = merge_invocation(lambda x, _: fn(x), connector)
    b.close()
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

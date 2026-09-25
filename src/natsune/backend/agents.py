from typing import Any, Iterator, Sequence

from natsune.backend.connector import Connector, ExpansionBuilder, FrozenExpansion
from natsune.backend.control_flow import (
    CloseAfterContingent,
    ConcurrentValueMerge,
    FlowControlInto,
    FlowInputInto,
    IfThenElse,
    IfThenElseStatement,
    Loop,
    SerialAnd,
    SerialOr,
)
from natsune.backend.invocations import Invocation, closer, pack_from, pack_into
from natsune.backend.registers import (
    FromInterfaceRegister,
    FromRegister,
    ToInterfaceRegister,
    ToRegister,
)
from natsune.first_order.adapters import Adapter
from natsune.first_order.ports import Graft, Port, Wire

type AgentImpl = (
    ExpansionBuilder
    | FrozenExpansion
    | PromiseExpansion
    | IfThenElseStatement
    | IfThenElse
    | Loop
    | SerialOr
    | SerialAnd
    | CloseAfterContingent
    | ConcurrentValueMerge
)


class PromiseExpansion:
    def __init__(
        self, name: str, input_adapter: Adapter, output_adapter: Adapter
    ) -> None:
        self.name = name
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.target: FrozenExpansion | None = None

    def __copy__(self) -> "PromiseExpansion":
        return self

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        assert (
            self.target is not None
        ), f"net for {self.name} used before compilation finished"
        self.target(executor, port, wires)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def try_iter(i: Iterator[Any]) -> tuple[Any, bool]:
    try:
        return next(i), True
    except StopIteration:
        return None, False


def callee_invocation(
    agent: FrozenExpansion | ExpansionBuilder | PromiseExpansion,
    arity: int,
    connector: Connector,
) -> tuple[Sequence[ToRegister], FromRegister]:
    input_adapter = agent.input_adapter
    output_adapter = agent.output_adapter
    expansion = agent

    graft = Graft(
        expansion,
        [Wire()],
    )

    inputs = ToInterfaceRegister(input_adapter, connector)
    outputs = FromInterfaceRegister(output_adapter, connector)

    connector.connect(graft, inputs.interface)
    connector.connect(graft.wires[0], outputs.interface)

    with closer(
        Invocation(
            pack_into(inputs, FlowInputInto), pack_from(outputs, FlowControlInto)
        )
    ) as invocation:
        variable_inputs = invocation.port.variables.readin().split()
        variable_inputs[0].close()
        for extra in variable_inputs[arity + 1 :]:
            extra.close()

        return (
            variable_inputs[1 : arity + 1],
            invocation.wire.return_value.readout(),
        )

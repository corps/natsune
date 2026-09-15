from typing import Sequence

from natsune.connector import Connector, ExpansionBuilder, FrozenExpansion
from natsune.control_flow import (
    CloseAfterContingent,
    IfThenElseStatement,
    SerialAnd,
    SerialOr,
)
from natsune.control_flow_generated import FlowControlInto, FlowInputInto
from natsune.invocations import Invocation, closer, pack_from, pack_into
from natsune.legacy import LegacyInetInterface
from natsune.ports import Graft, Wire
from natsune.registers import (
    FromInterfaceRegister,
    FromRegister,
    ToInterfaceRegister,
    ToRegister,
)

type AgentImpl = ExpansionBuilder | LegacyInetInterface | IfThenElseStatement | SerialOr | SerialAnd | CloseAfterContingent


def callee_invocation(
    agent: FrozenExpansion | AgentImpl, arity: int, connector: Connector
) -> tuple[Sequence[ToRegister], FromRegister]:
    if isinstance(agent, LegacyInetInterface):
        input_adapter = agent.compiled.input_adapter
        output_adapter = agent.compiled.output_adapter
        expansion = agent.compiled
    else:
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

    # The closer's exit is load-bearing (the legacy path's `with` does this
    # implicitly): it annihilates each interface register's dangling state
    # wire-end, which the golden-net oracle is sensitive to.
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

import dataclasses
from typing import Mapping

from natsune.backend.control_flow import (
    FlowControlInto,
    VariablesFlow,
)
from natsune.backend.registers import (
    as_from_register,
    send_value,
)
from natsune.first_order.ports import Erasure
from natsune.frontend.ir import Exits, VariableUsage


@dataclasses.dataclass(slots=True)
class ControlBranchFlow:
    containing_flow: VariablesFlow
    variables_usage: Mapping[str, VariableUsage]
    exits: Exits

    cur_control: FlowControlInto = dataclasses.field(init=False)

    def __post_init__(self):
        finish_variables = self.containing_flow.variables_readout()

        self.cur_control = FlowControlInto(
            return_value=as_from_register(
                Erasure(), self.containing_flow.return_adapter, self.containing_flow
            ).to_interface(),
            continue_variables=as_from_register(
                Erasure(), self.containing_flow.variables.adapter, self.containing_flow
            ).to_interface(),
            break_variables=as_from_register(
                Erasure(), self.containing_flow.variables.adapter, self.containing_flow
            ).to_interface(),
            finish_variables=finish_variables.to_interface(),
        )

    def apply_continuation(self, continuation: FlowControlInto) -> None:
        flow_return_out, flow_continue_out = continuation.return_value.readout().choice(
            continuation.continue_variables.readout()
        )
        (
            flow_continue_out,
            flow_break_out,
        ) = flow_continue_out.choice(continuation.break_variables.readout())

        flow_break_out, flow_finish_out = flow_break_out.choice(
            continuation.finish_variables.readout()
        )

        self.cur_control.finish_variables = flow_finish_out.to_interface()
        self.cur_control.continue_variables |= flow_continue_out
        self.cur_control.break_variables |= flow_break_out
        self.cur_control.return_value |= flow_return_out

    def close(self) -> None:
        self.cur_control.shortcut(self.exits)

        send_value(
            self.cur_control.finish_variables.readout(),
            self.containing_flow.control_output.finish_variables.readin(),
        )
        send_value(
            self.cur_control.return_value.readout(),
            self.containing_flow.control_output.return_value.readin(),
        )
        send_value(
            self.cur_control.continue_variables.readout(),
            self.containing_flow.control_output.continue_variables.readin(),
        )

        send_value(
            self.cur_control.break_variables.readout(),
            self.containing_flow.control_output.break_variables.readin(),
        )

        self.containing_flow.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def apply_new_layer(self) -> VariablesFlow:
        result = VariablesFlow(
            variables=self.containing_flow.variables,
            return_adapter=self.containing_flow.return_adapter,
        )
        with result.invocation(self.containing_flow) as invocation:
            send_value(
                self.cur_control.finish_variables.readout(),
                invocation.port.variables.readin(),
            )
            self.apply_continuation(invocation.wire)
        return result

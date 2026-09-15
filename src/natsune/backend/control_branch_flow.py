import dataclasses
from typing import Mapping

from natsune.control_flow import FlowMap, VariablesFlow
from natsune.control_flow_generated import FlowControlInto
from natsune.frontend.ir import Exits, VariableUsage
from natsune.ports import Erasure, Wire
from natsune.registers import (
    FlowRegisterUsage,
    as_from_register,
    as_to_register,
    send_value,
)


# TODO: On cutover, make this agree with VariablesFlow in terms of file positioning, esp when removing old concepts
#  like FlowMap.  I think VariablesFlow and ControlBranchFlow are conceptually related more to lowering than as atomic
# units in control_flow are.
@dataclasses.dataclass(slots=True)
class ControlBranchFlow:
    containing_flow: VariablesFlow
    variables_usage: Mapping[str, VariableUsage]
    exits: Exits

    cur_control: FlowControlInto = dataclasses.field(init=False)

    def __post_init__(self):
        # TODO: Refactor variable_usage to just take the str -> wire | read dict directly and drop
        # FlowMap after cutover.
        finish_variables = self.containing_flow.variables_readout(
            FlowMap(
                {
                    k: (
                        FlowRegisterUsage(flow_write=True)
                        if self.variables_usage.get(k) == "write"
                        else (
                            FlowRegisterUsage(flow_read=True)
                            if self.variables_usage.get(k) == "read"
                            else FlowRegisterUsage()
                        )
                    )
                    for k in self.containing_flow.variables.variables.keys()
                },
                False,
                False,
                False,
                False,
            )
        )

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
        x1, x2 = Wire.as_interface()
        flow_continue_in = as_to_register(
            x1, self.containing_flow.variables.adapter, self.containing_flow
        )
        flow_continue_out = as_from_register(
            x2, self.containing_flow.variables.adapter, self.containing_flow
        )

        y1, y2 = Wire.as_interface()
        flow_break_in = as_to_register(
            y1, self.containing_flow.variables.adapter, self.containing_flow
        )
        flow_break_out = as_from_register(
            y2, self.containing_flow.variables.adapter, self.containing_flow
        )

        z1, z2 = Wire.as_interface()
        flow_finish_in = as_to_register(
            z1, self.containing_flow.variables.adapter, self.containing_flow
        )
        flow_finish_out = as_from_register(
            z2, self.containing_flow.variables.adapter, self.containing_flow
        )

        control_out, control_in_ = (
            (continuation.continue_variables.readout() & ~flow_continue_in)
            | (continuation.break_variables.readout() & ~flow_break_in)
            | (continuation.finish_variables.readout() + ~flow_finish_in)
        ).split()

        send_value(control_out, ~control_in_)

        self.cur_control.finish_variables = flow_finish_out.to_interface()
        self.cur_control.continue_variables |= flow_continue_out
        self.cur_control.break_variables |= flow_break_out
        self.cur_control.return_value |= continuation.return_value.readout()

    def close(self) -> None:
        if not (self.exits & Exits.FALLTHROUGH):
            self.cur_control.finish_variables.shortcut()
        if not (self.exits & Exits.RETURN):
            self.cur_control.return_value.shortcut()
        if not (self.exits & Exits.CONTINUE):
            self.cur_control.continue_variables.shortcut()
        if not (self.exits & Exits.BREAK):
            self.cur_control.break_variables.shortcut()

        # TODO: Again, during cutover, reconcile all of this by removing flowmap and fixing how variables flow works.
        send_value(
            self.cur_control.finish_variables.readout(),
            self.containing_flow.mapped_variables_readin(
                FlowMap(
                    {
                        k: (
                            FlowRegisterUsage(flow_write=True)
                            if self.variables_usage.get(k) == "write"
                            else (
                                FlowRegisterUsage(flow_read=True)
                                if self.variables_usage.get(k) == "read"
                                else FlowRegisterUsage()
                            )
                        )
                        for k in self.containing_flow.variables.variables.keys()
                    },
                    False,
                    False,
                    False,
                    False,
                ),
                self.containing_flow.control_output.finish_variables.readin(),
            ),
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
        with result.invocation(self.containing_flow, internal=True) as invocation:
            send_value(
                self.cur_control.finish_variables.readout(),
                invocation.port.variables.readin(),
            )
            self.apply_continuation(invocation.wire)
        return result

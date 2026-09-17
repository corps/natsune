import dataclasses
from typing import Mapping

from natsune.control_flow import FlowControlInto, VariablesFlow
from natsune.frontend.ir import Exits, VariableUsage
from natsune.ports import Erasure, Wire
from natsune.registers import (
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
        # The layer receives the FULL live bundle (all-write extension): a
        # usage-erased input would poison this layer's neutral cells, and
        # every control path (finish/continue/break) forwards the bundle
        # onward. Linear values move through layers without copying.
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
        # Net law 4: can't-fire slots are shortcut, NEVER readout — a slot
        # shortcut per `exits` must not be wired onward (reading out a
        # shortcut slot fires an annihilation value, which the Loop
        # composite would consume as an iteration completion and recurse
        # forever). Each control path is wired only when `exits` says it
        # can fire. The bundles forward raw: layers receive the full live
        # bundle (see __post_init__), so every control path carries live
        # values for every variable.
        if not (self.exits & Exits.FALLTHROUGH):
            self.cur_control.finish_variables.shortcut()
        else:
            send_value(
                self.cur_control.finish_variables.readout(),
                self.containing_flow.control_output.finish_variables.readin(),
            )
        if not (self.exits & Exits.RETURN):
            self.cur_control.return_value.shortcut()
        else:
            send_value(
                self.cur_control.return_value.readout(),
                self.containing_flow.control_output.return_value.readin(),
            )
        if not (self.exits & Exits.CONTINUE):
            self.cur_control.continue_variables.shortcut()
        else:
            send_value(
                self.cur_control.continue_variables.readout(),
                self.containing_flow.control_output.continue_variables.readin(),
            )

        if not (self.exits & Exits.BREAK):
            self.cur_control.break_variables.shortcut()
        else:
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

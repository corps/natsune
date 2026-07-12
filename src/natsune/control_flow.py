from natsune.control_flow_generated import (
    FlowControlFrom,
    FlowInputFrom,
    FlowControlInto,
    FlowInputInto,
    IfThenElseOutputInto,
    MergeOutputInto,
    MergeOutputFrom,
    IfThenElseOutputFrom,
)
import dataclasses
from functools import cached_property
from typing import Sequence, Self, Callable, Any

from natsune.adapters import (
    Adapter,
    ValueAdapter,
    ParValueAdapter,
    Variables,
)
from natsune.ambiguous import AmbiguousPair
from natsune.connector import Connector, ExpansionBuilder
from natsune.executor import Executor
from natsune.invocations import (
    ExpansionWithAdapters,
    expansion_invocation,
    filter_invocation,
    send_parameter,
    merge_invocation,
    send_parameters,
    unpack_port_and_wires,
    unpack_wires,
    RHS,
    generate_register_pair_types,
    pack_into,
    pack_from,
    LHS,
    closer,
    Invocation,
)
from natsune.ports import Port, Wire, ValuePort, CombPort, Expansion, Erasure, WirePort
from natsune.registers import (
    as_to_register,
    send_value,
    as_from_register,
    FromRegister,
    FlowRegister,
    send_values,
    FromInterfaceRegister,
    ToInterfaceRegister,
)


@dataclasses.dataclass
class FlowInput:
    variables: LHS
    value: LHS

    @classmethod
    def adapter(cls, variables: Variables) -> Adapter:
        return ParValueAdapter(
            [
                variables.adapter,
                ValueAdapter(),
            ]
        )


generate_register_pair_types(FlowInput)


@dataclasses.dataclass
class FlowControl:
    return_value: RHS
    continue_variables: RHS
    break_variables: RHS
    finish_variables: RHS

    @classmethod
    def adapter(cls, return_adapter: Adapter, variables: Variables) -> Adapter:
        return ParValueAdapter(
            [
                return_adapter,
                variables.adapter,
                variables.adapter,
                variables.adapter,
            ]
        )


generate_register_pair_types(FlowControl)

IfThenElseInputInto = ToInterfaceRegister
IfThenElseInputFrom = FromInterfaceRegister


@dataclasses.dataclass
class IfThenElseOutput:
    context: LHS
    result: RHS


generate_register_pair_types(IfThenElseOutput)

MergeInputTo = ToInterfaceRegister
MergeInputFrom = FromInterfaceRegister


@dataclasses.dataclass
class MergeOutput:
    second_value: LHS
    result: RHS


generate_register_pair_types(MergeOutput)


@dataclasses.dataclass(frozen=True)
class WeakeningSelection(ExpansionWithAdapters):
    adapter: Adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return self.adapter

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.adapter,
                self.adapter,
            ]
        )

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[ToInterfaceRegister, MergeOutputInto]]:
        return expansion_invocation(self, invoker, ToInterfaceRegister, MergeOutputInto)

    def __copy__(self) -> Self:
        return self

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            for wire in wires:
                executor.annihilate(wire, port)
            return

        with unpack_wires(self, wires, executor, MergeOutputFrom) as outputs:
            send_value(outputs.second_value.readout(), outputs.result.readin())


@dataclasses.dataclass(kw_only=True)
class VariablesFlow(ExpansionBuilder):
    variables: Variables
    return_adapter: Adapter

    input_adapter: Adapter = dataclasses.field(init=False)
    output_adapter: Adapter = dataclasses.field(init=False)
    variable_registers: dict[str, FlowRegister] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.input_adapter = FlowInput.adapter(self.variables)
        self.output_adapter = FlowControl.adapter(self.return_adapter, self.variables)

        for name, variable_input in zip(
            self.variables.keys(), self.flow_input.variables.split()
        ):
            flow_register = self.variable_registers[name] = FlowRegister(
                self.variables[name], self
            )
            send_value(variable_input.readout(), flow_register.interface_readin())
            variable_input.close()

    def variables_readout(self) -> FromRegister:
        x1, x2 = Wire.as_interface()
        send_values(
            [r.readout() for r in self.variable_registers.values()],
            as_to_register(x1, self.variables.adapter, self).split(),
        )

        return as_from_register(
            x2,
            self.variables.adapter,
            self,
        )

    @cached_property
    def control_output(self) -> FlowControlFrom:
        return pack_into(self.output_interface, FlowControlFrom)

    @cached_property
    def flow_input(self) -> FlowInputFrom:
        return pack_from(self.input_interface, FlowInputFrom)

    def __copy__(self) -> Self:
        return self

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[FlowInputInto, FlowControlInto]]:
        return expansion_invocation(self, invoker, FlowInputInto, FlowControlInto)

    def close(self) -> None:
        for register in self.variable_registers.values():
            register.close()
        closer(self.flow_input).close()
        closer(self.control_output).close()


@dataclasses.dataclass
class Loop(ExpansionWithAdapters):
    # Takes in the loop's iterable value and the variable flows,
    # 'returns' whether the loop is continuing, along with the updated variable flows.
    iteration: VariablesFlow

    body: VariablesFlow
    orelse: VariablesFlow

    @cached_property
    def input_adapter(self) -> Adapter:
        return FlowInput.adapter(self.body.variables)

    @cached_property
    def output_adapter(self) -> Adapter:
        return FlowControl.adapter(self.body.return_adapter, self.body.variables)

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[FlowInputInto, FlowControlInto]]:
        return expansion_invocation(self, invoker, FlowInputInto, FlowControlInto)

    def __copy__(self) -> Self:
        return self

    @cached_property
    def conditional(self) -> IfThenElse:
        # conditonal test is wether iterator should keep going
        # context vlaue is the actual iterator value
        # context variables flow
        # output return returns the outside structure
        # output continue continues the outside structure
        # output break breaks the outside structure
        # output finish means this loop is over and continues
        return IfThenElse(
            self.true_case,
            self.false_case,
        )

    # In the true case, we are committed to calling the body, and that body's continue and break only affect next recursion
    # so we can only either recurse again, or finish.
    @cached_property
    def true_case(self) -> ExpansionWithAdapters:
        with (
            ExpansionBuilder(self.input_adapter, self.output_adapter) as builder,
            self.body.invocation(builder) as body_invocation,
            self.invocation(builder) as recurse,
        ):
            inputs = pack_from(builder.input_interface, FlowInputFrom)
            outputs = pack_into(builder.output_interface, FlowControlFrom)

            send_value(inputs.value.readout(), recurse.port.value.readin())

            with WeakeningSelection(self.body.return_adapter).invocation(
                builder
            ) as return_selection:
                send_value(
                    return_selection.wire.result.readout(),
                    outputs.return_value.readin(),
                )

                send_value(
                    body_invocation.wire.return_value.readout(),
                    return_selection.port.readin(),
                )
                send_value(
                    recurse.wire.return_value.readout(),
                    return_selection.wire.second_value.readin(),
                )

            with WeakeningSelection(self.body.variables.adapter).invocation(
                builder
            ) as recurse_variables_selection:
                send_value(
                    recurse_variables_selection.wire.result.readout(),
                    recurse.port.variables.readin(),
                )

                send_value(
                    body_invocation.wire.continue_variables.readout(),
                    recurse_variables_selection.port.readin(),
                )
                send_value(
                    body_invocation.wire.finish_variables.readout(),
                    recurse_variables_selection.wire.second_value.readin(),
                )

            send_value(
                body_invocation.wire.break_variables.readout(),
                outputs.finish_variables.readin(),
            )
            return builder

    # In the false case, our return, continue, break, and finish all flow as if this was a flattened context as this is
    # the else case.  We don't need to worry about the iterator anymore.
    @cached_property
    def false_case(self) -> ExpansionWithAdapters:
        with (
            ExpansionBuilder(self.input_adapter, self.output_adapter) as builder,
            expansion_invocation(
                self.orelse, builder, FlowInputInto, FromInterfaceRegister
            ) as else_body,
        ):
            inputs = pack_from(builder.input_interface, FlowInputFrom)
            outputs = builder.output_interface

            send_value(inputs.variables.readout(), else_body.port.variables.readin())
            send_value(else_body.wire.readout(), outputs.readin())
            return builder

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            for wire in wires:
                executor.annihilate(wire, port)
            return

        with (
            unpack_port_and_wires(
                self, port, wires, executor, FlowInputFrom, ToInterfaceRegister
            ) as this_invocation,
            self.iteration.invocation(executor) as iterable_invocation,
            self.conditional.invocation(executor) as conditional_invocation,
        ):
            iter_input_1, iter_input_2 = (
                this_invocation.port.value.readout().duplicate()
            )
            send_value(
                this_invocation.port.variables.readout(),
                iterable_invocation.port.variables.readin(),
            )
            send_value(iter_input_1, iterable_invocation.port.value.readin())
            send_value(
                iterable_invocation.wire.return_value.readout(),
                conditional_invocation.port.readin(),
            )

            with closer(
                pack_into(conditional_invocation.wire.context, FlowInputInto)
            ) as conditional_context:
                send_value(iter_input_2, conditional_context.value.readin())
                send_value(
                    iterable_invocation.wire.finish_variables.readout(),
                    conditional_context.variables.readin(),
                )

            send_value(
                conditional_invocation.wire.result.readout(),
                this_invocation.wire.readin(),
            )


@dataclasses.dataclass(slots=True, frozen=True)
class IfThenElse(ExpansionWithAdapters):
    true_case: ExpansionWithAdapters
    false_case: ExpansionWithAdapters

    def __copy__(self) -> Self:
        return self

    @cached_property
    def input_adapter(self) -> Adapter:
        return ValueAdapter()  # value

    @cached_property
    def expansion_inputs_adapter(self) -> Adapter:
        assert self.true_case.input_adapter == self.false_case.output_adapter
        return self.true_case.input_adapter

    @cached_property
    def expansion_outputs_adapter(self) -> Adapter:
        assert self.true_case.output_adapter == self.false_case.output_adapter
        return self.true_case.output_adapter

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.expansion_inputs_adapter,
                self.expansion_outputs_adapter,
            ]
        )

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[IfThenElseInputInto, IfThenElseOutputInto]]:
        return expansion_invocation(
            self, invoker, IfThenElseInputInto, IfThenElseOutputInto
        )

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        if not isinstance(port, ValuePort):
            for wire in wires:
                exec.annihilate(wire, port)
            for wire in port.wires:
                exec.annihilate(wire, port)
            return

        if port.value:
            case = self.true_case
        else:
            case = self.false_case

        with (
            unpack_wires(self, wires, exec, IfThenElseOutputFrom) as conditional,
            expansion_invocation(
                case, exec, ToInterfaceRegister, FromInterfaceRegister
            ) as case,
        ):
            send_value(conditional.context.readout(), case.port.readin())
            send_value(case.wire.readout(), conditional.result.readin())


@dataclasses.dataclass(frozen=True)
class ConcurrentMerge(ExpansionWithAdapters):
    should_short: Callable[[Any], bool]
    merge_operation: Callable[[Any, Any], Any]

    @cached_property
    def input_adapter(self) -> Adapter:
        return ParValueAdapter([ValueAdapter(), ValueAdapter()])

    @cached_property
    def output_adapter(self) -> Adapter:
        return ValueAdapter()

    def __copy__(self) -> Self:
        return self

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[MergeInputTo, MergeInputFrom]]:
        return expansion_invocation(self, invoker, MergeInputTo, MergeInputFrom)

    @cached_property
    def true_case(self) -> ExpansionBuilder:
        with ExpansionBuilder(self.input_adapter, self.output_adapter) as builder:
            v1, v2 = builder.input_interface.readout().split()

            send_value(
                v1,
                builder.output_interface.readin(),
            )

            return builder

    @cached_property
    def false_case(self) -> ExpansionBuilder:
        with ExpansionBuilder(self.input_adapter, self.output_adapter) as builder:
            v1, v2 = builder.input_interface.readout().split()

            send_value(
                send_parameters(
                    merge_invocation(self.merge_operation, builder),
                    (v1, v2),
                ),
                builder.output_interface.readin(),
            )

            return builder

    @cached_property
    def conditional(self) -> IfThenElse:
        return IfThenElse(self.true_case, self.false_case)

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        pair = AmbiguousPair()
        pair_invocation = pair.invocation(exec)
        left, right = as_from_register(port, self.input_adapter, exec).split()

        send_value(left, pair_invocation.i_value_1)
        send_value(right, pair_invocation.i_value_2)

        first_value_1, first_value_2 = pair_invocation.o_first_value.duplicate()
        with self.conditional.invocation(exec) as conditional:
            send_value(
                send_parameter(
                    filter_invocation(self.should_short, exec), first_value_1
                ),
                conditional.port.readin(),
            )
            a, b = conditional.wire.context.split()
            send_value(first_value_2, a.readin())
            send_value(pair_invocation.o_second_value, b.readin())
            send_value(
                conditional.wire.result.readout(),
                as_to_register(WirePort([wires[0]]), self.output_adapter, exec),
            )

from natsune.optimizer import optimize
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
from natsune.connector import (
    Connector,
    ExpansionBuilder,
    new_wires_cache,
    serialize_port,
)
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
from natsune.ports import (
    Port,
    Wire,
    ValuePort,
    Erasure,
    Graft,
)
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
            with unpack_wires(self, wires, executor, MergeOutputFrom) as outputs:
                send_value(outputs.second_value.readout(), outputs.result.readin())
            return

        with unpack_port_and_wires(
            self, port, wires, executor, MergeInputFrom, MergeOutputFrom
        ) as invocation:
            send_value(invocation.port.readout(), invocation.wire.result.readin())


@dataclasses.dataclass(frozen=True)
class GatedSelection(ExpansionWithAdapters):
    left: Adapter
    right: Adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return self.left

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.right,
                ParValueAdapter([self.left, self.right]),
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
                executor.annihilate(wire)
            return

        with unpack_port_and_wires(
            self, port, wires, executor, MergeInputFrom, MergeOutputFrom
        ) as invocation:
            send_values(
                [invocation.port.readout(), invocation.wire.second_value.readout()],
                invocation.wire.result.readin().split(),
            )


@dataclasses.dataclass(slots=True, frozen=True)
class Tracer:
    label: str
    wires_cache: dict[Wire, str] = dataclasses.field(default_factory=new_wires_cache)

    def __copy__(self) -> Self:
        return dataclasses.replace(self, wires_cache=new_wires_cache())

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        print(f"{self.label} from: {serialize_port(port, self.wires_cache, False)}")
        executor.connect(wires[0], port)


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
        x1, x2 = Wire.as_tautology()
        send_values(
            [r.readout() for k, r in self.variable_registers.items()],
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
        optimize(self, self.active_pairs)


@dataclasses.dataclass
class CloseAfterContingent(ExpansionWithAdapters):
    left: Adapter
    right: Adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return self.left

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter([self.right, self.left])

    def __copy__(self) -> Self:
        return self

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        with executor.sequenced_tuplate_from(wires[0]) as wire_iter:
            self.right.close(next(wire_iter), executor)
            executor.connect(port, next(wire_iter))


@dataclasses.dataclass
class Loop(ExpansionWithAdapters):
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
        return IfThenElse(
            self.true_case,
            self.false_case,
        )

    @cached_property
    def true_case(self) -> ExpansionWithAdapters:
        with ExpansionBuilder(self.input_adapter, self.output_adapter) as builder:

            with closer(pack_from(builder.input_interface, FlowInputFrom)) as inputs:
                input_variables = inputs.variables.readout()
                input_iter = inputs.value.readout()

            with self.body.invocation(builder) as body_invocation:
                send_value(
                    input_variables,
                    body_invocation.port.variables.readin(),
                )
                body_finish_variables = (
                    body_invocation.wire.continue_variables.readout()
                    | body_invocation.wire.finish_variables.readout()
                )
                body_break_variables = body_invocation.wire.break_variables.readout()
                body_return = body_invocation.wire.return_value.readout()

            recurse_variables, recurse_iter = (
                body_finish_variables & input_iter
            ).split()

            with self.invocation(builder) as recurse:
                send_value(
                    recurse_variables,
                    recurse.port.variables.readin(),
                )
                send_value(recurse_iter, recurse.port.value.readin())

                recurse_return = recurse.wire.return_value.readout()
                recurse_finish = recurse.wire.finish_variables.readout()
                recurse_break = recurse.wire.break_variables.readout()
                recurse_continue = recurse.wire.continue_variables.readout()

            with closer(
                pack_into(builder.output_interface, FlowControlFrom)
            ) as outputs:
                send_value(
                    body_return | recurse_return,
                    outputs.return_value.readin(),
                )
                send_value(
                    (body_break_variables | recurse_finish),
                    outputs.finish_variables.readin(),
                )
                send_value(
                    recurse_break,
                    outputs.break_variables.readin(),
                )
                send_value(
                    recurse_continue,
                    outputs.continue_variables.readin(),
                )

        return builder

    @cached_property
    def false_case(self) -> ExpansionWithAdapters:
        with (
            ExpansionBuilder(self.input_adapter, self.output_adapter) as builder,
            expansion_invocation(
                self.orelse, builder, FlowInputInto, FromInterfaceRegister
            ) as else_body,
            closer(pack_from(builder.input_interface, FlowInputFrom)) as inputs,
        ):
            outputs = builder.output_interface
            send_value(
                inputs.variables.readout(),
                else_body.port.variables.readin(),
            )
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
                self, port, wires, executor, FlowInputFrom, FlowControlFrom
            ) as this_invocation,
            self.iteration.invocation(executor) as iterable_invocation,
            self.conditional.invocation(executor) as conditional_invocation,
        ):
            input_iter, synchronized_iter = (
                this_invocation.port.value.readout().duplicate("share")
            )
            input_variables = this_invocation.port.variables.readout()

            send_value(
                input_variables,
                iterable_invocation.port.variables.readin(),
            )
            send_value(input_iter, iterable_invocation.port.value.readin())

            should_continue = iterable_invocation.wire.return_value.readout()

            send_value(
                should_continue,
                conditional_invocation.port.readin(),
            )

            with closer(
                pack_into(conditional_invocation.wire.context, FlowInputInto)
            ) as conditional_context:
                send_value(
                    synchronized_iter,
                    conditional_context.value.readin(),
                )
                send_value(
                    iterable_invocation.wire.finish_variables.readout(),
                    conditional_context.variables.readin(),
                )

            with closer(
                pack_from(conditional_invocation.wire.result, FlowControlInto)
            ) as conditional_result:
                send_value(
                    conditional_result.finish_variables.readout(),
                    this_invocation.wire.finish_variables.readin(),
                )
                send_value(
                    conditional_result.return_value.readout(),
                    this_invocation.wire.return_value.readin(),
                )
                send_value(
                    conditional_result.break_variables.readout(),
                    this_invocation.wire.break_variables.readin(),
                )
                send_value(
                    conditional_result.continue_variables.readout(),
                    this_invocation.wire.continue_variables.readin(),
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
        assert self.true_case.input_adapter == self.false_case.input_adapter
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
            ) as invocation,
        ):
            send_value(conditional.context.readout(), invocation.port.readin())
            send_value(invocation.wire.readout(), conditional.result.readin())


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
                as_to_register(wires[0], self.output_adapter, exec),
            )

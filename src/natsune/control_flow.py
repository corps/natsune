from multiprocessing import Value
import contextlib
import copy
import dataclasses
from functools import cached_property
from typing import Sequence, Self, Callable, Any, Iterator, Generator

from .adapters import (
    Adapter,
    ValueAdapter,
    ParValueAdapter,
    Variables,
)
from .ambiguous import AmbiguousPair
from .connector import Connector, ExpansionBuilder
from .executor import Executor
from .invocations import (
    ExpansionWithAdapters,
    expansion_invocation,
    filter_invocation,
    send_parameter,
    merge_invocation,
    send_parameters,
    unpack_port_and_wires,
    unpack_wires,
)
from .ports import Port, Wire, ValuePort, WirePort, Erasure, CombPort, Expansion, Graft
from .registers import (
    as_to_register,
    send_value,
    as_from_register,
    ToRegister,
    FromRegister,
    FlowRegister,
    InterfaceRegister,
    send_values,
)


@dataclasses.dataclass(slots=True, frozen=True)
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

    def invocation(self, invoker: Connector) -> FlowInvocation:
        inputs, outputs = expansion_invocation(self, invoker)

        return FlowInvocation(FlowInput(*inputs.split()), FlowControl(*outputs.split()))

    def __copy__(self) -> Self:
        return self

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if not isinstance(port, CombPort):
            for wire in wires:
                executor.annihilate(wire, port)
            for wire in port.wires:
                executor.annihilate(wire)
            return

        inputs, outputs = unpack_port_and_wires(self, port, wires, executor)

        with (
            FlowInvocation.split(inputs, outputs) as this_invocation,
            # this_invocation.control.share_to_interface() as (
            #     orelse_result,
            #     body_result,
            # ),
            self.iteration.invocation(executor) as iterable_invocation,
            self.body.invocation(executor) as body_invocation,
            self.orelse.invocation(executor) as orelse_invocation,
            # self.invocation(executor) as recurse,
        ):
            iter_input_1, iter_input_2 = (
                this_invocation.inputs.i_value.readout().duplicate()
            )
            send_value(
                this_invocation.inputs.i_variables.readout(),
                iterable_invocation.inputs.i_variables.readin(),
            )
            send_value(iter_input_1, iterable_invocation.inputs.i_value.readin())

            true_case = ExpansionBuilder(self.input_adapter, self.output_adapter)
            false_case = ExpansionBuilder(self.input_adapter, self.output_adapter)

            cond_invocation = IfThenElse(true_case, false_case).invocation(
                executor
            )
            cond_i_iter_context, cond_i_variables_context = cond_invocation.i_context.readin().split()

            send_value(
                iterable_invocation.control.o_return.readout(),
                cond_invocation.i_test_value.readin(),
            )
            send_value(
                iterable_invocation.control.o_finish.readout(),
                cond_i_variables_context,
            )
            send_value(iter_input_2, cond_i_iter_context)

            fi, fv = false_case.input_interface.readout().split()
            fi.close()

            send_value(
                cond_invocation.o_orelse.readout(),
                orelse_invocation.inputs.i_variables.readin(),
            )
            send_value(
                cond_invocation.o_body.readout(),
                body_invocation.inputs.i_variables.readin(),
            )

            orelse_invocation.control.send_to(orelse_result)

            # Option to return from parent
            # o_return_1, o_return_2 = AffineSelection.share_to_register(
            #     body_result.o_return.readin()
            # )
            # We return if either the current body returns or the recursion returns
            send_value(body_invocation.control.o_return.readout(), o_return_1)
            send_value(recurse.control.o_return.readout(), o_return_2)

            # Option to finish through parent
            # o_finished_1, o_finished_2 = AffineSelection.share_to_register(
            #     body_result.o_finish.readin()
            # )
            # We are finished if either the current body breaks or the recursion finishes.
            send_value(body_invocation.control.o_break.readout(), o_finished_1)
            send_value(recurse.control.o_finish.readout(), o_finished_2)

            # Option to recurse again
            # o_continue_1, o_continue_2 = AffineSelection.share_to_register(
            #     recurse.inputs.i_variables.readin()
            # )
            # We reurse if either the current body continues or finishes
            send_value(body_invocation.control.o_continue.readout(), o_continue_1)
            send_value(body_invocation.control.o_finish.readout(), o_continue_2)

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

    def invocation(self, invoker: Connector) -> IfThenElseInvocation:
        cond, outputs = expansion_invocation(self, invoker)
        case_inputs, case_outputs = outputs.split()
        return IfThenElseInvocation(cond, case_inputs, case_outputs)

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        if not isinstance(port, ValuePort):
            for wire in port.wires:
                exec.annihilate(wire, port)
            for wire in wires:
                exec.annihilate(wire, port)
            return

        case_inputs, case_outputs = unpack_wires(self, wires, exec).split()

        if port.value:
            inputs, outputs = expansion_invocation(self.true_case, exec)
        else:
            inputs, outputs = expansion_invocation(self.false_case, exec)
        send_value(case_inputs.readout(), inputs.readin())
        send_value(outputs.readout(), case_outputs.readin())


@dataclasses.dataclass(frozen=True, slots=True)
class IfThenElseInvocation:
    i_test_value: InterfaceRegister
    i_context: InterfaceRegister
    o_result: InterfaceRegister

    def __enter__(self) -> Self:
        return self

    def close(self) -> None:
        self.i_test_value.close()
        self.i_context.close()
        self.o_result.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@dataclasses.dataclass(frozen=True, slots=True)
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

    def invocation(self, invoker: Connector) -> ConcurrentMergeInvocation:
        inputs, outputs = expansion_invocation(self, invoker)
        return ConcurrentMergeInvocation.split(inputs, outputs)

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        pair = AmbiguousPair()
        pair_invocation = pair.invocation(exec)
        left, right = as_from_register(port, self.input_adapter, exec).split()

        send_value(left, pair_invocation.i_value_1)
        send_value(right, pair_invocation.i_value_2)

        first_value_1, first_value_2 = pair_invocation.o_first_value.duplicate()

        true_case = ExpansionBuilder(
            ParValueAdapter([ValueAdapter(), ValueAdapter()]),
            ValueAdapter(),
        )

        false_case = ExpansionBuilder(
            ParValueAdapter([ValueAdapter(), ValueAdapter()]),
            ValueAdapter(),
        )

        if_invocation = IfThenElse(
            true_case,
            false_case,
        ).invocation(exec)

        send_value(
            send_parameter(filter_invocation(self.should_short, exec), first_value_1),
            if_invocation.i_test_value.readin(),
        )

        if_first_value, if_second_value = if_invocation.i_context.readin().split()
        send_value(first_value_2, if_first_value)
        send_value(pair_invocation.o_second_value, if_second_value)

        t1, t2 = true_case.input_interface.readout().split()
        t2.close()
        send_value(t1, true_case.output_interface.readin())

        f1, f2 = false_case.input_interface.readout().split()

        send_value(
            send_parameters(
                merge_invocation(self.merge_operation, exec),
                (f1, f2),
            ),
            false_case.output_interface.readin(),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ConcurrentMergeInvocation:
    i_value_1: InterfaceRegister
    i_value_2: InterfaceRegister
    o_value: InterfaceRegister

    @classmethod
    def split(
        cls, inputs: InterfaceRegister, outputs: InterfaceRegister
    ) -> ConcurrentMergeInvocation:
        a, b = inputs.split()
        return cls(a, b, outputs)


@dataclasses.dataclass(slots=True, kw_only=True)
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
            self.variables.keys(), self.flow_input.i_variables.split()
        ):
            flow_register = self.variable_registers[name] = FlowRegister(
                self.variables[name], self
            )
            send_value(variable_input.readout(), flow_register.interface_readin())

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
    def o_control(self) -> FlowControl:
        return FlowControl.split(self.output_interface)

    @cached_property
    def flow_input(self) -> FlowInput:
        return FlowInput(*self.input_interface.split())

    def __copy__(self) -> Self:
        return self

    def invocation(self, invoker: Connector) -> FlowInvocation:
        inputs, outputs = expansion_invocation(self, invoker)
        return FlowInvocation.split(inputs, outputs)

    def close(self) -> None:
        for register in self.variable_registers.values():
            register.close()

        self.o_control.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclasses.dataclass(slots=True, frozen=True)
class FlowInvocation:
    inputs: FlowInput
    control: FlowControl

    @classmethod
    def split(
        cls, inputs: InterfaceRegister, outputs: InterfaceRegister
    ) -> FlowInvocation:
        return FlowInvocation(FlowInput(*inputs.split()), FlowControl.split(outputs))

    def close(self) -> None:
        self.inputs.close()
        self.control.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class FlowInput:
    i_variables: InterfaceRegister
    i_value: InterfaceRegister

    @classmethod
    def adapter(cls, variables: Variables) -> Adapter:
        return ParValueAdapter(
            [
                variables.adapter,
                ValueAdapter(),
            ]
        )

    def close(self) -> None:
        self.i_variables.close()
        self.i_value.close()


@dataclasses.dataclass(slots=True, frozen=True)
class FlowControl:
    o_return: InterfaceRegister
    o_continue: InterfaceRegister
    o_break: InterfaceRegister
    o_finish: InterfaceRegister

    # o_completion_errors: InterfaceRegister

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

    @classmethod
    def split(cls, source: InterfaceRegister) -> FlowControl:
        return FlowControl(*source.split())

    def send_to(self, i_control: FlowControl) -> None:
        send_values(
            [
                self.o_return.readout(),
                self.o_continue.readout(),
                self.o_break.readout(),
                self.o_finish.readout(),
            ],
            [
                i_control.o_return.readin(),
                i_control.o_continue.readin(),
                i_control.o_break.readin(),
                i_control.o_finish.readin(),
            ],
        )

    def close(self):
        self.o_return.close()
        self.o_continue.close()
        self.o_break.close()
        self.o_finish.close()

    def __iter__(self) -> Iterator[InterfaceRegister]:
        yield self.o_return
        yield self.o_continue
        yield self.o_break
        yield self.o_finish

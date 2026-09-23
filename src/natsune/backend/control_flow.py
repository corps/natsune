import dataclasses
from functools import cached_property
from typing import Any, Callable, Self, Sequence

from natsune.backend.connector import (
    Connector,
    ExpansionBuilder,
    global_wire_lock,
    new_wires_cache,
    serialize_port,
)
from natsune.backend.invocations import (
    ExpansionWithAdapters,
    Invocation,
    _pack_from,
    _pack_into,
    closer,
    expansion_invocation,
    filter_invocation,
    merge_invocation,
    pack_from,
    pack_into,
    send_parameter,
    send_parameters,
    unpack_port_and_wires,
    unpack_wires,
)
from natsune.backend.optimizer import optimize
from natsune.backend.registers import (
    FlowRegister,
    FromInterfaceRegister,
    FromRegister,
    ToInterfaceRegister,
    ToRegister,
    as_from_register,
    as_to_register,
    borrow_registers,
    send_value,
    send_values,
)
from natsune.first_order.adapters import (
    RA_VA,
    VA,
    Adapter,
    ParValueAdapter,
    Variables,
)
from natsune.first_order.ports import (
    Erasure,
    Expansion,
    Graft,
    Port,
    ValuePort,
    Wire,
)


@dataclasses.dataclass(slots=True)
class AmbiguousPair(Expansion):
    copied: int = 0

    def invocation(self, invoker: Connector, adapter: Adapter) -> "AmbiguousInvocation":
        graft_1 = Graft(self, wires=[Wire(), Wire()])
        graft_2 = Graft(self, wires=graft_1.wires)
        input_1 = as_to_register(graft_1, adapter, invoker)
        input_2 = as_to_register(graft_2, adapter, invoker)
        output_1 = as_from_register(graft_1.wires[0], adapter, invoker)
        output_2 = as_from_register(graft_1.wires[1], adapter, invoker)
        return AmbiguousInvocation(input_1, input_2, output_1, output_2)

    def __call__(self, exec: Connector, port: Port, wires: Sequence[Wire]) -> None:
        should_copy = False
        with global_wire_lock:
            if self.copied == 0:
                self.copied = 1
                should_copy = True

        if should_copy:
            exec.connect(port, wires[0])
        else:
            exec.connect(port, wires[1])

    def __copy__(self) -> "AmbiguousPair":
        return AmbiguousPair()


@dataclasses.dataclass(frozen=True, slots=True)
class AmbiguousInvocation:
    i_value_1: ToRegister
    i_value_2: ToRegister
    o_first_value: FromRegister
    o_second_value: FromRegister


def flow_input_adapter(variables: Variables) -> Adapter:
    return ParValueAdapter(
        [
            variables.adapter,
            VA,
        ]
    )


@dataclasses.dataclass(slots=True)
class FlowInputInto:
    variables: ToInterfaceRegister
    value: ToInterfaceRegister

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> FlowInputInto:
        assert isinstance(to_register.adapter, ParValueAdapter)
        variables_adapter, value_adapter = to_register.adapter.concurrent_items
        variables = ToInterfaceRegister(variables_adapter, to_register.connector)
        value = ToInterfaceRegister(value_adapter, to_register.connector)

        send_value(
            (~variables.interface_readin() & ~value.interface_readin()),
            to_register.readin(),
        )

        return FlowInputInto(variables, value)


@dataclasses.dataclass(slots=True)
class FlowInputFrom:
    variables: FromInterfaceRegister
    value: FromInterfaceRegister

    @classmethod
    def pack_from(cls, from_register: FromInterfaceRegister) -> Self:
        return _pack_from(from_register, cls)


def flow_control_adapter(return_adapter: Adapter, variables: Variables) -> Adapter:
    return ParValueAdapter(
        [
            return_adapter,
            variables.adapter,
            variables.adapter,
            variables.adapter,
        ]
    )


@dataclasses.dataclass(slots=True)
class FlowControlInto:
    return_value: FromInterfaceRegister
    continue_variables: FromInterfaceRegister
    break_variables: FromInterfaceRegister
    finish_variables: FromInterfaceRegister

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> Self:
        return _pack_into(to_register, cls)


@dataclasses.dataclass(slots=True)
class FlowControlFrom:
    return_value: ToInterfaceRegister
    continue_variables: ToInterfaceRegister
    break_variables: ToInterfaceRegister
    finish_variables: ToInterfaceRegister

    @classmethod
    def pack_from(cls, from_register: FromInterfaceRegister) -> Self:
        return _pack_from(from_register, cls)


IfThenElseInputInto = ToInterfaceRegister
IfThenElseInputFrom = FromInterfaceRegister


@dataclasses.dataclass(slots=True)
class IfThenElseOutputInto:
    context: ToInterfaceRegister
    result: FromInterfaceRegister

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> Self:
        return _pack_into(to_register, cls)


@dataclasses.dataclass(slots=True)
class IfThenElseOutputFrom:
    context: FromInterfaceRegister
    result: ToInterfaceRegister

    @classmethod
    def pack_from(cls, from_register: FromInterfaceRegister) -> Self:
        return _pack_from(from_register, cls)


@dataclasses.dataclass(slots=True)
class IfThenElseStatementOutputInto:
    context: FlowInputInto
    result: FlowControlInto

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> Self:
        return _pack_into(to_register, cls)


MergeInputTo = ToInterfaceRegister
MergeInputFrom = FromInterfaceRegister


@dataclasses.dataclass(slots=True)
class MergeOutputInto:
    second_value: ToInterfaceRegister
    result: FromInterfaceRegister

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> Self:
        return _pack_into(to_register, cls)


@dataclasses.dataclass(slots=True)
class MergeOutputFrom:
    second_value: FromInterfaceRegister
    result: ToInterfaceRegister

    @classmethod
    def pack_from(cls, from_register: FromInterfaceRegister) -> Self:
        return _pack_from(from_register, cls)


@dataclasses.dataclass(slots=True)
class ParallelMergeInputInto:
    first_value: ToInterfaceRegister
    second_value: ToInterfaceRegister

    @classmethod
    def pack_into(cls, to_register: ToInterfaceRegister) -> Self:
        return _pack_into(to_register, cls)


@dataclasses.dataclass(slots=True)
class ParallelMergeInputFrom:
    first_value: FromInterfaceRegister
    second_value: FromInterfaceRegister

    @classmethod
    def pack_from(cls, from_register: FromInterfaceRegister) -> Self:
        return _pack_from(from_register, cls)


ParallelMergeOutputTo = FromInterfaceRegister
ParallelMergeOutputFrom = ToInterfaceRegister


@dataclasses.dataclass(frozen=True)
class ParallelOr(ExpansionWithAdapters):
    adapter: Adapter

    @cached_property
    def output_adapter(self) -> Adapter:
        return self.adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.adapter,
                self.adapter,
            ]
        )

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[ParallelMergeInputInto, ParallelMergeOutputTo]]:
        return expansion_invocation(
            self, invoker, ParallelMergeInputInto, ParallelMergeOutputTo
        )

    def __copy__(self) -> Self:
        return self

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        pair = AmbiguousPair()
        pair_invocation = pair.invocation(executor, self.adapter)
        left, right = as_from_register(port, self.input_adapter, executor).split()

        send_value(left, pair_invocation.i_value_1)
        send_value(right, pair_invocation.i_value_2)

        with SerialOr(self.adapter).invocation(executor) as selection:
            send_value(
                pair_invocation.o_first_value,
                selection.port.readin(),
            )
            send_value(
                pair_invocation.o_second_value,
                selection.wire.second_value.readin(),
            )
            send_value(
                selection.wire.result.readout(),
                as_to_register(wires[0], self.output_adapter, executor),
            )


@dataclasses.dataclass(frozen=True)
class SerialOr(ExpansionWithAdapters):
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
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
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
class SerialLeftPass(ExpansionWithAdapters):
    lhs: Adapter
    rhs: Adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return self.lhs

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.rhs,
                self.rhs,
            ]
        )

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            executor.connect(wires[0], wires[1])
            return

        executor.annihilate(port)
        executor.annihilate(wires[0])
        executor.annihilate(wires[1])

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[ToInterfaceRegister, MergeOutputInto]]:
        return expansion_invocation(self, invoker, MergeInputTo, MergeOutputInto)


@dataclasses.dataclass(frozen=True)
class SerialAnd(ExpansionWithAdapters):
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
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
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


@dataclasses.dataclass(frozen=True)
class ConcurrentValueMerge(ExpansionWithAdapters):
    should_short: Callable[[Any], bool]
    merge_operation: Callable[[Any, Any], Any]

    @cached_property
    def input_adapter(self) -> Adapter:
        return ParValueAdapter([VA, VA])

    @cached_property
    def output_adapter(self) -> Adapter:
        return VA

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

    def __call__(self, exec: Connector, port: Port, wires: Sequence[Wire]) -> None:
        pair = AmbiguousPair()
        pair_invocation = pair.invocation(exec, VA)
        left, right = as_from_register(port, self.input_adapter, exec).split()

        send_value(left, pair_invocation.i_value_1)
        send_value(right, pair_invocation.i_value_2)

        first_value_1, first_value_2 = pair_invocation.o_first_value.duplicate("share")
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


@dataclasses.dataclass(slots=True, frozen=True)
class Tracer:
    label: str
    wires_cache: dict[Wire, str] = dataclasses.field(default_factory=new_wires_cache)

    def __copy__(self) -> Self:
        return dataclasses.replace(self, wires_cache=new_wires_cache())

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        print(f"{self.label} from: {serialize_port(port, self.wires_cache, False)}")
        executor.connect(wires[0], port)


@dataclasses.dataclass(kw_only=True)
class VariablesFlow(ExpansionBuilder):
    variables: Variables
    return_adapter: Adapter
    exceptions: FlowRegister = dataclasses.field(init=False)

    input_adapter: Adapter = dataclasses.field(init=False)
    output_adapter: Adapter = dataclasses.field(init=False)
    variable_registers: dict[str, FlowRegister] = dataclasses.field(
        default_factory=dict
    )

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.input_adapter = flow_input_adapter(self.variables)
        self.output_adapter = flow_control_adapter(self.return_adapter, self.variables)

        input_variables = iter(self.flow_input.variables.split())

        self.exceptions = FlowRegister(RA_VA, self)
        send_value(next(input_variables).readout(), self.exceptions.interface_readin())

        for name, variable_input in zip(
            self.variables.keys(), input_variables, strict=True
        ):
            flow_register = self.variable_registers[name] = FlowRegister(
                self.variables[name], self
            )
            send_value(variable_input.readout(), flow_register.interface_readin())
            variable_input.close()

    def variables_readout(self) -> FromRegister:
        x1, x2 = Wire.as_interface()
        readouts: list[FromRegister] = []

        readouts.append(self.exceptions.readout())
        for r in self.variable_registers.values():
            g, _ = r.extend()
            readouts.append(as_from_register(g, r.adapter, r.connector))

        send_values(
            readouts,
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
        self, invoker: Connector, internal: bool
    ) -> closer[Invocation[FlowInputInto, FlowControlInto]]:
        return expansion_invocation(
            self,
            invoker,
            FlowInputInto,
            FlowControlInto,
        )

    def close(self) -> None:
        for register in self.variable_registers.values():
            register.close()
        closer(self.flow_input).close()
        closer(self.control_output).close()
        self.exceptions.close()
        optimize(self, self.active_pairs)


def _push(l: list, v: Any) -> None:
    l.append(v)


@dataclasses.dataclass
class ExceptionSink:
    adapter: Adapter

    @cached_property
    def input_adapter(self) -> Adapter:
        return self.adapter

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter([self.ref_adapter, self.adapter])

    @cached_property
    def ref_adapter(self) -> Adapter:
        return RA_VA

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[MergeInputTo, MergeOutputInto]]:
        return expansion_invocation(self, invoker, MergeInputTo, MergeOutputInto)

    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        a, b = executor.tuplate(wires[0])
        executor.connect(b, port)

        if isinstance(port, Erasure):
            if port.value:
                (x, y), z = merge_invocation(_push, executor)
                [
                    input_x,
                ] = borrow_registers(
                    [as_from_register(a, self.ref_adapter, executor)], z
                )
                send_value(input_x, x)
                send_value(as_from_register(port.value, VA, executor), y)
                return

        self.ref_adapter.close(a, executor)


@dataclasses.dataclass(frozen=True, slots=True)
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
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        with executor.sequenced_tuplate_from(wires[0]) as wire_iter:
            self.right.close(next(wire_iter), executor)
            executor.connect(port, next(wire_iter))


@dataclasses.dataclass(frozen=True)
class Loop(ExpansionWithAdapters):
    iteration: VariablesFlow
    body: VariablesFlow
    orelse: VariablesFlow

    @cached_property
    def input_adapter(self) -> Adapter:
        return flow_input_adapter(self.body.variables)

    @cached_property
    def output_adapter(self) -> Adapter:
        return flow_control_adapter(self.body.return_adapter, self.body.variables)

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

            with self.body.invocation(builder, internal=True) as body_invocation:
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

            body_return, body_break_variables = body_return.choice(body_break_variables)
            body_break_variables, body_finish_variables = body_break_variables.choice(
                body_finish_variables
            )

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
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            for wire in wires:
                executor.annihilate(wire, port)
            return

        with (
            unpack_port_and_wires(
                self, port, wires, executor, FlowInputFrom, FlowControlFrom
            ) as this_invocation,
            self.iteration.invocation(executor, internal=True) as iterable_invocation,
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
class IfThenElseBase(ExpansionWithAdapters):
    true_case: ExpansionWithAdapters
    false_case: ExpansionWithAdapters

    def __copy__(self) -> Self:
        return self

    @cached_property
    def input_adapter(self) -> Adapter:
        return VA  # value

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

    def __call__(self, exec: Connector, port: Port, wires: Sequence[Wire]) -> None:
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


class IfThenElse(IfThenElseBase):
    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[IfThenElseInputInto, IfThenElseOutputInto]]:
        return expansion_invocation(
            self, invoker, IfThenElseInputInto, IfThenElseOutputInto
        )


@dataclasses.dataclass(frozen=True)
class IfThenElseStatement(IfThenElseBase):
    true_case: VariablesFlow
    false_case: VariablesFlow

    def invocation(
        self, invoker: Connector
    ) -> closer[Invocation[IfThenElseInputInto, IfThenElseStatementOutputInto]]:
        return expansion_invocation(
            self, invoker, IfThenElseInputInto, IfThenElseStatementOutputInto
        )

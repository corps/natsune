import contextlib
import copy
import dataclasses
from functools import cached_property
from typing import Sequence, Self, Callable, Any, Iterator, cast, Generator

from .adapters import (
    Adapter,
    ValueAdapter,
    ParValueAdapter,
    Variables,
    AlternativesAdapter,
)
from .ambiguous import AmbiguousPair
from .connector import Connector, BufferingConnector
from .executor import Executor
from .invocations import (
    ExpansionWithAdapters,
    expansion_invocation,
    filter_invocation,
    send_parameter,
    unpack_wires,
    merge_invocation,
    send_parameters,
    unpack_port_and_wires,
)
from .ports import Port, Wire, ValuePort, WirePort, Erasure
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
class OneOf(ExpansionWithAdapters):
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
    ) -> tuple[ToRegister, ToRegister, FromRegister]:
        inputs, outputs = expansion_invocation(self, invoker)
        alt, result = outputs.split(True)
        return inputs.readin(), alt.readin(), result.readout(True)

    def __copy__(self) -> Self:
        return self

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        a, b = exec.tuplate(wires[0])

        if isinstance(port, Erasure):
            if port.value is None:
                exec.connect(a, b)
            else:
                exec.annihilate(a, port)
                exec.annihilate(b, port)
            return

        exec.connect(port, b)
        exec.annihilate(a)

    @classmethod
    def share(
        cls, to: ToRegister, connector: Connector
    ) -> tuple[ToRegister, ToRegister]:
        a, b, c = cls(to.adapter).invocation(connector)
        send_value(c, to)
        return a, b


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
        return FlowInvocation.split(inputs, outputs)

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if not isinstance(port, ValuePort):
            for wire in wires:
                executor.annihilate(wire, port)
            for wire in port.wires:
                executor.annihilate(wire)
            return

        inputs, outputs = unpack_port_and_wires(self, port, wires, executor)

        with (
            FlowInvocation.split(inputs, outputs) as this_invocation,
            this_invocation.control.share() as (orelse_result, body_result),
            self.iteration.invocation(executor) as iterable_invocation,
            self.body.invocation(executor) as body_invocation,
            self.orelse.invocation(executor) as orelse_invocation,
            self.invocation(executor) as recurse,
        ):
            iter_input_1, iter_input_2 = this_invocation.inputs.i_value.readout(
                True
            ).duplicate()
            send_value(
                this_invocation.inputs.i_variables.readout(True),
                iterable_invocation.inputs.i_variables.readin(),
            )
            send_value(iter_input_1, iterable_invocation.inputs.i_value.readin())
            send_value(iter_input_2, recurse.inputs.i_value.readin())

            cond_invocation = IfThenElse(self.body.variables.adapter).invocation(
                executor
            )
            send_value(
                iterable_invocation.control.o_return.readout(True),
                cond_invocation.i_value.readin(),
            )
            send_value(
                iterable_invocation.control.o_finish.readout(True),
                cond_invocation.i_context.readin(),
            )

            send_value(
                cond_invocation.o_orelse.readout(True),
                orelse_invocation.inputs.i_variables.readin(),
            )
            send_value(
                cond_invocation.o_body.readout(True),
                body_invocation.inputs.i_variables.readin(),
            )

            orelse_invocation.control.send_to(orelse_result, True)

            # Option to return from parent
            o_return_1, o_return_2 = OneOf.share(
                body_result.o_return.readin(), executor
            )
            # We return if either the current body returns or the recursion returns
            send_value(body_invocation.control.o_return.readout(True), o_return_1)
            send_value(recurse.control.o_return.readout(True), o_return_2)

            # Option to finish through parent
            o_finished_1, o_finished_2 = OneOf.share(
                body_result.o_finish.readin(), executor
            )
            # We are finished if either the current body breaks or the recursion finishes.
            send_value(body_invocation.control.o_break.readout(True), o_finished_1)
            send_value(recurse.control.o_finish.readout(True), o_finished_2)

            # Option to recurse again
            o_continue_1, o_continue_2 = OneOf.share(
                recurse.inputs.i_variables.readin(), executor
            )
            # We reurse if either the current body continues or finishes
            send_value(body_invocation.control.o_continue.readout(True), o_continue_1)
            send_value(body_invocation.control.o_finish.readout(True), o_continue_2)


@dataclasses.dataclass(slots=True, frozen=True)
class IfThenElse(ExpansionWithAdapters):
    context_adapter: Adapter

    def __copy__(self) -> Self:
        return self

    @cached_property
    def input_adapter(self) -> Adapter:
        return ValueAdapter()  # value

    @cached_property
    def output_adapter(self) -> Adapter:
        return ParValueAdapter(
            [
                self.context_adapter,
                AlternativesAdapter(
                    [
                        self.context_adapter,
                        self.context_adapter,
                    ]
                ),
            ]
        )

    def invocation(self, invoker: Connector) -> IfThenElseInvocation:
        inputs, outputs = expansion_invocation(self, invoker)
        context, alternatives = outputs.split(True)
        a, b = alternatives.split(True)
        return IfThenElseInvocation(inputs, context, a, b)

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        if not isinstance(port, ValuePort):
            for wire in port.wires:
                exec.annihilate(wire, port)
            return

        context, alternatives = as_to_register(
            WirePort([wires[0]]), self.output_adapter, exec
        ).split()
        i_context = context.invert()
        a, b = alternatives.split()

        if port.value:
            send_value(i_context, a)
            b.close()
        else:
            send_value(i_context, b)
            a.close()


@dataclasses.dataclass(frozen=True, slots=True)
class IfThenElseInvocation:
    i_value: InterfaceRegister
    i_context: InterfaceRegister
    o_body: InterfaceRegister
    o_orelse: InterfaceRegister

    def __enter__(self) -> Self:
        return self

    def close(self) -> None:
        self.i_value.close()
        self.i_context.close()
        self.o_body.close()
        self.o_orelse.close()

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
        result1, result2 = OneOf.share(
            as_to_register(WirePort([wires[0]]), self.output_adapter, exec), exec
        )

        send_value(left, pair_invocation.i_value_1)
        send_value(right, pair_invocation.i_value_2)

        first_value_1, first_value_2 = pair_invocation.o_first_value.duplicate()

        if_invocation = IfThenElse(ValueAdapter()).invocation(exec)
        send_value(
            send_parameter(filter_invocation(self.should_short, exec), first_value_1),
            if_invocation.i_value.readin(),
        )

        send_value(first_value_2, if_invocation.i_context.readin())

        send_value(if_invocation.o_body.readout(True), result1)
        send_value(
            send_parameters(
                merge_invocation(self.merge_operation, exec),
                (if_invocation.o_orelse.readout(True), pair_invocation.o_second_value),
            ),
            result2,
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
        a, b = inputs.split(True)
        return cls(a, b, outputs)


@dataclasses.dataclass(slots=True)
class VariablesFlow(ExpansionWithAdapters):
    variables: Variables
    return_adapter: Adapter

    variable_registers: dict[str, FlowRegister] = dataclasses.field(
        default_factory=dict
    )
    buffer: BufferingConnector = dataclasses.field(default_factory=BufferingConnector)

    def __post_init__(self) -> None:
        for name, variable_input in zip(
            self.variables.keys(), self.flow_input.i_variables.split(True)
        ):
            flow_register = self.variable_registers[name] = FlowRegister(
                self.variables[name], self.buffer
            )
            send_value(variable_input.readout(True), flow_register.interface_readin())

    def variables_readout(self, owned: bool) -> FromRegister:
        x1, x2 = Wire.as_interface()
        send_values(
            [r.readout(owned) for r in self.variable_registers.values()],
            as_to_register(x1, self.variables.adapter, self.buffer).split(),
        )

        return as_from_register(
            x2,
            self.variables.adapter,
            self.buffer,
        )

    @cached_property
    def input_interface(self) -> InterfaceRegister:
        return InterfaceRegister(
            FlowInput.adapter(self.variables),
            self.buffer,
        )

    @cached_property
    def output_interface(self) -> InterfaceRegister:
        return InterfaceRegister(
            FlowControl.adapter(self.return_adapter, self.variables),
            self.buffer,
        )

    @cached_property
    def o_control(self) -> FlowControl:
        return FlowControl.split(self.output_interface)

    @cached_property
    def flow_input(self) -> FlowInput:
        return FlowInput.split(self.input_interface)

    @property
    def input_adapter(self) -> Adapter:
        return self.input_interface.adapter

    @property
    def output_adapter(self) -> Adapter:
        return self.output_interface.adapter

    def __copy__(self) -> Self:
        return self

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire], /) -> None:
        new_wire_identity: dict[int, Wire] = {}
        q: list[Port] = []
        pairs: list[tuple[Port, Port]] = []

        if isinstance(port, Erasure):
            for wire in wires:
                exec.annihilate(wire, port)
            return

        for l, r in self.buffer.active_pairs:
            ll = copy.copy(l)
            rr = copy.copy(r)
            pairs.append((ll, rr))
            q.append(ll)
            q.append(rr)

        return_port = copy.copy(self.output_interface.interface)
        exec.connect(return_port, wires[0])
        q.append(return_port)

        args_port = copy.copy(self.input_interface.interface)
        exec.connect(args_port, port)
        q.append(args_port)

        while q:
            head = q.pop()

            for i, wire in enumerate(head.wires):
                if id(wire) in new_wire_identity:
                    wire = new_wire_identity[id(wire)]
                    assert not wire.target
                else:
                    wire = copy.copy(wire)
                    new_wire_identity[id(wire)] = wire
                    if wire.target:
                        q.append(wire.target)

                head.wires[i] = wire

        for l, r in pairs:
            exec.connect_ports(l, r)

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
    def adapter(cls, return_adapter: Adapter, variables: Variables) -> Adapter:
        return ParValueAdapter(
            [
                FlowInput.adapter(variables),
                FlowControl.adapter(return_adapter, variables),
            ]
        )

    @classmethod
    def split(
        cls, inputs: InterfaceRegister, outputs: InterfaceRegister
    ) -> FlowInvocation:
        return FlowInvocation(FlowInput.split(inputs), FlowControl.split(outputs))

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

    @classmethod
    def split(cls, source: InterfaceRegister) -> FlowInput:
        return FlowInput(*source.split(True))

    def close(self) -> None:
        self.i_variables.close()
        self.i_value.close()


@dataclasses.dataclass(slots=True, frozen=True)
class FlowControl:
    o_return: InterfaceRegister
    o_continue: InterfaceRegister
    o_break: InterfaceRegister
    o_finish: InterfaceRegister

    @classmethod
    def adapter(cls, return_adapter: Adapter, variables: Variables) -> Adapter:
        return AlternativesAdapter(
            [
                return_adapter,
                variables.adapter,
                variables.adapter,
                variables.adapter,
            ]
        )

    @classmethod
    def split(cls, source: InterfaceRegister) -> FlowControl:
        return FlowControl(*source.split(True))

    def send_to(self, i_control: FlowControl, owned: bool) -> None:
        send_values(
            [
                self.o_return.readout(owned),
                self.o_continue.readout(owned),
                self.o_break.readout(owned),
                self.o_finish.readout(owned),
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
        for field in dataclasses.fields(self):
            if field.type is InterfaceRegister:
                yield cast(InterfaceRegister, getattr(self, field.name))

    @contextlib.contextmanager
    def share(self) -> Generator[tuple[FlowControl, FlowControl]]:
        a, b = tuple(
            FlowControl(*parts)
            for parts in zip(*(OneOf.share(i.readin(), i.connector) for i in self))
        )
        try:
            yield a, b
        finally:
            a.close()
            b.close()

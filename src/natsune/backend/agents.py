from typing import Any, Iterator, Sequence

from natsune.adapters import Adapter
from natsune.connector import Connector, ExpansionBuilder, FrozenExpansion
from natsune.control_flow import (
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
from natsune.invocations import Invocation, closer, pack_from, pack_into
from natsune.ports import Graft, Port, Wire
from natsune.registers import (
    FromInterfaceRegister,
    FromRegister,
    ToInterfaceRegister,
    ToRegister,
)

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
    """Late-bound callee net (CUTOVER.md §2): stands in for a
    FrozenExpansion that is still being compiled — recursive and mutually
    recursive inet calls. Call-site grafts baked with the promise
    materialize through it once the owning compile fills `target`.

    The adapters must match what the finished net will expose: the owner
    builds them from its collected symbols through the same
    flow_input_adapter/flow_control_adapter constructors the real net's
    VariablesFlow uses, and the fill asserts the match (CUTOVER.md §2).

    By-reference mutation of the promise is the accepted Python-target
    semantics. Fills always land before any net runs: a promise is only
    requested during its owner's callee pass, and a nested compile — fill
    included — completes before the caller's own lowering starts, while
    nets execute only from entry points after compilation.
    """

    def __init__(
        self, name: str, input_adapter: Adapter, output_adapter: Adapter
    ) -> None:
        self.name = name
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.target: FrozenExpansion | None = None

    def __copy__(self) -> "PromiseExpansion":
        return self  # every baked graft shares the promise; fills are global

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

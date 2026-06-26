from multiprocessing import Value
import dataclasses
from functools import cached_property
from typing import Any, Self, Sequence

from .adapters import ParValueAdapter, Adapter, ValueAdapter
from .connector import Connector
from .ports import Expansion
from .registers import as_from_register, as_to_register, FromRegister, ToRegister
from .executor import Executor
from .ports import ForkPort, Graft, Wire, Port, WirePort


@dataclasses.dataclass(slots=True)
class AmbiguousPair(Expansion):
    second_amb_half: ForkPort | None = None

    # Once one side is copied, the other side must agree to the copy.
    # In essence, a graft itself is only copied at most once, and
    # at most two graft share this expansion, thus the copy connects
    # in the same way
    copy: Any | None = None

    def invocation(self, invoker: Connector) -> AmbiguousInvocation:
        graft_1 = Graft(self, wires=[Wire(), Wire()])
        graft_2 = Graft(self, wires=graft_1.wires)
        input_1 = as_to_register(graft_1, ValueAdapter(), invoker)
        input_2 = as_to_register(graft_2, ValueAdapter(), invoker)
        output_1 = as_from_register(
            WirePort([graft_1.wires[0]]), ValueAdapter(), invoker
        )
        output_2 = as_from_register(
            WirePort([graft_1.wires[1]]), ValueAdapter(), invoker
        )
        return AmbiguousInvocation(input_1, input_2, output_1, output_2)

    def __copy__(self) -> Self:
        if self.copy is None:
            self.copy = dataclasses.replace(
                self,
                copy=None,
                second_amb_half=None,
            )
        return self.copy  # type: ignore

    def __call__(self, exec: Executor, port: Port, wires: Sequence[Wire]) -> None:
        if self.second_amb_half is not None:
            exec.connect_ports(port, self.second_amb_half)
            return

        forked = exec.fork()
        ambiguous_primary = ForkPort(forked)
        ambiguous_secondary = ForkPort(forked)
        ambiguous_primary_aux = ForkPort(forked)
        ambiguous_secondary_aux = ForkPort(forked)
        self.second_amb_half = ambiguous_secondary

        exec.connect(ambiguous_primary.wires[0], ambiguous_primary_aux.wires[0])
        exec.connect(ambiguous_primary.wires[1], ambiguous_secondary_aux.wires[1])

        exec.connect(ambiguous_secondary.wires[0], ambiguous_secondary_aux.wires[0])
        exec.connect(ambiguous_secondary.wires[1], ambiguous_primary_aux.wires[1])

        exec.connect(wires[0], ambiguous_primary_aux)
        exec.connect(wires[1], ambiguous_secondary_aux)
        exec.connect_ports(port, ambiguous_primary)

        return


@dataclasses.dataclass(frozen=True, slots=True)
class AmbiguousInvocation:
    i_value_1: ToRegister
    i_value_2: ToRegister
    o_first_value: FromRegister
    o_second_value: FromRegister

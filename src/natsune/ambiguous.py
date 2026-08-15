import ctypes
import dataclasses
from typing import Sequence

from natsune.adapters import Adapter
from natsune.connector import Connector
from natsune.deque import cas_ptr
from natsune.ports import Expansion, Graft, Port, Wire
from natsune.registers import FromRegister, ToRegister, as_from_register, as_to_register


@dataclasses.dataclass(slots=True)
class AmbiguousPair(Expansion):
    copied: ctypes.c_int64 = dataclasses.field(
        default_factory=lambda: ctypes.c_int64(0)
    )

    def invocation(self, invoker: Connector, adapter: Adapter) -> AmbiguousInvocation:
        graft_1 = Graft(self, wires=[Wire(), Wire()])
        graft_2 = Graft(self, wires=graft_1.wires)
        input_1 = as_to_register(graft_1, adapter, invoker)
        input_2 = as_to_register(graft_2, adapter, invoker)
        output_1 = as_from_register(graft_1.wires[0], adapter, invoker)
        output_2 = as_from_register(graft_1.wires[1], adapter, invoker)
        return AmbiguousInvocation(input_1, input_2, output_1, output_2)

    def __call__(self, exec: Connector, port: Port, wires: Sequence[Wire]) -> None:
        if cas_ptr(self.copied, 0, 1):
            exec.connect(port, wires[0])
            return

        exec.connect(port, wires[1])

    def __copy__(self) -> AmbiguousPair:
        return AmbiguousPair()


@dataclasses.dataclass(frozen=True, slots=True)
class AmbiguousInvocation:
    i_value_1: ToRegister
    i_value_2: ToRegister
    o_first_value: FromRegister
    o_second_value: FromRegister

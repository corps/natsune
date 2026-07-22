import dataclasses
from collections import defaultdict
from typing import Any, Callable, Iterator, Literal, Self, Sequence

from natsune.adapters import Adapter, ValueAdapter
from natsune.ambiguous import AmbiguousPair
from natsune.connector import serialize_active_pairs
from natsune.executor import Executor, SynchronizedExecutor
from natsune.optimizer import optimize
from natsune.ports import (
    CombPort,
    Erasure,
    Expansion,
    ExtMergeFuncPort,
    ExtSplitFuncPort,
    ForkPort,
    Graft,
    Port,
    Target,
    ValuePort,
    Wire,
    WirePort,
)
from natsune.registers import (
    FromRegister,
    ToRegister,
    as_constant_register,
    as_from_register,
    as_to_register,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TraceExpansion(Expansion):
    buffer: list[Port] = dataclasses.field(default_factory=list)

    def __copy__(self) -> Self:
        return self

    def __call__(
        self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        if isinstance(port, Erasure):
            if port.value:
                self.buffer.append(port.value)
        elif isinstance(port, ValuePort):
            self.buffer.append(port.value)
        for wire in port.wires:
            executor.connect(wire, Graft(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Calculus:
    _targets: list[Target] = dataclasses.field(default_factory=list)
    _wires: dict[int, Wire] = dataclasses.field(
        default_factory=lambda: defaultdict(Wire)
    )
    executor: SynchronizedExecutor = dataclasses.field(
        default_factory=SynchronizedExecutor
    )
    tracer: TraceExpansion = dataclasses.field(default_factory=TraceExpansion)

    def __setitem__(self, key: int | Target, value: Target) -> None:
        inner_wire = self[value]
        if not isinstance(key, int):
            self.executor.connect(key, inner_wire)
        else:
            self.executor.connect(self._wires[key], inner_wire)

    def __getitem__(self, key: int | Target) -> Wire:
        if not isinstance(key, int):
            port_id = id(key)
            if port_id not in self._wires:
                self._targets.append(key)
                wire = self._wires[port_id]
                self.executor.connect(wire, key)
            return self._wires[port_id]
        return self._wires[key]

    def readout(self, key: int | Target) -> Iterator[Any]:
        self[key] = Graft(self.tracer)
        yield from self.continue_readout()

    def continue_readout(self) -> Iterator[Any]:
        while self.executor.active_pairs:
            self.executor.process_pair()
            yield from self.tracer.buffer
            self.tracer.buffer.clear()

    def reduce(self, target: int | Target) -> Port | None:
        list(self.continue_readout())
        result = self[target].target
        while isinstance(result, WirePort):
            result = result.wires[0].target
        return result

    def reduce_to_value(self, target: int | Target) -> Any:
        p = self.reduce(target)
        assert isinstance(p, ValuePort), p
        return p.value

    def optimize(self, level: Literal[1, 2, 3] = 3) -> None:
        optimize(self.executor, self.executor.active_pairs, level=level)

    def take_step(self) -> list[str]:
        self.optimize()
        self.process_next_interaction()
        self.optimize()
        return self.serialize_active_pairs()

    def serialize_active_pairs(self) -> list[str]:
        return serialize_active_pairs(self.executor.active_pairs, self._wires)

    def process_next_interaction(self) -> None:
        self.executor.process_pair()

    def find_target(self, w: Wire) -> Port | None:
        target = w.target
        while isinstance(target, WirePort):
            target = target.wires[0].target
        return target

    def tup(self, a: Wire, b: Wire) -> Wire:
        return self[CombPort("x", [a, b])]

    def merge(self, f: Callable[[Any, Any], Any], a: Wire, b: Wire) -> Wire:
        return self[ExtMergeFuncPort(f, [a, b])]

    def split(self, f: Callable[[Any], tuple[Any, Any]], a: Wire, b: Wire) -> Wire:
        return self[ExtSplitFuncPort(f, [a, b])]

    def fork(self, a: Wire, b: Wire) -> Wire:
        return self[ForkPort(self.executor, [a, b])]

    def dup(self, a: Wire, b: Wire) -> Wire:
        return self[CombPort("dup", [a, b])]

    def v(self, a: Any) -> Wire:
        return self[ValuePort(a)]

    def amb(self, a: Wire, b: Wire, c: Wire, d: Wire) -> None:
        pair = AmbiguousPair()
        self[b] = self[Graft(pair, [c, d])]
        self[a] = self[Graft(pair, [c, d])]

    def e(self) -> Wire:
        return self[Erasure()]

    def to_key(self, key: int | Target, adapter: Adapter | None = None) -> ToRegister:
        return as_to_register(self[key], adapter or ValueAdapter(), self.executor)

    def from_key(
        self, key: int | Target, adapter: Adapter | None = None
    ) -> FromRegister:
        return as_from_register(self[key], adapter or ValueAdapter(), self.executor)

    def const(self, value: Any) -> FromRegister:
        return as_constant_register(value, self.executor)

    def erasure(self) -> FromRegister:
        return as_from_register(Erasure(), ValueAdapter(), self.executor)

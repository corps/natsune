import dataclasses
from collections import defaultdict
from typing import Any, Iterator, Self, Sequence, Callable, Literal

from natsune.adapters import Adapter, ValueAdapter
from natsune.ambiguous import AmbiguousPair
from natsune.executor import SynchronizedExecutor, Executor
from natsune.optimizer import optimize
from natsune.ports import (
    Wire,
    Port,
    Target,
    ExtSplitFuncPort,
    Erasure,
    Expansion,
    Graft,
    CombPort,
    ValuePort,
    ExtMergeFuncPort,
    ForkPort,
    WirePort,
)
from natsune.registers import (
    ToRegister,
    as_to_register,
    FromRegister,
    as_from_register,
    as_constant_register,
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

    def optimize(self, level: Literal[1, 2, 3] = 3) -> None:
        optimize(self.executor, self.executor.active_pairs, level=level)

    def serialize_active_pairs(self) -> list[str]:
        parts: list[str] = []
        cache = self.new_wires_cache()

        for i, k in self._wires.items():
            if i < 1000:
                cache[k] = f"[{i}]"

        for l, r in self.executor.active_pairs:
            parts.append(
                self.serialize_port(l, cache, True)
                + " = "
                + self.serialize_port(r, cache, False)
            )
        return parts

    def process_next_interaction(self) -> None:
        self.executor.process_pair()

    def find_target(self, w: Wire) -> Port | None:
        target = w.target
        while isinstance(target, WirePort):
            target = target.wires[0].target
        return target

    def new_wires_cache(self) -> dict[Wire, str]:
        cache: dict[Wire, str] = defaultdict(lambda: "w" + str(len(cache)))
        return cache

    def serialize_port(
        self, port: Port, wires_cache: dict[Wire, str], reverse: bool
    ) -> str:
        if isinstance(port, ValuePort):
            return (
                str(port.value)
                if isinstance(port.value, (int, str, type(None), bool, float))
                else "value"
            )
        elif isinstance(port, Erasure):
            return "Z"
        elif isinstance(port, ExtMergeFuncPort):
            front = "fnM"
        elif isinstance(port, ExtSplitFuncPort):
            front = "fnS"
        elif isinstance(port, ForkPort):
            front = "fork"
        elif isinstance(port, CombPort):
            front = port.label
        elif isinstance(port, Graft):
            front = "graft"
        elif isinstance(port, WirePort):
            wire_str = self.serialize_calculus_wire(port.wires[0], wires_cache, reverse)
            if reverse:
                return wire_str + ">-"
            return "-<" + wire_str
        else:
            raise NotImplementedError(str(type(port)))

        if port.wires:
            if reverse:
                return (
                    "("
                    + ", ".join(
                        [
                            self.serialize_calculus_wire(w, wires_cache, reverse)
                            for w in port.wires
                        ][::-1]
                    )
                    + ")"
                    + front
                )
            return (
                front
                + "("
                + ", ".join(
                    [
                        self.serialize_calculus_wire(w, wires_cache, reverse)
                        for w in port.wires
                    ]
                )
                + ")"
            )

        return front

    def serialize_calculus_wire(
        self, wire: Wire, wires_cache: dict[Wire, str], reverse: bool
    ) -> str:
        if wire.target is None:
            return wires_cache[wire]
        return self.serialize_port(wire.target, wires_cache, reverse)

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
        return as_to_register(
            WirePort([self[key]]), adapter or ValueAdapter(), self.executor
        )

    def from_key(
        self, key: int | Target, adapter: Adapter | None = None
    ) -> FromRegister:
        return as_from_register(
            WirePort([self[key]]), adapter or ValueAdapter(), self.executor
        )

    def const(self, value: Any) -> FromRegister:
        return as_constant_register(value, self.executor)

    def erasure(self) -> FromRegister:
        return as_from_register(Erasure(), ValueAdapter(), self.executor)

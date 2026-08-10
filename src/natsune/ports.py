import copy
import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    MutableSequence,
    Protocol,
    Self,
    Sequence,
    cast,
)

if TYPE_CHECKING:
    from .connector import Connector


class Port:
    wires: MutableSequence[Wire]

    def __copy__(self) -> Self:
        return copy.replace(self, wires=[*self.wires] if self.wires else self.wires)

    def __replace__(self, /, **kv: Any) -> Self:
        copy = self.__copy__()
        for k, v in kv.items():
            setattr(copy, k, v)
        return copy


@dataclasses.dataclass
class Wire:
    target: Port | None = None

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __copy__(self) -> Self:
        return dataclasses.replace(
            self,
            target=copy.copy(self.target) if self.target else None,
        )

    @classmethod
    def as_tautology(cls) -> tuple[WirePort, Wire]:
        wp = WirePort()
        return wp, wp.wires[0]


type Target = Wire | Port


@dataclasses.dataclass(frozen=True, slots=True)
class WirePort(Port):
    wires: MutableSequence[Wire] = dataclasses.field(default_factory=lambda: [Wire()])


@dataclasses.dataclass(frozen=True, slots=True)
class ValuePort(Port):
    value: Any
    wires: MutableSequence[Wire] = dataclasses.field(default_factory=list)

    def __copy__(self) -> ValuePort:
        return ValuePort(copy.copy(self.value))


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantValuePort(ValuePort):
    value: Any

    def __copy__(self) -> ValuePort:
        return ConstantValuePort(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class Erasure(Port):
    # May contain exception data
    value: Any = None
    wires: MutableSequence[Wire] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True, slots=True)
class CombPort(Port):
    label: Any
    wires: MutableSequence[Wire] = dataclasses.field(
        default_factory=lambda: [Wire(), Wire()]
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ExtMergeFuncPort(Port):
    fn: Callable[[Any, Any], Any]
    wires: list[Wire] = dataclasses.field(default_factory=lambda: [Wire(), Wire()])
    swapped: bool = False

    def __repr__(self) -> str:
        return f"ExtMergeFuncPort({getattr(self.fn, '__name__', str(self.fn))})"


@dataclasses.dataclass(frozen=True, slots=True)
class ExtSplitFuncPort(Port):
    fn: Callable[[Any], tuple[Any, Any]]
    wires: MutableSequence[Wire] = dataclasses.field(
        default_factory=lambda: [Wire(), Wire()]
    )

    def __repr__(self) -> str:
        return f"ExtSplitFuncPort({getattr(self.fn, '__name__', str(self.fn))})"


@dataclasses.dataclass(frozen=True, slots=True)
class ForkPort(Port):
    wires: MutableSequence[Wire] = dataclasses.field(
        default_factory=lambda: [Wire(), Wire()]
    )


class Expansion(Protocol):
    def __call__(
        self, executor: Connector, port: Port, wires: Sequence[Wire], /
    ) -> None: ...

    def __copy__(self) -> Self: ...


@dataclasses.dataclass(frozen=True, slots=True)
class Graft(Port):
    execute: Expansion
    wires: MutableSequence[Wire] = dataclasses.field(default_factory=lambda: [])

    def __copy__(self) -> Self:
        if hasattr(self.execute, "__copy__"):
            return dataclasses.replace(
                self,
                execute=copy.copy(self.execute),
                wires=[*self.wires] if self.wires else self.wires,
            )
        return dataclasses.replace(self)

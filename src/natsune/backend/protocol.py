import ast
import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from natsune.backend.agents import AgentImpl
from natsune.backend.connector import Connector, FrozenExpansion
from natsune.backend.registers import FromRegister, ToRegister
from natsune.first_order.adapters import Adapter
from natsune.frontend import IrFunction


@dataclasses.dataclass(frozen=True, slots=True)
class LoweredUnit:
    func: IrFunction
    main: FrozenExpansion
    agents: tuple[AgentImpl, ...]
    name: str


@runtime_checkable
class Backend(Protocol):
    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        function_globals: Mapping[str, Any],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister: ...

    def materialize_dynamic_to(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        function_globals: Mapping[str, Any],
        adapter: Adapter,
        connector: Connector,
    ) -> ToRegister: ...

    def finish(
        self,
        artifact: LoweredUnit,
    ) -> Any: ...

"""The compile-time Backend protocol (COMPILER_REFACTOR.md §5).

Deliberately compile-time only: VariablesFlow-level templating + agent
declarations. The runtime half (live per-invocation substrate) was removed
from the protocol until §8.1 settles the artifact question; lowering never
needed it — templates record into themselves.
"""

import ast
import dataclasses
from collections.abc import Mapping
from typing import Any, Collection, Protocol, runtime_checkable

from natsune.adapters import Adapter
from natsune.backend.agents import AgentImpl
from natsune.connector import Connector, FrozenExpansion
from natsune.frontend import IrFunction
from natsune.registers import FromRegister, ToRegister


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
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister: ...

    def materialize_dynamic_to(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> ToRegister: ...

    def finish(
        self,
        artifact: LoweredUnit,
    ) -> Any: ...

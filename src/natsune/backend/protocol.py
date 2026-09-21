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
        function_globals: Mapping[str, Any],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister:
        """Evaluate `source_text` with `captures` as its locals.

        `function_globals` is the compiled function's global namespace
        (IrFunction.globals): the names global references in the dynamic
        resolve through. Backends not implemented in terms of Python
        globals may still consult it to build their own view of the
        globals context the function is intended to bear.
        """

    def materialize_dynamic_to(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        function_globals: Mapping[str, Any],
        adapter: Adapter,
        connector: Connector,
    ) -> ToRegister:
        """Execute `source_text` as an assignment target: the value sent
        to the returned ToRegister arrives as `___from_input___`.

        `function_globals` is the compiled function's global namespace
        (IrFunction.globals) — see materialize_dynamic.
        """

    def finish(
        self,
        artifact: LoweredUnit,
    ) -> Any: ...

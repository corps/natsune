from typing import Protocol, runtime_checkable

from natsune.adapters import ParValueAdapter, TypeExpression
from natsune.control_flow import VariablesFlow


# This type is intended to match the InetFunctionCompiler without requiring an import from compiler
# Consider removing the runtime checkable when this is lifted away
@runtime_checkable
class LegacyInetInterface(Protocol):
    @property
    def compiled(self) -> VariablesFlow: ...

    @property
    def args_adapter(self) -> ParValueAdapter: ...

    @property
    def return_annot(self) -> TypeExpression | None: ...

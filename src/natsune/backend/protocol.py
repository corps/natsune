"""The compile-time Backend protocol (COMPILER_REFACTOR.md §5).

Deliberately compile-time only: VariablesFlow-level templating + agent
declarations. The runtime half (live per-invocation substrate) was removed
from the protocol until §8.1 settles the artifact question; lowering never
needed it — templates record into themselves.
"""

import ast
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from natsune.adapters import Adapter
from natsune.backend.types import AgentDef, AgentRef, Artifact
from natsune.connector import Connector
from natsune.control_flow import VariablesFlow
from natsune.ports import Port
from natsune.registers import FromRegister


@runtime_checkable
class Backend(Protocol):
    def declare_agent(self, name: str, defn: AgentDef) -> AgentRef:
        """Register a declaration; only impl RESOLUTION differs per target."""
        ...

    def resolve_call(self, ref: Any) -> AgentRef:
        """IrCallInet.ref (today: opaque old-style __inet__ objects) becomes
        an AgentRef here; metadata (arity, adapters) is copied (§6)."""
        ...

    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister:
        """Receives BOTH the original ast node and its unparsed text;
        backends use whichever is easier (Python evals the text, an emitter
        pattern-matches the node). Mirrors IrDynamic/IrTargetDynamic.ast_node;
        synthesized augassign dynamics carry a synthesized BinOp.

        ``captures`` is the already-lowered form: IrDynamic.captures is
        tuple[tuple[str, IrExpr], ...] at the IR, but lowering resolves each
        IrExpr to a FromRegister before calling — the backend never sees
        IrExpr. FromRegister (not Port) because the eval context needs
        register identity: serialize_values/borrow_registers consume value
        SOURCES (the legacy used_names was dict[str, FromRegister]).
        ``connector`` is the ambient flow the merge wires into (captures
        may be empty — constants stay in the source text).

        Returns a FromRegister: dynamics are inlined eagerly, not deferred
        grafts — matches the legacy shape."""
        ...

    def finish(self, flow: VariablesFlow, *, name: str = "main") -> Artifact:
        """Declares the function's own body as an AgentDef (NetTemplate
        extracted from the flow's active_pairs). What the artifact *is* —
        runnable vs. text — is §8.1, deferred with the runtime protocol."""
        ...

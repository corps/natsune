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
from natsune.control_flow import VariablesFlow
from natsune.ports import Graft, Port


@runtime_checkable
class Backend(Protocol):
    def declare_agent(self, name: str, defn: AgentDef) -> AgentRef:
        """Register a declaration; only impl RESOLUTION differs per target."""
        ...

    def resolve_call(self, ref: Any) -> AgentRef:
        """IrCallInet.ref (today: opaque old-style __inet__ objects) becomes
        an AgentRef here; metadata (arity, adapters) is copied (§6)."""
        ...

    def constant(self, value: Any, adapter: Adapter) -> Port:
        """Adapter-driven value discipline (§8.3) not yet enforced."""
        ...

    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, Port],
        adapter: Adapter,
    ) -> Graft:
        """Receives BOTH the original ast node and its unparsed text;
        backends use whichever is easier (Python evals the text, an emitter
        pattern-matches the node). ``captures`` is the already-lowered form:
        IrDynamic.captures is tuple[tuple[str, IrExpr], ...] at the IR, but
        lowering resolves each IrExpr to a port before calling."""
        ...

    def finish(self, flow: VariablesFlow, *, name: str = "main") -> Artifact:
        """Declares the function's own body as an AgentDef (NetTemplate
        extracted from the flow's active_pairs). What the artifact *is* —
        runnable vs. text — is §8.1, deferred with the runtime protocol."""
        ...

"""PythonBackend: a thin shell over the existing executor machinery.

Compile-time only for now (§5): templating + declarations. Resolution of
NetTemplate impls into live closures arrives with the recorder split
(§5.1 steps 1–2); the runtime half is deferred (§8.1).
"""

import ast
import dataclasses
from collections.abc import Mapping
from typing import Any

from natsune.adapters import VA, Adapter
from natsune.backend.types import (
    AgentDef,
    AgentRef,
    Artifact,
    LoweredUnit,
    NetTemplate,
    PythonCallable,
)
from natsune.control_flow import VariablesFlow
from natsune.ports import ConstantValuePort, Graft, Port


@dataclasses.dataclass
class PythonBackend:
    """The Python target, as far as declaration goes today.

    ``agents`` doubles as the symbol table the §5.1 registry will formalize;
    ``calls`` keys old-style callee objects by identity so refs stay stable
    within a unit (cross-run stability is the §6 nondeterminism note —
    cutover replaces refs with CompiledFunction anyway).
    """

    agents: dict[str, AgentDef] = dataclasses.field(default_factory=dict)
    calls: dict[Any, AgentRef] = dataclasses.field(default_factory=dict)

    def declare_agent(self, name: str, defn: AgentDef) -> AgentRef:
        existing = self.agents.get(name)
        if existing is not None and existing != defn:
            raise ValueError(f"agent {name!r} already declared differently")
        self.agents[name] = defn
        return AgentRef(name)

    def resolve_call(self, ref: Any) -> AgentRef:
        if isinstance(ref, AgentRef):
            return ref
        if ref not in self.calls:
            # Old-style __inet__ compiler object, duck-typed — no import of
            # natsune.compiler. Copy metadata when present (§6 opaque
            # callee refs); stable per-backend ordinal name. The args
            # ParValueAdapter IS the multi-param interface — no unpacking.
            args_adapter = getattr(ref, "args_adapter", None)
            name = f"call_{len(self.calls)}"
            self.calls[ref] = self.declare_agent(
                name,
                AgentDef(
                    name,
                    args_adapter if args_adapter is not None else VA,
                    getattr(ref, "return_adapter", VA),
                    PythonCallable(ref),
                ),
            )
        return self.calls[ref]

    def constant(self, value: Any, adapter: Adapter) -> Port:
        return ConstantValuePort(value)

    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, Port],
        adapter: Adapter,
    ) -> Graft:
        raise NotImplementedError(
            "deferred: the dynamic fallback lands with lowering (§8.4); "
            "its Python realization wraps today's eval_expression path"
        )

    def finish(self, flow: VariablesFlow, *, name: str = "main") -> Artifact:
        body_name = f"{name}__body"
        template = NetTemplate(
            flow.input_adapter,
            flow.output_adapter,
            tuple(flow.active_pairs),
        )
        self.declare_agent(
            body_name,
            AgentDef(body_name, flow.input_adapter, flow.output_adapter, template),
        )
        return LoweredUnit(
            name=name,
            body=AgentRef(body_name),
            agents=dict(self.agents),
        )

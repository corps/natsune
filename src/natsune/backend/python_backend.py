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
    net_template_of,
)
from natsune.compiler import construct_locals, eval_expression
from natsune.connector import Connector
from natsune.control_flow import VariablesFlow
from natsune.invocations import merge_invocation, send_parameters
from natsune.ports import ConstantValuePort, Port
from natsune.registers import (
    FromRegister,
    as_constant_register,
    borrow_registers,
    send_value,
    serialize_values,
)


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
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister:
        """The Python resolution of the dynamic fallback (§8.4): eval the
        rewritten source against a (globals, locals) context whose locals
        are the serialized captured values — the old
        evaluate_from_expression/construct_context, verbatim. CppBackend
        refuses this method (diagnostic per §4(2)).

        ``adapter`` is accepted for signature symmetry with the IR but the
        eval result carries VA, as in legacy; ``node`` goes unused here (an
        emitter backend is the consumer that pattern-matches it).
        ``eval_expression``/``construct_locals`` are legacy ext fns until
        they become Primitive/table entries (§4(1)); try-machinery wrapping
        (collect_exceptions) stays out — try is rejected outright."""

        (text_in, context_in), result = merge_invocation(eval_expression, connector)
        result_a, result_b = result.duplicate("share")
        send_value(as_constant_register(source_text, connector), text_in)

        value_readout = borrow_registers(list(captures.values()), result_b)
        # Evaluation order mirrors old construct_context exactly —
        # serialize2, then globals, then the construct_locals merge (which
        # evaluates serialize_n, then the keys constant). Invocation helpers
        # connect pairs eagerly, so creation order is net structure.
        context = send_parameters(
            serialize_values(connector, 2),
            (
                as_constant_register({}, connector),
                send_parameters(
                    merge_invocation(construct_locals, connector),
                    (
                        send_parameters(
                            serialize_values(connector, len(captures)),
                            value_readout,
                        ),
                        as_constant_register(tuple(captures.keys()), connector),
                    ),
                ),
            ),
        )
        send_value(context, context_in)
        return result_a

    def finish(self, flow: VariablesFlow, *, name: str = "main") -> Artifact:
        body_name = f"{name}__body"
        template = net_template_of(flow)
        self.declare_agent(
            body_name,
            AgentDef(body_name, flow.input_adapter, flow.output_adapter, template),
        )
        return LoweredUnit(
            name=name,
            body=AgentRef(body_name),
            agents=dict(self.agents),
        )

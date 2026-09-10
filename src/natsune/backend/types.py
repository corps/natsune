"""Declaration-layer types for the backend phase (COMPILER_REFACTOR.md §5).

Compile-time only: VariablesFlow-level templating + agent declarations.
Runtime — the live per-invocation substrate a net is instantiated into —
is deferred (§8.1); nothing here executes a net.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from natsune.adapters import Adapter
from natsune.connector import NetTemplateBuilder
from natsune.ports import AgentRef, Port


@dataclasses.dataclass(frozen=True, slots=True)
class InetCallable:
    """impl referencing a callable net object in the runtime environment
    (§7.2a) — target-neutral by design: the Python backend locates the
    object and constructs the invocation protocol against its flow; the
    C++ backend resolves same-unit callees to emitted functions and
    refuses foreign ones (§4 gating).

    Today ``ref`` is the old-style ``__inet__`` InetFunctionCompiler (the
    very object IrCallInet.ref carries); at cutover it becomes
    CompiledFunction. Resolution duck-types the runtime object: live
    Expansion instances resolve as themselves (agent_invocation path);
    legacy compilers resolve through callee_invocation's manual wiring —
    the same shape as NetTemplate resolution, with a live flow instead of
    recorded pairs and a slightly different way of filling in registers
    (variable-interface sorting vs. template data).
    """

    ref: Any


@dataclasses.dataclass(frozen=True, slots=True)
class Primitive:
    """Target-intrinsic op, keyed — never a closure. PythonBackend may
    resolve to a builtin; CppBackend to its intrinsic table (§5 sketch)."""

    kind: str


@dataclasses.dataclass(frozen=True, slots=True)
class NetTemplate:
    """A data-shaped net template, produced from a NetTemplateBuilder via
    net_template_of (§5.1). Knows nothing about executors."""

    input_adapter: Adapter
    output_adapter: Adapter
    pairs: tuple[tuple[Port, Port], ...]


def net_template_of(builder: NetTemplateBuilder) -> NetTemplate:
    """Extract the recorded template as data. The canonical producer once
    the recorder split (§5.1) landed."""

    return NetTemplate(
        builder.input_adapter,
        builder.output_adapter,
        tuple(builder.active_pairs),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AgentDef:
    """Declarative agent description (§4): literally the interaction-net
    symbol table entry. Registration is declaration; resolution is
    per-target.

    ``input_adapter`` is a single Adapter — multiplicity is expressed by the
    adapter lattice itself (ParValueAdapter), not by a Python tuple, so
    there is exactly one spelling per interface."""

    name: str
    input_adapter: Adapter
    output_adapter: Adapter
    impl: InetCallable | NetTemplate | Primitive


@dataclasses.dataclass(frozen=True, slots=True)
class LoweredUnit:
    """Compile-time artifact: everything known about one lowered function.
    No runtime half — §8.1 decides what wraps this (runnable vs. text).

    Deliberately carries no interface fields of its own: the body AgentDef
    records the flow interface (FlowInput/FlowControl over variable cells).
    The unit's *call* interface (Par of params + return) is a different
    adapter that coincides with it only while lowering is unwritten; it
    returns here with that role once lowering exists."""

    name: str
    body: AgentRef
    agents: Mapping[str, AgentDef]

    @property
    def body_def(self) -> AgentDef:
        """The declared body — one hop instead of table spelunking."""
        return self.agents[self.body.name]


#: What Backend.finish returns; renamed once §8.1 settles artifact kinds.
Artifact = LoweredUnit

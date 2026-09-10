"""Survey of the existing agent library as declarations (unit 1, §5.1).

The library splits two ways:

**Primitives** — self-contained expansions with static adapter signatures.
They translate 1:1 to static AgentDefs whose impl is InetCallable (the
live instance IS today's runtime-environment resolution). Parameterized primitives
(ConcurrentValueMerge(should_shortcircuit, merger), the ext-fn invocations
merge/split/filter in invocations.py) are factories: lowering constructs
them per-site, so they get AgentDefs at declaration time, not here.

**Composites** — parameterized by sub-templates (IfThenElse family, Loop).
Not declarable statically; lowering declares one per call-site after
lowering the bodies (impl=NetTemplate embedding AgentRefs to the branch
templates). The dispatch/recursion machinery underneath is primitive.
Their .invocation() side effects — flow_map mutation threading variable
cells — move into lowering, fed by IrBody.variable_usage (§6 first bullet).
"""

import dataclasses

from natsune.adapters import VA, Adapter
from natsune.backend.types import AgentDef, InetCallable
from natsune.control_flow import (
    CloseAfterContingent,
    ParallelOr,
    SerialAnd,
    SerialOr,
    Tracer,
)
from natsune.invocations import ExpansionWithAdapters


def agent_def_for(
    name: str,
    agent: ExpansionWithAdapters,
    *,
    input_adapter: Adapter | None = None,
    output_adapter: Adapter | None = None,
) -> AgentDef:
    """Extract an AgentDef from a live agent instance — anything satisfying
    the ExpansionWithAdapters protocol. The impl wraps the instance itself:
    today's invocation machinery is the Python resolution of the
    declaration."""

    return AgentDef(
        name,
        input_adapter if input_adapter is not None else agent.input_adapter,
        output_adapter if output_adapter is not None else agent.output_adapter,
        InetCallable(agent),
    )


def primitive_agents() -> dict[str, AgentDef]:
    """Static primitives constructible without lowering-time parameters.
    Adapter arguments below are placeholders for the survey — real sites
    declare with the adapters their context demands."""

    defs = [
        agent_def_for("parallel_or", ParallelOr(VA)),
        agent_def_for("serial_or", SerialOr(VA)),
        agent_def_for("serial_and", SerialAnd(VA, VA)),
        agent_def_for("close_after_contingent", CloseAfterContingent(VA, VA)),
        # Tracer is adapter-transparent (wires[0] <-> port) and carries no
        # adapters of its own, so it doesn't satisfy ExpansionWithAdapters;
        # declare it directly.
        AgentDef(
            "tracer",
            VA,
            VA,
            InetCallable(Tracer("trace")),
        ),
    ]
    return {d.name: d for d in defs}


@dataclasses.dataclass(frozen=True)
class CompositeAgent:
    """Classification entry for a composite agent: becomes a per-call-site
    declaration during lowering, not a static AgentDef."""

    name: str
    existing: str
    sub_templates: tuple[str, ...]
    note: str


COMPOSITE_AGENTS: tuple[CompositeAgent, ...] = (
    CompositeAgent(
        "if_then_else",
        "control_flow.IfThenElse",
        ("true_case", "false_case"),
        "Dispatch on the incoming pair is primitive; branches become "
        "per-site NetTemplates referenced by AgentRefs.",
    ),
    CompositeAgent(
        "if_then_else_statement",
        "control_flow.IfThenElseStatement",
        ("true_case", "false_case"),
        "invocation() mutates the invoker's flow_map; that variable-cell "
        "coordination moves into lowering, fed by IrBody.variable_usage.",
    ),
    CompositeAgent(
        "loop",
        "control_flow.Loop",
        ("iteration", "body", "orelse"),
        "Self-grafting recursion (invocation(builder) as recurse) becomes "
        "an AgentRef cycle in the template's own pairs; flow_map threading "
        "as with if_then_else_statement.",
    ),
)

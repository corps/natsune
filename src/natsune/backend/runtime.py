"""Python-side resolution of declarations to executable expansions (§5.1
step 3). The runtime owns "what a declaration runs as"; an emitter backend
instead walks NetTemplates and emits source. Grafts created here record
WHICH agent they instantiate (Graft.agent), so recorded nets surface agent
identity to the golden-net oracle and to the emitter."""

from natsune.backend.types import AgentDef, NetTemplate, Primitive, PythonCallable
from natsune.connector import Connector
from natsune.invocations import Invocation, closer, pack_from, pack_into
from natsune.ports import AgentRef, Expansion, Graft, Port, Wire
from natsune.registers import FromInterfaceRegister, ToInterfaceRegister


def resolve_impl(defn: AgentDef) -> Expansion:
    """The Python resolution of a declaration's impl. NetTemplate and
    Primitive resolutions land with the runtime protocol (§8.1) / the
    primitive table; PythonCallable requires an Expansion-protocol callable
    (the surveyed primitives qualify; old-style __inet__ callees are
    inlined at build time and never become Grafts)."""

    impl = defn.impl
    if isinstance(impl, PythonCallable):
        if not callable(impl.fn):
            raise TypeError(
                f"agent {defn.name!r}: PythonCallable impl is not an Expansion"
            )
        return impl.fn
    if isinstance(impl, NetTemplate):
        raise NotImplementedError(
            f"agent {defn.name!r}: NetTemplate resolution lands with the "
            "runtime protocol (§8.1)"
        )
    if isinstance(impl, Primitive):
        raise NotImplementedError(
            f"agent {defn.name!r}: no Python primitive table entry for "
            f"{impl.kind!r}"
        )
    raise TypeError(f"agent {defn.name!r}: unknown impl {impl!r}")


def agent_invocation[P, W](
    ref: AgentRef,
    defn: AgentDef,
    connector: Connector,
    p: type[P],
    w: type[W],
) -> closer[Invocation[P, W]]:
    """Declarative counterpart of expansion_invocation: the graft is tagged
    with the agent ref while executing the Python resolution of the
    declaration. Adapters come from the declaration, the expansion from
    resolve_impl — identical wiring to the legacy path otherwise."""

    expansion = resolve_impl(defn)
    graft = Graft(expansion, [Wire()], agent=ref)

    inputs = ToInterfaceRegister(defn.input_adapter, connector)
    outputs = FromInterfaceRegister(defn.output_adapter, connector)

    connector.connect(graft, inputs.interface)
    connector.connect(graft.wires[0], outputs.interface)

    return closer(Invocation(pack_into(inputs, p), pack_from(outputs, w)))

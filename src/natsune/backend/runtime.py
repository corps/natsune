"""Python-side resolution of declarations to executable expansions (§5.1
step 3). The runtime owns "what a declaration runs as"; an emitter backend
instead walks NetTemplates and emits source. Grafts created here record
WHICH agent they instantiate (Graft.agent), so recorded nets surface agent
identity to the golden-net oracle and to the emitter."""

from collections.abc import Sequence

from natsune.backend.types import AgentDef, InetCallable, NetTemplate, Primitive
from natsune.connector import Connector
from natsune.control_flow import FlowControlInto, FlowInputInto
from natsune.invocations import Invocation, closer, pack_from, pack_into
from natsune.ports import AgentRef, Expansion, Graft, Port, Wire
from natsune.registers import (
    FromInterfaceRegister,
    FromRegister,
    ToInterfaceRegister,
    ToRegister,
)


def resolve_impl(defn: AgentDef) -> Expansion:
    """The Python resolution of a declaration's impl. NetTemplate and
    Primitive resolutions land with the runtime protocol (§8.1) / the
    primitive table; an InetCallable whose runtime object is a live
    Expansion resolves as itself (the surveyed primitives qualify).
    Legacy __inet__ callees are NOT resolvable here — they go through
    callee_invocation's manual, agent-tagged wiring."""

    impl = defn.impl
    if isinstance(impl, InetCallable):
        if not callable(impl.ref):
            raise TypeError(
                f"agent {defn.name!r}: InetCallable impl is not an Expansion"
            )
        return impl.ref
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


def callee_invocation(
    defn: AgentDef, connector: Connector
) -> tuple[Sequence[ToRegister], FromRegister]:
    """Call-site resolution (§7.2a): the legacy ``invocation(connector) ->
    (inputs, output)`` contract, rebuilt by hand so the graft can carry the
    resolved AgentRef. The legacy method (compiler.py:293) creates its own
    untagged Graft deep inside ``VariablesFlow.invocation`` — unreachable
    for tagging — so the wiring is duplicated here against the callee's
    live flow. This is the same shape as NetTemplate resolution, with a
    slightly different way of filling in registers: the callee's variable
    interface is sorted into (exceptions slot closed, one register per
    parameter, locals closed) instead of instantiated from template data.

    The flow_map side effect of ``VariablesFlow.invocation`` (callee map
    merged into the invoker's) is deliberately skipped: flow_map
    coordination moves into lowering, fed by IrBody.variable_usage (§6).

    Only legacy InetCallable callees resolve today; NetTemplate impls
    (new-lowered callees linking to each other) are cutover territory —
    CompiledFunction refs, §7.4.
    """

    impl = defn.impl
    if not isinstance(impl, InetCallable):
        raise NotImplementedError(
            f"agent {defn.name!r}: {type(impl).__name__} call resolution is "
            "cutover territory (§7.4)"
        )

    callee = impl.ref
    flow = callee.compiled  # duck-typed InetFunctionCompiler -> VariablesFlow
    arity = len(callee.args)

    graft = Graft(flow, [Wire()], agent=AgentRef(defn.name))
    inputs = ToInterfaceRegister(flow.input_adapter, connector)
    outputs = FromInterfaceRegister(flow.output_adapter, connector)

    connector.connect(graft, inputs.interface)
    connector.connect(graft.wires[0], outputs.interface)

    # The closer's exit is load-bearing (the legacy path's `with` does this
    # implicitly): it annihilates each interface register's dangling state
    # wire-end, which the golden-net oracle is sensitive to.
    with closer(
        Invocation(
            pack_into(inputs, FlowInputInto), pack_from(outputs, FlowControlInto)
        )
    ) as invocation:
        # The register-sorting dance, duplicated from InetFunctionCompiler
        # .invocation: slot 0 is the callee's exceptions (stays with the
        # callee — should_capture_exceptions is a lowering decision, §8);
        # parameters follow; callee locals start unwritten and are erased.
        variable_inputs = invocation.port.variables.readin().split()
        variable_inputs[0].close()
        for extra in variable_inputs[arity + 1 :]:
            extra.close()

        return (
            variable_inputs[1 : arity + 1],
            invocation.wire.return_value.readout(),
        )

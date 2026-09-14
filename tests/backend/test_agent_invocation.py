"""§5.1 step 3: grafts carry references. agent_invocation is the
declarative counterpart of expansion_invocation — same wiring, but the
Graft records WHICH agent it instantiates and the expansion comes from
resolve_impl(defn)."""

import copy

import pytest

from natsune.adapters import VA
from natsune.backend.agents import primitive_agents, agent_invocation, callee_invocation
from natsune.backend.python_backend import PythonBackend
from natsune.backend.runtime import resolve_impl
from natsune.backend.types import AgentDef, InetCallable, NetTemplate, Primitive
from natsune.calculus import Calculus
from natsune.control_flow import MergeOutputInto, Tracer
from natsune.ports import AgentRef, Graft, Wire, WirePort
from natsune.registers import ToInterfaceRegister, send_value


@pytest.fixture(scope="function")
def c() -> Calculus:
    return Calculus()


def test_agent_invocation_runs_declared_agent(c: Calculus) -> None:
    # Mirrors tests/test_control_flow.py::test_weakening, but through the
    # declaration layer instead of the live agent object.
    backend = PythonBackend()
    defn = primitive_agents()["serial_or"]
    backend.declare_agent("serial_or", defn)

    with agent_invocation(
        AgentRef("serial_or"), defn, c.executor, ToInterfaceRegister, MergeOutputInto
    ) as invocation:
        send_value(c.erasure(), invocation.port.readin())
        send_value(c.const(30), invocation.wire.second_value.readin())
        send_value(invocation.wire.result.readout(), c.to_key(0))

    assert c.reduce_to_value(0) == 30


def test_declared_grafts_record_agent_ref(c: Calculus) -> None:
    defn = primitive_agents()["serial_or"]
    with agent_invocation(
        AgentRef("serial_or"), defn, c.executor, ToInterfaceRegister, MergeOutputInto
    ):
        pass

    # Grafts sit behind WirePort indirections (connect_to_target), so walk
    # wire targets too.
    found: list[Graft] = []

    def visit(port: object) -> None:
        if isinstance(port, Graft):
            found.append(port)
        elif isinstance(port, WirePort):
            for wire in port.wires:
                if wire.target is not None:
                    visit(wire.target)

    for pair in c.executor.active_pairs:
        for port in pair:
            visit(port)

    assert len(found) == 1
    assert found[0].agent == AgentRef("serial_or")


def test_serialization_surfaces_agent_identity(c: Calculus) -> None:
    defn = primitive_agents()["serial_or"]
    with agent_invocation(
        AgentRef("serial_or"), defn, c.executor, ToInterfaceRegister, MergeOutputInto
    ):
        pass
    assert any("graft:serial_or" in part for part in c.serialize_active_pairs())


def test_legacy_grafts_serialize_unchanged(c: Calculus) -> None:
    # Untagged (legacy) grafts render exactly as before the agent field.
    from natsune.control_flow import SerialOr

    with SerialOr(VA).invocation(c.executor) as invocation:
        send_value(c.erasure(), invocation.port.readin())
        send_value(c.const(30), invocation.wire.second_value.readin())
        send_value(invocation.wire.result.readout(), c.to_key(0))
    assert c.reduce_to_value(0) == 30
    assert all("graft:" not in part for part in c.serialize_active_pairs())


def test_graft_copy_preserves_agent_ref():
    graft = Graft(Tracer("t"), [Wire()], agent=AgentRef("tracer"))
    assert copy.copy(graft).agent == AgentRef("tracer")


def test_resolve_impl_resolves_live_expansions():
    defn = primitive_agents()["tracer"]
    impl = defn.impl
    assert isinstance(impl, InetCallable)
    assert resolve_impl(defn) is impl.ref


def test_resolve_impl_refuses_unresolvable():
    with pytest.raises(NotImplementedError):
        resolve_impl(AgentDef("t", VA, VA, NetTemplate(VA, VA, ())))
    with pytest.raises(NotImplementedError):
        resolve_impl(AgentDef("p", VA, VA, Primitive("foreach")))
    with pytest.raises(TypeError):
        resolve_impl(AgentDef("b", VA, VA, InetCallable(42)))

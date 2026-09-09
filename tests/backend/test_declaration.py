"""Unit-1 tests: the declaration layer (backend/types, protocol, shell,
agent survey). Compile-time only — nothing here executes a net."""

from natsune.adapters import ParValueAdapter, VA, Variables
from natsune.backend import (
    COMPOSITE_AGENTS,
    AgentDef,
    AgentRef,
    Backend,
    LoweredUnit,
    NetTemplate,
    Primitive,
    PythonBackend,
    PythonCallable,
    agent_def_for,
    primitive_agents,
)
from natsune.connector import serialize_active_pairs
from natsune.control_flow import SerialOr, VariablesFlow
from natsune.ports import ConstantValuePort
from natsune.compiler import inet

import pytest


def _hand_flow() -> VariablesFlow:
    # Same construction shape as the legacy compiler (compiler.py:182).
    return VariablesFlow(variables=Variables({"a": VA}), return_adapter=VA)


def test_python_backend_satisfies_protocol():
    assert isinstance(PythonBackend(), Backend)


def test_agent_def_for_extracts_adapters():
    agent = SerialOr(VA)
    d = agent_def_for("serial_or", agent)
    assert d.input_adapter is agent.input_adapter
    assert d.output_adapter is agent.output_adapter
    assert isinstance(d.impl, PythonCallable)
    assert d.impl.fn is agent


def test_primitive_agents_table():
    table = primitive_agents()
    assert set(table) == {
        "parallel_or",
        "serial_or",
        "serial_and",
        "close_after_contingent",
        "tracer",
    }
    for d in table.values():
        assert isinstance(d.impl, PythonCallable)


def test_composite_classification():
    by_name = {c.name: c for c in COMPOSITE_AGENTS}
    assert {"if_then_else", "if_then_else_statement", "loop"} <= set(by_name)
    # The flow_map coupling is the load-bearing note: it moves into
    # lowering, fed by IrBody.variable_usage.
    assert "flow_map" in by_name["if_then_else_statement"].note
    assert "flow_map" in by_name["loop"].note
    assert "AgentRef" in by_name["loop"].note
    # Composites are per-call-site declarations, not static primitives.
    assert not (set(by_name) & set(primitive_agents()))


def test_declare_agent_idempotent_and_conflict_detecting():
    backend = PythonBackend()
    d1 = AgentDef("x", VA, VA, Primitive("p"))
    d2 = AgentDef("x", VA, VA, PythonCallable(object()))
    assert backend.declare_agent("x", d1) == AgentRef("x")
    with pytest.raises(ValueError):
        backend.declare_agent("x", d2)
    # Re-declaring the identical defn is a no-op (e.g. re-finished flows).
    assert backend.declare_agent("x", d1) == AgentRef("x")


def test_constant_port():
    port = PythonBackend().constant(42, VA)
    assert isinstance(port, ConstantValuePort)
    assert port.value == 42


def test_resolve_call_mints_stable_ref_and_copies_metadata():
    class _DummyInet:
        args_adapter = ParValueAdapter([VA, VA])
        return_adapter = VA

    backend = PythonBackend()
    dummy = _DummyInet()
    ref = backend.resolve_call(dummy)
    assert isinstance(ref, AgentRef)
    assert backend.resolve_call(dummy) is ref

    defn = backend.agents[ref.name]
    # The args ParValueAdapter IS the interface — no unpacking.
    assert defn.input_adapter is _DummyInet.args_adapter
    assert defn.output_adapter is VA
    impl = defn.impl
    assert isinstance(impl, PythonCallable)
    assert impl.fn is dummy

    # Already-declared AgentRefs pass through untouched.
    named = backend.declare_agent("known", defn)
    assert backend.resolve_call(named) is named


def test_finish_snapshots_body_template():
    flow = _hand_flow()
    backend = PythonBackend()
    unit = backend.finish(flow, name="hand")

    assert isinstance(unit, LoweredUnit)
    assert unit.name == "hand"
    assert isinstance(unit.body, AgentRef)

    body = unit.body_def
    assert body.input_adapter is flow.input_adapter
    assert body.output_adapter is flow.output_adapter
    assert isinstance(body.impl, NetTemplate)
    # Snapshot, not alias: the template holds its own pair tuple.
    assert body.impl.pairs == tuple(flow.active_pairs)
    assert len(body.impl.pairs) > 0
    # The artifact is decoupled from the live registry.
    assert unit.agents is not backend.agents
    assert unit.agents[unit.body.name] is body


def test_same_flow_finishes_identically_across_backends():
    # Oracle smoke: serialization of the declared template is a pure
    # function of the flow, independent of which backend finished it.
    flow = _hand_flow()
    a = PythonBackend().finish(flow, name="f")
    b = PythonBackend().finish(flow, name="f")
    ta = a.agents[a.body.name].impl
    tb = b.agents[b.body.name].impl
    assert isinstance(ta, NetTemplate) and isinstance(tb, NetTemplate)
    assert serialize_active_pairs(list(ta.pairs), {}) == serialize_active_pairs(
        list(tb.pairs), {}
    )


# Legacy-compat smoke: a real compiled flow (old __inet__ machinery)
# finishes into a LoweredUnit through the shell.
@inet()
def _add_one(a: int) -> int:
    return a + 1


def test_finish_legacy_compiled_flow():
    compiled = _add_one.__inet__.compiled
    unit = PythonBackend().finish(compiled, name="add_one")

    assert isinstance(unit.body, AgentRef)
    body = unit.body_def
    assert isinstance(body.impl, NetTemplate)
    assert len(body.impl.pairs) > 0
    assert body.input_adapter is compiled.input_adapter
    assert body.output_adapter is compiled.output_adapter

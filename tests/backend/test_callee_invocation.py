"""§7.2a: callee_invocation — call-site resolution with agent-tagged
grafts. The legacy contract (invocation(connector) -> (inputs, output))
rebuilt by hand against the callee's live flow."""

import linecache
import textwrap

import pytest

from natsune.adapters import VA, Variables
from natsune.backend import PythonBackend
from natsune.backend.agents import callee_invocation
from natsune.backend.types import AgentDef, NetTemplate
from natsune.compiler import InetFunctionCompiler
from natsune.connector import serialize_active_pairs
from natsune.control_flow import VariablesFlow
from natsune.registers import _FromRegister, as_constant_register, send_value


def _compile(source: str, name: str):
    """Legacy-compile one function (same exec/linecache setup as the
    golden-net oracle's _legacy_flow)."""
    filename = "callee_invocation_case.py"
    text = textwrap.dedent(source)
    ns: dict = {}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    compiler = InetFunctionCompiler(ns[name], ns, filename)
    compiler.compile()
    return ns, compiler


def _hand_flow() -> VariablesFlow:
    return VariablesFlow(variables=Variables({"a": VA}), return_adapter=VA)


def test_callee_invocation_tags_graft_and_sorts_registers():
    ns, callee = _compile("def inc(x: int) -> int:\n    return x + 1", "inc")
    backend = PythonBackend()
    ref = backend.resolve_call(callee)
    defn = backend.agent_def(ref)

    flow = _hand_flow()
    inputs, output = callee_invocation(defn, flow)

    # One register per parameter; a FromRegister out.
    assert len(inputs) == 1
    assert isinstance(output, _FromRegister)

    # Flow a value through: the graft surfaces in the serialization,
    # tagged with the resolved ref (§5.1 step 3) — the whole point of the
    # manual wiring (legacy creates its graft untagged, out of reach).
    send_value(as_constant_register(5, flow), inputs[0])
    rendered = "\n".join(serialize_active_pairs(list(flow.active_pairs), {}))
    assert f"graft:{ref.name}" in rendered


def test_callee_invocation_redeclares_identically():
    ns, callee = _compile("def inc(x: int) -> int:\n    return x + 1", "inc")
    backend = PythonBackend()
    defn = backend.agent_def(backend.resolve_call(callee))
    # A second call site hits the same declaration (resolve_call keys by
    # object identity) and resolves without conflict.
    defn_again = backend.agent_def(backend.resolve_call(callee))
    assert defn is defn_again
    assert callee_invocation(defn, _hand_flow())[1] is not None


def test_callee_invocation_refuses_non_legacy_impls():
    # NetTemplate impls = new-lowered callees: cutover territory (§7.4).
    defn = AgentDef("t", VA, VA, NetTemplate(VA, VA, ()))
    with pytest.raises(NotImplementedError):
        callee_invocation(defn, _hand_flow())

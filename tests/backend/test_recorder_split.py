"""§5.1 steps 1–2: the recorder split. NetTemplateBuilder records data and
knows nothing about executors; instantiate_template is the liberated
closure; ExpansionBuilder/VariablesFlow delegate __call__ to it until
grafts carry AgentRefs (step 3)."""

from natsune.adapters import VA, Variables
from natsune.backend.types import net_template_of
from natsune.connector import (
    ExpansionBuilder,
    NetTemplateBuilder,
    instantiate_template,
    serialize_active_pairs,
)
from natsune.compiler import inet
from natsune.control_flow import VariablesFlow
from natsune.ports import ConstantValuePort, Wire


def _recorded(builder_cls):
    b = builder_cls(VA, VA)
    b.duplicate(ConstantValuePort(7))
    b.duplicate(ConstantValuePort(9))
    return b


def test_recorder_has_no_closure():
    assert not callable(NetTemplateBuilder(VA, VA))
    # Expansion-compatible by delegation until grafts carry AgentRefs.
    assert callable(ExpansionBuilder(VA, VA))
    flow = VariablesFlow(variables=Variables({"a": VA}), return_adapter=VA)
    assert callable(flow)


def test_instantiate_template_matches_method_delegation():
    template = _recorded(ExpansionBuilder)
    a = NetTemplateBuilder(VA, VA)
    instantiate_template(template, a, ConstantValuePort(1), [Wire(), Wire()])
    b = NetTemplateBuilder(VA, VA)
    template(b, ConstantValuePort(1), [Wire(), Wire()])
    assert serialize_active_pairs(a.active_pairs, {}) == serialize_active_pairs(
        b.active_pairs, {}
    )


def test_instantiate_copies_without_consuming_source():
    template = _recorded(NetTemplateBuilder)
    target = NetTemplateBuilder(VA, VA)
    instantiate_template(template, target, ConstantValuePort(1), [Wire(), Wire()])
    assert len(target.active_pairs) == len(template.active_pairs) > 0

    # The source is untouched: instantiating again produces the same shape.
    again = NetTemplateBuilder(VA, VA)
    instantiate_template(template, again, ConstantValuePort(1), [Wire(), Wire()])
    assert serialize_active_pairs(target.active_pairs, {}) == serialize_active_pairs(
        again.active_pairs, {}
    )


def test_net_template_of_extracts_data():
    template = _recorded(NetTemplateBuilder)
    nt = net_template_of(template)
    assert nt.input_adapter is VA
    assert nt.output_adapter is VA
    assert nt.pairs == tuple(template.active_pairs)


# Runtime contract: a legacy compiled function executes through the
# delegated closure (Graft -> VariablesFlow.__call__ -> instantiate_template).
@inet()
def _double(x: int) -> int:
    return x + x


def test_legacy_call_runs_through_delegated_closure():
    assert _double(21) == 42

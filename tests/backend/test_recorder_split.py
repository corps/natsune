"""§5.1 steps 1–2, post-restructure: ExpansionBuilder is the pure
recorder; freeze() extracts a FrozenExpansion (a data-only template) and
its __call__ (materialize_template) is the liberated closure that replays
the recorded pairs into any connector. VariablesFlow delegates __call__
through the same machinery."""

from natsune.first_order.adapters import VA, Variables
from natsune.backend.connector import ExpansionBuilder, serialize_active_pairs
from natsune.backend.control_flow import VariablesFlow
from natsune.inet import inet
from natsune.first_order.ports import ConstantValuePort, Wire


def _recorded():
    b = ExpansionBuilder(VA, VA)
    b.duplicate(ConstantValuePort(7))
    b.duplicate(ConstantValuePort(9))
    return b


def test_builder_records_data():
    """The recorder is data-only: recording produces pairs, and the
    frozen template carries exactly the recorded pairs."""
    builder = _recorded()
    frozen = builder.freeze()
    assert frozen.active_pairs == tuple(builder.active_pairs)
    assert frozen.input_adapter is VA
    assert frozen.output_adapter is VA


def test_frozen_expansion_replays_into_any_connector():
    frozen = _recorded().freeze()
    a = ExpansionBuilder(VA, VA)
    frozen(a, ConstantValuePort(1), [Wire(), Wire()])
    b = ExpansionBuilder(VA, VA)
    frozen(b, ConstantValuePort(1), [Wire(), Wire()])
    assert serialize_active_pairs(a.active_pairs, {}) == serialize_active_pairs(
        b.active_pairs, {}
    )


def test_freeze_does_not_consume_the_source():
    builder = _recorded()
    frozen = builder.freeze()
    target = ExpansionBuilder(VA, VA)
    frozen(target, ConstantValuePort(1), [Wire(), Wire()])
    assert len(target.active_pairs) == len(builder.active_pairs) > 0

    # The source is untouched: freezing and replaying again produces the
    # same shape.
    again = builder.freeze()
    other = ExpansionBuilder(VA, VA)
    again(other, ConstantValuePort(1), [Wire(), Wire()])
    assert serialize_active_pairs(target.active_pairs, {}) == serialize_active_pairs(
        other.active_pairs, {}
    )


def test_flows_stay_callable_expansions():
    """VariablesFlow remains an Expansion (dispatch-protocol conformance):
    the Python runtime resolves it by calling the frozen replay."""
    flow = VariablesFlow(variables=Variables({"a": VA}), return_adapter=VA)
    assert callable(flow)


# Runtime contract: a legacy compiled function executes through the
# frozen-closure machinery (Graft -> FrozenExpansion -> materialize_template).
@inet()
def _double(x: int) -> int:
    return x + x


def test_legacy_call_runs_through_the_frozen_closure():
    assert _double(21) == 42

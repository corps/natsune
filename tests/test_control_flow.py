import pytest

from natsune.adapters import ValueAdapter, Variables
from natsune.calculus import Calculus
from natsune.connector import ExpansionBuilder
from natsune.control_flow import WeakeningSelection, IfThenElse, Loop, VariablesFlow
from natsune.invocations import filter_invocation, send_parameter
from natsune.registers import send_value, as_constant_register


@pytest.fixture(scope='function')
def c() -> Calculus:
    return Calculus()

def test_weakening(c: Calculus) -> None:
    with WeakeningSelection(ValueAdapter()).invocation(c.executor) as invocation:
        send_value(c.erasure(), invocation.port.readin())
        send_value(c.const(30), invocation.wire.second_value.readin())
        send_value(invocation.wire.result.readout(), c.to_key(0))
    assert list(c.readout(0)) == [30]

def test_weakening_discard(c: Calculus) -> None:
    with WeakeningSelection(ValueAdapter()).invocation(c.executor) as invocation:
        send_value(invocation.wire.result.readout(), c.to_key(0))

    assert c.serialize_active_pairs() != []
    assert list(c.readout(0)) == []
    assert c.serialize_active_pairs() == []

def test_gated_and_sequence(c: Calculus) -> None:
    result = c.from_key(0) & c.from_key(1)
    send_value(result, c.to_key(2))
    assert c.serialize_active_pairs() != []
    assert list(c.readout(2)) == []
    assert c.serialize_active_pairs() == []

    send_value(c.const(1), c.to_key(1))
    assert c.serialize_active_pairs() != []
    assert list(c.continue_readout()) == []
    assert c.serialize_active_pairs() == []

    send_value(c.const(0), c.to_key(0))
    assert c.serialize_active_pairs() != []
    assert list(c.continue_readout()) == [(0, 1)]
    assert c.serialize_active_pairs() == []

def test_if_then_else(c: Calculus) -> None:
    with ExpansionBuilder(ValueAdapter(), ValueAdapter()) as true_case:
        a, b = filter_invocation(lambda x: x * 2, true_case)
        send_value(true_case.input_interface.readout(), a)
        send_value(b, true_case.output_interface.readin())

    with ExpansionBuilder(ValueAdapter(), ValueAdapter()) as false_case:
        a, b = filter_invocation(lambda x: x - 2, false_case)
        send_value(false_case.input_interface.readout(), a)
        send_value(b, false_case.output_interface.readin())

    with IfThenElse(true_case, false_case).invocation(c.executor) as conditional:
        send_value(c.const(True), conditional.port.readin())
        send_value(c.const(3), conditional.wire.context.readin())
        send_value(conditional.wire.result.readout(), c.to_key(0))

    assert list(c.readout(0)) == [6]


def test_if_then_else_false(c: Calculus) -> None:
    with ExpansionBuilder(ValueAdapter(), ValueAdapter()) as true_case:
        a, b = filter_invocation(lambda x: x * 2, true_case)
        send_value(true_case.input_interface.readout(), a)
        send_value(b, true_case.output_interface.readin())

    with ExpansionBuilder(ValueAdapter(), ValueAdapter()) as false_case:
        a, b = filter_invocation(lambda x: x - 2, false_case)
        send_value(false_case.input_interface.readout(), a)
        send_value(b, false_case.output_interface.readin())

    with IfThenElse(true_case, false_case).invocation(c.executor) as conditional:
        send_value(c.const(False), conditional.port.readin())
        send_value(c.const(3), conditional.wire.context.readin())
        send_value(conditional.wire.result.readout(), c.to_key(0))

    assert list(c.readout(0)) == [1]

def test_loop_basic(c: Calculus) -> None:
    variables = Variables({ "a" : ValueAdapter(), "b": ValueAdapter() })
    with VariablesFlow(variables=variables, return_adapter=ValueAdapter()) as iter:
        send_value(iter.variable_registers['a'].readout(), iter.control_output.return_value.readin())
        send_value(as_constant_register(False, iter), iter.variable_registers["a"].readin())
        send_value(iter.variables_readout(), iter.control_output.finish_variables.readin("finish for iter"))

    with VariablesFlow(variables=variables, return_adapter=ValueAdapter()) as body:
        send_value(
            send_parameter(filter_invocation(lambda x: x + 1, body), body.variable_registers["b"].readout()),
            body.variable_registers["b"].readin("setting b")
        )
        send_value(body.variables_readout(), body.control_output.finish_variables.readin("finish for body"))

    with VariablesFlow(variables=variables, return_adapter=ValueAdapter()) as orelse:
        send_value(orelse.variables_readout(), orelse.control_output.finish_variables.readin("finish for orelse"))

    with Loop(
        iter,
        body,
        orelse,
    ).invocation(c.executor) as loop_invocation:
        send_value(as_constant_register((True, 0), c.executor), loop_invocation.port.variables.readin("variables into loop"))
        send_value(loop_invocation.wire.finish_variables.readout(), c.to_key(0))

    assert list(c.readout(0)) == [1]


def test_variables_readout(c: Calculus) -> None:
    variables = Variables({"x": ValueAdapter(), "y": ValueAdapter()})

    with VariablesFlow(variables=variables, return_adapter=ValueAdapter()) as vf:
        send_value(as_constant_register(42, vf), vf.variable_registers["x"].readin())
        send_value(vf.variables_readout(), vf.control_output.finish_variables.readin())

    with vf.invocation(c.executor) as invocation:
        send_value(as_constant_register((10, 20), c.executor), invocation.port.variables.readin())
        send_value(invocation.wire.finish_variables.readout(), c.to_key(0))

    assert list(c.readout(0)) == [(42, 20)]
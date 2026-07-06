import pytest

from natsune.adapters import Variables, ValueAdapter
from natsune.calculus import Calculus
from natsune.control_flow import FlowInvocation, FlowInput, FlowControl, AffineSelection
from natsune.registers import InterfaceRegister, send_value


@pytest.fixture(scope='function')
def c() -> Calculus:
    return Calculus()

def make_flow_invocation(c: Calculus, variables: Variables) -> FlowInvocation:
    inputs = InterfaceRegister(FlowInput.adapter(variables), c.executor)
    outputs = InterfaceRegister(FlowControl.adapter(ValueAdapter(), variables), c.executor)
    return FlowInvocation.split(inputs, outputs)

def test_flow_invocation_send_to(c: Calculus) -> None:
    variables = Variables({})

    executing_invocation = make_flow_invocation(c, variables)
    wiring_invocation = make_flow_invocation(c, variables)

    wiring_invocation.control.send_to(executing_invocation.control)

    send_value(c.const(10), wiring_invocation.control.o_return.interface_readin())
    send_value(executing_invocation.control.o_return.interface_readin().invert(), c.to_key(0))

    assert list(c.readout(0)) == [10]

def test_affine_share(c: Calculus) -> None:
    i = InterfaceRegister(ValueAdapter(), c.executor)
    a, b = AffineSelection.share_to_register(i.readin())
    send_value(c.erasure(), a)
    send_value(c.const(30), b)
    send_value(i.interface_readin().invert(), c.to_key(0))
    assert list(c.readout(0)) == [30]

def test_affine_share_flow(c: Calculus) -> None:
    variables = Variables({"a": ValueAdapter()})

    executing_invocation = make_flow_invocation(c, variables)
    wiring_invocation = make_flow_invocation(c, variables)

    finish1, finish2 = AffineSelection.share_to_register(executing_invocation.control.o_finish.readin())
    send_value(wiring_invocation.control.o_break.readout(), finish2)
    send_value(wiring_invocation.control.o_finish.readout(), finish1)
    wiring_invocation.close()
    executing_invocation.close()

    send_value(executing_invocation.control.o_finish.interface_readin().invert(), c.to_key(0))
    send_value(c.const(10), wiring_invocation.control.o_break.interface_readin())
    wiring_invocation.control.o_finish.interface_readin().close()

    assert list(c.readout(0)) == [10]

def test_affine_discard(c: Calculus) -> None:
    inter = InterfaceRegister(ValueAdapter(), c.executor)
    a, b = AffineSelection.share_to_register(inter.readin())
    a.close()
    b.close()
    send_value(inter.interface_readin().invert(), c.to_key(0))

    assert list(c.readout(0)) == []

def test_share_control_flow(c: Calculus) -> None:
    variables = Variables({"a": ValueAdapter()})

    executing_invocation = make_flow_invocation(c, variables)
    wiring_invocation = make_flow_invocation(c, variables)

    with executing_invocation, executing_invocation.control.share_to_interface() as (control1, control2), wiring_invocation:
        send_value(wiring_invocation.control.o_break.readout(), control2.o_finish.readin())
        send_value(wiring_invocation.control.o_finish.readout(), control1.o_finish.readin())

    send_value(executing_invocation.control.o_finish.interface_readin().invert(), c.to_key(0))
    send_value(c.const(10), wiring_invocation.control.o_break.interface_readin())
    wiring_invocation.control.o_finish.interface_readin().close()

    assert list(c.readout(0)) == [10]

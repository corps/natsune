import pytest

from natsune.adapters import Variables, ValueAdapter
from natsune.calculus import Calculus
from natsune.control_flow import FlowInput, FlowControl, WeakeningSelection
from natsune.registers import InterfaceRegister, send_value


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
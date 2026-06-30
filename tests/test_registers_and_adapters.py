from natsune.optimizer import optimize
from typing import Sequence, Self

from natsune.executor import Executor
from natsune.invocations import ExpansionWithAdapters, expansion_invocation, unpack_port_and_wires
from natsune.ports import ValuePort, Port, Wire
from natsune.control_flow import VariablesFlow, FlowInvocation
from natsune.adapters import ValueAdapter, ReferenceAdapter, Variables, InverseAdapter, ParValueAdapter, Adapter
from natsune.calculus import Calculus
import pytest

from natsune.registers import serialize_values, send_value, send_values, parallelize_value, InterfaceRegister, \
    as_from_register, as_to_register, as_constant_register


@pytest.fixture(scope="function")
def c() -> Calculus:
    return Calculus()


def test_serialize_values(c: Calculus) -> None:
    inputs, output = serialize_values(c.executor, 3)
    send_values([c.const(5), c.const(3), c.const(2)], inputs)
    send_value(output, c.to_key(0))
    assert list(c.readout(0)) == [(5, 3, 2)]


def test_parallelize_value(c: Calculus) -> None:
    input, outputs = parallelize_value(c.executor, 3)
    send_value(c.const((9, 4, 3)), input)
    send_value(outputs[0], c.to_key(0))
    c.optimize()
    assert c.serialize_active_pairs() == ['None = -<w0',
                                          'w1>- = fnS(w2, w3)',
                                          'w3>- = fnS(w4, w5)',
                                          'w5>- = fnS(w6, Z)',
                                          '(9, 4, 3) = fnM(w0, w1)',
                                          'w2>- = -<w7']
    assert list(c.readout(0)) == [9]
    send_value(outputs[1], c.to_key(1))
    assert list(c.readout(1)) == [4]
    send_value(outputs[2], c.to_key(2))
    assert c.serialize_active_pairs() == ['3>- = -<w0']
    assert list(c.readout(2)) == [3]

def test_interface_register(c: Calculus) -> None:
    register = InterfaceRegister(ValueAdapter(), c.executor)
    send_value(register.readout(False), c.to_key(0))
    assert list(c.readout(0)) == []

    send_value(c.const(40), register.interface_readin())
    assert list(c.continue_readout()) == [40]

    register = InterfaceRegister(ValueAdapter(), c.executor)
    send_value(c.const(11), register.readin())
    send_value(c.from_key(33), register.interface_readin())
    assert list(c.readout(33)) == [11]


def test_interface_register_with_multiple_accesses(c: Calculus) -> None:
    register = InterfaceRegister(ValueAdapter(), c.executor)
    send_value(register.readout(False), c.to_key(0))
    send_value(register.readout(False), c.to_key(1))
    send_value(register.readout(False), c.to_key(3))
    assert list(c.readout(0)) == []

    send_value(c.const(40), register.interface_readin())
    assert list(c.continue_readout()) == [40]

    register = InterfaceRegister(ValueAdapter(), c.executor)
    send_value(c.const(11), register.readin())
    send_value(c.const(52), register.readin())
    send_value(c.from_key(33), register.interface_readin())
    assert list(c.readout(33)) == [11]

def test_flow_invocation_simple(c: Calculus) -> None:
    variables = Variables(dict(a=ValueAdapter(), b=ReferenceAdapter(ValueAdapter()), c=InverseAdapter(ValueAdapter()), d=ParValueAdapter([ValueAdapter(), ValueAdapter()])))
    flow = VariablesFlow(variables, ValueAdapter())

    send_value(as_from_register(ValuePort(10), ValueAdapter(), flow.buffer), flow.o_control.o_return.readin())
    with flow.invocation(c.executor) as invocation:
        send_value(invocation.control.o_return.readout(False), c.to_key(0))
    c.optimize()
    assert list(c.readout(0)) == [10]

def test_interface_register_split_from(c: Calculus) -> None:
    register = InterfaceRegister(ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor)
    parts = register.split(True)

    send_value(as_constant_register((1, 2), c.executor), register.interface_readin())
    send_value(parts[0].readout(True), c.to_key(0))
    send_value(parts[1].readout(True), c.to_key(1))

    assert list(c.readout(0)) == [1]
    assert list(c.readout(1)) == [2]

def test_from_register_split(c: Calculus) -> None:
    register = InterfaceRegister(ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor)
    parts = register.readout(True).split()

    send_value(as_constant_register((1, 2), c.executor), register.interface_readin())
    send_value(parts[0], c.to_key(0))
    send_value(parts[1], c.to_key(1))

    assert list(c.readout(0)) == [1]
    assert list(c.readout(1)) == [2]

def test_to_register_split(c: Calculus) -> None:
    register = InterfaceRegister(ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor)
    parts = register.readin().split()
    send_value(c.const(1), parts[0])
    send_value(c.const(2), parts[1])
    send_value(
    register.interface_readin().invert(),
    c.to_key(0),
    )

    assert list(c.readout(0)) == [(1, 2)]

def test_variables_flow_invocation(c: Calculus) -> None:
    variables = Variables(dict(a=ValueAdapter(), b=ReferenceAdapter(ValueAdapter()), c=InverseAdapter(ValueAdapter()), d=ParValueAdapter([ValueAdapter(), ValueAdapter()])))
    flow = VariablesFlow(variables, ValueAdapter())

    send_value(as_from_register(ValuePort(10), ValueAdapter(), flow.buffer), flow.o_control.o_return.readin())
    optimize(flow.buffer, flow.buffer.active_pairs)
    with flow.invocation(c.executor) as invocation:
        send_value(invocation.control.o_return.readout(False), c.to_key(0))
    while any('graft' in line for line in c.serialize_active_pairs()):
        c.process_next_interaction()
    assert list(c.readout(0)) == [10]

class TestExpansion(ExpansionWithAdapters):
    @property
    def input_adapter(self) -> Adapter:
        return ValueAdapter()

    @property
    def output_adapter(self) -> Adapter:
        return ValueAdapter()

    def __call__(
            self, executor: Executor, port: Port, wires: Sequence[Wire], /
    ) -> None:
        inputs, outputs = unpack_port_and_wires(self, port, wires, executor)
        send_value(inputs.readout(True), outputs.readin())

    def __copy__(self) -> Self:
        return self

def test_expansion_invocation(c: Calculus) -> None:
    inputs, outputs = expansion_invocation(TestExpansion(), c.executor)
    send_value(c.const(10), inputs.readin())
    send_value(outputs.readout(True), c.to_key(0))
    assert list(c.readout(0)) == [10]


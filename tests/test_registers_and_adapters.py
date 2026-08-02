from typing import Sequence, Self

import pytest

from natsune.adapters import (
    ValueAdapter,
    ReferenceAdapter,
    Variables,
    InverseAdapter,
    ParValueAdapter,
    Adapter,
)
from natsune.calculus import Calculus
from natsune.control_flow import VariablesFlow
from natsune.executor import Executor
from natsune.invocations import (
    ExpansionWithAdapters,
    expansion_invocation,
    unpack_port_and_wires,
    filter_invocation,
    send_parameter,
    merge_invocation,
    send_parameters,
)
from natsune.optimizer import optimize
from natsune.ports import ValuePort, Port, Wire
from natsune.registers import (
    serialize_values,
    send_value,
    send_values,
    parallelize_value,
    as_from_register,
    as_constant_register,
    FlowRegister,
    join_to_registers,
    join_from_registers,
    FromInterfaceRegister,
    ToInterfaceRegister,
    borrow_registers,
)


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
    assert c.serialize_active_pairs() == [
        "value = fnM(None, fnS(-<[0], fnS(w1, fnS(w2, Z))))",
        "[0]",
    ]
    assert list(c.readout(0)) == [9]
    send_value(outputs[1], c.to_key(1))
    assert list(c.readout(1)) == [4]
    send_value(outputs[2], c.to_key(2))
    assert c.serialize_active_pairs() == ["3 = -<[2]", "9", "9", "4", "4", "[2]"]
    assert list(c.readout(2)) == [3]


def test_interface_register(c: Calculus) -> None:
    register = FromInterfaceRegister(ValueAdapter(), c.executor)
    send_value(register.readout(), c.to_key(0))
    assert list(c.readout(0)) == []

    send_value(c.const(40), register.interface_readin())
    assert list(c.continue_readout()) == [40]

    register2 = ToInterfaceRegister(ValueAdapter(), c.executor)
    send_value(c.const(11), register2.readin())
    send_value(c.from_key(33), register2.interface_readin())
    assert list(c.readout(33)) == [11]


def test_interface_register_with_multiple_accesses(c: Calculus) -> None:
    register = FromInterfaceRegister(ValueAdapter(), c.executor)
    send_value(register.readout(), c.to_key(0))
    send_value(register.readout(), c.to_key(1))
    send_value(register.readout(), c.to_key(3))
    assert list(c.readout(0)) == []

    send_value(c.const(40), register.interface_readin())
    assert list(c.continue_readout()) == [40]

    register = ToInterfaceRegister(ValueAdapter(), c.executor)
    send_value(c.const(11), register.readin())
    send_value(c.const(52), register.readin())
    send_value(c.from_key(33), register.interface_readin())
    assert list(c.readout(33)) == [11]


def test_interface_register_split_from(c: Calculus) -> None:
    register = FromInterfaceRegister(
        ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor
    )
    parts = register.split()

    send_value(as_constant_register((1, 2), c.executor), register.interface_readin())
    send_value(parts[0].readout(), c.to_key(0))
    send_value(parts[1].readout(), c.to_key(1))

    assert list(c.readout(0)) == [1]
    assert list(c.readout(1)) == [2]


def test_from_register_split(c: Calculus) -> None:
    register = FromInterfaceRegister(
        ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor
    )
    parts = register.readout().split()

    send_value(as_constant_register((1, 2), c.executor), register.interface_readin())
    send_value(parts[0], c.to_key(0))
    send_value(parts[1], c.to_key(1))

    assert list(c.readout(0)) == [1]
    assert list(c.readout(1)) == [2]


def test_to_register_split(c: Calculus) -> None:
    register = ToInterfaceRegister(
        ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor
    )
    parts = register.readin().split()
    send_value(c.const(1), parts[0])
    send_value(c.const(2), parts[1])
    send_value(
        ~register.interface_readin(),
        c.to_key(0),
    )

    assert list(c.readout(0)) == [(1, 2)]


def test_variables_flow_invocation(c: Calculus) -> None:
    variables = Variables(
        dict(
            a=ValueAdapter(),
            b=ReferenceAdapter(ValueAdapter()),
            c=InverseAdapter(ValueAdapter()),
            d=ParValueAdapter([ValueAdapter(), ValueAdapter()]),
        )
    )
    flow = VariablesFlow(variables=variables, return_adapter=ValueAdapter())
    send_value(
        as_from_register(ValuePort(10), ValueAdapter(), flow),
        flow.control_output.return_value.readin(),
    )
    optimize(flow, flow.active_pairs)
    with flow.invocation(c.executor) as invocation:
        send_value(invocation.wire.return_value.readout(), c.to_key(0))
        send_value(c.const(10), invocation.port.value.readin())
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
        with unpack_port_and_wires(
            self, port, wires, executor, FromInterfaceRegister, ToInterfaceRegister
        ) as invocation:
            send_value(invocation.port.readout(), invocation.wire.readin())

    def __copy__(self) -> Self:
        return self


def test_expansion_invocation(c: Calculus) -> None:
    with expansion_invocation(
        TestExpansion(), c.executor, ToInterfaceRegister, FromInterfaceRegister
    ) as invocation:
        send_value(c.const(10), invocation.port.readin())
        send_value(invocation.wire.readout(), c.to_key(0))
        assert list(c.readout(0)) == [10]


def test_flow_register(c: Calculus) -> None:
    register = FlowRegister(ValueAdapter(), c.executor)
    send_value(c.const(10), register.readin())
    send_value(register.readout(), c.to_key(0))
    assert list(c.readout(0)) == [10]

    send_value(
        send_parameter(
            filter_invocation(lambda x: x + 10, c.executor), register.readout()
        ),
        register.readin(),
    )

    send_value(register.readout(), c.to_key(1))
    register.close()

    assert list(c.readout(1)) == [20]


def test_join_registers_are_parallel(c: Calculus) -> None:
    to_register = join_to_registers([c.to_key(0), c.to_key(1)], c.executor)
    from_register = join_from_registers([c.from_key(2), c.from_key(3)], c.executor)

    send_value(from_register, to_register)
    send_value(c.const(10), c.to_key(2))

    assert list(c.readout(0)) == [10]
    assert list(c.readout(1)) == []


def test_reference_adapter(c: Calculus) -> None:
    ref1 = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)
    ref2 = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)

    send_value(c.const(1), ref1.readin())
    send_value(ref1.readout(), ref2.readin())
    send_value(c.const(20), ref2.readin())
    ref2.close()

    send_value(ref1.readout(), c.to_key(0))
    send_value(ref2.readout(), c.to_key(1))
    ref1.close()

    assert list(c.readout(0)) == [20]
    assert list(c.readout(1)) == [20]


def test_inverse_adapter_with_close(c: Calculus) -> None:
    value = FlowRegister(ValueAdapter(), c.executor)
    inv1 = FlowRegister(InverseAdapter(ValueAdapter()), c.executor)
    inv2 = FlowRegister(InverseAdapter(ValueAdapter()), c.executor)

    send_value(inv1.readout(), value.readin())
    send_value(inv1.readout(), inv2.readin())
    inv1.close()
    send_value(c.const(1), inv2.readin())
    inv2.close()

    send_value(value.readout(), c.to_key(0))
    assert list(c.readout(0)) == [1]


def test_inverse_compound_adapter(c: Calculus) -> None:
    inv1 = FlowRegister(InverseAdapter(ReferenceAdapter(ValueAdapter())), c.executor)
    ref1 = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)
    ref2 = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)

    send_value(
        as_from_register(inv1.adapter.initialize(c.executor), inv1.adapter, c.executor),
        inv1.interface_readin(),
    )

    send_value(inv1.readout(), ref2.readin())
    send_value(c.const(10), ref2.readin())

    send_value(ref1.readout(), inv1.readin())
    send_value(ref1.readout(), c.to_key(0))

    ref1.close()
    ref2.close()
    inv1.close()

    assert list(c.readout(0)) == [10]


def test_borrow_registers(c: Calculus) -> None:
    value_r = FlowRegister(ValueAdapter(), c.executor)
    ref_r = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)
    par_r = FlowRegister(ParValueAdapter([ValueAdapter(), ValueAdapter()]), c.executor)

    send_value(c.const(10), value_r.readin())
    send_value(c.const([]), ref_r.readin())
    send_value(c.const((1, 2)), par_r.readin())

    r = FromInterfaceRegister(ValueAdapter(), c.executor)

    results = borrow_registers(
        [value_r.readout(), ref_r.readout(), par_r.readout()], r.readout()
    )

    send_values([results[0], results[2]], [c.to_key(0), c.to_key(2)])
    assert list(c.readout(0)) == [10]
    assert list(c.readout(2)) == [(1, 2)]

    send_parameter(filter_invocation(lambda x: x.append(1), c.executor), results[1])

    send_values(
        [value_r.readout(), ref_r.readout(), par_r.readout()],
        [c.to_key(3), c.to_key(4), c.to_key(5)],
    )
    assert list(c.readout(3)) == [10]
    assert list(c.readout(4)) == []
    assert list(c.readout(5)) == [(1, 2)]

    send_value(c.const(1), r.interface_readin())
    assert list(c.continue_readout()) == [[1]]


def test_borrow_registers_inverse(c: Calculus) -> None:
    ref_r = FlowRegister(ReferenceAdapter(ValueAdapter()), c.executor)
    inv_r = FlowRegister(InverseAdapter(ValueAdapter()), c.executor)

    send_value(c.const([]), ref_r.readin())

    r = FromInterfaceRegister(ValueAdapter(), c.executor)

    results = borrow_registers([ref_r.readout(), inv_r.readout()], r.readout())

    result = send_parameters(
        merge_invocation(lambda x, y: (x, y), c.executor), (results[0], results[1])
    )

    send_value(result, c.to_key(0))
    assert list(c.readout(0)) == []

    send_value(c.const(1), r.interface_readin())
    assert list(c.continue_readout()) == []

    send_value(c.const(1), inv_r.readin())
    assert list(c.continue_readout()) == [([], 1)]

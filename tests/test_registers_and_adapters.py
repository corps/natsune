from natsune.calculus import Calculus
import pytest

from natsune.registers import serialize_values, send_value, send_values, parallelize_value


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

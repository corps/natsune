import pytest

from natsune.calculus import Calculus


@pytest.fixture(scope="function")
def c() -> Calculus:
    return Calculus()


def test_tuple_of_values(c: Calculus) -> None:
    c[0] = c.tup(c[1], c[2])
    c[c.tup(c.v(1), c.v(2))] = c.tup(c[1], c[2])
    assert list(c.readout(0)) == [2, 1]


def test_dup(c: Calculus) -> None:
    c[0] = c.tup(c[1], c[2])
    c[c.v(3)] = c.dup(c[1], c[2])
    assert list(c.readout(0)) == [3, 3]


def test_dup_of_tup(c: Calculus) -> None:
    c[0] = c.tup(c[1], c[2])
    c[c.tup(c.v(3), c.v(5))] = c.dup(c[1], c[2])
    assert list(c.readout(0)) == [3, 5, 3, 5]

def test_dup_of_tup_erasure(c: Calculus) -> None:
    c[0] = c.tup(c[1], c[2])
    c[c.tup(c.v(3), c.e())] = c.dup(c[1], c[2])
    assert list(c.readout(0)) == [3, 3]

def test_merge_fn_out_of_order(c: Calculus) -> None:
    c[c.v(1)] = c.merge(lambda x, y: (x, y), c[2], c[0])
    assert list(c.readout(0)) == []

    c[2] = c.v(2)
    assert list(c.continue_readout()) == [(1, 2)]

def test_merge_fn_order(c: Calculus) -> None:
    c[c.v(1)] = c.merge(lambda x, y: (x, y), c[2], c[0])
    c[2] = c.v(2)
    assert list(c.readout(0)) == [(1, 2)]

def test_split_fn(c: Calculus) -> None:
    c[c.v(9)] = c.split(lambda x: (x, 8), c[2], c[0])
    assert list(c.readout(c.tup(c[2], c[0]))) == [8, 9]

def test_fork(c: Calculus) -> None:
    c[c.fork(c.v(1), c.v(2))] = c.tup(c.v(3), c.v(4))
    c.optimize()
    assert c.serialize_active_pairs() == [
        "(4, 3)x = fork(1, 2)"
    ]
    c.process_next_interaction()
    c.optimize()
    c.process_next_interaction()
    c.process_next_interaction()
    assert c.serialize_active_pairs() == [
        '3 = fork(1, 2)',
        '4 = fork(1, 2)',
    ]

def test_amb(c: Calculus) -> None:
    c.amb(c.v(1), c.v(2), c[1], c[2])
    c[1] = c.merge(lambda x, y: (x, y), c[2], c[0])
    assert list(c.readout(0)) == [(2, 1), (1, 2)]

def test_amb_just_one(c: Calculus) -> None:
    c.amb(c.v(1), c[3], c[1], c.e())
    c[1] = c.merge(lambda x, _: x, c.v(9), c[0])
    assert list(c.readout(0)) == [1]

def test_amb_of_amb(c: Calculus) -> None:
    c.amb(c.v(1), c.v(2), c[0], c[1])
    c.amb(c[0], c[1], c[2], c[3])
    c[2] = c.merge(lambda x, y: (x, y), c[3], c[4])
    assert list(c.readout(4)) == [1]

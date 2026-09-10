import pytest

from natsune.calculus import Calculus
from natsune.ports import ValuePort


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


def test_amb(c: Calculus) -> None:
    c.amb(c.v(1), c.v(2), c[1], c[2])
    c[1] = c.merge(lambda x, y: (x, y), c[2], c[0])
    assert list(c.readout(0)) == [(1, 2)]


def test_amb_just_one(c: Calculus) -> None:
    c.amb(c.v(1), c[3], c[1], c.e())
    c[1] = c.merge(lambda x, _: x, c.v(9), c[0])
    assert list(c.readout(0)) == [1]


# ---------------------------------------------------------------------------
# Duplication through nested structures
#
# `execute_commute_or_anihilate` re-parents "straight": the agent's own
# outputs/branches stay on the copies of the same symbol.  For `dup x tup`
# this means the source tree's internal tups meet the *tup copies* (same
# label -> annihilate) instead of the *dup copies* (different label ->
# commute), so the tree dissolves and its leaves surface exactly once,
# partitioned across the dup's two outputs.  Cloning never fires.
#
# Crossing the re-parenting (outputs -> the new tups, branches -> the new
# dups) makes `dup` distributive: each output receives a full copy of the
# tree with cloned leaves, at every depth.
# ---------------------------------------------------------------------------


def _tup_tree(c: Calculus, lo: int, hi: int):
    if hi - lo == 1:
        return c.v(lo)
    m = (lo + hi) // 2
    return c.tup(_tup_tree(c, lo, m), _tup_tree(c, m, hi))


def _drain(c: Calculus) -> None:
    while c.executor.active_pairs:
        c.executor.process_pair()


def _fringe(c: Calculus, key: int) -> list:
    """Graft-free observation: walk the live graph below interface `key`.

    readout() is invasive (its Grafts propagate onto aux wires and it drains
    the whole net), so tests that want a passive observation of one output
    should drain a fresh net and walk it instead.
    """
    seen: set[int] = set()
    vals: list = []
    stack = [c._wires[key].target]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, ValuePort):
            vals.append(p.value)
        for w in p.wires:
            stack.append(w.target)
    return sorted(vals)


def test_dup_depth_1_duplicates(c: Calculus) -> None:
    c[c.dup(c[100], c[200])] = _tup_tree(c, 0, 2)
    _drain(c)
    assert _fringe(c, 100) == [0, 1]
    assert _fringe(c, 200) == [0, 1]


def test_dup_nested_tree_distributes(c: Calculus) -> None:
    """Duplicating a tup tree yields a full copy of the tree (leaves cloned)
    on each of the dup's outputs, at any depth."""
    c[c.dup(c[100], c[200])] = _tup_tree(c, 0, 4)
    _drain(c)
    assert _fringe(c, 100) == [0, 1, 2, 3]
    assert _fringe(c, 200) == [0, 1, 2, 3]


def test_readout_drains_whole_net_and_is_not_independent(c: Calculus) -> None:
    """readout() attaches a propagating Graft and drains *every* active pair,
    so reading two interfaces on one Calculus observes two different nets:
    the second readout sees the already-quiesced result.  Observe outputs on
    separate Calculi (or drain first and walk the graph) when independence
    matters."""
    c[c.dup(c[100], c[200])] = _tup_tree(c, 0, 4)
    first = list(c.readout(100))
    assert not c.executor.active_pairs  # whole net was drained
    second = list(c.readout(200))
    assert sorted(first) == [0, 1, 2, 3]
    assert sorted(second) == [0, 1, 2, 3]

    # A fresh calculus reads output 200 identically to the drained net (this
    # program's grafts happen not to change the outcome), but the two
    # sequential readouts were still not independent observations.
    fresh = Calculus()
    fresh[fresh.dup(fresh[100], fresh[200])] = _tup_tree(fresh, 0, 4)
    # Same output, same values, different yield order: readout yields in
    # executor stack (LIFO) order, not any canonical tree order.
    assert sorted(list(fresh.readout(200))) == sorted(second)
    # ... and readout is destructive: the Grafts replace the output subtree,
    # so nothing is walkable below the interface afterwards.
    assert _fringe(fresh, 200) == []

from natsune.deque import LockFreeDeque


def test_basic_single_threaded() -> None:
    q = LockFreeDeque[int]()
    q.push(1)
    q.push(2)
    q.push(3)

    assert q.pop() == 3
    assert q.pop() == 2
    assert q.pop() == 1

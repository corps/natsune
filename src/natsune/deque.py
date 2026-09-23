import threading
from typing import Literal


class IdleCounter:
    __slots__ = ("_value", "_lock", "total")

    def __init__(self, total: int):
        self._value = 0
        self._lock = threading.Lock()
        self.total = total

    def signal(self, worker_id: int) -> bool:
        with self._lock:
            self._value = worker_id + 1
            # This can only happen when all workers have signaled idle consecutively
            return self._value == self.total

    def reset(self) -> None:
        with self._lock:
            self._value = 0


class WorkStealingDeque[A]:
    """
    A thread-safe deque with work stealing.
    Uses standard Python threading.Lock for synchronization instead of CAS operations.
    Under CPython's GIL, this provides equivalent performance to lock-free approaches
    while being simpler, more maintainable, and more portable.
    """

    __slots__ = (
        "left",
        "right",
        "store",
        "capacity",
        "max_reentrant_push",
        "wrap_mask",
        "_lock",
    )

    store: list[A | None]
    left: int
    right: int
    max_reentrant_push: int
    capacity: int
    wrap_mask: int
    _lock: threading.Lock

    def __init__(self, storage_hint: int, max_reentrant_push: int):
        if storage_hint < max_reentrant_push or max_reentrant_push < 1:
            raise ValueError(
                "storage_hint must be greater than max_reentrant_push and max_reentrant_push must be greater than 0"
            )

        storage_hint -= 1
        for i in (1, 2, 4, 8, 16, 32):
            storage_hint |= storage_hint >> i
        self.capacity = storage_hint + 1
        self.wrap_mask = storage_hint

        self.store = [None] * self.capacity
        # Thief side
        self.left = 0
        # Owner side
        self.right = 0
        self.max_reentrant_push = max_reentrant_push
        self._lock = threading.Lock()

    def push(self, task: A) -> None:
        with self._lock:
            cur_left = self.left
            cur_right = self.right

            # Workers should avoid this case as much as possible by avoiding pop or steal when
            # the capacity to push does not exist.
            assert (
                cur_right - cur_left < self.capacity
            ), "Re-entrant push greater than expected capacity"

            self.right = cur_right + 1
            self.store[cur_right & self.wrap_mask] = task

    def pop(self) -> A | None | Literal[0]:
        with self._lock:
            cur_left = self.left
            cur_right = self.right

            if cur_right <= cur_left:
                return None

            # Don't pop work that we won't be able to complete with potential additional push
            if cur_right - cur_left - 1 >= self.capacity - self.max_reentrant_push:
                return 0

            if cur_right - cur_left == 1:
                self.left = cur_left + 1
                return self.store[cur_left & self.wrap_mask]

            self.right = cur_right - 1
            return self.store[(cur_right - 1) & self.wrap_mask]

    def steal(self) -> A | None:
        with self._lock:
            cur_left = self.left
            cur_right = self.right

            if cur_right <= cur_left:
                return None

            self.left = cur_left + 1
            return self.store[cur_left & self.wrap_mask]

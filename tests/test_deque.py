"""
Comprehensive tests for the lock-free deque with work stealing.

Tests both the C extension (when available) and the pure Python fallback.
"""

import threading
import time
from natsune.deque import LockFreeDeque


def test_push_and_pop_single_item():
    dq = LockFreeDeque(16, 1)
    dq.push("task1")
    assert dq.pop() == "task1"
    assert dq.pop() is None


def test_push_and_pop_multiple_items():
    dq = LockFreeDeque(16, 1)
    items = ["task1", "task2", "task3"]

    for item in items:
        dq.push(item)

    for item in reversed(items):
        result = dq.pop()
        assert result == item


def test_pop_from_empty_deque():
    dq = LockFreeDeque(16, 1)
    result = dq.pop()
    assert result is None


def test_steal_from_empty_deque():
    dq = LockFreeDeque(16, 1)
    result = dq.steal()
    assert result is None


def test_steal_single_item():
    dq = LockFreeDeque(16, 1)
    dq.push("task1")
    result = dq.steal()
    assert result == "task1"
    assert dq.steal() is None
    assert dq.pop() is None


def test_steal_multiple_items():
    dq = LockFreeDeque(16, 1)
    items = ["task1", "task2", "task3"]

    for item in items:
        dq.push(item)

    result = dq.steal()
    assert result == "task1"


def test_mixed_push_pop_steal():
    dq = LockFreeDeque(16, 1)

    dq.push("a")
    dq.push("b")
    dq.push("c")

    assert dq.pop() == "c"
    assert dq.steal() == "a"
    assert dq.pop() == "b"
    assert dq.pop() is None


def test_concurrent_steal():
    dq = LockFreeDeque(128, 1)
    stolen = []
    errors = []

    for i in range(100):
        dq.push(f"task{i}")

    def stealer(stealer_id):
        try:
            while True:
                item = dq.steal()
                if item is None:
                    break
                stolen.append((stealer_id, item))
        except Exception as e:
            errors.append(e)

    n_stealers = 4
    threads = []

    for i in range(n_stealers):
        t = threading.Thread(target=stealer, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(stolen) == 100


def test_producer_consumer():
    dq = LockFreeDeque(1024, 1)
    total_items = 1000
    consumed = []
    errors = []

    def producer():
        try:
            for i in range(total_items):
                dq.push(f"item{i}")
        except Exception as e:
            errors.append(f"Producer error: {e}")

    def consumer():
        try:
            count = 0
            while count < total_items // 2:
                item = dq.pop()
                if item is not None:
                    consumed.append(item)
                    count += 1
        except Exception as e:
            errors.append(f"Consumer error: {e}")

    def stealer():
        try:
            count = 0
            while count < total_items // 2:
                item = dq.steal()
                if item is not None:
                    consumed.append(item)
                    count += 1
        except Exception as e:
            errors.append(f"Stealer error: {e}")

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    stealer_thread = threading.Thread(target=stealer)

    producer_thread.start()
    time.sleep(0.1)  # Give producer a head start
    consumer_thread.start()
    stealer_thread.start()

    producer_thread.join(timeout=10)
    consumer_thread.join(timeout=10)
    stealer_thread.join(timeout=10)

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(consumed) == total_items


def test_stress_test():
    dq = LockFreeDeque(1024, 1)
    errors = []

    def worker(worker_id, operations):
        try:
            for i in range(operations):
                if i % 3 == 0:
                    dq.push(f"worker{worker_id}-op{i}")
                elif i % 3 == 1:
                    dq.pop()
                else:
                    dq.steal()
        except Exception as e:
            errors.append(f"Worker {worker_id} error: {e}")

    n_workers = 8
    ops_per_worker = 500
    threads = []

    for i in range(n_workers):
        t = threading.Thread(target=worker, args=(i, ops_per_worker))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=30)

    assert len(errors) == 0, f"Errors occurred: {errors}"

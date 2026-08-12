import concurrent.futures
import dataclasses
import os
from functools import cached_property
from typing import Any, Self, cast

from natsune.connector import Connector
from natsune.deque import LockFreeDeque
from natsune.interactions import execute_interaction
from natsune.ports import Port


class DeadLetter(Connector):
    def connect_ports(self, l: Port, r: Port) -> None:
        pass


@dataclasses.dataclass
class DeterministicSerialExecutor(Connector):
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    def connect_ports(self, l: Port, r: Port) -> None:
        self.active_pairs.append((l, r))

    def process_pair(self) -> None:
        if self.active_pairs:
            l, r = self.active_pairs.pop()
            execute_interaction(self, l, r)

    def run(self) -> None:
        while self.active_pairs:
            self.process_pair()


@dataclasses.dataclass(slots=True)
class ThreadPoolExecutor(Connector):
    # None implies let the system decide.
    workers: int = cast(Any, None)
    queues: list[LockFreeDeque[tuple[Port, Port]]] = dataclasses.field(
        default_factory=list
    )
    pool: concurrent.futures.ThreadPoolExecutor = cast(Any, None)

    def __post_init__(self) -> None:
        # We reserve one queue for ourselves that can be stolen from by other workers.
        if self.workers is None:
            self.workers = min(32, (os.process_cpu_count() or 1) + 4)

        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)

        for i in range(self.workers):
            self.queues.append(LockFreeDeque())
            thread_exec = ThreadExecutor(self.queues, i)
            future = self.pool.submit(lambda f=future: thread_exec(f))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self.pool.shutdown(cancel_futures=True)


@dataclasses.dataclass(slots=True)
class ThreadExecutor(Connector):
    queues: list[LockFreeDeque[tuple[Port, Port]]]
    worker_id: int

    def connect_ports(self, l: Port, r: Port) -> None:
        self.queues[self.worker_id].push((l, r))

    def process_pair(self) -> bool:
        q = self.queues[self.worker_id]
        next_pair = q.pop()
        if next_pair is None:
            idx = self.worker_id + 1 % len(self.queues)
            while idx != self.worker_id:
                q = self.queues[idx]
                next_pair = q.steal()
                if next_pair:
                    break
                idx = idx + 1 % len(self.queues)

        if next_pair is not None:
            l, r = next_pair
            execute_interaction(self, l, r)
            return True

        return False

    def __call__(self, future: concurrent.futures.Future) -> None:
        while not future.cancelled():
            self.process_pair()

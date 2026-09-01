import concurrent.futures
import dataclasses
import os
import random
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from natsune.connector import Connector
from natsune.deque import IdleCounter, WorkStealingDeque
from natsune.interactions import execute_interaction
from natsune.ports import Port


class Executor(Connector, ABC):
    @abstractmethod
    def run(self, end_event: threading.Event) -> None: ...


@dataclasses.dataclass
class DeterministicSerialExecutor(Executor):
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    def connect_ports(self, l: Port, r: Port) -> None:
        self.active_pairs.append((l, r))

    def process_pair(self) -> None:
        if self.active_pairs:
            l, r = self.active_pairs.pop()
            execute_interaction(self, l, r)

    def run(self, end_event: threading.Event) -> None:
        while self.active_pairs and not end_event.is_set():
            self.process_pair()


@dataclasses.dataclass(slots=True)
class ThreadPoolExecutor(Executor):
    workers: int = cast(Any, None)
    worker_queue_size_hint: int = 1024
    max_reentrant: int = 16

    queues: list[WorkStealingDeque[tuple[Port, Port]]] = dataclasses.field(
        default_factory=list
    )
    running: bool = False
    idle_counter: IdleCounter = cast(Any, None)

    def connect_ports(self, l: Port, r: Port) -> None:
        # Select the next free worker queue and set i.
        assert not self.running
        i = random.randint(0, len(self.queues) - 1)
        self.queues[i].push((l, r))

    def __post_init__(self) -> None:
        # Borrowed from how threadpoolexecutor sees sane default worker counts
        if self.workers is None:
            self.workers = min(32, (os.process_cpu_count() or 1) + 4)
        self.idle_counter = IdleCounter(self.workers)
        # Initialize queues for connect_ports to work before run()
        # The actual worker threads will reuse these queues
        for _ in range(self.workers):
            self.queues.append(
                WorkStealingDeque(self.worker_queue_size_hint, self.max_reentrant)
            )

    def run(self, end_event: threading.Event) -> None:
        def print_worker_exception(future: concurrent.futures.Future[None]) -> None:
            try:
                future.result()
            except Exception:
                traceback.print_exc()
                end_event.set()

        self.running = True
        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
            with pool:
                for i in range(self.workers):
                    worker = ThreadWorker(self.idle_counter, self.queues, i, end_event)
                    pool.submit(worker).add_done_callback(print_worker_exception)
        finally:
            self.running = False


@dataclasses.dataclass(slots=True)
class ThreadWorker(Connector):
    idle_counter: IdleCounter
    queues: list[WorkStealingDeque[tuple[Port, Port]]]
    worker_id: int
    end_event: threading.Event

    def connect_ports(self, l: Port, r: Port) -> None:
        self.queues[self.worker_id].push((l, r))

    def process_pair(self) -> bool:
        q = self.queues[self.worker_id]
        next_pair = q.pop()

        if next_pair is None:
            idx = (self.worker_id + 1) % len(self.queues)
            while idx != self.worker_id:
                if self.end_event.is_set():
                    return True
                q = self.queues[idx]
                next_pair = q.steal()
                if next_pair:
                    break
                idx = (idx + 1) % len(self.queues)

        if next_pair:
            if self.end_event.is_set():
                return True
            self.idle_counter.reset()
            l, r = next_pair
            execute_interaction(self, l, r)
            return True

        return False

    def __call__(self) -> None:
        while not self.end_event.is_set():
            if not self.process_pair():
                # All workers are idle, stop
                if self.idle_counter.signal(self.worker_id):
                    return
                # Prevent busy sleep
                # time.sleep(0.001)

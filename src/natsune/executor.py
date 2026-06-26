import abc
import contextlib
import dataclasses
from contextlib import AbstractContextManager
from queue import Queue
from threading import RLock
from typing import Generator

from .connector import Connector
from .interactions import execute_interaction
from .ports import Port


class Executor(Connector):
    @abc.abstractmethod
    def lock(self) -> AbstractContextManager: ...

    @abc.abstractmethod
    def fork(self) -> Executor: ...


class DeadLetter(Executor):
    def connect_ports(self, l: Port, r: Port) -> None:
        pass

    def fork(self) -> Executor:
        return self

    @contextlib.contextmanager
    def lock(self) -> Generator:
        yield


@dataclasses.dataclass
class SynchronizedExecutor(Executor):
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    @contextlib.contextmanager
    def lock(self) -> Generator:
        yield

    def fork(self) -> Executor:
        return self

    def connect_ports(self, l: Port, r: Port) -> None:
        with self.lock():
            self.active_pairs.append((l, r))

    def take_active_pair(self) -> tuple[Port, Port] | None:
        with self.lock():
            try:
                return self.active_pairs.pop()
            except IndexError:
                return None

    def process_pair(self) -> None:
        with self.lock():
            if self.active_pairs:
                l, r = self.active_pairs.pop()
                execute_interaction(self, l, r)

    def run(self) -> None:
        with self.lock():
            while self.active_pairs:
                self.process_pair()


@dataclasses.dataclass
class ThreadSynchronizedExecutor(SynchronizedExecutor):
    pairs_lock: RLock = RLock()
    active_executor_queue: Queue = dataclasses.field(default_factory=Queue)

    @contextlib.contextmanager
    def lock(self) -> Generator:
        with self.pairs_lock:
            yield

    def fork(self) -> Executor:
        return ThreadSynchronizedExecutor(
            active_executor_queue=self.active_executor_queue
        )

    def connect_ports(self, l: Port, r: Port) -> None:
        with self.lock():
            self.active_pairs.append((l, r))
            if len(self.active_pairs) == 1:
                self.active_executor_queue.put(self)

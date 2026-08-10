import abc
import contextlib
import dataclasses
from contextlib import AbstractContextManager
from threading import RLock
from typing import Generator

from natsune.connector import Connector
from natsune.interactions import execute_interaction
from natsune.ports import Port


class DeadLetter(Connector):
    def connect_ports(self, l: Port, r: Port) -> None:
        pass


@dataclasses.dataclass
class SynchronizedExecutor(Connector):
    active_pairs: list[tuple[Port, Port]] = dataclasses.field(default_factory=list)

    @contextlib.contextmanager
    def lock(self) -> Generator:
        yield

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

    @contextlib.contextmanager
    def lock(self) -> Generator:
        with self.pairs_lock:
            yield

    def connect_ports(self, l: Port, r: Port) -> None:
        with self.lock():
            self.active_pairs.append((l, r))

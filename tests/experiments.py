import random
import threading
from typing import Self
import dataclasses
from natsune.compiler import inet
from natsune.executor import ThreadPoolExecutor
from natsune.special_forms import Par, Ref, Inverse


@dataclasses.dataclass(frozen=True, slots=True)
class Singleton[A]:
    inner: A

    def __copy__(self) -> Self:
        return self


@dataclasses.dataclass(slots=True)
class Universe:
    total_workers: int
    idle_counter: int = 0
    lock: threading.Lock = dataclasses.field(default_factory=lambda: threading.Lock())

    def mark_idle(self, worker: int) -> bool:
        with self.lock:
            if self.idle_counter == worker:
                self.idle_counter += 1

        return self.converged()

    def mark_active(self) -> None:
        with self.lock:
            self.idle_counter = 0

    def converged(self) -> bool:
        with self.lock:
            return self.idle_counter >= self.total_workers


@dataclasses.dataclass(slots=True)
class SortingWorld:
    words: list[int]
    agent_positions: list[int] = dataclasses.field(init=False)
    position_agents: list[int] = dataclasses.field(init=False)
    lock: threading.Lock = dataclasses.field(default_factory=lambda: threading.Lock())

    def __post_init__(self) -> None:
        self.agent_positions = list(range(len(self.words)))
        self.position_agents = list(self.agent_positions)

    def execute_swap(self, identity: int, target_identity: int) -> None:
        with self.lock:
            print(f"Swapping {identity} and {target_identity}")
            a = self.agent_positions[identity]
            b = self.agent_positions[target_identity]
            self.agent_positions[identity] = a
            self.agent_positions[target_identity] = b
            self.position_agents[a] = identity
            self.position_agents[b] = target_identity
            c = self.words[b]
            self.words[b] = self.words[a]
            self.words[a] = c

    def ask_self(self, identity: int) -> int:
        with self.lock:
            print("meee?")
            return self.words[self.agent_positions[identity]]

    def ask_point(self, identity: int, delta: int) -> tuple[int, int] | None:
        with self.lock:
            a = self.agent_positions[identity] + delta
            if delta < 0 or delta >= len(self.words):
                return None
            return self.position_agents[a], self.words[a]


@inet
def bubble_sort(world: SortingWorld, identity: int, universe: Universe) -> None:
    while True:
        print("looping")
        self_val = world.ask_self(0)
        right = world.ask_point(identity, 1)

        if right is not None:
            right_identity, right_val = right
            if self_val > right_val:
                world.execute_swap(identity, right_identity)
                universe.mark_active()
                continue

        self_val = world.ask_self(0)
        left = world.ask_point(identity, -1)
        if left is not None:
            left_identity, left_val = left
            if self_val < left_val:
                world.execute_swap(identity, left_identity)
                universe.mark_active()
                continue

        if universe.mark_idle(identity):
            break


@inet(executor=ThreadPoolExecutor())
def main() -> None:
    # world = SortingWorld([random.randint(1, 1000) for _ in range(30)])
    # print(f"World {id(world)}")
    # universe = Universe(len(world.words))
    result = 0
    for identity in range(20):
        print(identity)
        result += identity
    #     result = result or bubble_sort(world, identity, universe)
    #
    # return result


main()

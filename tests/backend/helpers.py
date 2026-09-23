"""Shared machinery for the backend lowering tests: cases are REAL
Python functions, not source strings.

A single-function case is just the function object; a multi-function
case is a `Program` — an entry function plus the callees that must be
`__inet__`-attached before the entry is compiled or IR-linked (exactly
what @inet does in production). Everything else is derived from those
objects through the genuine pipeline paths (`inspect.getsourcelines`
reads the real test module, the same way the `tests/frontend/programs.py`
programs are consumed by the snapshot suite), so there is no exec, no
linecache registration, and no string-source copies to keep in sync.
Adding a case is writing (or reusing) a function; its syntax is checked
at import like any other code.

Callees attach as deferred `InetFunction` markers (the post-cutover
decoration shape): the driver's callee pass compiles them on demand.
"""

from collections.abc import Generator
from contextlib import contextmanager
from types import FunctionType
from typing import Any

from natsune.backend.lowering import lower_function
from natsune.backend.python_backend import PythonBackend
from natsune.frontend.ir import IrFunction
from natsune.inet import InetFunction, build_ir_for_function


def _filename(func: FunctionType) -> str:
    return func.__code__.co_filename


@contextmanager
def markers_attached(
    callees: tuple[FunctionType, ...],
) -> Generator[None]:
    attached: list[FunctionType] = []
    try:
        for callee in callees:
            setattr(callee, "__inet__", InetFunction(callee))
            attached.append(callee)
        yield
    finally:
        for callee in attached:
            delattr(callee, "__inet__")


class Program:
    entry: FunctionType
    callees: tuple[FunctionType, ...]

    def __init__(self, entry: FunctionType, *callees: FunctionType) -> None:
        self.entry = entry
        self.callees = callees

    @property
    def name(self) -> str:
        return self.entry.__name__

    def call(self, *args: Any) -> Any:
        return self.entry(*args)

    def build_ir(self) -> IrFunction:
        with markers_attached(self.callees):
            return build_ir_for_function(self.entry)

    def lower(self, backend: PythonBackend | None = None) -> Any:
        return lower_function(
            self.build_ir(),
            backend if backend is not None else PythonBackend(),
        )


def program_ids(value: Any) -> str | None:
    if isinstance(value, Program):
        return value.name
    if isinstance(value, tuple):
        return "-".join("None" if arg is None else str(arg) for arg in value)
    return None


class CapturingBackend(PythonBackend):
    def __init__(self) -> None:
        super().__init__()
        self.artifact = None

    def finish(self, artifact):
        self.artifact = artifact
        return artifact


def capturing_lower(program: Program):
    backend = CapturingBackend()
    program.lower(backend)
    assert backend.artifact is not None
    return backend.artifact

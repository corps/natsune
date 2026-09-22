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

Two attachment modes, one attribute: `build_ir`/`lower` attach deferred
`InetFunction` markers (the post-cutover shape — the driver's callee
pass compiles them on demand); `compile_legacy` legacy-compiles callees
for the differential legs. They never run nested, so the `__inet__`
attribute handoff is safe.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from types import FunctionType
from typing import Any

from natsune.backend.lowering import lower_function
from natsune.backend.python_backend import PythonBackend
from natsune.compiler import InetFunctionCompiler
from natsune.control_flow import FlowControlInto, FlowInputInto
from natsune.executor import DeterministicSerialExecutor
from natsune.frontend.ir import IrFunction
from natsune.inet import InetFunction, build_ir_for_function
from natsune.invocations import expansion_invocation, filter_invocation
from natsune.registers import as_constant_register, send_value


def _filename(func: FunctionType) -> str:
    return func.__code__.co_filename


@contextmanager
def _inet_attached(
    callees: tuple[FunctionType, ...],
) -> Generator[None]:
    """Legacy-compile each callee and attach its `__inet__` for the
    duration of the block (what the old @inet did), then remove the
    attachment — module-level case functions are shared between tests.
    Serves the differential (`compile_legacy`) legs."""
    attached: list[FunctionType] = []
    try:
        for callee in callees:
            compiler = InetFunctionCompiler(
                callee, callee.__globals__, _filename(callee)
            )
            compiler.compile()
            setattr(callee, "__inet__", compiler)
            attached.append(callee)
        yield
    finally:
        for callee in attached:
            delattr(callee, "__inet__")


@contextmanager
def _markers_attached(
    callees: tuple[FunctionType, ...],
) -> Generator[None]:
    """Attach deferred `InetFunction` markers (the post-cutover
    decoration shape: mark only, compile on demand) for the duration of
    the block."""
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
    """A test program: an entry function plus optional callees.

    The callees are legacy-compiled and `__inet__`-attached before the
    entry is compiled or IR-linked, so multi-function cases link for
    real instead of falling back to eval. A single-function case is
    `Program(fn)`.
    """

    entry: FunctionType
    callees: tuple[FunctionType, ...]

    def __init__(self, entry: FunctionType, *callees: FunctionType) -> None:
        self.entry = entry
        self.callees = callees

    @property
    def name(self) -> str:
        return self.entry.__name__

    def call(self, *args: Any) -> Any:
        """Plain Python semantics: the source's own function is the
        oracle the compilers must match."""
        return self.entry(*args)

    def build_ir(self) -> IrFunction:
        """The deferred pipeline (natsune.inet): callees attached as
        markers, compiled on demand by the driver's callee pass."""
        with _markers_attached(self.callees):
            return build_ir_for_function(self.entry)

    def lower(self, backend: PythonBackend | None = None) -> Any:
        """The new lowering, through the backend: lower_function returns
        the finished artifact (a callable for the Python backend)."""
        return lower_function(
            self.build_ir(),
            backend if backend is not None else PythonBackend(),
        )

    def compile_legacy(self) -> InetFunctionCompiler:
        """The legacy compiler over the entry, with callees compiled and
        attached first (multi-function cases need the attachment to
        link; single-function cases are unaffected by it)."""
        with _inet_attached(self.callees):
            compiler = InetFunctionCompiler(
                self.entry, self.entry.__globals__, _filename(self.entry)
            )
            compiler.compile()
            return compiler


def program_ids(value: Any) -> str | None:
    """Parametrize ids: a `Program` ids as its entry function's name, an
    args tuple as its elements joined with '-'; every other value falls
    back to pytest's automatic id."""
    if isinstance(value, Program):
        return value.name
    if isinstance(value, tuple):
        return "-".join("None" if arg is None else str(arg) for arg in value)
    return None


class CapturingBackend(PythonBackend):
    """finish-capturing backend: the LoweredUnit is internal to
    lower_function, so introspection tests recover it here — conforming
    to the protocol, no src hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.artifact = None

    def finish(self, artifact):
        self.artifact = artifact
        return artifact


def capturing_lower(program: Program):
    """Lower through a finish-capturing backend to recover the
    LoweredUnit."""
    backend = CapturingBackend()
    program.lower(backend)
    assert backend.artifact is not None
    return backend.artifact


def run_legacy(expansion, *args, executor=None):
    """Drive a legacy-compiled flow with concrete args through a
    deterministic executor (mirrors the inet decorator's default
    runtime); pass an executor to mirror @inet(executor=...) cases —
    the old suite's infinite-value programs only terminate their output
    under ThreadPoolExecutor."""
    exec = executor if executor is not None else DeterministicSerialExecutor()

    with expansion_invocation(
        expansion, exec, FlowInputInto, FlowControlInto
    ) as invocation:
        variable_inputs = invocation.port.variables.readin().split()
        variable_inputs[0].close()
        for extra in variable_inputs[len(args) + 1 :]:
            extra.close()
        for register, arg in zip(variable_inputs[1 : len(args) + 1], args, strict=True):
            send_value(as_constant_register(arg, exec), register)

        outputs: list = []
        end_event = threading.Event()

        def output_callback(x):
            outputs.append(x)
            end_event.set()

        to_register, from_register = filter_invocation(output_callback, exec)
        from_register.close()
        send_value(invocation.wire.return_value.readout(), to_register)

    exec.run(end_event)
    if not outputs:
        raise ValueError("No output produced by the function")
    return outputs[0]

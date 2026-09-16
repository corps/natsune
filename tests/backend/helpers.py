"""Shared machinery for the backend lowering tests: cases are REAL
Python functions, not source strings.

A single-function case is just the function object; a multi-function
case is a `Program` — an entry function plus the callees that must be
legacy-compiled and `__inet__`-attached before the entry is compiled or
IR-linked (exactly what @inet does in production). Everything else is
derived from those objects through the genuine pipeline paths
(`inspect.getsourcelines` reads the real test module, the same way the
`tests/frontend/programs.py` programs are consumed by the snapshot
suite), so there is no exec, no linecache registration, and no
string-source copies to keep in sync. Adding a case is writing (or
reusing) a function; its syntax is checked at import like any other
code.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from types import FunctionType
from typing import Any

from natsune.backend.lowering import lower_function
from natsune.backend.python_backend import PythonBackend
from natsune.compiler import InetFunctionCompiler
from natsune.control_flow_generated import FlowControlInto, FlowInputInto
from natsune.executor import DeterministicSerialExecutor
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import IrFunction, build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.invocations import expansion_invocation, filter_invocation
from natsune.registers import as_constant_register, send_value


def _filename(func: FunctionType) -> str:
    return func.__code__.co_filename


@contextmanager
def _inet_attached(
    callees: tuple[FunctionType, ...],
) -> Generator[None]:
    """Legacy-compile each callee and attach its `__inet__` for the
    duration of the block (what @inet does), then remove the attachment —
    module-level case functions are shared between tests."""
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
        with _inet_attached(self.callees):
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


def build_ir_for_function(func: FunctionType) -> IrFunction:
    """The phase-1 pipeline over a real function object (no exec):
    extract_source → signature → symbols → call links → IR."""
    source = extract_source(func, globals=func.__globals__, filename=_filename(func))
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())
    links = collect_call_links(source.func_def.body, source.globals)
    sink = DiagnosticSink()
    ir = build_ir(source, signature, symbols, links, sink)
    assert not sink.diagnostics, tuple(str(d) for d in sink.diagnostics)
    return ir


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


def run_legacy(expansion, *args):
    """Drive a legacy-compiled flow with concrete args through a
    deterministic executor (mirrors the inet decorator's runtime)."""
    exec = DeterministicSerialExecutor()

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

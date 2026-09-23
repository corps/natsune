"""The inet decorator and deferred compilation driver (CUTOVER.md §2).

`@inet` marks a function for the compiler: it attaches an `InetFunction`
artifact as `__inet__` and returns a thin entry-point wrapper. Nothing
compiles at decoration time — compilation happens on first use, when
every import is guaranteed in scope:

- a call from standard Python goes through the entry point
  (`ensure_compiled`, then the net-driven callable);
- a call from inside another inet function is linked by the frontend and
  grafted into the caller's net at lowering time; the driver's callee
  pass compiles each callee on demand first, so definition/import order
  never matters.

Recursive and mutually recursive inet functions compile through
`PromiseExpansion` (natsune.backend.agents): a call site wired while the
callee is still mid-compile grafts the callee's promise, which the
callee's compile fills with the finished `FrozenExpansion` before any
net can run.
"""

import functools
import threading
from types import FunctionType
from typing import Any, Callable

from natsune.backend.agents import PromiseExpansion
from natsune.backend.connector import FrozenExpansion
from natsune.backend.control_flow import flow_control_adapter, flow_input_adapter
from natsune.backend.lowering import lower_function
from natsune.backend.protocol import LoweredUnit
from natsune.backend.python_backend import PythonBackend, as_callable
from natsune.executor import DeterministicSerialExecutor, Executor
from natsune.first_order.adapters import Adapter, ParValueAdapter, Variables
from natsune.frontend import IrFunction
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import (
    collect_call_links,
    collect_called_names,
)
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import SymbolsTable, collect_symbols

# One lock for all compilations: compiles are milliseconds, and a global
# lock makes cross-thread first-call races on mutually recursive pairs
# serialize instead of deadlocking (A holds its lock wanting B while B
# holds its lock wanting A). It is an RLock because the driver's callee
# pass compiles callees WHILE holding it — same-thread nesting is the
# depth-first compilation itself and must progress. What stops cycles is
# the per-artifact state checked BEFORE the lock is touched: `_compiling`
# (a mid-compile owner's ensure_compiled returns its promise) and
# `args_adapter` (already filled for a mid-compile owner's
# ensure_signature).
_compile_lock = threading.RLock()


class InetFunction:
    """The `__inet__` artifact: marker, deferred compile state, and the
    callee surface the compiler links against (natsune.frontend.inet_ref.InetRef).

    Field progression: `expansion` is None until compilation starts, then
    promise-or-frozen forever after; `entry` (the as_callable product) is
    the compiled flag.
    """

    def __init__(
        self, func: FunctionType, default_executor: Executor | None = None
    ) -> None:
        self.func = func
        self.default_executor = default_executor
        self.name = func.__name__

        # Signature metadata, filled during compilation (lazily: full
        # deferral also defers annotation resolution past imports).
        self.args_adapter: ParValueAdapter | None = None
        self.return_adapter: Adapter | None = None
        self.ir: IrFunction | None = None

        self.expansion: FrozenExpansion | PromiseExpansion | None = None
        self.entry: Callable[..., Any] | None = None

        self._compiling = False
        self._symbols: SymbolsTable | None = None
        self._promise_expansion: PromiseExpansion | None = None

    # -- compilation -------------------------------------------------------

    @property
    def compiled(self) -> bool:
        return self.entry is not None

    def ensure_compiled(self) -> "InetFunction":
        if self.entry is not None or self._compiling:
            # Compiled already, or mid-compile on this thread: the caller
            # reads `expansion` (the outstanding promise for a cycle).
            return self
        with _compile_lock:
            if self.entry is not None:
                return self
            self._compiling = True
            try:
                self._compile()
            finally:
                self._compiling = False
        return self

    def ensure_signature(self) -> "InetFunction":
        """Fill `args_adapter`/`return_adapter` without compiling: the
        cheapest data callers' LINKING needs (symbol collection types a
        call from the callee's return adapter). A mid-compile owner
        already has both, as does a compiled one."""
        if self.args_adapter is not None or self._compiling:
            return self
        with _compile_lock:
            if self.args_adapter is not None:
                return self
            source = extract_source(
                self.func,
                globals=self.func.__globals__,
                filename=self.func.__code__.co_filename,
            )
            sink = DiagnosticSink()
            signature = analyze_signature(source, sink)
            sink.raise_if_errors("inet compilation failed")
            self.args_adapter = signature.args_adapter
            self.return_adapter = signature.return_adapter
        return self

    def _compile(self) -> None:
        # A retry after a failed attempt must not keep a stale promise:
        # nothing valid can have embedded it (nets only run post-compile).
        self._promise_expansion = None
        self.expansion = None

        sink = DiagnosticSink()
        source = extract_source(
            self.func,
            globals=self.func.__globals__,
            filename=self.func.__code__.co_filename,
        )
        signature = analyze_signature(source, sink)
        self.args_adapter = signature.args_adapter
        self.return_adapter = signature.return_adapter

        # Signature pass first: collect_symbols types calls against callee
        # return adapters, so every marker callee must have its signature
        # filled before symbols are collected.
        markers = _resolve_markers(source)
        for _, ref in markers:
            ref.ensure_signature()

        self._symbols = collect_symbols(source, signature, sink)

        # From here until the end of the compile, `expansion` is
        # promise-or-frozen so lowering can graft every call uniformly.
        # (Symbols exist by now — a cyclic request during the pass below
        # can always materialize this promise.)
        self.expansion = self._promise()

        # Compile pass: compile every marked callee depth-first. A cyclic
        # ref is mid-compile on this thread and returns immediately,
        # leaving its promise in `ref.expansion` for the graft.
        for _, ref in markers:
            ref.ensure_compiled()

        links = collect_call_links(source.func_def.body, source.globals)
        ir = build_ir(source, signature, self._symbols, links, sink)
        sink.raise_if_errors("inet compilation failed")
        self.ir = ir

        capture = _UnitCapture(self._resolved_default_executor())
        self.entry = lower_function(ir, capture)

        assert capture.unit is not None
        frozen = capture.unit.main
        promise = self._promise_expansion
        assert promise is not None
        # The promise's adapters were built from the collected symbols;
        # the frozen net's from its own variable walk. They must agree —
        # this assert is the pinned invariant from CUTOVER.md §2.
        assert promise.input_adapter == frozen.input_adapter, (
            f"promise input adapter mismatch for {self.name}: "
            f"{promise.input_adapter} != {frozen.input_adapter}"
        )
        assert promise.output_adapter == frozen.output_adapter, (
            f"promise output adapter mismatch for {self.name}: "
            f"{promise.output_adapter} != {frozen.output_adapter}"
        )
        promise.target = frozen
        self.expansion = frozen

    def _promise(self) -> PromiseExpansion:
        if self._promise_expansion is None:
            # Only reachable once compilation is underway: the symbols and
            # return adapter are what the promise's adapters derive from.
            assert self._symbols is not None and self.return_adapter is not None
            # The real net's interface adapters derive from its full
            # variable bundle (params + locals, RA_VA slot first). The
            # collected symbols are that same set in that same order —
            # the constructor pair below is exactly what the net's
            # VariablesFlow will use.
            variables = Variables(dict(self._symbols.variables))
            self._promise_expansion = PromiseExpansion(
                self.name,
                flow_input_adapter(variables),
                flow_control_adapter(self.return_adapter, variables),
            )
        return self._promise_expansion

    def _resolved_default_executor(self) -> Executor:
        if self.default_executor is not None:
            return self.default_executor
        return DeterministicSerialExecutor()

    # -- entry point ---------------------------------------------------------

    def __call__(self, *args: Any, executor: Executor | None = None) -> Any:
        self.ensure_compiled()
        assert not isinstance(
            self.expansion, PromiseExpansion
        ), f"{self.name} is still compiling"
        if executor is None:
            assert self.entry is not None
            return self.entry(*args)
        assert self.ir is not None and self.expansion is not None
        return as_callable(self.ir, self.expansion, executor)(*args)


class _UnitCapture(PythonBackend):
    """finish-capturing backend: the driver needs the LoweredUnit's frozen
    main flow (to fill promises and expose `expansion`) as well as the
    finished entry callable that finish() returns."""

    def __init__(self, executor: Executor | None = None) -> None:
        super().__init__(executor)
        self.unit: LoweredUnit | None = None

    def finish(self, artifact: LoweredUnit) -> Any:
        self.unit = artifact
        return super().finish(artifact)


def _resolve_markers(source) -> list[tuple[str, Any]]:
    """(name, marker) for every called name that resolves to an inet
    artifact, in first-appearance order. Names that fail to evaluate are
    skipped here — link_name reports them with its own precision."""
    markers: list[tuple[str, Any]] = []
    for name in collect_called_names(source.func_def.body):
        try:
            value = eval(
                name, source.globals
            )  # noqa: S307 — compile-time linking by design
        except Exception:
            continue
        ref = getattr(value, "__inet__", None)
        if ref is not None:
            markers.append((name, ref))
    return markers


def build_ir_for_function(func: FunctionType) -> IrFunction:
    sink = DiagnosticSink()
    source = extract_source(
        func, globals=func.__globals__, filename=func.__code__.co_filename
    )
    signature = analyze_signature(source, sink)

    markers = _resolve_markers(source)
    for _, ref in markers:
        ref.ensure_signature()

    symbols = collect_symbols(source, signature, sink)

    for _, ref in markers:
        ref.ensure_compiled()

    links = collect_call_links(source.func_def.body, source.globals)
    ir = build_ir(source, signature, symbols, links, sink)
    sink.raise_if_errors("inet compilation failed")
    return ir


def inet(
    f: FunctionType | None = None, *, executor: Executor | None = None
) -> Callable[..., Any]:
    def decorate(func: FunctionType) -> Callable[..., Any]:
        artifact = InetFunction(func, executor)

        @functools.wraps(func)
        def impl(*args: Any, executor: Executor | None = None) -> Any:
            return artifact(*args, executor=executor)

        setattr(impl, "__inet__", artifact)
        return impl

    if f is not None:
        return decorate(f)
    return decorate

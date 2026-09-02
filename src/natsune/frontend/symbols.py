"""Phase 4 — symbol collection.

Rework of the old `InetVariablesEvaluator` (a `NodeVisitor` interleaved with
compiler-object mutation) into flat walk functions over explicit state,
producing a plain `SymbolTable`.

Absorbed behavior (verified against the old compiler; see §10):

- Parameters are seeded as locals from the phase-2 `Signature` adapters.
- `Assign` marks targets with the value's inferred adapter; tuple targets
  match element-wise against a same-size Par (§10 row 4: size mismatch is now
  a diagnostic — the old code silently fell back to VA). Chained assignment
  marks every target (the read-modify-write wiring question of §10 row 6 is
  a lowering concern, not a symbol concern).
- `AnnAssign` registers simple-name targets only, with the adapter from
  phase-3 `eval_annotation` (its consistent failure format, positioned at the
  target). The RHS is now walked (§10 row 11: the old collector skipped it,
  so a global RHS crashed at lowering with a raw KeyError).
- `AugAssign` introduces its Name target as a local with adapter VA — a
  documented feature (`x += 1` on an undeclared name), though Python would
  raise UnboundLocalError (§10 row 2). The value is now walked (same fix as
  AnnAssign).
- `For` targets are marked with VA (§10 row 3: typing loop targets from the
  iterable is deferred until lowering semantics are settled); tuple targets
  recurse; exotic targets (e.g. `for x[0] in ...`) are now diagnosed instead
  of silently marking the root name.
- Any other `Name` — Load or Store — not already a local is marked
  `used_as_globals` (§10 row 2 decision: the order-dependent global fallback
  is preserved). A read that precedes a normal assignment therefore trips the
  "Assign target is also a global variable" conflict, exactly like the old
  compiler (verified by probe).
- New diagnostic (§10 row 2): a value that reads a name first bound by the
  very same statement (`a = a + 1` with `a` otherwise unknown) is flagged —
  the old compiler silently created an uninitialized local where Python
  raises UnboundLocalError.

Try blocks are skipped entirely (both `ast.Try` and `ast.TryStar` are
rejected as unsupported): the try implementation is not trusted yet and gets
revisited with the paused lowering phases (§10 row 12).

Assignment-value adapter inference delegates to phase-5 `infer_adapter`; call
targets are pre-resolved once via phase-3 `collect_call_links`, so collection
stays deterministic and eval-free.

Deviations from the plan's written contract: `collect_symbols` takes the
phase-1 `FunctionSource` instead of bare `(func_def, globals)` — it bundles
them with the `SourceMap` the diagnostics need.

None of this imports `natsune.compiler`.
"""

import ast
import dataclasses
from typing import Any

from natsune.adapters import VA, Adapter, ParValueAdapter, adapter_from_type
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.infer import infer_adapter
from natsune.frontend.link import (
    EvaluationFailure,
    LinkResult,
    collect_call_links,
    eval_annotation,
)
from natsune.frontend.signature import Signature
from natsune.frontend.source import FunctionSource

# Absorbed from the old compiler's module-level tuples (§3.3 re-validates
# these during IR construction). `ast.TryStar` is new: the old collector
# rejected `try` but silently walked `try/except*`.
UNSUPPORTED_EXPR: tuple[type[ast.expr], ...] = (
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Starred,
    ast.Lambda,
    ast.NamedExpr,
)

UNSUPPORTED_STMT: tuple[type[ast.stmt], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Match,
    ast.Assert,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.TryStar,
    ast.TypeAlias,
    ast.Delete,
)


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolTable:
    """The function's variables and dynamic-fallback routing, collected.

    `variables` is a plain dict for practicality; by convention it is treated
    as read-only once returned (the paused lowering phases will freeze it
    into net-level `Variables`).
    """

    variables: dict[str, Adapter]
    used_as_globals: frozenset[str]


@dataclasses.dataclass
class _Collector:
    """Mutable per-function state, threaded explicitly through the walks."""

    source: FunctionSource
    sink: DiagnosticSink
    globals: dict[str, Any]
    links: dict[str, LinkResult]
    variables: dict[str, Adapter]
    used_as_globals: set[str] = dataclasses.field(default_factory=set)

    def error(self, message: str, node: ast.AST) -> None:
        # Caller-owned fallback (phase 0 decision): the FunctionDef is the
        # parent context available at this callsite.
        source_map = self.source.source_map
        position = source_map.resolve(node) or source_map.resolve(self.source.func_def)
        assert position is not None  # parsed nodes always carry locations
        self.sink.add_at(message, source_map, position)

    def note_name(self, node: ast.Name) -> None:
        """Old `visit_Name`: any unknown name routes through globals."""
        if node.id not in self.variables:
            self.used_as_globals.add(node.id)

    def mark_target(
        self,
        target: ast.Name,
        adapter: Adapter,
        just_declared: set[str] | None = None,
    ) -> None:
        if target.id in self.used_as_globals:
            self.error("Assign target is also a global variable", target)
            return
        if target.id not in self.variables:
            self.variables[target.id] = adapter
            if just_declared is not None:
                just_declared.add(target.id)


def collect_symbols(
    source: FunctionSource, signature: Signature, sink: DiagnosticSink
) -> SymbolTable:
    """Collect the function's variables and global reads into a SymbolTable."""
    collector = _Collector(
        source=source,
        sink=sink,
        globals=source.globals,
        links=collect_call_links(source.func_def.body, source.globals),
        variables={
            name: adapter
            for (name, _), adapter in zip(
                signature.args, signature.args_adapter.concurrent_items
            )
        },
    )
    for stmt in source.func_def.body:
        _visit_stmt(stmt, collector)
    return SymbolTable(
        variables=dict(collector.variables),
        used_as_globals=frozenset(collector.used_as_globals),
    )


# --- statement dispatch ------------------------------------------------------


def _visit_stmt(node: ast.stmt, state: _Collector) -> None:
    if isinstance(node, UNSUPPORTED_STMT):
        state.error("Unsupported statement type", node)
        return
    match node:
        case ast.Assign():
            _on_assign(node, state)
        case ast.AnnAssign():
            _on_annassign(node, state)
        case ast.AugAssign():
            _on_augassign(node, state)
        case ast.For():
            _on_for(node, state)
        case _:
            _visit_children(node, state)


def _visit_children(node: ast.stmt, state: _Collector) -> None:
    """Old `generic_visit`: recurse, collecting names, rejecting unsupported."""
    for child in ast.iter_child_nodes(node):
        _walk(child, state)


def _walk(node: ast.AST, state: _Collector) -> None:
    if isinstance(node, ast.Name):
        state.note_name(node)
        return
    if isinstance(node, UNSUPPORTED_EXPR):
        state.error("Unsupported expression type", node)
        return
    if isinstance(node, ast.stmt):
        _visit_stmt(node, state)
        return
    for child in ast.iter_child_nodes(node):
        _walk(child, state)


# --- specialized handlers ----------------------------------------------------


def _on_assign(node: ast.Assign, state: _Collector) -> None:
    value_adapter = infer_adapter(node.value, state.variables, state.links)
    just_declared: set[str] = set()
    for target in node.targets:
        _mark_target_expr(target, value_adapter, state, just_declared)
    # Walk the value after marking (old order), flagging reads of names this
    # statement itself just bound (`a = a + 1` with `a` otherwise unknown —
    # Python: UnboundLocalError; old compiler: silent uninitialized local).
    for sub in ast.walk(node.value):
        if (
            isinstance(sub, ast.Name)
            and isinstance(sub.ctx, ast.Load)
            and sub.id in just_declared
        ):
            state.error("Read of variable before assignment", sub)
    _walk(node.value, state)


def _on_annassign(node: ast.AnnAssign, state: _Collector) -> None:
    if not node.simple:
        state.error("Annotations must be simple in inet functions", node)
        return
    if not isinstance(node.target, ast.Name):
        return  # simple implies Name; belt and suspenders, as in the old code
    result = eval_annotation(node.annotation, state.globals)
    if isinstance(result, EvaluationFailure):
        state.error(result.message, node.target)
        return
    just_declared: set[str] = set()
    state.mark_target(node.target, adapter_from_type(result.value), just_declared)
    if node.value is not None:
        _walk(node.value, state)
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and sub.id in just_declared
            ):
                state.error("Read of variable before assignment", sub)


def _on_augassign(node: ast.AugAssign, state: _Collector) -> None:
    if isinstance(node.target, ast.Name):
        # Introduces the name as a local (documented §10 row 2 asymmetry).
        state.mark_target(node.target, VA)
    _walk(node.value, state)


def _on_for(node: ast.For, state: _Collector) -> None:
    _mark_loop_target(node.target, state)
    _walk(node.iter, state)
    for stmt in node.body:
        _visit_stmt(stmt, state)
    for stmt in node.orelse:
        _visit_stmt(stmt, state)


# --- targets -----------------------------------------------------------------


def _mark_target_expr(
    target: ast.expr,
    adapter: Adapter,
    state: _Collector,
    just_declared: set[str],
) -> None:
    if isinstance(target, ast.Name):
        state.mark_target(target, adapter, just_declared)
    elif isinstance(target, ast.Tuple):
        _mark_tuple_target(target, adapter, state, just_declared)
    elif isinstance(target, ast.List):
        state.error("List deconstructors in assignment not supported", target)
    # Attribute/Subscript targets are ignored, as in the old collector: the
    # names inside route through the dynamic path at lowering.


def _mark_tuple_target(
    target: ast.Tuple,
    adapter: Adapter,
    state: _Collector,
    just_declared: set[str],
) -> None:
    # §10 row 4: arity/type mismatch is a diagnostic (old: silent VA).
    if isinstance(adapter, ParValueAdapter) and len(adapter.concurrent_items) == len(
        target.elts
    ):
        for element, item_adapter in zip(target.elts, adapter.concurrent_items):
            _mark_target_expr(element, item_adapter, state, just_declared)
    else:
        state.error(
            "Tuple assignment targets do not match the value's Par size", target
        )
        for element in target.elts:
            _mark_target_expr(element, VA, state, just_declared)


def _mark_loop_target(target: ast.expr, state: _Collector) -> None:
    # §10 row 3: loop targets keep adapter VA (typing from the iterable is
    # deferred until lowering semantics are settled); exotic leaves are
    # diagnosed instead of silently marking a root name.
    if isinstance(target, ast.Name):
        state.mark_target(target, VA)
    elif isinstance(target, ast.Tuple):
        for element in target.elts:
            _mark_loop_target(element, state)
    else:
        state.error("Unsupported for-loop target", target)


# --- targets -----------------------------------------------------------------

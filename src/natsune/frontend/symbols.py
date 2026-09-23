import ast
import dataclasses
from typing import Any

from natsune.first_order.adapters import VA, Adapter, ParValueAdapter, adapter_from_type
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
from natsune.frontend.unsupported import UNSUPPORTED_EXPR, UNSUPPORTED_STMT


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolsTable:
    variables: dict[str, Adapter]
    used_as_globals: frozenset[str]


@dataclasses.dataclass
class SymbolsCollector:
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
) -> SymbolsTable:
    """Collect the function's variables and global reads into a SymbolTable."""
    collector = SymbolsCollector(
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
    return SymbolsTable(
        variables=dict(collector.variables),
        used_as_globals=frozenset(collector.used_as_globals),
    )


def _visit_stmt(node: ast.stmt, state: SymbolsCollector) -> None:
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
            for child in ast.iter_child_nodes(node):
                _walk(child, state)


def _walk(node: ast.AST, state: SymbolsCollector) -> None:
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


def _on_assign(node: ast.Assign, state: SymbolsCollector) -> None:
    value_adapter = infer_adapter(node.value, state.variables, state.links)
    just_declared: set[str] = set()
    for target in node.targets:
        _mark_target_expr(target, value_adapter, state, just_declared)
    for sub in ast.walk(node.value):
        if (
            isinstance(sub, ast.Name)
            and isinstance(sub.ctx, ast.Load)
            and sub.id in just_declared
        ):
            state.error("Read of variable before assignment", sub)
    _walk(node.value, state)


def _on_annassign(node: ast.AnnAssign, state: SymbolsCollector) -> None:
    if not node.simple:
        state.error("Annotations must be simple in inet functions", node)
        return
    if not isinstance(node.target, ast.Name):
        return
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


def _on_augassign(node: ast.AugAssign, state: SymbolsCollector) -> None:
    if isinstance(node.target, ast.Name):
        # Introduces the name as a local (documented §10 row 2 asymmetry).
        state.mark_target(node.target, VA)
    _walk(node.value, state)


def _on_for(node: ast.For, state: SymbolsCollector) -> None:
    _mark_loop_target(node.target, state)
    _walk(node.iter, state)
    for stmt in node.body:
        _visit_stmt(stmt, state)
    for stmt in node.orelse:
        _visit_stmt(stmt, state)


def _mark_target_expr(
    target: ast.expr,
    adapter: Adapter,
    state: SymbolsCollector,
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
    state: SymbolsCollector,
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


def _mark_loop_target(target: ast.expr, state: SymbolsCollector) -> None:
    if isinstance(target, ast.Name):
        state.mark_target(target, VA)
    elif isinstance(target, ast.Tuple):
        for element in target.elts:
            _mark_loop_target(element, state)
    else:
        state.error("Unsupported for-loop target", target)

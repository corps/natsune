import ast
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from natsune.adapters import VA, ParValueAdapter
from natsune.frontend.diagnostics import DiagnosticSink, Position
from natsune.frontend.infer import ParSubscriptError, infer_adapter, par_subscript_index
from natsune.frontend.ir.nodes import (
    IrAssign,
    IrAugAssign,
    IrBody,
    IrBoolOp,
    IrBreak,
    IrCallInet,
    IrConst,
    IrContinue,
    IrDynamic,
    IrExpr,
    IrExprStmt,
    IrFor,
    IrFunction,
    IrIf,
    IrParIndex,
    IrReturn,
    IrStmt,
    IrTarget,
    IrTargetDynamic,
    IrTargetName,
    IrTargetTuple,
    IrTuple,
    IrVar,
    IrWhile,
    analyze_ir_body,
)
from natsune.frontend.link import LinkedInet, LinkResult
from natsune.frontend.signature import Signature
from natsune.frontend.source import FunctionSource
from natsune.frontend.symbols import SymbolsTable
from natsune.frontend.unsupported import UNSUPPORTED_EXPR, UNSUPPORTED_STMT


@dataclasses.dataclass
class IrBuilder:
    source: FunctionSource
    symbols: SymbolsTable
    links: Mapping[str, LinkResult]
    sink: DiagnosticSink
    fresh_name: Callable[[], str]

    # Builder can have a parent context, TODO
    def error(self, message: str, node: ast.AST) -> None:
        source_map = self.source.source_map
        position = source_map.resolve(node) or source_map.resolve(self.source.func_def)
        assert position is not None  # parsed nodes always carry locations
        self.sink.add_at(message, source_map, position)

    def position_of(self, node: ast.AST) -> Position | None:
        return self.source.source_map.resolve(node)


def _default_name_factory(used: set[str]) -> Callable[[], str]:
    counter = 0

    def factory() -> str:
        nonlocal counter
        while True:
            name = f"__natsune_{counter}__"
            counter += 1
            if name not in used:
                used.add(name)
                return name

    return factory


def build_ir(
    source: FunctionSource,
    signature: Signature,
    symbols: SymbolsTable,
    links: Mapping[str, LinkResult],
    sink: DiagnosticSink,
    *,
    name_factory: Callable[[], str] | None = None,
) -> IrFunction:
    """Build the `IrFunction` for an extracted, analyzed, collected source."""
    if name_factory is None:
        name_factory = _default_name_factory(
            set(symbols.variables) | set(symbols.used_as_globals)
        )
    builder = IrBuilder(
        source=source,
        symbols=symbols,
        links=links,
        sink=sink,
        fresh_name=name_factory,
    )
    params = tuple(
        (name, adapter)
        for (name, _), adapter in zip(
            signature.args, signature.args_adapter.concurrent_items
        )
    )
    return IrFunction(
        name=source.func_def.name,
        params=params,
        return_adapter=signature.return_adapter,
        body=_build_body(source.func_def.body, builder),
        position=builder.position_of(source.func_def),
    )


def _build_body(body: list[ast.stmt], builder: IrBuilder) -> IrBody:
    statements: list[IrStmt] = []
    for stmt in body:
        built = _build_stmt(stmt, builder)
        if built is not None:
            statements.append(built)
    built = tuple(statements)
    usage, disjunctives, exit = analyze_ir_body(built)
    return IrBody(
        statements=built,
        variable_usage=usage,
        disjunctives=disjunctives,
        exit=exit,
    )


# --- statements ---------------------------------------------------------------


def _build_stmt(node: ast.stmt, builder: IrBuilder) -> IrStmt | None:
    """Build one statement; None means dropped (pass, bare AnnAssign,
    unsupported — unsupported statements are diagnosed, never raised)."""
    position = builder.position_of(node)
    if isinstance(node, UNSUPPORTED_STMT):
        builder.error("Unsupported statement type", node)
        return None
    match node:
        case ast.Assign():
            value = _build_expr(node.value, builder)
            targets = []
            for target in node.targets:
                built = _build_target(target, builder)
                if built is not None:
                    targets.append(built)
            _validate_tuple_targets(node.targets, value, builder)
            return IrAssign(targets=tuple(targets), value=value, position=position)
        case ast.AnnAssign():
            # Desugar: with a value it is an ordinary single-target assign;
            # without one it is a pure declaration and lowers to nothing.
            if node.value is None:
                return None
            target = _build_target(node.target, builder)
            if target is None:
                return None
            return IrAssign(
                targets=(target,),
                value=_build_expr(node.value, builder),
                position=position,
            )
        case ast.AugAssign():
            target = _build_target(node.target, builder)
            if target is None:
                return None
            return IrAugAssign(
                target=target,
                op=node.op,
                value=_build_expr(node.value, builder),
                position=position,
            )
        case ast.For():
            target = _build_target(node.target, builder)
            if target is None:
                return None
            return IrFor(
                target=target,
                iter=_build_expr(node.iter, builder),
                body=_build_body(node.body, builder),
                orelse=_build_body(node.orelse, builder),
                position=position,
            )
        case ast.While():
            return IrWhile(
                test=_build_expr(node.test, builder),
                body=_build_body(node.body, builder),
                orelse=_build_body(node.orelse, builder),
                position=position,
            )
        case ast.If():
            # elif chains arrive pre-nested: the parser puts the next If in
            # orelse, which is exactly the IR shape (§3.3).
            return IrIf(
                test=_build_expr(node.test, builder),
                then_body=_build_body(node.body, builder),
                else_body=_build_body(node.orelse, builder),
                position=position,
            )
        case ast.Return():
            value = _build_expr(node.value, builder) if node.value is not None else None
            return IrReturn(value=value, position=position)
        case ast.Expr():
            return IrExprStmt(value=_build_expr(node.value, builder), position=position)
        case ast.Break():
            return IrBreak(position=position)
        case ast.Continue():
            return IrContinue(position=position)
        case ast.Pass():
            return None
        case _:
            builder.error("Unsupported statement type", node)
            return None


# --- targets ------------------------------------------------------------------


def _build_target(node: ast.expr, builder: IrBuilder) -> IrTarget | None:
    position = builder.position_of(node)
    if isinstance(node, ast.Name):
        return IrTargetName(
            name=node.id,
            adapter=builder.symbols.variables.get(node.id, VA),
            is_global=node.id in builder.symbols.used_as_globals,
            position=position,
        )
    if isinstance(node, ast.Tuple):
        elements = []
        for element in node.elts:
            built = _build_target(element, builder)
            if built is not None:
                elements.append(built)
        return IrTargetTuple(elements=tuple(elements), position=position)
    if isinstance(node, ast.List):
        builder.error("List deconstructors in assignment not supported", node)
        return None
    # Attribute/Subscript lvalues: the old lowering routed them through the
    # exec fallback; IrTargetDynamic carries what that fallback needs.
    dynamic = _scan_dynamic(node, builder)
    return IrTargetDynamic(
        source_text=dynamic.source_text,
        captures=dynamic.captures,
        ast_node=node,
        position=position,
    )


def _validate_tuple_targets(
    targets: list[ast.expr], value: IrExpr, builder: IrBuilder
) -> None:
    """§10 row 4: tuple targets must match the value's Par size (same message
    as the collector; the sink deduplicates the pipeline's double check)."""
    for target in targets:
        if isinstance(target, ast.Tuple) and not (
            isinstance(value.adapter, ParValueAdapter)
            and len(value.adapter.concurrent_items) == len(target.elts)
        ):
            builder.error(
                "Tuple assignment targets do not match the value's Par size", target
            )


# --- expressions --------------------------------------------------------------


def _build_expr(node: ast.expr, builder: IrBuilder) -> IrExpr:
    if isinstance(node, UNSUPPORTED_EXPR):
        # The rewritten source of these does not re-parse in eval mode; the
        # diagnostic is what keeps them from ever lowering.
        builder.error("Unsupported expression type", node)
        return IrDynamic(
            source_text=ast.unparse(node),
            position=builder.position_of(node),
            ast_node=node,
        )
    typed = _try_typed(node, builder)
    if typed is not None:
        return typed
    return _scan_dynamic(node, builder)


def _fold_unaryop(node: ast.UnaryOp) -> Any:
    """Fold a unary operation over a literal constant, or None.

    Mirrors CPython's optimizer: fold only when the operand type implements
    the operation (`-"str"` and `~1.5` stay dynamic and raise at runtime).
    """
    if not isinstance(node.operand, ast.Constant):
        return None
    value = node.operand.value
    # bool is an int subclass: `-True` folds to -1, exactly like CPython.
    match node.op:
        case ast.USub():
            if isinstance(value, (int, float, complex)):
                return -value
        case ast.UAdd():
            if isinstance(value, (int, float, complex)):
                return +value
        case ast.Invert():
            if isinstance(value, int):
                return ~value
        case ast.Not():
            return not value
    return None


def _boolop_name(op: ast.operator | ast.boolop) -> str:
    return "or" if isinstance(op, ast.Or) else "and"


def _try_typed(node: ast.expr, builder: IrBuilder) -> IrExpr | None:
    """Type an expression as one of the IR kinds, or None (dynamic route).

    Diagnostics fired here (inet keywords/arity, Par subscripts) are the
    §3.3 validation; the caller decides the fallback shape.
    """
    position = builder.position_of(node)
    match node:
        case ast.Constant():
            return IrConst(value=node.value, position=position)
        case ast.UnaryOp():
            # Constant folding: `-1` is a constant, not an exec fallback
            # (CPython's own optimizer folds these too). Unfoldable operand
            # types (`~1.5`) stay dynamic.
            folded = _fold_unaryop(node)
            if folded is not None:
                return IrConst(value=folded, position=position)
            return None
        case ast.Name():
            if node.id in builder.symbols.variables:
                return IrVar(
                    name=node.id,
                    adapter=builder.symbols.variables[node.id],
                    position=position,
                )
            return IrVar(name=node.id, is_global=True, position=position)
        case ast.Tuple():
            return IrTuple(
                elements=tuple(_build_expr(element, builder) for element in node.elts),
                adapter=infer_adapter(node, builder.symbols.variables, builder.links),
                position=position,
            )
        case ast.Subscript():
            base_adapter = infer_adapter(
                node.value, builder.symbols.variables, builder.links
            )
            if not isinstance(base_adapter, ParValueAdapter):
                return None
            index = par_subscript_index(base_adapter, node.slice)
            if isinstance(index, ParSubscriptError):
                builder.error(index.message, node.slice)
                return None
            return IrParIndex(
                base=_build_expr(node.value, builder),
                index=index,
                adapter=base_adapter.concurrent_items[index],
                position=position,
            )
        case ast.Call():
            if not isinstance(node.func, ast.Name):
                return None
            link = builder.links.get(node.func.id)
            if not isinstance(link, LinkedInet):
                return None
            if node.keywords:
                builder.error(
                    "Keyword arguments are currently not supported for inet"
                    " invocations",
                    node,
                )
                return None
            if len(node.args) != link.arity:
                builder.error(
                    f"Expected {link.arity} arguments, got {len(node.args)}", node
                )
                return None
            return IrCallInet(
                ref=link.ref,
                args=tuple(_build_expr(arg, builder) for arg in node.args),
                arity=link.arity,
                arg_adapters=link.arg_adapters,
                return_adapter=link.return_adapter,
                adapter=link.return_adapter,
                position=position,
            )
        case ast.BoolOp():
            op = _boolop_name(node.op)
            values: list[IrExpr] = []
            for operand in node.values:
                if isinstance(operand, ast.BoolOp) and _boolop_name(operand.op) == op:
                    values.extend(_flatten_boolop(operand, builder, op))
                else:
                    values.append(_build_expr(operand, builder))
            return IrBoolOp(op=op, values=tuple(values), position=position)
        case _:
            return None


def _flatten_boolop(node: ast.BoolOp, builder: IrBuilder, op: str) -> list[IrExpr]:
    values: list[IrExpr] = []
    for operand in node.values:
        if isinstance(operand, ast.BoolOp) and _boolop_name(operand.op) == op:
            values.extend(_flatten_boolop(operand, builder, op))
        else:
            values.append(_build_expr(operand, builder))
    return values


# --- dynamic fallback ---------------------------------------------------------


def _capturable(typed: IrExpr) -> bool:
    """The old special-form set: what the exec fallback captures as a value.

    Local reads, inet calls, tuples, and valid Par indexes are captured;
    constants and boolean expressions stay in the rewritten source (pure
    Python), and global names stay in the source (they resolve through the
    exec context).
    """
    if isinstance(typed, IrVar):
        return not typed.is_global
    return isinstance(typed, (IrCallInet, IrTuple, IrParIndex))


class _DynamicRewriter(ast.NodeTransformer):
    """Mirrors the old `ReplaceWithSerializedVariables`, minus all wiring."""

    def __init__(self, builder: IrBuilder, captures: dict[str, IrExpr]) -> None:
        self.builder = builder
        self.captures = captures

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr):
            typed = _try_typed(node, self.builder)
            if typed is not None and _capturable(typed):
                if isinstance(node, ast.Name):
                    name = node.id
                else:
                    name = self.builder.fresh_name()
                if name not in self.captures:
                    self.captures[name] = typed
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return super().generic_visit(node)


def _scan_dynamic(node: ast.expr, builder: IrBuilder) -> IrDynamic:
    """Scan the ORIGINAL tree (like the old `rewriter.visit(expr)`): the
    rewritten source unparses back from it, and any diagnostics fired inside
    the scan carry true positions and deduplicate against the ones fired
    while typing the same node."""
    captures: dict[str, IrExpr] = {}
    rewritten = _DynamicRewriter(builder, captures).visit(node)
    return IrDynamic(
        source_text=ast.unparse(rewritten),
        captures=tuple(captures.items()),
        position=builder.position_of(node),
        ast_node=node,
    )

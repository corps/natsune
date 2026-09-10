import ast
import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from enum import IntFlag
from typing import Any, Literal, assert_never

from natsune.adapters import VA, Adapter, read_independently
from natsune.frontend.diagnostics import Position


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrNode:
    position: Position | None = dataclasses.field(
        default=None, kw_only=True, compare=False
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetName(IrNode):
    name: str
    adapter: Adapter
    is_global: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetTuple(IrNode):
    elements: tuple[IrTarget, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetDynamic(IrNode):
    source_text: str
    captures: tuple[tuple[str, IrExpr], ...] = ()
    # The original ast node this target was built from, so consumers can
    # inspect it without re-parsing source_text. Not part of equality/repr.
    ast_node: ast.expr = dataclasses.field(compare=False, repr=False)


type IrTarget = IrTargetName | IrTargetTuple | IrTargetDynamic


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrVar(IrNode):
    name: str
    is_global: bool = False
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrConst(IrNode):
    value: Any
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTuple(IrNode):
    elements: tuple[IrExpr, ...]
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrParIndex(IrNode):
    base: IrExpr
    index: int
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBoolOp(IrNode):
    op: str
    values: tuple[IrExpr, ...]
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrCallInet(IrNode):
    ref: Any
    args: tuple[IrExpr, ...]
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrDynamic(IrNode):
    source_text: str
    captures: tuple[tuple[str, IrExpr], ...] = ()
    adapter: Adapter = VA
    # The original ast node this dynamic was built from, so consumers can
    # inspect it without re-parsing source_text (mirrors IrTargetDynamic).
    # Not part of equality/repr.
    ast_node: ast.expr = dataclasses.field(compare=False, repr=False)


type IrExpr = (
    IrVar | IrConst | IrTuple | IrParIndex | IrBoolOp | IrCallInet | IrDynamic
)


class Exits(IntFlag):
    """Ways a node can terminate, as a may-analysis: each set bit is a
    possible exit kind. FALLTHROUGH means flow returns to the enclosing
    statement list (the node is a disjunctive).
    """

    BOTTOM = 0
    FALLTHROUGH = 1
    RETURN = 2
    BREAK = 4
    CONTINUE = 8


# Bits of a loop body's exits that pass through the loop unchanged: RETURN
# escapes the function, FALLTHROUGH keeps flow alive. CONTINUE is dropped
# (it only re-tests the loop header); BREAK is promoted to FALLTHROUGH
# separately (see _loop_exits).
LOOP_BODY_MASK = Exits.FALLTHROUGH | Exits.RETURN


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBody(IrNode):
    """A statement list with derived fields computed at build time by
    analyze_ir_body: variable_usage merges nested bodies' own usages
    ("write" wins over "read"); disjunctives are the IrIf/IrWhile/IrFor
    statements before exit that return flow to this list; closer is the first
    statement that never returns flow (None when all statements fall
    through). Invalid flow structures raise IrStructureError at build time.
    """

    statements: tuple[IrStmt, ...] = ()
    variable_usage: Mapping[str, VariableUsage] = dataclasses.field(
        default_factory=dict, compare=False
    )
    disjunctives: Sequence[IrIf | IrWhile | IrFor] = dataclasses.field(
        default=(), compare=False, repr=False
    )
    closer: IrBodyExit | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    @property
    def exits(self) -> Exits:
        result = Exits.BOTTOM
        if self.closer is not None:
            if isinstance(self.closer, (IrIf, IrWhile, IrFor)):
                result = self.closer.exits
            elif isinstance(self.closer, IrReturn):
                result = Exits.RETURN
            elif isinstance(self.closer, IrBreak):
                result = Exits.BREAK
            elif isinstance(self.closer, IrContinue):
                result = Exits.CONTINUE
            else:
                assert_never(self.closer)

        for disjunctive in self.disjunctives:
            result |= disjunctive.exits
        return result or Exits.FALLTHROUGH


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAssign(IrNode):
    targets: tuple[IrTarget, ...]
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAugAssign(IrNode):
    target: IrTarget
    op: ast.operator
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrIf(IrNode):
    test: IrExpr
    then_body: IrBody
    else_body: IrBody

    @property
    def exits(self) -> Exits:
        return self.then_body.exits | self.else_body.exits


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFor(IrNode):
    target: IrTarget
    iter: IrExpr
    body: IrBody
    orelse: IrBody

    @property
    def exits(self) -> Exits:
        return _loop_exits(self.body, self.orelse)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrWhile(IrNode):
    test: IrExpr
    body: IrBody
    orelse: IrBody

    @property
    def exits(self) -> Exits:
        return _loop_exits(self.body, self.orelse)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrReturn(IrNode):
    value: IrExpr | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBreak(IrNode):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrContinue(IrNode):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrExprStmt(IrNode):
    value: IrExpr


type IrStmt = (
    IrAssign
    | IrAugAssign
    | IrIf
    | IrFor
    | IrWhile
    | IrReturn
    | IrBreak
    | IrContinue
    | IrExprStmt
)

type IrBodyExit = IrIf | IrWhile | IrFor | IrReturn | IrContinue | IrBreak


class IrStructureError(Exception):
    """An IrBody statement list has an invalid exit structure."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFunction(IrNode):
    name: str
    params: tuple[tuple[str, Adapter], ...] = ()
    return_adapter: Adapter = VA
    body: IrBody

    @property
    def exits(self) -> Exits:
        return self.body.exits


def iter_child_expressions(node: IrExpr | IrStmt) -> Iterator[IrExpr]:
    """Yield the direct IrExpr children of an expression or statement (shallow)."""
    match node:
        case IrVar() | IrConst():
            return
        case IrTuple(elements=elements):
            yield from elements
        case IrParIndex(base=base):
            yield base
        case IrBoolOp(values=values):
            yield from values
        case IrCallInet(args=args):
            yield from args
        case IrDynamic(captures=captures):
            for _, capture in captures:
                yield capture
        case IrAssign(value=value) | IrAugAssign(value=value) | IrExprStmt(value=value):
            yield value
        case IrIf(test=test) | IrWhile(test=test):
            yield test
        case IrFor(iter=iter_):
            yield iter_
        case IrReturn(value=value):
            if value is not None:
                yield value
        case IrBreak() | IrContinue():
            return
        case _:
            assert_never(node)


type VariableUsage = Literal["read", "write"]


def _record_variable_usage(
    usage: dict[str, VariableUsage], name: str, how: VariableUsage
) -> None:
    """Record a usage; "write" takes priority over "read"."""
    if how == "write" or name not in usage:
        usage[name] = how


def _merge_variable_usage(
    into: dict[str, VariableUsage], source: Mapping[str, VariableUsage]
) -> None:
    for name, how in source.items():
        _record_variable_usage(into, name, how)


def _collect_expr_usage(
    expr: IrExpr,
    usage: dict[str, VariableUsage],
    adapters: Mapping[str, Adapter],
) -> None:
    if isinstance(expr, IrVar) and not expr.is_global:
        # Usage classifies the EFFECT on the variable's cell, not the
        # syntactic position (§1): a read of a variable whose adapter
        # linearizes reads (Reference/Inverse/Par today) is effectually a
        # write — the cell advances.
        adapter = adapters.get(expr.name)
        how = (
            "read"
            if adapter is None or read_independently(adapter.adapter_wiring_type())
            else "write"
        )
        _record_variable_usage(usage, expr.name, how)
    for child in iter_child_expressions(expr):
        _collect_expr_usage(child, usage, adapters)


def _collect_target_usage(
    target: IrTarget,
    usage: dict[str, VariableUsage],
    adapters: Mapping[str, Adapter],
) -> None:
    match target:
        case IrTargetName(is_global=False, name=name):
            _record_variable_usage(usage, name, "write")
        case IrTargetName():
            return  # global names are not tracked
        case IrTargetTuple(elements=elements):
            for element in elements:
                _collect_target_usage(element, usage, adapters)
        case IrTargetDynamic(captures=captures):
            for _, capture in captures:
                _collect_expr_usage(capture, usage, adapters)
        case _:
            assert_never(target)


def _iter_child_bodies(stmt: IrStmt) -> Iterator[IrBody]:
    match stmt:
        case IrIf(then_body=then_body, else_body=else_body):
            yield then_body
            yield else_body
        case IrFor(body=body, orelse=orelse) | IrWhile(body=body, orelse=orelse):
            yield body
            yield orelse
        case (
            IrAssign()
            | IrAugAssign()
            | IrReturn()
            | IrBreak()
            | IrContinue()
            | IrExprStmt()
        ):
            return
        case _:
            assert_never(stmt)


def _collect_stmt_usage(
    stmt: IrStmt,
    usage: dict[str, VariableUsage],
    adapters: Mapping[str, Adapter],
) -> None:
    match stmt:
        case IrAssign(targets=targets):
            for target in targets:
                _collect_target_usage(target, usage, adapters)
        case IrAugAssign(target=target):
            _collect_target_usage(target, usage, adapters)
        case IrFor(target=target):
            _collect_target_usage(target, usage, adapters)
        case IrIf() | IrWhile() | IrBreak() | IrContinue() | IrReturn() | IrExprStmt():
            pass
        case _:
            assert_never(stmt)
    for expr in iter_child_expressions(stmt):
        _collect_expr_usage(expr, usage, adapters)
    for body in _iter_child_bodies(stmt):
        _merge_variable_usage(usage, body.variable_usage)


def _loop_exits(body: IrBody, orelse: IrBody) -> Exits:
    """Exits of IrWhile/IrFor.

    RETURN propagates past the loop; FALLTHROUGH (an iteration completes)
    and CONTINUE (jumps to the re-test) keep the loop alive. The orelse
    runs whenever the loop completes normally — conditions and iterables
    are opaque, so its exits always contribute. BREAK terminates the loop
    while skipping the orelse: flow resumes after the loop, so it
    contributes FALLTHROUGH for the loop as a whole.
    """
    body_exits = body.exits
    result = (body_exits & LOOP_BODY_MASK) | orelse.exits
    if body_exits & Exits.BREAK:
        result |= Exits.FALLTHROUGH
    return result


def _validate_post_exit(stmt: IrStmt) -> None:
    """Check one statement placed after the body's exit has been determined.

    Post-exit statements run concurrently: a bare IrReturn/IrContinue/IrBreak,
    or a disjunctive with an exiting path, would race the determined exit.
    """
    if isinstance(stmt, (IrReturn, IrContinue, IrBreak)):
        raise IrStructureError(
            f"{type(stmt).__name__} after the statement list's close: post-close "
            "statements run concurrently and cannot exit themselves"
        )
    if isinstance(stmt, (IrIf, IrWhile, IrFor)) and stmt.exits != Exits.FALLTHROUGH:
        raise IrStructureError(
            f"{type(stmt).__name__} after the statement list's close has a closing "
            "path: post-close disjunctives must fall through on every path"
        )


def _analyze_body_flow(
    statements: tuple[IrStmt, ...],
) -> tuple[Sequence[IrIf | IrWhile | IrFor], IrBodyExit | None]:
    """Scan a statement list for its disjunctives and closer.

    A disjunctive (IrIf/IrWhile/IrFor) that can fall through (its exits
    carries Exits.FALLTHROUGH) returns flow to the statement list and is
    collected; the first statement that never returns flow is the closer.
    Statements after the closer are validated by _validate_post_exit. Exits
    are read from the stored exits properties, so the analysis runs
    bottom-up: children must already carry their computed fields.
    """
    disjunctives: list[IrIf | IrWhile | IrFor] = []
    body_exit: IrBodyExit | None = None
    exit_index = -1
    for index, stmt in enumerate(statements):
        if isinstance(stmt, (IrReturn, IrContinue, IrBreak)):
            body_exit, exit_index = stmt, index
            break
        if isinstance(stmt, (IrIf, IrWhile, IrFor)):
            if Exits.FALLTHROUGH & stmt.exits:
                disjunctives.append(stmt)  # returns flow to this list
            else:
                body_exit, exit_index = stmt, index  # every path exits
                break
    if body_exit is not None:
        for stmt in statements[exit_index + 1 :]:
            _validate_post_exit(stmt)
    return tuple(disjunctives), body_exit


def analyze_ir_body(
    statements: tuple[IrStmt, ...],
    adapters: Mapping[str, Adapter],
) -> tuple[
    Mapping[str, VariableUsage], Sequence[IrIf | IrWhile | IrFor], IrBodyExit | None
]:
    """Compute an IrBody's derived fields (see the IrBody docstring).

    Called on a body's statements as the builder constructs it: child bodies
    inside the statements must already carry their own computed fields,
    keeping the whole computation bottom-up. Invalid exit structures raise
    IrStructureError here, at build time.

    ``adapters`` is the function-wide name→adapter table (params and
    annotated locals — SymbolsTable.variables): reads are classified by
    the variable's wiring discipline, so the table must be complete,
    which it is — it is built ahead of IR construction, order-free.
    """
    usage: dict[str, VariableUsage] = {}
    for stmt in statements:
        _collect_stmt_usage(stmt, usage, adapters)
    disjunctives, body_exit = _analyze_body_flow(statements)
    return usage, disjunctives, body_exit

import ast
import dataclasses
from collections.abc import Iterator
from typing import Any, assert_never

from natsune.adapters import VA, Adapter
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


IrTarget = IrTargetName | IrTargetTuple | IrTargetDynamic


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


IrExpr = IrVar | IrConst | IrTuple | IrParIndex | IrBoolOp | IrCallInet | IrDynamic


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBody(IrNode):
    statements: tuple[IrStmt, ...] = ()


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFor(IrNode):
    target: IrTarget
    iter: IrExpr
    body: IrBody
    orelse: IrBody


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrWhile(IrNode):
    test: IrExpr
    body: IrBody
    orelse: IrBody


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


IrStmt = (
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFunction(IrNode):
    name: str
    params: tuple[tuple[str, Adapter], ...] = ()
    return_adapter: Adapter = VA
    body: IrBody


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

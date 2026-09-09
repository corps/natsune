import ast
import dataclasses
from typing import Any

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
class IrExpr(IrNode):
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrVar(IrExpr):
    name: str
    is_global: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrConst(IrExpr):
    value: Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTuple(IrExpr):
    elements: tuple[IrExpr, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrParIndex(IrExpr):
    base: IrExpr
    index: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBoolOp(IrExpr):
    op: str
    values: tuple[IrExpr, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrCallInet(IrExpr):
    ref: Any
    args: tuple[IrExpr, ...]
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrDynamic(IrExpr):
    source_text: str
    captures: tuple[tuple[str, IrExpr], ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrStmt(IrNode):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBody(IrNode):
    statements: tuple[IrStmt, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAssign(IrStmt):
    targets: tuple[IrTarget, ...]
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAugAssign(IrStmt):
    target: IrTarget
    op: ast.operator
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrIf(IrStmt):
    test: IrExpr
    then_body: IrBody
    else_body: IrBody


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFor(IrStmt):
    target: IrTarget
    iter: IrExpr
    body: IrBody
    orelse: IrBody


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrWhile(IrStmt):
    test: IrExpr
    body: IrBody
    orelse: IrBody


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrReturn(IrStmt):
    value: IrExpr | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBreak(IrStmt):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrContinue(IrStmt):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrExprStmt(IrStmt):
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFunction(IrNode):
    name: str
    params: tuple[tuple[str, Adapter], ...] = ()
    return_adapter: Adapter = VA
    body: IrBody

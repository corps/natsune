"""Phase 6a — the IR node inventory (COMPILER_REFACTOR.md §3.2).

Frozen dataclasses only; every node carries its resolved source position
(phase-0 `Position`, None for synthetic nodes), and every expression node
carries its `Adapter`. All fields are keyword-only: IR nodes are constructed
by name, which keeps the ~20-field surface self-documenting. The IR is the
inspectable layer — rendering, testing, and the paused lowering phases read
these nodes; nothing here creates registers, wires, or flows.

Deliberate departures from the §3.2 sketch (recorded in §5, phase 6):
- `IrTryStar` is deferred with the rest of the try machinery (owner decision,
  §10 row 12): the builder rejects `try`/`except*` statements outright.
- Assignment targets come in three shapes, not two: `IrTargetName`,
  `IrTargetTuple`, and `IrTargetDynamic` — the old lowering routed
  non-Name/Tuple lvalues (`x[0] = v`, `obj.a = v`) through the exec fallback,
  and the IR has to carry what that fallback needs (rewritten source plus
  captured sub-expressions), exactly like `IrDynamic` does for values.
"""

import ast
import dataclasses
from typing import Any

from natsune.adapters import VA, Adapter
from natsune.frontend.diagnostics import Position


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrNode:
    """Common base: resolved source position (None for synthetic nodes).

    Positions are excluded from equality: node identity is structural
    content, so tests and golden renders compare shape wherever the snippet
    sat in its file.
    """

    position: Position | None = dataclasses.field(
        default=None, kw_only=True, compare=False
    )


# --- targets (lvalues; §3.2 "no IR node for lvalues" — patterns only) --------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTarget(IrNode):
    """An assignment/binding target pattern."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetName(IrTarget):
    name: str
    adapter: Adapter
    is_global: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetTuple(IrTarget):
    elements: tuple[IrTarget, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTargetDynamic(IrTarget):
    """An lvalue routed through the exec fallback at lowering.

    `source_text` is the (possibly rewritten) target expression;
    `captures` maps placeholder names to the net-wirable sub-expressions the
    scan found inside it — same shape as `IrDynamic.captures`.
    """

    source_text: str
    captures: tuple[tuple[str, IrExpr], ...] = ()


# --- expressions --------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrExpr(IrNode):
    adapter: Adapter = VA


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrVar(IrExpr):
    """A variable read; `is_global` marks names routed through the dynamic
    path (reads of module globals)."""

    name: str
    is_global: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrConst(IrExpr):
    value: Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrTuple(IrExpr):
    """Par construction."""

    elements: tuple[IrExpr, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrParIndex(IrExpr):
    """A constant-int Par subscript; the builder only produces it for valid
    indices (§3.2)."""

    base: IrExpr
    index: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBoolOp(IrExpr):
    """`and`/`or` with a flattened, n-ary operand list (`op` is "and"/"or")."""

    op: str
    values: tuple[IrExpr, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrCallInet(IrExpr):
    """A call to a linked inet function.

    `ref` is opaque (the old compiler object behind `__inet__` until
    cutover); the metadata is copied at link time. `adapter` mirrors
    `return_adapter` so generic expression handling can read either.
    """

    ref: Any
    args: tuple[IrExpr, ...]
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrDynamic(IrExpr):
    """The eval/exec escape hatch as a first-class node (§3.2).

    `source_text` is the rewritten expression (net-wirable sub-expressions
    replaced by placeholder names — including local reads, which keep their
    own names); `captures` maps each placeholder to the corresponding
    sub-expression as IR, in deterministic scan order. Globals are not
    captured: they stay in the source text and resolve through the exec
    context at lowering.
    """

    source_text: str
    captures: tuple[tuple[str, IrExpr], ...] = ()


# --- statements ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrStmt(IrNode):
    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrBody(IrNode):
    statements: tuple[IrStmt, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAssign(IrStmt):
    """One or more targets (chained assignment modeled explicitly; the
    value→target wiring order of §10 row 6 is a lowering decision)."""

    targets: tuple[IrTarget, ...]
    value: IrExpr


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrAugAssign(IrStmt):
    """Kept as its own node, not desugared to read-modify-write (§10 row 1:
    Ref/InPlace semantics are a lowering decision)."""

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


# --- function level -----------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IrFunction(IrNode):
    name: str
    params: tuple[tuple[str, Adapter], ...] = ()
    return_adapter: Adapter = VA
    body: IrBody

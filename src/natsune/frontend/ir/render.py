import ast
from collections.abc import Callable
from typing import Any

from natsune.adapters import (
    InverseAdapter,
    ParValueAdapter,
    ReferenceAdapter,
    ValueAdapter,
)
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
    IrNode,
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
)

_OPERATOR_TOKENS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "@",
}


def render_adapter(adapter) -> str:
    if isinstance(adapter, ParValueAdapter):
        items = ", ".join(render_adapter(item) for item in adapter.concurrent_items)
        return f"par[{items}]"
    if isinstance(adapter, ReferenceAdapter):
        return f"ref[{render_adapter(adapter.inner)}]"
    if isinstance(adapter, InverseAdapter):
        return f"inv[{render_adapter(adapter.inner)}]"
    if isinstance(adapter, ValueAdapter):
        return "va"
    return type(adapter).__name__


def render_function(function: IrFunction, *, include_positions: bool = False) -> str:
    renderer = IrRenderer(include_positions)
    lines = renderer.function(function)
    return "\n".join(lines) + "\n"


class IrRenderer:
    def __init__(self, include_positions: bool) -> None:
        self.include_positions = include_positions

    def block(
        self,
        node: IrNode | None,
        depth: int,
        header: str,
        children: list[list[str]] | None = None,
    ) -> list[str]:
        children = children or []
        if self.include_positions and node is not None and node.position is not None:
            header = f"{header} @{node.position.lineno}:{node.position.col_offset}"
        if not children:
            return ["  " * depth + header + ")"]
        lines = ["  " * depth + header]
        flat = [line for child in children for line in child]
        last = flat.pop()
        lines.extend(flat)
        lines.append(last + ")")
        return lines

    # -- function -------------------------------------------------------------

    def function(self, node: IrFunction) -> list[str]:
        children: list[list[str]] = []
        for name, adapter in node.params:
            children.append([f"  (param {name} {render_adapter(adapter)})"])
        children.append([f"  (returns {render_adapter(node.return_adapter)})"])
        children.append(self.body(node.body, 1))
        return self.block(node, 0, f"(function {node.name}", children)

    # -- bodies ----------------------------------------------------------------

    def body(self, node: IrBody, depth: int) -> list[str]:
        return self.block(
            node,
            depth,
            "(body",
            [self.stmt(stmt, depth + 1) for stmt in node.statements],
        )

    # -- statements --------------------------------------------------------------

    def stmt(self, node: IrStmt, depth: int) -> list[str]:
        match node:
            case IrAssign():
                targets = self.block(
                    node,
                    depth + 1,
                    "(targets",
                    [self.target(target, depth + 2) for target in node.targets],
                )
                return self.block(
                    node,
                    depth,
                    "(assign",
                    [targets, self.labeled("value", node.value, depth + 1)],
                )
            case IrAugAssign():
                token = _OPERATOR_TOKENS.get(type(node.op), repr(node.op))
                return self.block(
                    node,
                    depth,
                    f"(augassign {token}",
                    [
                        self.labeled(
                            "target", node.target, depth + 1, render=self.target
                        ),
                        self.labeled("value", node.value, depth + 1),
                    ],
                )
            case IrIf():
                return self.block(
                    node,
                    depth,
                    "(if",
                    [
                        self.labeled("test", node.test, depth + 1),
                        self.branch("then", node.then_body, depth + 1),
                        self.branch("else", node.else_body, depth + 1),
                    ],
                )
            case IrFor():
                return self.block(
                    node,
                    depth,
                    "(for",
                    [
                        self.labeled(
                            "target", node.target, depth + 1, render=self.target
                        ),
                        self.labeled("iter", node.iter, depth + 1),
                        self.branch("body", node.body, depth + 1),
                        self.branch("orelse", node.orelse, depth + 1),
                    ],
                )
            case IrWhile():
                return self.block(
                    node,
                    depth,
                    "(while",
                    [
                        self.labeled("test", node.test, depth + 1),
                        self.branch("body", node.body, depth + 1),
                        self.branch("orelse", node.orelse, depth + 1),
                    ],
                )
            case IrReturn():
                if node.value is None:
                    return self.block(node, depth, "(return")
                return self.block(
                    node, depth, "(return", [self.expr(node.value, depth + 1)]
                )
            case IrBreak():
                return self.block(node, depth, "(break")
            case IrContinue():
                return self.block(node, depth, "(continue")
            case IrExprStmt():
                return self.block(
                    node, depth, "(exprstmt", [self.expr(node.value, depth + 1)]
                )
            case _:
                return self.block(node, depth, f"(?{type(node).__name__})")

    def branch(self, label: str, body: IrBody, depth: int) -> list[str]:
        return self.block(
            body,
            depth,
            f"({label}",
            [self.stmt(stmt, depth + 1) for stmt in body.statements],
        )

    def labeled(
        self,
        label: str,
        node: Any,
        depth: int,
        render: Callable | None = None,
    ) -> list[str]:
        render_fn = render or self.expr
        return self.block(None, depth, f"({label}", [render_fn(node, depth + 1)])

    # -- targets -----------------------------------------------------------------

    def target(self, node: IrTarget, depth: int) -> list[str]:
        match node:
            case IrTargetName():
                marker = " :global" if node.is_global else ""
                return self.block(
                    node,
                    depth,
                    f"(name {node.name}{marker} :{render_adapter(node.adapter)}",
                )
            case IrTargetTuple():
                return self.block(
                    node,
                    depth,
                    "(tuple",
                    [self.target(element, depth + 1) for element in node.elements],
                )
            case IrTargetDynamic():
                children: list[list[str]] = [
                    ["  " * (depth + 1) + f'(source "{node.source_text}")']
                ]
                if node.captures:
                    children.append(self.captures(node.captures, depth + 1))
                return self.block(node, depth, "(dynamic", children)
            case _:
                return self.block(node, depth, f"(?{type(node).__name__})")

    # -- expressions ---------------------------------------------------------------

    def expr(self, node: IrExpr, depth: int) -> list[str]:
        adapter = f" :{render_adapter(node.adapter)}"
        match node:
            case IrConst():
                return self.block(node, depth, f"(const {node.value!r}{adapter}")
            case IrVar():
                marker = " :global" if node.is_global else ""
                return self.block(node, depth, f"(var {node.name}{marker}{adapter}")
            case IrTuple():
                return self.block(
                    node,
                    depth,
                    f"(tuple{adapter}",
                    [self.expr(element, depth + 1) for element in node.elements],
                )
            case IrParIndex():
                return self.block(
                    node,
                    depth,
                    f"(par-index {node.index}{adapter}",
                    [self.expr(node.base, depth + 1)],
                )
            case IrBoolOp():
                return self.block(
                    node,
                    depth,
                    f"(boolop {node.op}{adapter}",
                    [self.expr(value, depth + 1) for value in node.values],
                )
            case IrCallInet():
                sigs = " ".join(render_adapter(a) for a in node.arg_adapters)
                return self.block(
                    node,
                    depth,
                    f"(call-inet :arity {node.arity} :sigs ({sigs})"
                    f" :ret {render_adapter(node.return_adapter)}",
                    [self.expr(arg, depth + 1) for arg in node.args],
                )
            case IrDynamic():
                children: list[list[str]] = [
                    ["  " * (depth + 1) + f'(source "{node.source_text}")']
                ]
                if node.captures:
                    children.append(self.captures(node.captures, depth + 1))
                return self.block(node, depth, f"(dynamic{adapter}", children)
            case _:
                return self.block(node, depth, f"(?{type(node).__name__}{adapter}")

    def captures(
        self, captures: tuple[tuple[str, IrExpr], ...], depth: int
    ) -> list[str]:
        entries = [
            self.block(None, depth + 1, f"({name}", [self.expr(captured, depth + 2)])
            for name, captured in captures
        ]
        return self.block(None, depth, "(captures", entries)

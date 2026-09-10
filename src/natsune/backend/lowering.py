"""The new Ir → VariablesFlow lowering — migration step 2.

Scope: straight-line bodies plus inet calls. Statements: IrAssign (chained
targets mirror the old compiler's target-to-target chain), IrAugAssign
(rebind interpretation per §8.6, synthesized as the old compiler's read-
modify-write dynamic), IrReturn (stop at first return, mirroring
parse_statement_body), IrExprStmt, and the default_return_none tail.
Expressions include IrCallInet via runtime.callee_invocation (§7.2a).
Composites (IrIf/IrFor/IrWhile) and multi-target tuple assigns are next.

Deliberate protocol routing:
- Dynamics route through Backend.materialize_dynamic — the Python
  resolution (the old eval fallback, verbatim) lives in PythonBackend;
  lowering stays target-agnostic. Captures are Mapping[str, FromRegister]
  (the eval context needs register identity), and synthesized augassign
  dynamics carry a synthesized BinOp node (the old compiler.py:833
  construction) — node is always a real ast.expr.
- eval_expression/construct_locals are resolved inside PythonBackend from
  the legacy compiler until they become Primitive/ext-fn-table entries
  (§4(1)).

Oracle: serialize_active_pairs equality against the legacy compiler
(tests/backend/test_lowering_oracle.py).
"""

import ast

from natsune.adapters import VA, Adapter, Variables
from natsune.backend.protocol import Backend
from natsune.backend.runtime import callee_invocation
from natsune.backend.types import LoweredUnit
from natsune.control_flow import VariablesFlow
from natsune.frontend.ir import IrFunction
from natsune.frontend.ir.nodes import (
    IrAssign,
    IrAugAssign,
    IrCallInet,
    IrConst,
    IrDynamic,
    IrExpr,
    IrExprStmt,
    IrReturn,
    IrTarget,
    IrTargetDynamic,
    IrTargetName,
    IrVar,
)
from natsune.ports import ConstantValuePort
from natsune.registers import (
    FromRegister,
    as_constant_register,
    as_from_register,
    send_value,
)

_AUG_OPS: dict[type[ast.operator], str] = {
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


def lower_function(ir: IrFunction, backend: Backend) -> LoweredUnit:
    """Lower an IrFunction: body template + declared agents, via backend.
    Names derive from the IR (stable per §6); caller-side renaming, if
    linking ever demands it, belongs in declare_agent — not here."""

    lowering = _FunctionLowering(ir, backend)
    return lowering.run(ir.name)


class _FunctionLowering:
    def __init__(self, ir: IrFunction, backend: Backend) -> None:
        self.ir = ir
        self.backend = backend

    def run(self, name: str) -> LoweredUnit:
        flow = VariablesFlow(
            variables=Variables(self._collect_variables()),
            return_adapter=self.ir.return_adapter,
        )
        self._lower_body(flow, default_return_none=True)
        return self.backend.finish(flow, name=name)

    # -- variables ------------------------------------------------------------

    def _collect_variables(self) -> dict[str, Adapter]:
        """Params first, then locals in statement order — mirroring the old
        compiler's collection pass (interface position is order-bearing)."""

        variables: dict[str, Adapter] = dict(self.ir.params)
        for stmt in self.ir.body.statements:
            if isinstance(stmt, IrAssign):
                for target in stmt.targets:
                    if isinstance(target, IrTargetName) and not target.is_global:
                        variables.setdefault(target.name, target.adapter)
            elif isinstance(stmt, IrAugAssign):
                target = stmt.target
                if isinstance(target, IrTargetName) and not target.is_global:
                    # Old visit_AugAssign marks fresh targets VA unconditionally.
                    variables.setdefault(target.name, VA)
        return variables

    # -- statements -----------------------------------------------------------

    def _lower_body(self, flow: VariablesFlow, *, default_return_none: bool) -> None:
        with flow:
            for stmt in self.ir.body.statements:
                if isinstance(stmt, IrReturn):
                    flow.flow_map.return_output = True
                    if stmt.value is None:
                        # Old bare return: evaluate_from_expression(None)
                        send_value(
                            as_constant_register(None, flow),
                            flow.control_output.return_value.readin(),
                        )
                    else:
                        send_value(
                            self._from_expr(stmt.value, flow),
                            flow.control_output.return_value.readin(),
                        )
                    return  # old parse stops at the first return
                if isinstance(stmt, IrAssign):
                    self._lower_assign(stmt, flow)
                elif isinstance(stmt, IrAugAssign):
                    self._lower_augassign(stmt, flow)
                elif isinstance(stmt, IrExprStmt):
                    self._from_expr(stmt.value, flow).close()
                else:
                    raise NotImplementedError(
                        f"{type(stmt).__name__} lowering lands with the "
                        "composite prototype"
                    )
            if default_return_none:
                flow.flow_map.return_output = True
                send_value(
                    as_from_register(ConstantValuePort(None), VA, flow),
                    flow.control_output.return_value.readin(),
                )

    def _lower_assign(self, stmt: IrAssign, flow: VariablesFlow) -> None:
        # The old chain: value -> target[0] -> target[1] -> ...
        # (target[i] wires from target[i-1]'s register — §8-open aliasing
        # question; the prototype mirrors legacy.)
        if not stmt.targets:
            return
        send_value(
            self._from_expr(stmt.value, flow), self._to_target(stmt.targets[0], flow)
        )
        for prev, target in zip(stmt.targets, stmt.targets[1:]):
            if not isinstance(prev, IrTargetName):
                raise NotImplementedError(
                    f"chaining from {type(prev).__name__} targets"
                )
            send_value(self._var_read(prev.name, flow), self._to_target(target, flow))

    def _lower_augassign(self, stmt: IrAugAssign, flow: VariablesFlow) -> None:
        # Rebind interpretation (§8.6): synthesize the old compiler's
        # read-modify-write dynamic — the same BinOp node (compiler.py:833),
        # text, and capture order included.
        target = stmt.target
        if not isinstance(target, IrTargetName) or target.is_global:
            raise NotImplementedError("augassign targets are plain locals in scope")
        try:
            op = _AUG_OPS[type(stmt.op)]
        except KeyError:
            raise NotImplementedError(f"augassign operator {stmt.op!r}") from None

        used: dict[str, FromRegister] = {target.name: self._var_read(target.name, flow)}
        value_node: ast.expr
        if isinstance(stmt.value, IrVar):
            value_node = ast.Name(id=stmt.value.name, ctx=ast.Load())
            if not stmt.value.is_global:
                used.setdefault(stmt.value.name, self._var_read(stmt.value.name, flow))
        elif isinstance(stmt.value, IrConst):
            value_node = ast.Constant(value=stmt.value.value)
        elif isinstance(stmt.value, IrDynamic):
            # The node carries the value's original expression (the text
            # derives from the node below); captures wire the sub-values.
            # Name-preserving invariant: capture keys are the identifiers
            # appearing in source_text — the rewriter keeps ast.Name ids
            # verbatim and only mints fresh __natsune_N__ names for
            # non-Name capturables (calls/tuples/Par indexes), where the
            # fresh name replaces the sub-expression in the text.
            value_node = stmt.value.ast_node
            for sub_name, sub_expr in stmt.value.captures:
                if sub_name not in used:
                    used[sub_name] = self._from_expr(sub_expr, flow)
        else:
            raise NotImplementedError(f"augassign value {type(stmt.value).__name__}")

        node = ast.BinOp(
            left=ast.Name(id=target.name, ctx=ast.Load()), op=stmt.op, right=value_node
        )
        # Text derives from the node — legacy also ast.unparse'd the BinOp
        # (parenthesized nested operands included), so node and text can
        # never disagree.
        text = ast.unparse(node)
        send_value(
            self.backend.materialize_dynamic(node, text, used, VA, flow),
            self._to_target(target, flow),
        )

    # -- expressions ----------------------------------------------------------

    def _from_expr(self, expr: IrExpr, flow: VariablesFlow) -> FromRegister:
        if isinstance(expr, IrConst):
            return as_constant_register(expr.value, flow)
        if isinstance(expr, IrVar):
            if expr.is_global:
                raise NotImplementedError(
                    "global reads need the exec-context globals hook (§4(2))"
                )
            return self._var_read(expr.name, flow)
        if isinstance(expr, IrDynamic):
            used: dict[str, FromRegister] = {}
            for name, sub in expr.captures:
                if isinstance(sub, IrVar) and sub.is_global:
                    continue  # stays in the source; resolves through globals
                used[name] = self._from_expr(sub, flow)
            return self.backend.materialize_dynamic(
                expr.ast_node, expr.source_text, used, expr.adapter, flow
            )
        if isinstance(expr, IrCallInet):
            # Mirror old compiler.py:405–425: resolve the callee, wire args
            # left-to-right, return the output register. Arity/keywords are
            # frontend-validated (§3.3); the zip is strict anyway.
            ref = self.backend.resolve_call(expr.ref)
            inputs, output = callee_invocation(self.backend.agent_def(ref), flow)
            for input_register, arg in zip(inputs, expr.args, strict=True):
                send_value(self._from_expr(arg, flow), input_register)
            # should_capture_exceptions is False for now: IrTry does not
            # exist yet, so no body can request capture. When it lands, its
            # presence in the body decides the gate (legacy: the identity
            # when False, an ExceptionSink invocation when True).
            return output
        raise NotImplementedError(f"{type(expr).__name__} lowering is not in scope")

    def _var_read(self, name: str, flow: VariablesFlow) -> FromRegister:
        return flow.variable_registers[name].readout()

    def _to_target(self, target: IrTarget, flow: VariablesFlow):
        if isinstance(target, IrTargetName):
            if target.is_global:
                raise NotImplementedError("global assignment is refused (§10.10)")
            return flow.variable_registers[target.name].readin()
        if isinstance(target, IrTargetDynamic):
            raise NotImplementedError("dynamic targets land with composites")
        raise NotImplementedError(f"{type(target).__name__} targets")

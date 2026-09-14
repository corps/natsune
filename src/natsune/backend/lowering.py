import ast
from typing import Any, assert_never

from natsune.adapters import VA, Adapter, Variables
from natsune.backend.agents import (
    AgentImpl,
    callee_invocation,
)
from natsune.backend.control_branch_flow import ControlBranchFlow
from natsune.backend.protocol import Backend
from natsune.connector import ExpansionBuilder
from natsune.control_flow import (
    IfThenElseStatement,
    VariablesFlow,
)
from natsune.frontend.ir import IrBody, IrFunction
from natsune.frontend.ir.nodes import (
    Exits,
    IrAssign,
    IrAugAssign,
    IrCallInet,
    IrConst,
    IrDynamic,
    IrExpr,
    IrExprStmt,
    IrFor,
    IrIf,
    IrReturn,
    IrStmt,
    IrTarget,
    IrTargetDynamic,
    IrTargetName,
    IrTargetTuple,
    IrVar,
    IrWhile,
)
from natsune.invocations import closer
from natsune.ports import Expansion, Graft
from natsune.registers import (
    FromRegister,
    as_constant_register,
    as_from_register,
    send_value,
)


def lower_function(ir: IrFunction, backend: Backend) -> Any:
    lowering = _FunctionLowering(ir, backend)
    return lowering.run()


class _FunctionLowering:
    def __init__(self, ir: IrFunction, backend: Backend) -> None:
        self.ir = ir
        self.backend = backend
        self.variables = self.collect_variables()

    def run(self) -> Any:
        flow = self.branch_flow(self.ir.body, default_return_none=True)
        seen_agents: set[AgentImpl] = set()
        expansions: list[Expansion] = [
            p.execute
            for pair in flow.active_pairs
            for p in pair
            if isinstance(p, Graft)
        ]

        while expansions:
            expansion = expansions.pop()
            if isinstance(expansion, ExpansionBuilder):
                if expansion not in seen_agents:
                    expansions.extend(
                        p.execute
                        for pair in expansion.active_pairs
                        for p in pair
                        if isinstance(p, Graft)
                    )
                    seen_agents.add(expansion)
            elif isinstance(expansion, IfThenElseStatement):
                if expansion not in seen_agents:
                    expansions.extend((expansion.true_case, expansion.false_case))
                    seen_agents.add(expansion)
            else:
                raise NotImplementedError(
                    f"Compiled graft agent {expansion} not implemented in lowering"
                )

        return self.backend.finish(
            self.ir, flow.freeze(), [*seen_agents], name=self.ir.name
        )

    def collect_variables(self) -> dict[str, Adapter]:
        """Params first, then locals in statement order — mirroring the old
        compiler's collection pass (interface position is order-bearing).

        The walk is recursive, like the old InetVariablesEvaluator: branch
        and loop bodies contribute their targets at the enclosing
        statement's position (then before else, a for-loop's target before
        its body), and tuple targets contribute their element names
        left-to-right (the frontend stamped each element's inferred Par
        adapter into the IrTargetName). Dynamic targets declare nothing —
        attribute/subscript lvalues route through the exec fallback, and
        the names inside their captures are reads, not locals.
        """

        variables: dict[str, Adapter] = dict(self.ir.params)
        self.collect_from_statements(self.ir.body.statements, variables)
        return variables

    def collect_from_statements(
        self, statements: tuple[IrStmt, ...], variables: dict[str, Adapter]
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, IrAssign):
                for target in stmt.targets:
                    self.collect_from_target(target, variables)
            elif isinstance(stmt, IrAugAssign):
                # Old visit_AugAssign marks fresh targets VA unconditionally.
                # A fresh augassign target is never in the symbols table, so
                # the frontend already stamped VA into its IrTargetName.
                self.collect_from_target(stmt.target, variables)
            elif isinstance(stmt, IrIf):
                self.collect_from_statements(stmt.then_body.statements, variables)
                self.collect_from_statements(stmt.else_body.statements, variables)
            elif isinstance(stmt, IrFor):
                self.collect_from_target(stmt.target, variables)
                self.collect_from_statements(stmt.body.statements, variables)
                self.collect_from_statements(stmt.orelse.statements, variables)
            elif isinstance(stmt, IrWhile):
                self.collect_from_statements(stmt.body.statements, variables)
                self.collect_from_statements(stmt.orelse.statements, variables)
            # IrReturn/IrBreak/IrContinue/IrExprStmt declare nothing.

    def collect_from_target(
        self, target: IrTarget, variables: dict[str, Adapter]
    ) -> None:
        match target:
            case IrTargetName(is_global=False):
                variables.setdefault(target.name, target.adapter)
            case IrTargetTuple(elements=elements):
                for element in elements:
                    self.collect_from_target(element, variables)
            case IrTargetName() | IrTargetDynamic():
                pass  # globals are refused at lowering; dynamics declare nothing

    def lower_statements(
        self,
        branch_flow: ControlBranchFlow,
        body: IrBody,
        *,
        default_return_none: bool = False,
    ) -> None:
        """
        Lowers the given body from the entry_flow provided, mapping in variables from the entry_flow that match the body's
        actual usage, and returning a flow that represents the last sequenced flow, after variables have been reintegrated
        from the entry_flow.
        """
        flow = self.new_flow(branch_flow)

        for index, stmt in enumerate(body.statements):
            if isinstance(stmt, IrReturn):
                if stmt.value is None:
                    # Old bare return: evaluate_from_expression(None)
                    send_value(
                        as_constant_register(None, flow),
                        flow.control_output.return_value.readin(),
                    )
                else:
                    send_value(
                        self.from_expr(stmt.value, flow),
                        flow.control_output.return_value.readin(),
                    )
            elif isinstance(stmt, IrIf):
                then_exits = stmt.then_body.exits
                else_exits = stmt.else_body.exits
                union = then_exits | else_exits
                if union & (Exits.BREAK | Exits.CONTINUE):
                    raise NotImplementedError(
                        "break/continue land with the loop composite (§7.2c)"
                    )
                self.lower_if(flow, stmt)
                flow = self.new_flow(branch_flow)
            elif isinstance(stmt, IrAssign):
                self.lower_assign(stmt, flow)
            elif isinstance(stmt, IrAugAssign):
                self.lower_augassign(stmt, flow)
            elif isinstance(stmt, IrExprStmt):
                self.from_expr(stmt.value, flow).close()
            else:
                raise NotImplementedError(
                    f"{type(stmt).__name__} lowering lands with the "
                    "composite prototype"
                )

        if body.exits & Exits.FALLTHROUGH:
            if default_return_none:
                send_value(
                    as_constant_register(None, flow),
                    flow.control_output.return_value.readin(),
                )
            else:
                send_value(
                    flow.variables_readout(),
                    flow.control_output.finish_variables.readin(),
                )

    def lower_if(self, flow: VariablesFlow, stmt: IrIf) -> None:
        true_flow = self.branch_flow(stmt.then_body, default_return_none=False)
        false_flow = self.branch_flow(stmt.else_body, default_return_none=False)
        if_agent = IfThenElseStatement(true_case=true_flow, false_case=false_flow)

        with if_agent.invocation(flow) as if_invocation:
            send_value(
                self.from_expr(stmt.test, flow),
                if_invocation.port.readin(),
            )

            cells = [flow.exceptions, *flow.variable_registers.values()]
            gives: list = []
            with closer(if_invocation.wire.context) as context:
                slots = context.variables.readin().split()
                for register, slot in zip(cells, slots, strict=True):
                    taken, give = register.extend()
                    gives.append(give)
                    send_value(as_from_register(taken, register.adapter, flow), slot)

            send_value(
                if_invocation.wire.result.finish_variables.readout(),
                flow.control_output.finish_variables.readin(),
            )

            send_value(
                if_invocation.wire.result.break_variables.readout(),
                flow.control_output.break_variables.readin(),
            )

            send_value(
                if_invocation.wire.result.return_value.readout(),
                flow.control_output.return_value.readin(),
            )

            send_value(
                if_invocation.wire.result.continue_variables.readout(),
                flow.control_output.continue_variables.readin(),
            )

    def new_flow(
        self, branching_flow: ControlBranchFlow | None = None
    ) -> VariablesFlow:
        result = VariablesFlow(
            variables=Variables(self.variables),
            return_adapter=self.ir.return_adapter,
        )

        if branching_flow is not None:
            with result.invocation(
                branching_flow.containing_flow, internal=True
            ) as invocation:
                send_value(
                    branching_flow.cur_control.finish_variables.readout(),
                    invocation.port.variables.readin(),
                )
                branching_flow.apply_continuation(invocation.wire)

        return result

    def branch_flow(self, body: IrBody, *, default_return_none: bool) -> VariablesFlow:
        with ControlBranchFlow(
            self.new_flow(), body.variable_usage, body.exits
        ) as branch_flow:
            self.lower_statements(
                branch_flow, body, default_return_none=default_return_none
            )
        return branch_flow.containing_flow

    def lower_assign(self, stmt: IrAssign, flow: VariablesFlow) -> None:
        # The old chain: value -> target[0] -> target[1] -> ...
        # (target[i] wires from target[i-1]'s register — §8-open aliasing
        # question; the prototype mirrors legacy.)
        if not stmt.targets:
            return
        send_value(
            self.from_expr(stmt.value, flow), self.to_target(stmt.targets[0], flow)
        )
        for prev, target in zip(stmt.targets, stmt.targets[1:]):
            if not isinstance(prev, IrTargetName):
                raise NotImplementedError(
                    f"chaining from {type(prev).__name__} targets"
                )
            send_value(
                flow.variable_registers[prev.name].readout(),
                self.to_target(target, flow),
            )

    def lower_augassign(self, stmt: IrAugAssign, flow: VariablesFlow) -> None:
        # Rebind interpretation (§8.6): synthesize the old compiler's
        # read-modify-write dynamic — the same BinOp node (compiler.py:833),
        # text, and capture order included.
        target = stmt.target
        if not isinstance(target, IrTargetName) or target.is_global:
            raise NotImplementedError("augassign targets are plain locals in scope")

        used: dict[str, FromRegister] = {
            target.name: flow.variable_registers[target.name].readout()
        }
        value_node: ast.expr
        if isinstance(stmt.value, IrVar):
            value_node = ast.Name(id=stmt.value.name, ctx=ast.Load())
            if not stmt.value.is_global:
                used.setdefault(
                    stmt.value.name, flow.variable_registers[stmt.value.name].readout()
                )
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
                    used[sub_name] = self.from_expr(sub_expr, flow)
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
            self.to_target(target, flow),
        )

    # -- expressions ----------------------------------------------------------

    def from_expr(self, expr: IrExpr, flow: VariablesFlow) -> FromRegister:
        if isinstance(expr, IrConst):
            return as_constant_register(expr.value, flow)
        if isinstance(expr, IrVar):
            if expr.is_global:
                raise NotImplementedError(
                    "global reads need the exec-context globals hook (§4(2))"
                )
            return flow.variable_registers[expr.name].readout()
        if isinstance(expr, IrDynamic):
            used: dict[str, FromRegister] = {}
            for name, sub in expr.captures:
                if isinstance(sub, IrVar) and sub.is_global:
                    continue  # stays in the source; resolves through globals
                used[name] = self.from_expr(sub, flow)
            return self.backend.materialize_dynamic(
                expr.ast_node, expr.source_text, used, expr.adapter, flow
            )
        if isinstance(expr, IrCallInet):
            # Mirror old compiler.py:405–425: resolve the callee, wire args
            # left-to-right, return the output register. Arity/keywords are
            # frontend-validated (§3.3); the zip is strict anyway.
            agent = expr.ref
            inputs, output = callee_invocation(
                agent, len(agent.args_adapter.concurrent_items), flow
            )
            for input_register, arg in zip(inputs, expr.args, strict=True):
                send_value(self.from_expr(arg, flow), input_register)
            # should_capture_exceptions is False for now: IrTry does not
            # exist yet, so no body can request capture. When it lands, its
            # presence in the body decides the gate (legacy: the identity
            # when False, an ExceptionSink invocation when True).
            return output
        raise NotImplementedError(f"{type(expr).__name__} lowering is not in scope")

    def to_target(self, target: IrTarget, flow: VariablesFlow):
        if isinstance(target, IrTargetName):
            if target.is_global:
                raise NotImplementedError("global assignment is refused (§10.10)")
            return flow.variable_registers[target.name].readin()
        elif isinstance(target, IrTargetDynamic):
            raise NotImplementedError("dynamic targets land with composites")
        elif isinstance(target, IrTargetTuple):
            raise NotImplementedError("composite targets land with composites")
        else:
            assert_never(target)

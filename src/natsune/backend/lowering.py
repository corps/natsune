"""The new Ir → VariablesFlow lowering — migration step 2.

Scope: straight-line bodies, inet calls, and falling-through ifs.
Statements: IrAssign (chained targets mirror the old compiler's target-to-
target chain), IrAugAssign (rebind interpretation per §8.6, synthesized as
the old compiler's read-modify-write dynamic), IrReturn (stop at first
return, mirroring parse_statement_body), IrExprStmt, IrIf (per-branch
scheme, §7.2b), and the default_return_none tail. Expressions include
IrCallInet via runtime.callee_invocation (§7.2a). Exiting branches,
IrWhile/IrFor, and multi-target tuple assigns are next.

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
from natsune.backend.runtime import agent_invocation, callee_invocation
from natsune.backend.types import AgentDef, InetCallable, LoweredUnit
from natsune.control_flow import (
    IfThenElseInputInto,
    IfThenElseStatement,
    IfThenElseStatementOutputInto,
    VariablesFlow,
)
from natsune.frontend.ir import IrFunction
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
from natsune.ports import ConstantValuePort
from natsune.registers import (
    FromRegister,
    as_constant_register,
    as_from_register,
    as_to_register,
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
        self._if_count = 0
        self._variables: dict[str, Adapter] = {}

    def run(self, name: str) -> LoweredUnit:
        self._variables = self._collect_variables()
        self.flow = flow = VariablesFlow(
            variables=Variables(self._variables),
            return_adapter=self.ir.return_adapter,
        )
        self._lower_body(flow, default_return_none=True)
        return self.backend.finish(flow, name=name)

    # -- variables ------------------------------------------------------------

    def _collect_variables(self) -> dict[str, Adapter]:
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
        self._collect_from_statements(self.ir.body.statements, variables)
        return variables

    def _collect_from_statements(
        self, statements: tuple[IrStmt, ...], variables: dict[str, Adapter]
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, IrAssign):
                for target in stmt.targets:
                    self._collect_from_target(target, variables)
            elif isinstance(stmt, IrAugAssign):
                # Old visit_AugAssign marks fresh targets VA unconditionally.
                # A fresh augassign target is never in the symbols table, so
                # the frontend already stamped VA into its IrTargetName.
                self._collect_from_target(stmt.target, variables)
            elif isinstance(stmt, IrIf):
                self._collect_from_statements(stmt.then_body.statements, variables)
                self._collect_from_statements(stmt.else_body.statements, variables)
            elif isinstance(stmt, IrFor):
                self._collect_from_target(stmt.target, variables)
                self._collect_from_statements(stmt.body.statements, variables)
                self._collect_from_statements(stmt.orelse.statements, variables)
            elif isinstance(stmt, IrWhile):
                self._collect_from_statements(stmt.body.statements, variables)
                self._collect_from_statements(stmt.orelse.statements, variables)
            # IrReturn/IrBreak/IrContinue/IrExprStmt declare nothing.

    def _collect_from_target(
        self, target: IrTarget, variables: dict[str, Adapter]
    ) -> None:
        match target:
            case IrTargetName(is_global=False):
                variables.setdefault(target.name, target.adapter)
            case IrTargetTuple(elements=elements):
                for element in elements:
                    self._collect_from_target(element, variables)
            case IrTargetName() | IrTargetDynamic():
                pass  # globals are refused at lowering; dynamics declare nothing

    # -- statements -----------------------------------------------------------

    def _lower_body(self, flow: VariablesFlow, *, default_return_none: bool) -> None:
        with flow:
            closed = self._lower_statements(flow, self.ir.body.statements)
            if default_return_none and not closed:
                # Same derivable bookkeeping as the IrReturn case (§6
                # flow_map bullet).
                flow.flow_map.return_output = True
                send_value(
                    as_from_register(ConstantValuePort(None), VA, flow),
                    flow.control_output.return_value.readin(),
                )

    def _lower_statements(
        self, flow: VariablesFlow, statements: tuple[IrStmt, ...]
    ) -> bool:
        """Lower a statement list; returns True when a closer (IrReturn)
        terminated it — the caller must not add a fall-through tail."""
        for stmt in statements:
            if isinstance(stmt, IrReturn):
                # MARKED post-cutover removal (§6 flow_map bullet):
                # bookkeeping, not wiring — the send below is what the net
                # records (the oracle never serializes flow_map). Derivable
                # from IrBody.exits; unconsumed until slice 2 moves shortcut
                # decisions into lowering.
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
                return True  # old parse stops at the first return
            if isinstance(stmt, IrAssign):
                self._lower_assign(stmt, flow)
            elif isinstance(stmt, IrAugAssign):
                self._lower_augassign(stmt, flow)
            elif isinstance(stmt, IrIf):
                self._lower_if(stmt, flow)
            elif isinstance(stmt, IrExprStmt):
                self._from_expr(stmt.value, flow).close()
            else:
                raise NotImplementedError(
                    f"{type(stmt).__name__} lowering lands with the "
                    "composite prototype"
                )
        return False

    def _lower_if(self, stmt: IrIf, flow: VariablesFlow) -> None:
        """Per-branch scheme (§7.2b): the union is never formed. The
        parent wires the full variables bundle into the composite context
        (every cell extended — current value out, fresh state continues);
        each branch runs over its own registers and emits every variable's
        final state on fall-through (untouched cells pass through); the
        taken branch's bundle is the only live one (the composite
        dispatches). The continuation is the parent flow itself: branch
        outputs feed the extended cells, and later reads see them
        normally."""
        for body in (stmt.then_body, stmt.else_body):
            if body.exits != Exits.FALLTHROUGH:
                raise NotImplementedError(
                    "branches that exit (return/break/continue) land with "
                    "the closer machinery (§7.2b slice 2)"
                )

        composite = IfThenElseStatement(
            true_case=self._lower_branch(stmt.then_body),
            false_case=self._lower_branch(stmt.else_body),
        )
        name = f"if_{self._if_count}"
        self._if_count += 1
        ref = self.backend.declare_agent(
            name,
            AgentDef(
                name,
                composite.input_adapter,
                composite.output_adapter,
                InetCallable(composite),
            ),
        )

        with agent_invocation(
            ref,
            self.backend.agent_def(ref),
            flow,
            IfThenElseInputInto,
            IfThenElseStatementOutputInto,
        ) as if_invocation:
            send_value(self._from_expr(stmt.test, flow), if_invocation.port.readin())

            # Context: the full bundle, unclassified — the either-or split
            # is the composite's job, not the parent's.
            cells = [flow.exceptions, *flow.variable_registers.values()]
            gives: list = []
            with closer(if_invocation.wire.context) as context:
                slots = context.variables.readin().split()
                for register, slot in zip(cells, slots, strict=True):
                    taken, give = register.extend()
                    gives.append(give)
                    send_value(as_from_register(taken, register.adapter, flow), slot)

            # Continuation: the taken branch's final states feed the
            # extended cells; later parent reads see them normally. Slots
            # for control outputs the branches cannot emit (slice 1
            # branches always fall through) are shortcut, mirroring the
            # legacy wire_continuation.
            result = if_invocation.wire.result
            result.return_value.shortcut()
            result.continue_variables.shortcut()
            result.break_variables.shortcut()
            finish = result.finish_variables.readout().split()
            for register, give, out in zip(cells, gives, finish, strict=True):
                send_value(out, as_to_register(give, register.adapter, flow))

    def _lower_branch(self, body) -> VariablesFlow:
        """Lower one branch body as a self-contained flow over the full
        function variables (the legacy new_branch().parse_statement_body
        shape). Fall-through emits every variable's final state; untouched
        cells pass through by construction of the register plumbing."""
        branch = VariablesFlow(
            variables=Variables(self._variables),
            return_adapter=self.ir.return_adapter,
        )
        with branch:
            self._lower_statements(branch, body.statements)
            # Same derivable bookkeeping as the IrReturn case (§6 flow_map
            # bullet): load-bearing only once slice 2 lets branches exit.
            branch.flow_map.finish_output = True
            send_value(
                branch.variables_readout(),
                branch.control_output.finish_variables.readin(),
            )
        return branch

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

import ast
import operator
from typing import Any, assert_never

from natsune.backend.agents import (
    AgentImpl,
    PromiseExpansion,
    callee_invocation,
    try_iter,
)
from natsune.backend.connector import ExpansionBuilder, FrozenExpansion
from natsune.backend.control_branch_flow import ControlBranchFlow
from natsune.backend.control_flow import (
    CloseAfterContingent,
    ConcurrentValueMerge,
    FlowControlInto,
    FlowInputInto,
    IfThenElse,
    IfThenElseStatement,
    Loop,
    SerialAnd,
    SerialOr,
    VariablesFlow,
)
from natsune.backend.invocations import (
    closer,
    filter_invocation,
    pack_from,
    pack_into,
    send_parameter,
    split_invocation,
)
from natsune.backend.protocol import Backend, LoweredUnit
from natsune.backend.registers import (
    FromRegister,
    ToRegister,
    as_constant_register,
    as_from_register,
    join_from_registers,
    join_to_registers,
    send_value,
)
from natsune.first_order.adapters import VA, Adapter, ParValueAdapter, Variables
from natsune.first_order.ports import Expansion, Graft, Port, Target, Wire
from natsune.frontend.ir import IrBody, IrBoolOp, IrFunction, IrParIndex
from natsune.frontend.ir.nodes import (
    Exits,
    IrAssign,
    IrAugAssign,
    IrBreak,
    IrCallInet,
    IrConst,
    IrContinue,
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
    IrTuple,
    IrVar,
    IrWhile,
)


def true_and(a: Any, b: Any) -> Any:
    return a and b


def true_or(a: Any, b: Any) -> Any:
    return a or b


def lower_function(ir: IrFunction, backend: Backend) -> Any:
    lowering = _FunctionLowering(ir, backend)
    return lowering.run()


class _FunctionLowering:
    def __init__(self, ir: IrFunction, backend: Backend) -> None:
        self.ir = ir
        self.backend = backend
        self.variables = self.collect_variables()

    def run(self) -> Any:
        flow = self.branch_flow(self.ir.body, exits=self.ir.exits)

        seen_agents = self.walk_agents(flow)

        return self.backend.finish(
            LoweredUnit(self.ir, flow.freeze(), tuple(seen_agents), self.ir.name)
        )

    def walk_agents(self, flow: VariablesFlow) -> set[AgentImpl]:
        targets: list[Target] = []
        seen_agents: set[AgentImpl] = set()
        expansions: list[Expansion] = [flow]

        while expansions:
            expansion = expansions.pop()
            if isinstance(expansion, ExpansionBuilder):
                if expansion not in seen_agents:
                    targets.extend(p for pair in expansion.active_pairs for p in pair)
                    targets.append(expansion.input_interface.interface)
                    targets.append(expansion.output_interface.interface)
                    seen_agents.add(expansion)
            elif isinstance(expansion, (FrozenExpansion, PromiseExpansion)):
                # Compiled callees, grafted whole. Leaves in the catalog:
                # their internal agents belong to the callee's own unit,
                # and a promise has nothing behind it yet.
                seen_agents.add(expansion)
            elif isinstance(expansion, IfThenElseStatement):
                if expansion not in seen_agents:
                    expansions.extend((expansion.true_case, expansion.false_case))
                    seen_agents.add(expansion)
            elif isinstance(expansion, (IfThenElse, Loop)):
                if expansion not in seen_agents:
                    seen_agents.add(expansion)
                    if isinstance(expansion, Loop):
                        expansions.extend(
                            (expansion.iteration, expansion.body, expansion.orelse)
                        )
                    else:
                        expansions.extend((expansion.true_case, expansion.false_case))
            elif isinstance(
                expansion,
                (SerialOr, SerialAnd, CloseAfterContingent, ConcurrentValueMerge),
            ):
                if expansion not in seen_agents:
                    seen_agents.add(expansion)
            else:
                raise NotImplementedError(
                    f"Compiled graft agent {expansion} not implemented in lowering"
                )

            while targets:
                target = targets.pop()

                if isinstance(target, Graft):
                    expansions.append(target.execute)

                if isinstance(target, Port):
                    targets.extend(target.wires)
                elif isinstance(target, Wire):
                    if target.target:
                        targets.append(target.target)

        return seen_agents

    def collect_variables(self) -> dict[str, Adapter]:
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
            elif isinstance(stmt, (IrReturn, IrBreak, IrContinue, IrExprStmt)):
                continue
            else:
                assert_never(stmt)

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
        exits: Exits,
    ) -> None:
        flow = self.new_flow(branch_flow)

        for index, stmt in enumerate(body.statements):
            if isinstance(stmt, IrReturn):
                assert body.closer is stmt
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
                self.lower_if(flow, stmt)
                flow = self.new_flow(branch_flow)
            elif isinstance(stmt, (IrFor, IrWhile)):
                self.lower_loop(flow, stmt)
                flow = self.new_flow(branch_flow)
            elif isinstance(stmt, IrBreak):
                assert body.closer is stmt
                send_value(
                    flow.variables_readout(),
                    flow.control_output.break_variables.readin(),
                )
            elif isinstance(stmt, IrContinue):
                assert body.closer is stmt
                send_value(
                    flow.variables_readout(),
                    flow.control_output.continue_variables.readin(),
                )
            elif isinstance(stmt, IrAssign):
                self.lower_assign(stmt, flow)
            elif isinstance(stmt, IrAugAssign):
                self.lower_augassign(stmt, flow)
            elif isinstance(stmt, IrExprStmt):
                self.from_expr(stmt.value, flow).close()
            else:
                assert_never(stmt)

        if body.closer is None:
            if exits & Exits.FALLTHROUGH:
                send_value(
                    flow.variables_readout(),
                    flow.control_output.finish_variables.readin(),
                )
            else:
                assert exits & Exits.RETURN
                send_value(
                    as_constant_register(None, flow),
                    flow.control_output.return_value.readin(),
                )

        flow.close()

    def lower_if(self, flow: VariablesFlow, stmt: IrIf) -> None:
        true_flow = self.branch_flow(stmt.then_body, exits=stmt.then_body.exits)
        false_flow = self.branch_flow(stmt.else_body, exits=stmt.else_body.exits)
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

        flow.close()

    def lower_loop(self, flow: VariablesFlow, stmt: IrFor | IrWhile) -> None:
        """Lower IrWhile/IrFor through the Loop composite (§6.1).

        The iteration flow is the per-iteration re-test: for a For it
        pulls the next element off the iterator and deconstructs it into
        the target; for a While it evaluates the test. The composite's
        finish slot (= exhaustion, including body break) is what sequences
        the trailing region — the next layer starts off cur_control.finish,
        so "loop exhausted" is exactly what feeds it. All four result
        slots forward unconditionally (the §3.2 rule, loops included);
        consumption/shortcut stays exits-driven in ControlBranchFlow.
        """
        body_flow = self.branch_flow(stmt.body, exits=stmt.body.exits)
        orelse_flow = self.branch_flow(stmt.orelse, exits=stmt.orelse.exits)

        loop = Loop(
            (
                self.deconstruct_iteration_flow(stmt.target)
                if isinstance(stmt, IrFor)
                else self.test_iteration_flow(stmt.test)
            ),
            body_flow,
            orelse_flow,
        )

        with loop.invocation(flow) as loop_invocation:
            if isinstance(stmt, IrFor):
                # iter(<iterable>) — the builtin, through the eval-filter
                # gadget (mirrors legacy compiler.py:857–860).
                send_value(
                    send_parameter(
                        filter_invocation(iter, flow),
                        self.from_expr(stmt.iter, flow),
                    ),
                    loop_invocation.port.value.readin(),
                )
            send_value(
                flow.variables_readout(),
                loop_invocation.port.variables.readin(),
            )

            send_value(
                loop_invocation.wire.finish_variables.readout(),
                flow.control_output.finish_variables.readin(),
            )
            send_value(
                loop_invocation.wire.break_variables.readout(),
                flow.control_output.break_variables.readin(),
            )
            send_value(
                loop_invocation.wire.return_value.readout(),
                flow.control_output.return_value.readin(),
            )
            send_value(
                loop_invocation.wire.continue_variables.readout(),
                flow.control_output.continue_variables.readin(),
            )

        flow.close()

    def deconstruct_iteration_flow(self, target: IrTarget) -> VariablesFlow:
        """The For loop's iteration flow: pull the next element and
        deconstruct it into the target, reporting (element-matched) as the
        re-test. Mirrors legacy parse_deconstruct_iter (compiler.py:534).
        """
        with VariablesFlow(
            variables=Variables(self.variables), return_adapter=VA
        ) as true_case:
            send_value(
                true_case.flow_input.value.readout(),
                self.to_target(target, true_case),
            )
            send_value(
                as_constant_register(True, true_case),
                true_case.control_output.return_value.readin(),
            )
            send_value(
                true_case.variables_readout(),
                true_case.control_output.finish_variables.readin(),
            )

        with VariablesFlow(
            variables=Variables(self.variables), return_adapter=VA
        ) as false_case:
            send_value(
                as_constant_register(False, false_case),
                false_case.control_output.return_value.readin(),
            )
            send_value(
                false_case.variables_readout(),
                false_case.control_output.finish_variables.readin(),
            )

        with VariablesFlow(
            variables=Variables(self.variables), return_adapter=VA
        ) as flow:
            input_variables = flow.variables_readout()
            input_iter = flow.flow_input.value.readout()

            try_iter_in, (next_value, should_continue) = split_invocation(
                try_iter, flow
            )
            send_value(input_iter, try_iter_in)

            with IfThenElse(true_case, false_case).invocation(flow) as conditional:
                send_value(should_continue, conditional.port.readin())

                with closer(
                    pack_into(conditional.wire.context, FlowInputInto)
                ) as context:
                    send_value(next_value, context.value.readin())
                    send_value(input_variables, context.variables.readin())

                with closer(
                    pack_from(conditional.wire.result, FlowControlInto)
                ) as result:
                    send_value(
                        result.return_value.readout(),
                        flow.control_output.return_value.readin(),
                    )
                    send_value(
                        result.finish_variables.readout(),
                        flow.control_output.finish_variables.readin(),
                    )

        return flow

    def test_iteration_flow(self, test: IrExpr) -> VariablesFlow:
        """The While loop's iteration flow: evaluate the test per
        iteration; its value is the re-test. Mirrors legacy parse_test
        (compiler.py:730)."""
        with VariablesFlow(
            variables=Variables(self.variables), return_adapter=VA
        ) as flow:
            send_value(
                self.from_expr(test, flow),
                flow.control_output.return_value.readin(),
            )
            send_value(
                flow.variables_readout(),
                flow.control_output.finish_variables.readin(),
            )
        return flow

    def new_flow(
        self, branching_flow: ControlBranchFlow | None = None
    ) -> VariablesFlow:
        if branching_flow is None:
            return VariablesFlow(
                variables=Variables(self.variables),
                return_adapter=self.ir.return_adapter,
            )
        else:
            return branching_flow.apply_new_layer()

    def branch_flow(self, body: IrBody, *, exits: Exits) -> VariablesFlow:
        with ControlBranchFlow(
            self.new_flow(), body.variable_usage, exits
        ) as branch_flow:
            self.lower_statements(branch_flow, body, exits=exits)
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
            self.backend.materialize_dynamic(
                node, text, used, self.ir.globals, VA, flow
            ),
            self.to_target(target, flow),
        )

    # -- expressions ----------------------------------------------------------

    def from_expr(self, expr: IrExpr, flow: VariablesFlow) -> FromRegister:
        if isinstance(expr, IrConst):
            return as_constant_register(expr.value, flow)
        elif isinstance(expr, IrVar):
            if expr.is_global:
                raise NotImplementedError(
                    "global reads need the exec-context globals hook (§4(2))"
                )
            return flow.variable_registers[expr.name].readout()
        elif isinstance(expr, IrDynamic):
            used: dict[str, FromRegister] = {}
            for name, sub in expr.captures:
                if isinstance(sub, IrVar) and sub.is_global:
                    continue  # stays in the source; resolves through globals
                used[name] = self.from_expr(sub, flow)
            return self.backend.materialize_dynamic(
                expr.ast_node,
                expr.source_text,
                used,
                self.ir.globals,
                expr.adapter,
                flow,
            )
        elif isinstance(expr, IrCallInet):
            # Mirror old compiler.py:405–425: wire args left-to-right,
            # return the output register. Arity/keywords are
            # frontend-validated (§3.3); the zip is strict anyway. The
            # ref's expansion is promise-or-frozen (CUTOVER.md §2) — the
            # driver compiled every non-cyclic callee before lowering,
            # and cyclic ones graft the promise the same way.
            expansion = expr.ref.expansion
            assert isinstance(
                expansion, (FrozenExpansion, ExpansionBuilder, PromiseExpansion)
            ), "callee net missing: the driver compiles (or promises) every callee before lowering"
            inputs, output = callee_invocation(expansion, expr.arity, flow)
            for input_register, arg in zip(inputs, expr.args, strict=True):
                send_value(self.from_expr(arg, flow), input_register)
            # should_capture_exceptions is False for now: IrTry does not
            # exist yet, so no body can request capture. When it lands, its
            # presence in the body decides the gate (legacy: the identity
            # when False, an ExceptionSink invocation when True).
            return output
        elif isinstance(expr, IrTuple):
            # Par construction, mirroring old compiler.py's
            # evaluate_from_expression ast.Tuple branch: the tuple's
            # adapter derives from the ELEMENT registers (not the
            # frontend-inferred annotation), the elements fan out into
            # the Par's split, and the interface's far side is the value.
            return join_from_registers(
                [self.from_expr(element, flow) for element in expr.elements],
                flow,
            )
        elif isinstance(expr, IrBoolOp):
            merger = true_and if expr.op == "and" else true_or
            should_shortcircuit = operator.not_ if expr.op == "and" else operator.truth
            acc = self.from_expr(expr.values[0], flow)
            for n in expr.values[1:]:
                with ConcurrentValueMerge(should_shortcircuit, merger).invocation(
                    flow
                ) as invocation:
                    a, b = invocation.port.readin().split()
                    send_value(acc, a)
                    send_value(self.from_expr(n, flow), b)
                    acc = invocation.wire.readout()

            return acc
        elif isinstance(expr, IrParIndex):
            # Par element read, mirroring old compiler.py's
            # evaluate_from_expression ast.Subscript branch: split the
            # base's Par and keep the indexed element, closing the
            # neighbors — a linearizing read of the whole cell, exactly
            # what the usage cross-check's "Par reads linearize" pin
            # documents (element-pass-through reads are the marked
            # post-cutover relaxation, not this). The frontend only
            # builds IrParIndex for constant in-range integer subscripts
            # of Par-typed bases, so the guards below fire on net
            # corruption, not on user input.
            inner = self.from_expr(expr.base, flow)
            assert isinstance(
                inner.adapter, ParValueAdapter
            ), f"Par index into non-Par register: {inner.adapter}"
            for index, element in enumerate(inner.split()):
                if index == expr.index:
                    return element
                element.close()
            raise AssertionError(
                f"Par index {expr.index} out of range for {inner.adapter}"
            )
        else:
            assert_never(expr)

    def to_target(self, target: IrTarget, flow: VariablesFlow) -> ToRegister:
        if isinstance(target, IrTargetName):
            if target.is_global:
                raise NotImplementedError("global assignment is refused (§10.10)")
            return flow.variable_registers[target.name].readin()
        elif isinstance(target, IrTargetTuple):
            # Par deconstruction, mirroring old compiler.py's
            # evaluate_to_expression ast.Tuple branch: the element
            # targets' readins join into a Par-typed ToRegister, and
            # send_value fans the incoming Par out through its split.
            # Element adapters come from the bundle (the frontend marks
            # unpack targets with the value Par's concurrent items), so
            # a matching-shape Par wires port-for-port.
            return join_to_registers(
                [self.to_target(element, flow) for element in target.elements],
                flow,
            )
        elif isinstance(target, IrTargetDynamic):
            used: dict[str, FromRegister] = {}
            for name, sub in target.captures:
                if isinstance(sub, IrVar) and sub.is_global:
                    continue  # stays in the source; resolves through globals
                used[name] = self.from_expr(sub, flow)
            return self.backend.materialize_dynamic_to(
                target.ast_node,
                target.source_text,
                used,
                self.ir.globals,
                VA,
                flow,
            )
        else:
            assert_never(target)

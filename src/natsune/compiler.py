import ast
import dataclasses
import functools
import inspect
import random
import string
import sys
import textwrap
from functools import cached_property
from typing import cast, Any, get_type_hints, Iterable, Sequence, Callable, Iterator

from natsune.adapters import (
    Adapter,
    ValueAdapter,
    adapter_from_type,
    TypeExpression,
    ParValueAdapter,
    Variables, ReferenceAdapter,
)
from natsune.connector import Connector, serialize_active_pairs
from natsune.control_flow import (
    VariablesFlow,
    Loop,
    IfThenElse,
    ConcurrentMerge,
    WeakeningSelection, CloseAfterContingent, MergeInputTo,
)
from natsune.control_flow_generated import (
    FlowInputInto,
    FlowControlInto, MergeOutputInto,
)
from natsune.executor import SynchronizedExecutor
from natsune.invocations import (
    merge_invocation,
    send_parameters,
    filter_invocation,
    send_parameter,
    closer,
    pack_into,
    pack_from,
    split_invocation, expansion_invocation,
)
from natsune.ports import Wire, ConstantValuePort, Erasure
from natsune.registers import (
    FromRegister,
    ToRegister,
    as_to_register,
    as_from_register,
    send_value,
    send_values,
    serialize_values,
    as_constant_register,
    join_to_registers,
    InterfaceRegister,
    ToInterfaceRegister, FlowRegister, parallelize_value, join_from_registers,
)

unsupported_expr: tuple[type[ast.expr], ...] = (
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Starred,
    ast.Lambda,
    ast.NamedExpr,
)

unsupported_stmt: tuple[type[ast.stmt], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Match,
    ast.Try,
    ast.Assert,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.TryStar,
    ast.TypeAlias,
    ast.Delete,
)


@dataclasses.dataclass(frozen=True, slots=True)
class ReplaceWithSerializedVariables(ast.NodeTransformer):
    branch_compiler: InetBranchCompiler
    used_names: dict[str, FromRegister] = dataclasses.field(default_factory=dict)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr):
            special_form = self.branch_compiler.evaluate_special_form_from_expression(
                node
            )
            if special_form:
                if isinstance(node, ast.Name):
                    name = node.id
                else:
                    name = self.random_identifier()
                    node = ast.Name(id=name, ctx=ast.Load())
                if (
                    name not in self.used_names
                    and name
                    not in self.branch_compiler.function_compiler.used_as_globals
                ):
                    self.used_names[name] = special_form
            else:
                return super().generic_visit(node)
        return node

    def random_identifier(self, length=8) -> str:
        while True:
            first_char = random.choice(string.ascii_letters + "_")
            rest_chars = "".join(
                random.choices(string.ascii_letters + string.digits + "_", k=length - 1)
            )
            candidate = "__" + first_char + rest_chars + "__"
            if candidate not in self.used_names:
                return candidate

    def construct_context(self, finished_from: FromRegister) -> FromRegister:
        held_context = [
            FlowRegister(ReferenceAdapter(ValueAdapter()), self.branch_compiler.flow) for _ in self.used_names.values()
        ]

        # "cast" values into a reference
        send_values(
            list(self.used_names.values()),
            [r.readin() for r in held_context]
        )

        value_readout = [r.readout() for r in held_context]

        # Implicit close here as we transfer the state directly
        joined_content = join_from_registers([
            as_from_register(r.state, r.adapter, r.connector) for r in held_context
        ], self.branch_compiler.flow)

        with expansion_invocation(CloseAfterContingent(finished_from.adapter, joined_content.adapter), self.branch_compiler.flow, MergeInputTo, MergeOutputInto) as invocation:
            send_value(finished_from, invocation.port.readin())
            send_value(joined_content, invocation.wire.second_value.readin())
            invocation.wire.result.readout().close()


        return send_parameters(
            serialize_values(self.branch_compiler.flow, 2),
            (
                as_constant_register(
                    self.branch_compiler.function_compiler.globals,
                    self.branch_compiler.flow,
                ),
                (
                    send_parameters(
                        merge_invocation(construct_locals, self.branch_compiler.flow),
                        (
                            (
                                send_parameters(
                                    serialize_values(
                                        self.branch_compiler.flow,
                                        len(self.used_names),
                                    ),
                                    value_readout,
                                )
                            ),
                            as_constant_register(
                                tuple(self.used_names.keys()),
                                self.branch_compiler.flow,
                            ),
                        ),
                    )
                ),
            ),
        )


@dataclasses.dataclass
class InetFunctionCompiler:
    func: Any
    globals: dict[str, Any]
    filename: str

    variables: dict[str, Adapter] = dataclasses.field(default_factory=dict)
    used_as_globals: set[str] = dataclasses.field(default_factory=set)
    lineno: int = dataclasses.field(init=False)

    def __post_init__(self):
        for k, te in self.args:
            self.variables[k] = adapter_from_type(te)
        for stmt in self.func_def.body:
            InetVariablesEvaluator(self).visit(stmt)

    def new_branch(self) -> InetBranchCompiler:
        return InetBranchCompiler(
            self,
            VariablesFlow(
                variables=Variables(self.variables),
                return_adapter=adapter_from_type(self.return_annot),
            ),
        )

    def new_test(self) -> InetBranchCompiler:
        return InetBranchCompiler(
            self,
            VariablesFlow(
                variables=Variables(self.variables),
                return_adapter=ValueAdapter(),
            ),
        )

    def syntax_error(self, node: ast.AST, message: str) -> SyntaxError:
        lineno = self.func_def.lineno
        offset = self.func_def.col_offset
        if hasattr(node, "lineno"):
            lineno = cast(int, node.lineno)
        if hasattr(node, "col_offset"):
            offset = cast(int, node.col_offset)
        return SyntaxError(
            message, (self.filename, lineno + self.lineno - 1, offset, None)
        )

    def evaluate_call_adapter(self, node: ast.Call) -> Adapter:
        inet = self.lookup_inet(node)
        if inet is not None:
            return adapter_from_type(inet.return_annot)
        return ValueAdapter()

    def evaluate_subscript(self, base: ParValueAdapter, slice: ast.expr) -> int:
        if not isinstance(slice, ast.Constant) or not isinstance(slice.value, int):
            raise self.syntax_error(
                slice, "Subscript index of Par must be a constant integer"
            )
        if slice.value < 0 or slice.value >= len(base.concurrent_items):
            raise self.syntax_error(
                slice, "Subscript index of Par must be in range [0, par_size)"
            )
        return slice.value

    def lookup_inet(self, node: ast.Call) -> InetFunctionCompiler | None:
        is_decidable = False
        if isinstance(node.func, ast.Name):
            is_decidable = node.func.id in self.used_as_globals

        if is_decidable:
            try:
                global_f = eval(ast.unparse(node.func), self.globals)
            except Exception as e:
                raise self.syntax_error(
                    node, "Could not evaluate global function invocation"
                ) from e
            if hasattr(global_f, "__inet__"):
                inet: InetFunctionCompiler = getattr(global_f, "__inet__")
                return inet
        return None

    def infer_expression_adapter(self, node: ast.expr) -> Adapter:
        if isinstance(node, ast.Call):
            return self.evaluate_call_adapter(node)
        elif isinstance(node, ast.Name):
            return self.variables[node.id]
        elif isinstance(node, ast.Tuple):
            return ParValueAdapter(
                [self.infer_expression_adapter(n) for n in node.elts]
            )
        elif isinstance(node, ast.Subscript):
            base = self.infer_expression_adapter(node.value)
            if isinstance(base, ParValueAdapter):
                slice_solution = self.evaluate_subscript(base, node.slice)
                return base.concurrent_items[slice_solution]

        return ValueAdapter()

    @cached_property
    def return_annot(self) -> TypeExpression | None:
        return get_type_hints(self.func).get("return", None)

    @cached_property
    def func_def(self) -> ast.FunctionDef:
        lines, self.lineno = inspect.getsourcelines(self.func)
        module = ast.parse("".join(lines))
        func = module.body[0]
        assert isinstance(func, ast.FunctionDef)
        return func

    @cached_property
    def args(self) -> list[tuple[str, TypeExpression | None]]:
        th = get_type_hints(self.func)
        if (
            self.func_def.args.kw_defaults
            or self.func_def.args.kwonlyargs
            or self.func_def.args.kwarg
            or self.func_def.args.vararg
        ):
            raise SyntaxError(
                "Keyword, var, and defaulte args are not currently supported"
            )

        result = []
        for arg in self.func_def.args.args:
            result.append((arg.arg, th.get(arg.arg, None)))
        return result

    @cached_property
    def args_adapter(self) -> ParValueAdapter:
        return ParValueAdapter([adapter_from_type(te) for te in self.args])

    @cached_property
    def compiled(self) -> VariablesFlow:
        return self.new_branch().parse_statement_body(
            self.func_def.body, default_return_none=True
        )

    def compile(self) -> None:
        getattr(self, "compiled")

    def invocation(
        self, connector: Connector
    ) -> tuple[Sequence[ToRegister], FromRegister]:
        with self.compiled.invocation(connector) as invocation:
            variable_inputs = invocation.port.variables.readin().split()
            for input in variable_inputs[len(self.args) :]:
                input.close()

            return (
                variable_inputs[: len(self.args)],
                invocation.wire.return_value.readout(),
            )


def _try_iter(i: Iterator) -> Any:
    try:
        return next(i), True
    except StopIteration:
        return None, False


@dataclasses.dataclass(frozen=True, slots=True)
class InetBranchCompiler:
    function_compiler: InetFunctionCompiler
    flow: VariablesFlow

    def evaluate_to_expression(self, expr: ast.expr) -> ToRegister:
        if isinstance(expr, ast.Name):
            if expr.id not in self.function_compiler.used_as_globals:
                return self.flow.variable_registers[expr.id].readin()

        if isinstance(expr, ast.Tuple):
            inner_registers = [
                self.evaluate_to_expression(element) for element in expr.elts
            ]
            return join_to_registers(inner_registers, self.flow)

        if isinstance(expr, ast.List):
            raise self.function_compiler.syntax_error(
                expr, "List deconstructors in assignment not supported"
            )

        rewriter = ReplaceWithSerializedVariables(self)
        inner = textwrap.dedent(ast.unparse(rewriter.visit(expr)))
        assigned = rewriter.random_identifier()
        inner += " = " + assigned

        x1, x2 = Wire.as_tautology()
        rewriter.used_names[assigned] = as_from_register(x2, ValueAdapter(), self.flow)
        rhs_register = as_to_register(x1, ValueAdapter(), self.flow)

        (a, b), c = merge_invocation(exec_expression, self.flow)

        send_value(as_constant_register(inner, self.flow), a)
        send_value(rewriter.construct_context(c), b)

        return rhs_register

    def evaluate_special_form_from_expression(
        self, expr: ast.expr
    ) -> FromRegister | None:
        if isinstance(expr, ast.Call):
            inet = self.function_compiler.lookup_inet(expr)
            if inet:
                if expr.keywords:
                    raise self.function_compiler.syntax_error(
                        expr, "Keyword arguments are currently not supported for inet invocations"
                    )

                if len(expr.args) != len(inet.args_adapter.concurrent_items):
                    raise self.function_compiler.syntax_error(
                        expr,
                        f"Expected {len(inet.args_adapter.concurrent_items)} arguments, got {len(expr.args)}",
                    )

                inputs, output = inet.invocation(self.flow)
                for input_register, arg in zip(inputs, expr.args, strict=True):
                    send_value(self.evaluate_from_expression(arg), input_register)

                return output

        if isinstance(expr, ast.Name):
            if expr.id not in self.function_compiler.used_as_globals:
                return self.flow.variable_registers[expr.id].readout()

        if isinstance(expr, ast.Subscript):
            inner_adapter = self.function_compiler.infer_expression_adapter(expr.value)
            if isinstance(inner_adapter, ParValueAdapter):
                slice_idx = self.function_compiler.evaluate_subscript(
                    inner_adapter, expr.slice
                )
                inner = self.evaluate_from_expression(expr.value)
                assert inner.adapter == inner_adapter

                resulting_register: FromRegister
                for i, target in enumerate(inner.split()):
                    if i == slice_idx:
                        resulting_register = target
                        break
                    else:
                        target.close()
                else:
                    raise self.function_compiler.syntax_error(
                        expr, "Invalid par subscript"
                    )

                return resulting_register

        if isinstance(expr, ast.Tuple):
            inner_registers = [
                self.evaluate_from_expression(element) for element in expr.elts
            ]

            par_adapter = ParValueAdapter(
                [register.adapter for register in inner_registers]
            )
            x1, x2 = Wire.as_tautology()
            to_register = as_to_register(x1, par_adapter, self.flow)
            send_values(inner_registers, to_register.split())

            return as_from_register(
                x2,
                par_adapter,
                self.flow,
            )

        if isinstance(expr, ast.BoolOp):
            op = expr.op
            merger: Callable[[Any, Any], Any] = lambda x, y: (
                x & y if isinstance(op, ast.And) else lambda x, y: x | y
            )
            should_shortcircuit = (
                (lambda x: not x) if isinstance(op, ast.And) else lambda x: bool(x)
            )
            acc = self.evaluate_from_expression(expr.values[0])
            for n in expr.values[1:]:
                with ConcurrentMerge(should_shortcircuit, merger).invocation(
                    self.flow
                ) as invocation:
                    a, b = invocation.port.readin().split()
                    send_value(acc, a)
                    send_value(self.evaluate_from_expression(n), b)
                    acc = invocation.wire.readout()

            return acc

        return None

    def evaluate_from_expression(self, expr: ast.expr | None) -> FromRegister:
        if expr is None:
            return as_constant_register(None, self.flow)

        solution = self.evaluate_special_form_from_expression(expr)
        if solution is not None:
            return solution

        if isinstance(expr, ast.Constant):
            return as_constant_register(expr.value, self.flow)

        rewriter = ReplaceWithSerializedVariables(self)
        inner = textwrap.dedent(ast.unparse(rewriter.visit(expr)))


        (a, b), c = merge_invocation(eval_expression, self.flow)
        c1, c2 = c.duplicate("share")
        send_value(as_constant_register(inner, self.flow), a)
        send_value(rewriter.construct_context(c2), b)
        return c1

    def parse_deconstruct_iter(self, deconstructor_expr: ast.expr) -> VariablesFlow:
        with VariablesFlow(
            variables=self.flow.variables,
            return_adapter=ValueAdapter(),
        ) as true_case:
            true_branch = InetBranchCompiler(self.function_compiler, true_case)
            send_value(
                true_case.flow_input.value.readout(),
                true_branch.evaluate_to_expression(deconstructor_expr),
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
            variables=self.flow.variables,
            return_adapter=ValueAdapter(),
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
            variables=self.flow.variables,
            return_adapter=ValueAdapter(),
        ) as flow:
            input_variables = flow.variables_readout()
            input_iter = flow.flow_input.value.readout()

            # this branch receives the whole iter
            try_iter_in, (next_value, should_continue) = split_invocation(
                _try_iter,
                flow,
            )
            # it receives the iter
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

    def parse_test(self, test_expr: ast.expr) -> VariablesFlow:
        branch = self.function_compiler.new_test()
        test_result = branch.evaluate_from_expression(test_expr)
        send_value(
            test_result,
            branch.flow.control_output.return_value.readin(),
        )
        send_value(
            branch.flow.variables_readout(),
            branch.flow.control_output.finish_variables.readin(),
        )
        branch.flow.close()
        return branch.flow

    def wire_continuation(
        self, control: FlowControlInto, body_iter: Iterable[ast.stmt]
    ):
        with (
            self.function_compiler.new_branch()
            .parse_statement_body(body_iter)
            .invocation(self.flow) as continuation
        ):
            send_value(
                control.finish_variables.readout(),
                continuation.port.variables.readin(),
            )

            # This finishes iff the continuation finishes
            send_value(
                continuation.wire.finish_variables.readout(),
                self.flow.control_output.finish_variables.readin(),
            )

            send_value(
                control.return_value.readout()
                | continuation.wire.return_value.readout(),
                self.flow.control_output.return_value.readin(),
            )

            send_value(
                control.continue_variables.readout()
                | continuation.wire.continue_variables.readout(),
                self.flow.control_output.continue_variables.readin(),
            )

            send_value(
                control.break_variables.readout()
                | continuation.wire.break_variables.readout(),
                self.flow.control_output.break_variables.readin(),
            )

    def parse_statement_body(
        self, body: Iterable[ast.stmt], default_return_none: bool = False
    ) -> VariablesFlow:
        body_iter = iter(body)

        with self.flow:
            for stmt in body_iter:
                if isinstance(stmt, ast.Return):
                    from_register = self.evaluate_from_expression(stmt.value)
                    send_value(
                        from_register, self.flow.control_output.return_value.readin()
                    )
                    return self.flow
                elif isinstance(stmt, ast.Assign):
                    for from_expr, to_expr in zip(
                        [stmt.value, *stmt.targets], stmt.targets
                    ):
                        send_value(
                            self.evaluate_from_expression(from_expr),
                            self.evaluate_to_expression(to_expr),
                        )
                elif isinstance(stmt, ast.AugAssign):
                    from_expr = ast.BinOp(stmt.target, stmt.op, stmt.value)
                    send_value(
                        self.evaluate_from_expression(from_expr),
                        self.evaluate_to_expression(stmt.target),
                    )
                elif isinstance(stmt, ast.AnnAssign):
                    if stmt.value is not None:
                        send_value(
                            self.evaluate_from_expression(stmt.value),
                            self.evaluate_to_expression(stmt.target),
                        )
                elif isinstance(stmt, (ast.For, ast.While)):
                    with (
                        Loop(
                            (
                                self.parse_deconstruct_iter(stmt.target)
                                if isinstance(stmt, ast.For)
                                else self.parse_test(stmt.test)
                            ),
                            self.function_compiler.new_branch().parse_statement_body(
                                stmt.body
                            ),
                            self.function_compiler.new_branch().parse_statement_body(
                                stmt.orelse
                            ),
                        ).invocation(self.flow) as for_invocation,
                    ):

                        if isinstance(stmt, ast.For):
                            send_value(
                                send_parameter(
                                    filter_invocation(iter, self.flow),
                                    self.evaluate_from_expression(stmt.iter),
                                ),
                                for_invocation.port.value.readin(),
                            )

                        send_value(
                            self.flow.variables_readout(),
                            for_invocation.port.variables.readin(),
                        )

                        self.wire_continuation(for_invocation.wire, body_iter)

                        return self.flow

                elif isinstance(stmt, ast.If):
                    true_case = (
                        self.function_compiler.new_branch().parse_statement_body(
                            stmt.body
                        )
                    )
                    false_case = (
                        self.function_compiler.new_branch().parse_statement_body(
                            stmt.orelse
                        )
                    )

                    with (
                        IfThenElse(true_case, false_case).invocation(
                            self.flow
                        ) as if_invocation,
                    ):
                        send_value(
                            self.evaluate_from_expression(stmt.test),
                            if_invocation.port.readin(),
                        )

                        with closer(
                            pack_into(if_invocation.wire.context, FlowInputInto)
                        ) as conditional_context:
                            send_value(
                                self.flow.variables_readout(),
                                conditional_context.variables.readin(),
                            )

                        self.wire_continuation(
                            pack_from(if_invocation.wire.result, FlowControlInto),
                            body_iter,
                        )

                        return self.flow
                elif isinstance(stmt, ast.Pass):
                    continue
                elif isinstance(stmt, ast.Expr):
                    self.evaluate_from_expression(stmt.value).close()
                elif isinstance(stmt, ast.Break):
                    send_value(
                        self.flow.variables_readout(),
                        self.flow.control_output.break_variables.readin(),
                    )
                    return self.flow
                elif isinstance(stmt, ast.Continue):
                    send_value(
                        self.flow.variables_readout(),
                        self.flow.control_output.continue_variables.readin(),
                    )
                    return self.flow
                else:
                    raise NotImplementedError

            if default_return_none:
                send_value(
                    as_from_register(
                        ConstantValuePort(None), ValueAdapter(), self.flow
                    ),
                    self.flow.control_output.return_value.readin(),
                )
            else:
                send_value(
                    self.flow.variables_readout(),
                    self.flow.control_output.finish_variables.readin(),
                )
            return self.flow


@dataclasses.dataclass(frozen=True, slots=True)
class InetVariablesEvaluator(ast.NodeVisitor):
    compiler: InetFunctionCompiler

    def generic_visit(self, node):
        if isinstance(node, unsupported_expr):
            raise self.compiler.syntax_error(node, "Unsupported expression type")
        if isinstance(node, unsupported_stmt):
            raise self.compiler.syntax_error(node, "Unsupported statement type")

        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self.compiler.variables:
            self.compiler.used_as_globals.add(node.id)

    def mark_assign_target(self, target: ast.Name, adapter: Adapter):
        if target.id in self.compiler.used_as_globals:
            raise self.compiler.syntax_error(
                target, "Assign target is also a global variable"
            )
        if target.id not in self.compiler.variables:
            self.compiler.variables[target.id] = adapter

    def visit_For(self, node):
        adapter = ValueAdapter()

        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                self.mark_assign_target(target, adapter)

        for target in ast.walk(node.iter):
            if isinstance(target, ast.Name):
                self.visit_Name(target)

        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AnnAssign(self, node):
        if not node.simple:
            raise self.compiler.syntax_error(
                node, "Annotations must be simple in inet functions"
            )
        if isinstance(node.target, ast.Name):
            try:
                te = eval(ast.unparse(node.annotation), self.compiler.globals)
            except Exception as e:
                raise self.compiler.syntax_error(
                    node.target, "Could not evaluate annotation in inet"
                ) from e
            adapter = adapter_from_type(te)
            self.mark_assign_target(node.target, adapter)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            adapter = ValueAdapter()
            self.mark_assign_target(node.target, adapter)

    def visit_Assign(self, node):
        adapter = self.infer_expression_adapter(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.mark_assign_target(target, adapter)
            elif isinstance(target, ast.Tuple):
                if isinstance(adapter, ParValueAdapter) and len(
                    adapter.concurrent_items
                ) == len(target.elts):
                    for target, adapter in zip(target.elts, adapter.concurrent_items):
                        if isinstance(target, ast.Name):
                            self.mark_assign_target(target, adapter)
                else:
                    for target in target.elts:
                        if isinstance(target, ast.Name):
                            self.mark_assign_target(target, ValueAdapter())

    def infer_expression_adapter(self, node: ast.expr) -> Adapter:
        if isinstance(node, ast.Call):
            self.generic_visit(node)
        return self.compiler.infer_expression_adapter(node)


def construct_locals(locals_values: tuple, locals_keys: tuple) -> dict:
    return dict(zip(locals_keys, locals_values))


def exec_expression(expr_str: str, context: tuple[dict, dict]) -> None:
    globals, locals = context
    exec(expr_str, globals=globals, locals=locals)


def eval_expression(expr_str: str, context: tuple[dict, dict]) -> Any:
    globals, locals = context
    return eval(expr_str, globals=globals, locals=locals)


def inet[C: Callable](f: C) -> C:
    g = sys._getframe(1).f_globals
    compiler = InetFunctionCompiler(f, g, g.get("__file__", "<anonymous>"))
    compiler.compile()
    setattr(f, "__inet__", compiler)

    @functools.wraps(f)
    def impl(*args: Any) -> Any:
        outputs: list = []
        executor = SynchronizedExecutor()
        inputs, output = compiler.invocation(executor)
        for to_register, arg in zip(inputs, args):
            send_value(as_constant_register(arg, executor), to_register)

        to_register, from_register = filter_invocation(
            lambda x: outputs.append(x), executor
        )
        from_register.close()
        send_value(output, to_register)

        while not outputs and executor.active_pairs:
            executor.process_pair()

        if not outputs:
            raise ValueError("No output produced by the function")

        return outputs[0]

    return cast(C, impl)

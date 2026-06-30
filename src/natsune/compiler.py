import ast
import dataclasses
import functools
import inspect
import random
import string
import sys
import textwrap
from functools import cached_property
from typing import cast, Any, get_type_hints, Iterable, Sequence, Callable

from .adapters import (
    Adapter,
    ValueAdapter,
    adapter_from_type,
    TypeExpression,
    ParValueAdapter,
    Variables,
)
from .connector import Connector
from .control_flow import VariablesFlow, Loop, OneOf, IfThenElse, ConcurrentMerge
from .executor import SynchronizedExecutor
from .invocations import (
    merge_invocation,
    send_parameters,
    filter_invocation,
    send_parameter,
)
from .ports import Wire
from .registers import (
    FromRegister,
    ToRegister,
    as_to_register,
    as_from_register,
    send_value,
    send_values,
    serialize_values,
    as_constant_register,
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

    def construct_context(self) -> FromRegister:
        return send_parameters(
            serialize_values(self.branch_compiler.flow.buffer, 2),
            (
                as_constant_register(
                    self.branch_compiler.function_compiler.globals,
                    self.branch_compiler.flow.buffer,
                ),
                (
                    send_parameters(
                        merge_invocation(
                            construct_locals, self.branch_compiler.flow.buffer
                        ),
                        (
                            (
                                send_parameters(
                                    serialize_values(
                                        self.branch_compiler.flow.buffer,
                                        len(self.used_names),
                                    ),
                                    list(self.used_names.values()),
                                )
                            ),
                            as_constant_register(
                                tuple(self.used_names.keys()),
                                self.branch_compiler.flow.buffer,
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
                Variables(self.variables), adapter_from_type(self.return_annot)
            ),
        )

    def new_test(self) -> InetBranchCompiler:
        return InetBranchCompiler(
            self, VariablesFlow(Variables(self.variables), ValueAdapter())
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
        return self.new_branch().parse_statement_body(self.func_def.body)

    def compile(self) -> None:
        getattr(self, "compiled")

    def invocation(
        self, connector: Connector
    ) -> tuple[Sequence[ToRegister], FromRegister]:
        with self.compiled.invocation(connector) as invocation:
            variable_inputs = invocation.inputs.i_variables.readin().split()
            for input in variable_inputs[len(self.args) :]:
                input.close()
            return (
                variable_inputs[: len(self.args)],
                invocation.control.o_return.readout(),
            )


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
            par_adapter = ParValueAdapter(
                [register.adapter for register in inner_registers]
            )
            x1, x2 = Wire.as_interface()
            from_register = as_from_register(x1, par_adapter, self.flow.buffer)
            send_values(from_register.split(), inner_registers)

            return as_to_register(
                x2,
                par_adapter,
                self.flow.buffer,
            )

        if isinstance(expr, ast.List):
            raise self.function_compiler.syntax_error(
                expr, "List deconstructors in assignment not supported"
            )

        rewriter = ReplaceWithSerializedVariables(self)
        inner = textwrap.dedent(ast.unparse(rewriter.visit(expr)))
        assigned = rewriter.random_identifier()
        inner += " = " + assigned

        x1, x2 = Wire.as_interface()
        rewriter.used_names[assigned] = as_from_register(
            x2, ValueAdapter(), self.flow.buffer
        )
        rhs_register = as_to_register(x1, ValueAdapter(), self.flow.buffer)

        send_parameters(
            merge_invocation(exec_expression, self.flow.buffer),
            (
                as_constant_register(inner, self.flow.buffer),
                rewriter.construct_context(),
            ),
        ).close()

        return rhs_register

    def evaluate_special_form_from_expression(
        self, expr: ast.expr
    ) -> FromRegister | None:
        if isinstance(expr, ast.Call):
            if expr.keywords:
                raise self.function_compiler.syntax_error(
                    expr, "Keyword arguments are currently not supported"
                )

            inet = self.function_compiler.lookup_inet(expr)
            if inet:
                if len(expr.args) != len(inet.args_adapter.concurrent_items):
                    raise self.function_compiler.syntax_error(
                        expr,
                        f"Expected {len(inet.args_adapter.concurrent_items)} arguments, got {len(expr.args)}",
                    )

                inputs, output = inet.invocation(self.flow.buffer)
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
            x1, x2 = Wire.as_interface()
            to_register = as_to_register(x1, par_adapter, self.flow.buffer)
            send_values(inner_registers, to_register.split())

            return as_from_register(
                x2,
                par_adapter,
                self.flow.buffer,
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
                invocation = ConcurrentMerge(should_shortcircuit, merger).invocation(
                    self.flow.buffer
                )
                send_value(acc, invocation.i_value_1.readin())
                send_value(
                    self.evaluate_from_expression(n), invocation.i_value_2.readin()
                )
                acc = invocation.o_value.readout()

            return acc

        return None

    def evaluate_from_expression(self, expr: ast.expr | None) -> FromRegister:
        if expr is None:
            return as_constant_register(None, self.flow.buffer)

        solution = self.evaluate_special_form_from_expression(expr)
        if solution is not None:
            return solution

        if isinstance(expr, ast.Constant):
            return as_constant_register(expr.value, self.flow.buffer)

        rewriter = ReplaceWithSerializedVariables(self)
        inner = textwrap.dedent(ast.unparse(rewriter.visit(expr)))

        return send_parameters(
            merge_invocation(eval_expression, self.flow.buffer),
            (
                as_constant_register(inner, self.flow.buffer),
                rewriter.construct_context(),
            ),
        )

    def parse_deconstructor(self, deconstructor_expr: ast.expr) -> VariablesFlow:
        branch = self.function_compiler.new_branch()
        send_value(
            branch.flow.flow_input.i_value.readout(),
            branch.evaluate_to_expression(deconstructor_expr),
        )
        send_value(
            branch.flow.variables_readout(True), branch.flow.o_control.o_finish.readin()
        )
        branch.flow.close()
        return branch.flow

    def parse_test(self, test_expr: ast.expr) -> VariablesFlow:
        branch = self.function_compiler.new_test()
        send_value(
            branch.evaluate_from_expression(test_expr),
            branch.flow.o_control.o_return.readin(),
        )
        send_value(
            branch.flow.variables_readout(True), branch.flow.o_control.o_finish.readin()
        )
        branch.flow.close()
        return branch.flow

    def parse_statement_body(
        self,
        body: Iterable[ast.stmt],
    ) -> VariablesFlow:
        body_iter = iter(body)

        with self.flow:
            for stmt in body_iter:
                if isinstance(stmt, ast.Return):
                    from_register = self.evaluate_from_expression(stmt.value)
                    send_value(from_register, self.flow.o_control.o_return.readin())
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
                                self.parse_deconstructor(stmt.target)
                                if isinstance(stmt, ast.For)
                                else self.parse_test(stmt.test)
                            ),
                            self.function_compiler.new_branch().parse_statement_body(
                                stmt.body
                            ),
                            self.function_compiler.new_branch().parse_statement_body(
                                stmt.orelse
                            ),
                        ).invocation(self.flow.buffer) as for_invocation,
                        self.function_compiler.new_branch().parse_statement_body(
                            body_iter
                        ) as continuation,
                    ):

                        if isinstance(stmt, ast.For):
                            send_value(
                                send_parameter(
                                    filter_invocation(iter, self.flow.buffer),
                                    self.evaluate_from_expression(stmt.iter),
                                ),
                                for_invocation.inputs.i_value.readin(),
                            )

                        send_value(
                            self.flow.variables_readout(True),
                            for_invocation.inputs.i_variables.readin(),
                        )

                        send_value(
                            for_invocation.control.o_finish.readout(),
                            continuation.flow_input.i_variables.readin(),
                        )

                        # This finishes iff the continuation finishes
                        send_value(
                            continuation.o_control.o_finish.readout(),
                            self.flow.o_control.o_finish.readin(),
                        )

                        o_return_1, o_return_2 = OneOf.share(
                            self.flow.o_control.o_return.readin(), self.flow.buffer
                        )
                        send_value(
                            for_invocation.control.o_return.readout(), o_return_1
                        )
                        send_value(
                            continuation.o_control.o_return.readout(), o_return_1
                        )

                        o_break_1, o_break_2 = OneOf.share(
                            self.flow.o_control.o_break.readin(), self.flow.buffer
                        )
                        send_value(for_invocation.control.o_break.readout(), o_break_1)
                        send_value(continuation.o_control.o_break.readout(), o_break_1)

                        o_continue_1, o_continue_2 = OneOf.share(
                            self.flow.o_control.o_continue.readin(), self.flow.buffer
                        )
                        send_value(
                            for_invocation.control.o_continue.readout(),
                            o_continue_1,
                        )
                        send_value(
                            continuation.o_control.o_continue.readout(),
                            o_continue_1,
                        )

                        return self.flow

                elif isinstance(stmt, ast.If):
                    with (
                        IfThenElse(self.flow.variables.adapter).invocation(
                            self.flow.buffer
                        ) as if_invocation,
                        self.function_compiler.new_branch()
                        .parse_statement_body(stmt.body)
                        .invocation(self.flow.buffer) as body_invocation,
                        self.function_compiler.new_branch()
                        .parse_statement_body(stmt.orelse)
                        .invocation(self.flow.buffer) as orelse_invocation,
                    ):
                        send_value(
                            self.evaluate_from_expression(stmt.test),
                            if_invocation.i_value.readin(),
                        )
                        send_value(
                            self.flow.variables_readout(True),
                            if_invocation.i_context.readin(),
                        )
                        send_value(
                            if_invocation.o_body.readout(),
                            body_invocation.inputs.i_variables.readin(),
                        )
                        send_value(
                            if_invocation.o_orelse.readout(),
                            orelse_invocation.inputs.i_variables.readin(),
                        )
                        return self.flow
                elif isinstance(stmt, ast.Pass):
                    continue
                elif isinstance(stmt, ast.Expr):
                    self.evaluate_from_expression(stmt.value).close()
                elif isinstance(stmt, ast.Break):
                    send_value(
                        self.flow.variables_readout(True),
                        self.flow.o_control.o_break.readin(),
                    )
                    return self.flow
                elif isinstance(stmt, ast.Continue):
                    send_value(
                        self.flow.variables_readout(True),
                        self.flow.o_control.o_continue.readin(),
                    )
                    return self.flow
                else:
                    raise NotImplementedError

            send_value(
                self.flow.variables_readout(True), self.flow.o_control.o_finish.readin()
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

    def visit_AnnAssign(self, node):
        if not node.simple:
            raise self.compiler.syntax_error(
                node, "Annotations must be simple in inet functions"
            )
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                try:
                    te = eval(ast.unparse(node.annotation), self.compiler.globals)
                except Exception as e:
                    raise self.compiler.syntax_error(
                        node.target, "Could not evaluate annotation in inet"
                    ) from e
                adapter = adapter_from_type(te)
                self.mark_assign_target(target, adapter)

    def visit_AugAssign(self, node):
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                adapter = ValueAdapter()
                self.mark_assign_target(target, adapter)

    def visit_Assign(self, node):
        adapter = self.infer_expression_adapter(node.value)
        for target in node.targets:
            for subnode in ast.walk(target):
                if isinstance(subnode, ast.Name):
                    self.mark_assign_target(subnode, adapter)

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

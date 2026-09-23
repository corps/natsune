import ast
import dataclasses

import pytest

from natsune.first_order.adapters import VA, ParValueAdapter, adapter_from_type
from natsune.frontend.ir import (
    Exits,
    IrAssign,
    IrAugAssign,
    IrBoolOp,
    IrBreak,
    IrCallInet,
    IrConst,
    IrContinue,
    IrDynamic,
    IrExprStmt,
    IrFor,
    IrIf,
    IrParIndex,
    IrReturn,
    IrTargetDynamic,
    IrTargetName,
    IrTargetTuple,
    IrTuple,
    IrVar,
    IrWhile,
    build_ir,
)
from natsune.frontend.link import LinkedInet
from natsune.special_forms import Par
from tests.frontend.helpers import build_ir_for, make_position


def _inet_function(arg_types=(int,), return_type=int):
    from types import SimpleNamespace

    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_adapter=adapter_from_type(return_type),
        expansion=None,
    )

    def make(x):
        return 0, 0

    setattr(make, "__inet__", compiler)
    return make


def _body_of(snippet: str, **kwargs):
    ir, sink = build_ir_for(snippet, **kwargs)
    assert ir.body is not None
    return ir, ir.body.statements, sink


def test_function_shape_params_and_return():
    ir, _ = build_ir_for("""
        def f(a: int, b: str) -> int:
            return a
        """)

    assert ir.name == "f"
    assert ir.params == (("a", adapter_from_type(int)), ("b", adapter_from_type(str)))
    assert ir.return_adapter == adapter_from_type(int)
    assert len(ir.body.statements) == 1


# --- statements ----------------------------------------------------------------


def test_assign_const_and_chained_targets():
    _, stmts, sink = _body_of("""
        def f() -> int:
            a = b = 10
            return a
        """)

    assert sink.diagnostics == set()
    assign = stmts[0]
    assert isinstance(assign, IrAssign)
    assert assign.targets == (
        IrTargetName(name="a", adapter=VA),
        IrTargetName(name="b", adapter=VA),
    )
    assert assign.value == IrConst(value=10)


def test_assign_from_param_and_from_global():
    _, stmts, sink = _body_of(
        """
        def f(b: int) -> int:
            a = b
            g = b_global
            return a
        """,
        namespace={"b_global": 3},
    )

    assert sink.diagnostics == set()
    from_param, from_global = stmts[0], stmts[1]
    assert isinstance(from_param, IrAssign)
    assert from_param.value == IrVar(name="b", adapter=adapter_from_type(int))
    assert isinstance(from_global, IrAssign)
    assert from_global.value == IrVar(name="b_global", is_global=True)


def test_tuple_assign_targets():
    _, stmts, sink = _body_of("""
        def f(c: int, d: str) -> int:
            a, b = c, d
            return a
        """)

    assert sink.diagnostics == set()
    assign = stmts[0]
    assert isinstance(assign, IrAssign)
    [target] = assign.targets
    assert isinstance(target, IrTargetTuple)
    assert target.elements == (
        IrTargetName(name="a", adapter=VA),
        IrTargetName(name="b", adapter=VA),
    )


def test_augassign_kept_as_own_node():
    _, stmts, sink = _body_of("""
        def f(a: int) -> int:
            a += 1
            return a
        """)

    assert sink.diagnostics == set()
    aug = stmts[0]
    assert isinstance(aug, IrAugAssign)
    assert aug.target == IrTargetName(name="a", adapter=adapter_from_type(int))
    assert isinstance(aug.op, ast.Add)
    assert aug.value == IrConst(value=1)


def test_if_and_elif_nesting():
    _, stmts, sink = _body_of("""
        def f(c: int, d: int) -> int:
            if c:
                a = 1
            elif d:
                a = 2
            else:
                a = 3
            return a
        """)

    assert sink.diagnostics == set()
    [first_if, ret] = stmts
    assert isinstance(first_if, IrIf)
    assert first_if.test == IrVar(name="c", adapter=adapter_from_type(int))
    assert isinstance(first_if.else_body.statements[0], IrIf)
    [nested_if] = first_if.else_body.statements
    assert nested_if.test == IrVar(name="d", adapter=adapter_from_type(int))
    assert nested_if.else_body.statements[0].value == IrConst(value=3)


def test_if_without_else_has_empty_else_body():
    _, stmts, _ = _body_of("""
        def f(c: int) -> int:
            if c:
                a = 1
            return 0
        """)

    if_stmt = stmts[0]
    assert if_stmt.else_body.statements == ()  # type: ignore[union-attr]


def test_while_break_continue_orelse():
    _, stmts, sink = _body_of("""
        def f(c: int) -> int:
            while c:
                if c:
                    break
                else:
                    continue
            else:
                d = 1
            return 0
        """)

    assert sink.diagnostics == set()
    while_stmt = stmts[0]
    assert isinstance(while_stmt, IrWhile)
    # break/continue live on opposite branches of an if: a bare continue
    # after the body's break would be an invalid post-exit statement.
    branch = while_stmt.body.statements[0]
    assert isinstance(branch, IrIf)
    assert isinstance(branch.then_body.statements[0], IrBreak)
    assert isinstance(branch.else_body.statements[0], IrContinue)
    assert isinstance(while_stmt.orelse.statements[0], IrAssign)


def test_while_body_break_skips_orelse_and_falls_through():
    ir, stmts, sink = _body_of("""
        def f(c: int) -> int:
            while c:
                break
            else:
                return 1
            return 0
        """)

    assert sink.diagnostics == set()
    [while_stmt, ret] = stmts
    assert isinstance(while_stmt, IrWhile)
    # break terminates the loop while skipping the orelse: flow resumes
    # after the loop, so the loop can fall through even though the orelse
    # always returns. (Promoting BREAK to FALLTHROUGH; masking it out
    # entirely classified this loop as a closer and rejected the trailing
    # return as a post-close statement.)
    assert while_stmt.exits == (Exits.FALLTHROUGH | Exits.RETURN)
    assert while_stmt.body.exits == Exits.BREAK
    # The loop returns flow to the list: disjunctive, with the trailing
    # return as the closer.
    assert ir.body.disjunctives == (while_stmt,)
    assert ir.body.closer is ret
    assert ir.body.exits == (Exits.FALLTHROUGH | Exits.RETURN)


def test_for_body_break_skips_orelse_and_falls_through():
    ir, stmts, sink = _body_of(
        """
        def f() -> int:
            for i in xs:
                break
            else:
                return 1
            return 0
        """,
        namespace={"xs": [1]},
    )

    assert sink.diagnostics == set()
    [for_stmt, ret] = stmts
    assert isinstance(for_stmt, IrFor)
    # Same rule as while: break skips the orelse and resumes after the loop.
    assert for_stmt.exits == (Exits.FALLTHROUGH | Exits.RETURN)
    assert ir.body.disjunctives == (for_stmt,)
    assert ir.body.closer is ret


def test_while_body_continue_only_exits_via_orelse():
    ir, stmts, sink = _body_of("""
        def f(c: int) -> int:
            while c:
                continue
            else:
                return 1
        """)

    assert sink.diagnostics == set()
    [while_stmt] = stmts
    assert isinstance(while_stmt, IrWhile)
    # continue only re-tests the condition, so the loop terminates via the
    # orelse alone: it never falls through and is the body's closer.
    assert while_stmt.exits == Exits.RETURN
    assert ir.body.disjunctives == ()
    assert ir.body.closer is while_stmt
    assert ir.body.exits == Exits.RETURN


def test_for_target_and_global_iter():
    _, stmts, sink = _body_of(
        """
        def f() -> int:
            total = 0
            for i in xs:
                print(i)
            else:
                pass
            return total
        """,
        namespace={"xs": [1]},
    )

    assert sink.diagnostics == set()
    for_stmt = stmts[1]
    assert isinstance(for_stmt, IrFor)
    assert for_stmt.target == IrTargetName(name="i", adapter=VA)
    assert for_stmt.iter == IrVar(name="xs", is_global=True)
    assert for_stmt.orelse.statements == ()


def test_for_tuple_target():
    _, stmts, sink = _body_of(
        """
        def f() -> int:
            for a, b in pairs:
                print(a)
            return 0
        """,
        namespace={"pairs": [(1, 2)]},
    )

    assert sink.diagnostics == set()
    for_stmt = stmts[0]
    assert for_stmt.target.elements == (  # type: ignore[union-attr]
        IrTargetName(name="a", adapter=VA),
        IrTargetName(name="b", adapter=VA),
    )


def test_return_with_and_without_value():
    _, stmts, sink = _body_of("""
        def f() -> int:
            return 1
        """)

    assert sink.diagnostics == set()
    [ret] = stmts
    assert ret == IrReturn(value=IrConst(value=1))


def test_return_none_value():
    _, stmts, _ = _body_of("""
        def f() -> int:
            return
        """)

    assert stmts == (IrReturn(value=None),)


def test_exprstmt_and_pass_dropped():
    _, stmts, sink = _body_of("""
        def f() -> int:
            pass
            print(1)
            return 0
        """)

    assert sink.diagnostics == set()
    assert len(stmts) == 2
    assert isinstance(stmts[0], IrExprStmt)
    assert isinstance(stmts[1], IrReturn)


def test_inet_call_node_with_copied_metadata():
    make = _inet_function(arg_types=(int,), return_type=Par[int, str])
    ir, sink = build_ir_for(
        """
        def f() -> int:
            a, b = make(1)
            return a
        """,
        namespace={"make": make},
    )

    assert sink.diagnostics == set()
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    # The call itself returns the Par — no wrapping tuple node.
    assert isinstance(assign.value, IrCallInet)
    call = assign.value
    assert call.ref is make.__inet__
    assert call.arity == 1
    assert call.arg_adapters == (VA,)
    assert call.return_adapter == ParValueAdapter(
        [adapter_from_type(int), adapter_from_type(str)]
    )
    assert call.adapter is call.return_adapter
    assert call.args == (IrConst(value=1),)
    assert assign.targets == (
        IrTargetTuple(
            elements=(
                IrTargetName(name="a", adapter=adapter_from_type(int)),
                IrTargetName(name="b", adapter=adapter_from_type(str)),
            )
        ),
    )


def test_par_index_from_valid_subscript():
    ir, sink = build_ir_for(
        """
        def f(p: Par[int, str]) -> int:
            return p[1]
        """,
        namespace={"Par": Par},
    )

    assert sink.diagnostics == set()
    ret = ir.body.statements[0]
    assert isinstance(ret, IrReturn)
    assert ret.value == IrParIndex(
        base=IrVar(name="p", adapter=ParValueAdapter([VA, VA])),
        index=1,
        adapter=VA,
    )


def test_boolop_flattens_same_op_chains():
    _, stmts, sink = _body_of("""
        def f(a: int, b: int, c: int) -> int:
            d = a and (b and c)
            e = a and b or c
            return d
        """)

    assert sink.diagnostics == set()
    [flat, mixed, _] = stmts
    assert isinstance(flat, IrAssign) and isinstance(flat.value, IrBoolOp)
    # Parenthesized same-op chains flatten to one n-ary node.
    assert flat.value.op == "and"
    assert [v.name for v in flat.value.values if isinstance(v, IrVar)] == [
        "a",
        "b",
        "c",
    ]
    assert isinstance(mixed, IrAssign) and isinstance(mixed.value, IrBoolOp)
    assert mixed.value.op == "or"
    [inner_and, c_var] = mixed.value.values
    assert isinstance(inner_and, IrBoolOp) and inner_and.op == "and"


def test_positions_are_resolved():
    ir, _ = build_ir_for("""
        def f() -> int:
            a = 10
            return a
        """)

    assert ir.position is not None
    assert ir.position == make_position(1, 0)
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    assert assign.position == make_position(2, 4)
    assert assign.value.position == make_position(2, 8)  # type: ignore[union-attr]
    ret = ir.body.statements[1]
    assert ret.position == make_position(3, 4)


def test_exprstmt_dynamic_captures_locals_keeps_globals():
    _, stmts, sink = _body_of(
        """
        def f(b: int) -> int:
            print(b + b_global)
            return 0
        """,
        namespace={"b_global": 7},
    )

    assert sink.diagnostics == set()
    exprstmt = stmts[0]
    assert isinstance(exprstmt, IrExprStmt)
    dynamic = exprstmt.value
    assert isinstance(dynamic, IrDynamic)
    # `print` (global) and `b_global` (global) stay in the source; the local
    # read `b` is captured under its own name.
    assert dynamic.source_text == "print(b + b_global)"
    assert dynamic.captures == (("b", IrVar(name="b", adapter=VA)),)


def test_dynamic_replaces_nested_inet_call_with_placeholder():
    make = _inet_function()
    _, stmts, sink = _body_of(
        """
        def f() -> int:
            print(make(1))
            return 0
        """,
        namespace={"make": make},
    )

    assert sink.diagnostics == set()
    exprstmt = stmts[0]
    dynamic = exprstmt.value  # type: ignore[union-attr]
    assert isinstance(dynamic, IrDynamic)
    assert dynamic.source_text == "print(__natsune_0__)"
    [(name, captured)] = dynamic.captures
    assert name == "__natsune_0__"
    assert isinstance(captured, IrCallInet)
    assert captured.arity == 1
    assert captured.args == (IrConst(value=1),)
    assert captured.return_adapter == VA


def test_dynamic_capture_order_and_dedup():
    make = _inet_function()
    _, stmts, sink = _body_of(
        """
        def f() -> int:
            print(make(1), make(2))
            return 0
        """,
        namespace={"make": make},
    )

    assert sink.diagnostics == set()
    exprstmt = stmts[0]
    dynamic = exprstmt.value  # type: ignore[union-attr]
    assert isinstance(dynamic, IrDynamic)
    # Two separate calls → two placeholders, in scan order.
    assert dynamic.source_text == "print(__natsune_0__, __natsune_1__)"
    assert [name for name, _ in dynamic.captures] == [
        "__natsune_0__",
        "__natsune_1__",
    ]


def test_dynamic_local_read_captured_once():
    _, stmts, sink = _body_of("""
        def f(b: int) -> int:
            print(b + b)
            return 0
        """)

    assert sink.diagnostics == set()
    exprstmt = stmts[0]
    dynamic = exprstmt.value  # type: ignore[union-attr]
    assert isinstance(dynamic, IrDynamic)
    # Same-name local reads share one capture (old `used_names` behavior).
    assert dynamic.source_text == "print(b + b)"
    assert dynamic.captures == (("b", IrVar(name="b", adapter=VA)),)


def test_dynamic_lvalue_target():
    _, stmts, sink = _body_of(
        """
        def f(p: Par[int, str]) -> int:
            p[0] = 5
            return 0
        """,
        namespace={"Par": Par},
    )

    assert sink.diagnostics == set()
    assign = stmts[0]
    assert isinstance(assign, IrAssign)
    [target] = assign.targets
    assert isinstance(target, IrTargetDynamic)
    # The Par index is net-wirable, so the whole target is captured.
    assert target.source_text == "__natsune_0__"
    [(name, captured)] = target.captures
    assert name == "__natsune_0__"
    assert isinstance(captured, IrParIndex)
    assert captured.index == 0


def test_default_name_factory_avoids_used_names():
    # The function reads a global literally named `__natsune_0__`, so the
    # default factory must skip to `__natsune_1__` for its placeholder.
    _, stmts, sink = _body_of(
        """
        def f() -> int:
            print(make(1) + __natsune_0__)
            return 0
        """,
        namespace={"make": _inet_function()},
    )

    assert sink.diagnostics == set()
    dynamic = stmts[0].value  # type: ignore[union-attr]
    assert isinstance(dynamic, IrDynamic)
    assert dynamic.source_text == "print(__natsune_1__ + __natsune_0__)"
    assert [name for name, _ in dynamic.captures] == ["__natsune_1__"]


def test_inet_call_keywords_validated():
    make = _inet_function()
    ir, sink = build_ir_for(
        """
        def f() -> int:
            a = make(x=1)
            return 0
        """,
        namespace={"make": make},
    )

    [diagnostic] = sink.diagnostics  # collector never checks this; builder once
    assert diagnostic.message == (
        "Keyword arguments are currently not supported for inet invocations"
    )
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    assert isinstance(assign.value, IrDynamic)
    assert assign.value.source_text == "make(x=1)"


def test_inet_call_arity_validated():
    make = _inet_function()
    ir, sink = build_ir_for(
        """
        def f() -> int:
            a = make()
            return 0
        """,
        namespace={"make": make},
    )

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Expected 1 arguments, got 0"
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    assert isinstance(assign.value, IrDynamic)


def test_par_subscript_out_of_range_validated():
    ir, sink = build_ir_for(
        """
        def f(p: Par[int, str]) -> int:
            return p[5]
        """,
        namespace={"Par": Par},
    )

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Subscript index of Par must be in range [0, par_size)"
    ret = ir.body.statements[0]
    assert isinstance(ret, IrReturn)
    assert isinstance(ret.value, IrDynamic)
    # The base remains net-wirable and is captured inside the dynamic.
    assert ret.value.source_text == "p[5]"
    assert ret.value.captures[0][0] == "p"


def test_par_subscript_non_constant_validated():
    ir, sink = build_ir_for(
        """
        def f(p: Par[int, str], i: int) -> int:
            return p[i]
        """,
        namespace={"Par": Par},
    )

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Subscript index of Par must be a constant integer"


def test_par_subscript_negative_literal_validated():
    _, sink = build_ir_for(
        """
        def f(p: Par[int, str]) -> int:
            return p[-1]
        """,
        namespace={"Par": Par},
    )

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Subscript index of Par must be a constant integer"


def test_list_lvalue_validated_once_across_pipeline():
    ir, sink = build_ir_for("""
        def f() -> int:
            a, [b] = 1, 2
            return 0
        """)

    # Collector and builder both check; the sink dedupes to one finding.
    assert [d.message for d in sink.diagnostics] == [
        "List deconstructors in assignment not supported"
    ]
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    # The list element is dropped from the target pattern.
    [target] = assign.targets
    assert isinstance(target, IrTargetTuple)
    assert target.elements == (IrTargetName(name="a", adapter=VA),)


def test_unsupported_statement_validated_once_across_pipeline():
    ir, sink = build_ir_for("""
        def f() -> int:
            assert False
            return 0
        """)

    assert [d.message for d in sink.diagnostics] == ["Unsupported statement type"]
    assert [type(s) for s in ir.body.statements] == [IrReturn]


def test_try_blocks_rejected():
    _, sink = build_ir_for("""
        def f() -> int:
            try:
                pass
            except* ValueError:
                pass
            return 0
        """)

    assert [d.message for d in sink.diagnostics] == ["Unsupported statement type"]


def test_unsupported_expression_validated():
    ir, sink = build_ir_for("""
        def f() -> int:
            g = lambda: 1
            return 0
        """)

    assert [d.message for d in sink.diagnostics] == ["Unsupported expression type"]
    assign = ir.body.statements[0]
    assert isinstance(assign, IrAssign)
    assert isinstance(assign.value, IrDynamic)


def test_unary_constant_folds_to_const():
    ir, sink = build_ir_for("""
        def f() -> int:
            a = -1
            b = not True
            c = -a
            d = ~1.5
            print(-1)
            return a
        """)

    assert sink.diagnostics == set()
    [neg, not_, neg_var, invert, exprstmt, _] = ir.body.statements

    assert isinstance(neg, IrAssign)
    assert neg.value == IrConst(value=-1)

    assert isinstance(not_, IrAssign)
    assert not_.value == IrConst(value=False)

    # Operands that are not constants stay dynamic, with the variable
    # captured.
    assert isinstance(neg_var, IrAssign)
    assert isinstance(neg_var.value, IrDynamic)
    assert neg_var.value.source_text == "-a"
    assert neg_var.value.captures == (("a", IrVar(name="a", adapter=VA)),)

    # `~1.5` raises TypeError if attempted — left for the exec path.
    assert isinstance(invert, IrAssign)
    assert isinstance(invert.value, IrDynamic)
    assert invert.value.source_text == "~1.5"

    # Inside a dynamic scan, folded constants are not captured: `-1` stays
    # in the source text exactly as written.
    assert isinstance(exprstmt, IrExprStmt)
    assert isinstance(exprstmt.value, IrDynamic)
    assert exprstmt.value.source_text == "print(-1)"
    assert exprstmt.value.captures == ()


def test_arbitrary_expression_becomes_dynamic_binop():
    _, stmts, sink = _body_of("""
        def f(b: int) -> int:
            a = b * 2 + 1
            return a
        """)

    assert sink.diagnostics == set()
    assign = stmts[0]
    assert isinstance(assign, IrAssign)
    value = assign.value
    assert isinstance(value, IrDynamic)
    assert value.source_text == "b * 2 + 1"
    assert value.captures == (("b", IrVar(name="b", adapter=VA)),)

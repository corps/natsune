"""Phase 5 tests — adapter inference.

Pure functions over plain mappings: no collector, no compilation. The plan's
named cases (in/out-of-range subscripts, non-constant slices, call adapters
for linked vs unlinked callees) plus the absorbed quirks (negative literals,
bool indices) and the `collect_call_links` pre-pass.
"""

import ast
from types import SimpleNamespace

from natsune.adapters import VA, ParValueAdapter, adapter_from_type
from natsune.frontend.infer import (
    ParSubscriptError,
    infer_adapter,
    par_subscript_index,
)
from natsune.frontend.link import (
    LinkedInet,
    LinkedValue,
    LinkNotFound,
    collect_call_links,
)
from natsune.special_forms import Par
from tests.frontend.helpers import parse_function


def _expr(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def _slice(index_text: str) -> ast.expr:
    """The slice expression of `p[index_text]` (what the check takes)."""
    subscript = _expr(f"p[{index_text}]")
    assert isinstance(subscript, ast.Subscript)
    return subscript.slice


INT, STR = adapter_from_type(int), adapter_from_type(str)
PAR_INT_STR = ParValueAdapter([INT, STR])


def _inet_link(return_adapter) -> LinkedInet:
    return LinkedInet(
        ref=None, arity=1, arg_adapters=(VA,), return_adapter=return_adapter
    )


# --- infer_adapter -----------------------------------------------------------


def test_name_resolves_from_variables():
    assert infer_adapter(_expr("x"), {"x": INT}, {}) is INT


def test_unknown_name_yields_va():
    assert infer_adapter(_expr("nope"), {"x": INT}, {}) is VA


def test_tuple_builds_par():
    result = infer_adapter(_expr("(x, y)"), {"x": INT, "y": STR}, {})
    assert result == PAR_INT_STR


def test_call_linked_inet_returns_copied_adapter():
    links = {"f": _inet_link(PAR_INT_STR)}
    assert infer_adapter(_expr("f(1)"), {}, links) == PAR_INT_STR


def test_call_unlinked_callee_yields_va():
    assert infer_adapter(_expr("g(1)"), {}, {"g": LinkedValue(5)}) is VA
    assert infer_adapter(_expr("g(1)"), {}, {"g": LinkNotFound()}) is VA
    assert infer_adapter(_expr("g(1)"), {}, {}) is VA


def test_call_non_name_callee_yields_va():
    # `obj.m()` — the old lookup only ever considered bare names.
    assert infer_adapter(_expr("obj.m()"), {"obj": INT}, {}) is VA


def test_subscript_constant_in_range():
    variables = {"p": PAR_INT_STR}
    assert infer_adapter(_expr("p[0]"), variables, {}) is INT
    assert infer_adapter(_expr("p[1]"), variables, {}) is STR


def test_subscript_out_of_range_yields_va():
    assert infer_adapter(_expr("p[2]"), {"p": PAR_INT_STR}, {}) is VA


def test_subscript_non_par_base_yields_va():
    # Silently VA in both old and new; IR construction validates explicitly.
    assert infer_adapter(_expr("x[0]"), {"x": INT}, {}) is VA


def test_subscript_of_unknown_base_yields_va():
    assert infer_adapter(_expr("nope[0]"), {}, {}) is VA


def test_subscript_of_tuple_par_element():
    # `pair[0][1]`: Par[int, Par[str, ...]] — element adapters compose.
    variables = {"pair": ParValueAdapter([INT, PAR_INT_STR])}
    assert infer_adapter(_expr("pair[0]"), variables, {}) is INT
    assert infer_adapter(_expr("pair[1]"), variables, {}) is PAR_INT_STR


# --- par_subscript_index (reused by IR construction) --------------------------


def test_subscript_index_valid():
    assert par_subscript_index(PAR_INT_STR, _slice("0")) == 0
    assert par_subscript_index(PAR_INT_STR, _slice("1")) == 1


def test_subscript_index_non_constant_slice():
    assert par_subscript_index(PAR_INT_STR, _slice("i")) == ParSubscriptError(
        "Subscript index of Par must be a constant integer"
    )


def test_subscript_index_non_int_constant():
    assert par_subscript_index(PAR_INT_STR, _slice('"zero"')) == ParSubscriptError(
        "Subscript index of Par must be a constant integer"
    )


def test_subscript_index_negative_literal_is_not_constant():
    # §10 row 13: `-1` is UnaryOp(USub, Constant), so the old check rejected
    # it as "not a constant integer" — absorbed exactly. The discrimination
    # (vs out-of-range) is consumed by phase 6's builder diagnostics.
    assert par_subscript_index(PAR_INT_STR, _slice("-1")) == ParSubscriptError(
        "Subscript index of Par must be a constant integer"
    )


def test_subscript_index_out_of_range():
    assert par_subscript_index(PAR_INT_STR, _slice("2")) == ParSubscriptError(
        "Subscript index of Par must be in range [0, par_size)"
    )


def test_subscript_index_bool_constant_acts_as_int():
    # Absorbed quirk: bool is an int subclass, old behavior accepted it.
    assert par_subscript_index(PAR_INT_STR, _slice("True")) == 1


# --- collect_call_links -------------------------------------------------------


def test_collect_call_links_resolves_body_calls_once():
    func_def = parse_function("""
        def f():
            a = make(1)
            b = make(2)
            c = obj.m()
            return a
        """)
    links = collect_call_links(func_def.body, {})

    assert set(links) == {"make"}  # deduplicated; `obj.m()` not a bare name
    assert isinstance(links["make"], LinkNotFound)
    assert links["make"] is links["make"] or links["make"] == LinkNotFound()


def test_collect_call_links_resolves_inet_and_values():
    # A compiler-shaped object behind `__inet__` (what `from_ref` reads).
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([VA]),
        return_annot=Par[int, int],
    )

    def make(x):
        return 0, 0

    setattr(make, "__inet__", compiler)

    func_def = parse_function("""
        def f():
            a = make(1)
            b = print(2)
            return a
        """)
    links = collect_call_links(func_def.body, {"make": make})

    assert links["make"] == LinkedInet(
        ref=compiler,
        arity=1,
        arg_adapters=(VA,),
        return_adapter=ParValueAdapter([VA, VA]),
    )
    assert links["print"] == LinkedValue(print)

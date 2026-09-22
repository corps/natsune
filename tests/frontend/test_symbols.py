from types import SimpleNamespace

from natsune.adapters import VA, ParValueAdapter, ReferenceAdapter, adapter_from_type
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.signature import analyze_signature
from natsune.frontend.symbols import SymbolsTable, collect_symbols
from natsune.special_forms import Par, Ref
from tests.frontend.helpers import make_source


def _collect(snippet: str, **kwargs) -> tuple[SymbolsTable, DiagnosticSink]:
    source = make_source(snippet, **kwargs)
    signature = analyze_signature(source, DiagnosticSink())
    sink = DiagnosticSink()
    table = collect_symbols(source, signature, sink)
    return table, sink


def _inet_function(arg_types=(int,), return_type=int):
    """A callable with an `__inet__` attribute shaped like the old compiler."""
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_adapter=adapter_from_type(return_type),
        expansion=None,
    )

    def make(x):
        return 0, 0

    # Dynamic attribute, exactly as the old `inet` decorator attaches it.
    setattr(make, "__inet__", compiler)
    return make


# --- locals and globals ------------------------------------------------------


def test_params_are_seeded_locals():
    table, sink = _collect("""
        def f(a: int, b: str) -> int:
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables == {
        "a": adapter_from_type(int),
        "b": adapter_from_type(str),
    }
    assert table.used_as_globals == frozenset()


def test_global_read_marks_used_as_globals():
    table, sink = _collect(
        """
        def f() -> int:
            print(b_global)
            return 0
        """,
        namespace={"b_global": 7},
    )

    assert sink.diagnostics == set()
    assert "b_global" in table.used_as_globals
    assert "b_global" not in table.variables


def test_assign_from_param_infers_adapter():
    table, sink = _collect("""
        def f(b: int) -> int:
            a = b
            return a
        """)
    assert sink.diagnostics == set()
    assert table.variables["a"] == adapter_from_type(int)


def test_assign_from_global_marked_not_crashed():
    table, sink = _collect(
        """
        def f() -> int:
            a = b_global
            return a
        """,
        namespace={"b_global": 7},
    )

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA
    assert "b_global" in table.used_as_globals


def test_simple_assignment_introduces_va_local():
    table, sink = _collect("""
        def f() -> int:
            a = 10
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA


def test_chained_assignment_marks_all_targets():
    table, sink = _collect("""
        def f() -> int:
            a = b = 10
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA
    assert table.variables["b"] is VA


def test_tuple_assignment_matches_par_elements():
    table, sink = _collect("""
        def f(c: int, d: str) -> int:
            a, b = c, d
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] == adapter_from_type(int)
    assert table.variables["b"] == adapter_from_type(str)


def test_tuple_assignment_mismatch_diagnosed():
    table, sink = _collect("""
        def f(c: int) -> int:
            a, b = c
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == (
        "Tuple assignment targets do not match the value's Par size"
    )
    assert table.variables["a"] is VA
    assert table.variables["b"] is VA


def test_tuple_assignment_from_inet_call_uses_return_adapter():
    make = _inet_function(return_type=Par[int, int])
    table, sink = _collect(
        """
        def f() -> int:
            a, b = make(1)
            return a
        """,
        namespace={"make": make},
    )

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA
    assert table.variables["b"] is VA


def test_augassign_introduces_local():
    table, sink = _collect("""
        def f() -> int:
            x += 1
            return x
        """)

    assert sink.diagnostics == set()
    assert table.variables["x"] is VA


def test_augassign_on_param_keeps_adapter():
    table, sink = _collect("""
        def f(a: int) -> int:
            a += 1
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] == adapter_from_type(int)


def test_augassign_value_names_are_marked():
    table, sink = _collect(
        """
        def f() -> int:
            x += b_global
            return x
        """,
        namespace={"b_global": 7},
    )

    assert sink.diagnostics == set()
    assert "b_global" in table.used_as_globals


# --- AnnAssign ---------------------------------------------------------------


def test_annassign_registers_with_evaluated_adapter():
    table, sink = _collect("""
        def f() -> int:
            a: int = 20
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] == adapter_from_type(int)


def test_annassign_with_ref_annotation():
    table, sink = _collect(
        """
        def f() -> int:
            a: Ref[int] = 20
            return a
        """,
        namespace={"Ref": Ref},
    )

    assert sink.diagnostics == set()
    assert isinstance(table.variables["a"], ReferenceAdapter)


def test_annassign_failure_uses_consistent_format():
    table, sink = _collect("""
        def f() -> int:
            a: Missing = 5
            return 0
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == (
        "Could not evaluate annotation: Missing (name 'Missing' is not defined)"
    )
    assert "a" not in table.variables


def test_annassign_rhs_names_are_marked():
    table, sink = _collect(
        """
        def f() -> int:
            a: int = b_global
            return a
        """,
        namespace={"b_global": 7},
    )

    assert sink.diagnostics == set()
    assert "b_global" in table.used_as_globals


def test_annassign_non_simple_rejected():
    _, sink = _collect("""
        def f() -> int:
            (a): int = 5
            return 0
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Annotations must be simple in inet functions"


# --- for targets -------------------------------------------------------------


def test_for_target_introduces_va_local_and_iter_global():
    table, sink = _collect(
        """
        def f() -> int:
            for i in xs:
                print(i)
            return 0
        """,
        namespace={"xs": [1, 2]},
    )

    assert sink.diagnostics == set()
    assert table.variables["i"] is VA
    assert "xs" in table.used_as_globals


def test_for_tuple_target_marks_all_leaves():
    table, sink = _collect(
        """
        def f() -> int:
            for a, b in pairs:
                print(a)
            return 0
        """,
        namespace={"pairs": [(1, 2)]},
    )

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA
    assert table.variables["b"] is VA


def test_for_exotic_target_diagnosed():
    table, sink = _collect(
        """
        def f() -> int:
            for x[0] in xs:
                print(x)
            return 0
        """,
        namespace={"xs": [1, 2]},
    )

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Unsupported for-loop target"
    assert "x" not in table.variables


def test_read_before_assignment_conflicts_with_later_assignment():
    table, sink = _collect("""
        def f() -> int:
            print(a)
            a = 1
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Assign target is also a global variable"
    assert "a" in table.used_as_globals
    assert "a" not in table.variables


# TODO: Not sure if this makes sense when we pull in inverted, drop diagnostic
def test_self_referential_first_binding_is_diagnosed():
    table, sink = _collect("""
        def f() -> int:
            a = a + 1
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Read of variable before assignment"
    assert table.variables["a"] is VA  # still introduced, best effort


def test_self_reference_after_real_binding_is_fine():
    table, sink = _collect("""
        def f(b: int) -> int:
            a = 1
            a = a + b
            return a
        """)

    assert sink.diagnostics == set()
    assert table.variables["a"] is VA


def test_tuple_swap_of_fresh_names_is_diagnosed():
    table, sink = _collect("""
        def f() -> int:
            a, b = b, a
            return a
        """)

    assert [d.message for d in sink.diagnostics] == [
        "Read of variable before assignment",
        "Read of variable before assignment",
    ]


def test_unsupported_stmt_diagnosed():
    _, sink = _collect("""
        def f() -> int:
            assert False
            return 0
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Unsupported statement type"


def test_unsupported_expr_diagnosed():
    _, sink = _collect("""
        def f() -> int:
            g = lambda: 1
            return 0
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Unsupported expression type"

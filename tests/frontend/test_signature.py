"""Phase 2 tests — signature analysis.

Snippet in, `Signature` out — via the phase-1 `make_source` fixture, no
executor. Covers the plan's named cases (0/1/n args, missing annotations,
rejection of kw-only/vararg/default) plus the two old-code holes found while
absorbing (positional defaults ignored, positional-only params dropped —
§10 rows 8–9).
"""

import pytest

from natsune.adapters import VA, ParValueAdapter, adapter_from_type
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.signature import Signature, analyze_signature
from natsune.special_forms import Par
from tests.frontend.helpers import make_source


def _analyze(snippet: str, **kwargs) -> tuple[Signature, DiagnosticSink]:
    source = make_source(snippet, **kwargs)
    sink = DiagnosticSink()
    return analyze_signature(source, sink), sink


def test_zero_args():
    signature, sink = _analyze("""
        def f() -> int:
            return 1
        """)

    assert sink.diagnostics == set()
    assert signature.args == ()
    assert signature.arity == 0
    assert signature.args_adapter == ParValueAdapter([])
    assert signature.return_type is int
    assert signature.return_adapter == adapter_from_type(int)


def test_one_annotated_arg():
    signature, sink = _analyze("""
        def f(a: int) -> int:
            return a
        """)

    assert sink.diagnostics == set()
    assert signature.args == (("a", int),)
    assert signature.args_adapter == ParValueAdapter([adapter_from_type(int)])
    assert signature.return_adapter == adapter_from_type(int)


def test_n_args_with_par_return():
    signature, sink = _analyze(
        """
        def f(a: int, b: str) -> Par[int, int]:
            return a, b
        """,
        namespace={"Par": Par},
    )

    assert sink.diagnostics == set()
    assert signature.args == (("a", int), ("b", str))
    assert signature.arity == 2
    assert signature.args_adapter == ParValueAdapter(
        [adapter_from_type(int), adapter_from_type(str)]
    )
    # Par[int, int] resolves to a two-slot Par of value adapters.
    assert signature.return_adapter == ParValueAdapter([VA, VA])


def test_missing_annotations_become_none_with_va_adapters():
    signature, sink = _analyze("""
        def f(a, b):
            return a
        """)

    assert sink.diagnostics == set()
    assert signature.args == (("a", None), ("b", None))
    assert signature.args_adapter == ParValueAdapter([VA, VA])
    assert signature.return_type is None
    assert signature.return_adapter is VA


def test_mixed_annotated_and_missing():
    signature, sink = _analyze("""
        def f(a: int, b) -> int:
            return a
        """)

    assert signature.args == (("a", int), ("b", None))
    assert signature.args_adapter == ParValueAdapter([adapter_from_type(int), VA])


# --- structural rejection ----------------------------------------------------


def test_rejects_vararg_with_position():
    signature, sink = _analyze("""
        def f(*rest):
            return 1
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "*args is not supported in inet functions"
    # Positioned at the `rest` parameter node in the snippet (base_lineno 1).
    assert (diagnostic.lineno, diagnostic.col_offset) == (1, 7)
    # Best-effort signature is still produced; the phase boundary raises.
    assert signature.args == ()


def test_rejects_kwarg():
    _, sink = _analyze("""
        def f(**rest):
            return 1
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "**kwargs is not supported in inet functions"


def test_rejects_kwonly_args():
    # `kw_defaults` is subsumed: every entry pairs with a kw-only arg.
    _, sink = _analyze("""
        def f(*, flag=3):
            return 1
        """)

    [diagnostic] = sink.diagnostics
    assert (
        diagnostic.message
        == "Keyword-only arguments are not supported in inet functions"
    )


def test_rejects_positional_defaults_old_code_ignored():
    # §10 row 8: `def f(a, b=5)` used to compile silently while the default
    # could never be supplied through the net interface.
    signature, sink = _analyze("""
        def f(a, b=5):
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message == "Default values are not supported in inet functions"
    assert (diagnostic.lineno, diagnostic.col_offset) == (1, 9)  # `b`
    assert signature.args == (("a", None), ("b", None))  # best effort


def test_rejects_positional_only_params_old_code_dropped():
    # §10 row 9: `def f(a, /, b)` used to silently drop `a` from the arg list
    # (wrong arity).
    signature, sink = _analyze("""
        def f(a, /, b):
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert (
        diagnostic.message
        == "Positional-only parameters are not supported in inet functions"
    )
    assert signature.args == (("b", None),)  # best effort


def test_rejections_accumulate_in_feature_order():
    signature, sink = _analyze("""
        def f(a=1, *rest, **kw):
            return a
        """)

    assert {d.message for d in sink.diagnostics} == {
        "*args is not supported in inet functions",
        "**kwargs is not supported in inet functions",
        "Default values are not supported in inet functions",
    }
    # The boundary hook raises all diagnostics as an ExceptionGroup.
    with pytest.raises(ExceptionGroup) as excinfo:
        sink.raise_if_errors("signature errors")
    assert len(excinfo.value.exceptions) == 3
    exceptions = excinfo.value.exceptions
    assert all(isinstance(e, SyntaxError) for e in exceptions)
    messages = {e.msg for e in exceptions}  # type: ignore
    assert messages == {
        "*args is not supported in inet functions",
        "**kwargs is not supported in inet functions",
        "Default values are not supported in inet functions",
    }
    assert signature.arity == 1  # `a` — best effort despite diagnostics


# --- annotation resolution failures ------------------------------------------


def test_unresolvable_annotation_is_diagnostic_not_exception():
    # Old code: raw NameError escaped `get_type_hints`. Python 3.14's lazy
    # annotations let the def itself succeed, so this surfaces here.
    signature, sink = _analyze("""
        def f(a: Missing) -> int:
            return a
        """)

    [diagnostic] = sink.diagnostics
    assert diagnostic.message.startswith("Could not resolve annotations:")
    assert "name 'Missing' is not defined" in diagnostic.message
    # `get_type_hints` is all-or-nothing: with one unresolvable annotation,
    # everything is treated as absent (old code: raw NameError, no signature).
    assert signature.args == (("a", None),)
    assert signature.return_type is None
    assert signature.return_adapter is VA

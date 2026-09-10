"""Step-0 cross-check for §7.2b (COMPILER_REFACTOR.md §6, first bullet):
IrBody.variable_usage vs the legacy flow_map usage flags.

Semantics (§1): usage classifies the EFFECT on the variable's cell, not
the syntactic position — "read" = independent copy (cell identity
preserved), "write" = linear use that advances the cell. Classification
is adapter-declared via adapters.read_independently over the wiring
type; deliberately VALUE-leaf-only while Par reads linearize (§6).

Findings (why the checks below are shaped the way they are):

Legacy's flow_map flags are NOT a complete usage analysis — they are
wherever evaluation happened to run. Fragment flows whose flags never
reach the parent: if/while tests evaluate in a `new_test()` flow that is
closed and grafted, never merged (test reads invisible — is_it_even's
`input`); for-target writes bypass FlowRegister.readin via the
deconstruct case flows' interface (sum_it_up's `i`); iterable captures
land on case flows too (sum_it_up's `start`/`end`).

The IR, post-normalization, now AGREES with legacy everywhere legacy is
complete — including the Ref/Inverse read-as-write behavior, which the
IR adopted as intended semantics (linear reads ARE writes) rather than
as a quirk. The soundness check below is therefore unconditional:
every legacy-flagged name appears in the IR usage, and a legacy write
is an IR write. The remaining disagreements are exactly legacy's
under-reporting sites, pinned individually.

Programs excluded from the parametrized check entirely: cross-program
calls (invoke_an_inet, use_references, test_delayed_inverse,
drops_infinite_loop, and_or_with_finites_and_infinites) — their callees
carry the frontend's fake __inet__ objects (SimpleNamespace, no
.invocation), so the legacy compiler cannot lower them.
"""

import linecache
import textwrap

import pytest

from natsune.compiler import InetFunctionCompiler
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.special_forms import Par, Ref
from tests.frontend.helpers import build_ir_for
from tests.frontend.programs import PROGRAMS

CALL_PROGRAMS = frozenset(
    {
        "invoke_an_inet",
        "use_references",
        "test_delayed_inverse",
        "drops_infinite_loop",
        "and_or_with_finites_and_infinites",
    }
)

CROSSCHECK_PROGRAMS = [func for func in PROGRAMS if func.__name__ not in CALL_PROGRAMS]


def program(name: str):
    return next(f for f in PROGRAMS if f.__name__ == name)


def _legacy_flow(func):
    """Compile an undecorated module-level program through the legacy
    compiler (its own globals carry the annotations and fakes)."""
    compiler = InetFunctionCompiler(func, func.__globals__, f"{func.__name__}.py")
    compiler.compile()
    return compiler.compiled


def _build_ir(func):
    """The genuine phase-1 path, as in test_snapshots.compile_program."""
    source = extract_source(
        func, globals=func.__globals__, filename=f"{func.__name__}.py"
    )
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())
    links = collect_call_links(source.func_def.body, source.globals)
    sink = DiagnosticSink()
    ir = build_ir(source, signature, symbols, links, sink)
    assert not sink.diagnostics
    return ir


def _legacy_flags(flow) -> dict[str, tuple[bool, bool]]:
    return {
        name: (usage.flow_read, usage.flow_write)
        for name, usage in flow.flow_map.usage.items()
    }


def test_legacy_flags_are_a_sound_subset_of_ir_usage() -> None:
    """Every legacy-flagged name must exist in the IR usage, and a legacy
    write must be an IR write — unconditionally, now that the IR
    normalizes linear reads (Reference/Inverse/Par) to writes."""

    for func in CROSSCHECK_PROGRAMS:
        ir_usage = dict(_build_ir(func).body.variable_usage)
        flow = _legacy_flow(func)
        context = func.__name__

        for name, (read, write) in _legacy_flags(flow).items():
            if not (read or write):
                continue  # untouched variable — the IR records nothing
            assert name in ir_usage, (
                f"{context}: legacy flags {name!r} "
                f"(flow_read={read}, flow_write={write}); IR usage has no entry"
            )
            if write:
                assert ir_usage[name] == "write", (
                    f"{context}: legacy flow_write on {name!r} but IR says "
                    f"{ir_usage[name]!r}"
                )


def test_test_flow_reads_never_reach_parent_flags():
    # Pinned legacy under-reporting: the if-test is evaluated in a
    # new_test() flow that is closed and grafted, never merged into the
    # parent — so `input`'s read (the entire predicate!) is invisible to
    # legacy's parent flow_map. The IR records it.
    ir_usage = dict(_build_ir(program("is_it_even")).body.variable_usage)
    legacy = _legacy_flags(_legacy_flow(program("is_it_even")))

    assert ir_usage == {"input": "read"}
    assert legacy["input"] == (False, False)


def test_for_target_writes_and_iterable_reads_missed():
    # Pinned legacy under-reporting: sum_it_up's `i` is written every
    # iteration (for target) but legacy shows read-only — the write went
    # through the deconstruct case flows' interface, bypassing
    # FlowRegister.readin. The iterable's `range(start, end)` reads
    # landed on case flows as well and never merged.
    ir_usage = dict(_build_ir(program("sum_it_up")).body.variable_usage)
    legacy = _legacy_flags(_legacy_flow(program("sum_it_up")))

    assert ir_usage == {
        "start": "read",
        "end": "read",
        "total": "write",
        "i": "write",
    }
    assert legacy["i"] == (True, False)  # read flagged, write missed
    assert legacy["total"] == (True, True)
    assert legacy["start"] == (False, False)  # iterable read never merged
    assert legacy["end"] == (False, False)


def test_ref_read_normalizes_to_write_and_agrees():
    # Reading a Ref-typed variable — here `a.append(1)`, a dynamic
    # capture, with NO assignment anywhere — is a linear use: the IR
    # normalizes it to "write" (adapters.read_independently refuses on
    # REFERENCE wiring), agreeing with legacy's flow_write.
    source = "def append_to(a: Ref[list]) -> None:\n    a.append(1)"

    text = textwrap.dedent(source)
    filename = "usage_ref_case.py"
    ns: dict = {"Ref": Ref}  # the annotation is evaluated at compile time
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )

    ir, sink = build_ir_for(source, namespace={"Ref": Ref})
    assert not sink.diagnostics
    legacy = _legacy_flags(_legacy_flow(ns["append_to"]))

    assert dict(ir.body.variable_usage) == {"a": "write"}
    assert legacy["a"] == (False, True)


def test_par_reads_linearize():
    # Par is the conservative edge (§6): even an all-copyable Par
    # linearizes on read, because reads go through a readout of the whole
    # Par with the neighbor elements closed — matching legacy's flags
    # exactly. Post-cutover, element-pass-through reads relax this to the
    # recursive discipline rule; this pin (and the doc marker) is what
    # that improvement updates.
    source = "def second(p: Par[int, int]) -> int:\n    return p[1]"

    text = textwrap.dedent(source)
    filename = "usage_par_case.py"
    ns: dict = {"Par": Par}
    exec(compile(text, filename, "exec"), ns)  # noqa: S102 — test source
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )

    ir, sink = build_ir_for(source, namespace={"Par": Par})
    assert not sink.diagnostics
    legacy = _legacy_flags(_legacy_flow(ns["second"]))

    assert dict(ir.body.variable_usage) == {"p": "write"}
    assert legacy["p"] == (False, True)

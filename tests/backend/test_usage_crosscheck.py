"""Step-0 cross-check for §7.2b (COMPILER_REFACTOR.md §6, first bullet):
IrBody.variable_usage vs the legacy flow_map usage flags.

Findings (the reason this file asserts soundness, not equality):

The legacy flow_map flags are NOT a usage analysis — they are wherever
evaluation happened to run. FlowRegister.readout/readin set flags only on
the flow whose registers they touched, and several legacy constructs run
on fragment flows whose flags never reach the parent:

- if/while tests are evaluated in a fresh `new_test()` flow that is
  closed and grafted, never invoked-into the parent (compiler.py:723) —
  test reads are invisible to the parent's map (is_it_even: `input`).
- for-target writes go through the deconstruct case flows' variable
  interface, bypassing FlowRegister.readin (sum_it_up: `i` has
  flow_write=False despite being written every iteration); iterable
  reads (`range(start, end)` captures) land on case flows too.
- Ref/Inverse reads set flow_write, not flow_read — FlowRegister.readout
  branches on ValueAdapter (registers.py:291) — so a Ref-typed read
  looks like a write.

The IR's variable_usage is the first complete statement-level analysis
here: it walks the actual tree, merges nested bodies (write wins), and
records globals-aware reads. 2b will source wiring decisions from it.

What we can and do assert — legacy flags are a SOUND SUBSET of the IR:

1. every legacy-flagged name appears in the IR usage (both directions of
   the flag), and
2. a legacy write-flag on a ValueAdapter-typed variable implies the IR
   says "write" (readin only fires on assignments/targets). Ref/Inverse
   names are exempt: their read-as-write quirk is pinned separately.

Plus pinned divergences (the §6 bullet promises disagreements are
surfaced, not smoothed over): the three fragment-flow sites above, each
with a concrete program.

Programs excluded from the parametrized check entirely: cross-program
calls (invoke_an_inet, use_references, test_delayed_inverse,
drops_infinite_loop, and_or_with_finites_and_infinites) — their callees
carry the frontend's fake __inet__ objects (SimpleNamespace, no
.invocation), so the legacy compiler cannot lower them.
"""

import linecache
import textwrap

import pytest

from natsune.adapters import ValueAdapter
from natsune.compiler import InetFunctionCompiler
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.special_forms import Ref
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
    """Every legacy-flagged name must exist in the IR usage; a legacy
    write on a value-typed variable must be an IR write."""

    for func in CROSSCHECK_PROGRAMS:
        ir = _build_ir(func)
        flow = _legacy_flow(func)
        legacy = _legacy_flags(flow)
        ir_usage = dict(ir.body.variable_usage)
        context = func.__name__

        for name, (read, write) in legacy.items():
            if not (read or write):
                continue  # untouched variable — the IR records nothing
            assert name in ir_usage, (
                f"{context}: legacy flags {name!r} "
                f"(flow_read={read}, flow_write={write}); IR usage has no entry"
            )
            if write and isinstance(flow.variables[name], ValueAdapter):
                assert ir_usage[name] == "write", (
                    f"{context}: legacy flow_write on {name!r} but IR says "
                    f"{ir_usage[name]!r}"
                )


def test_test_flow_reads_never_reach_parent_flags():
    # Pinned: the if-test is evaluated in a new_test() flow that is closed
    # and grafted, never merged into the parent — so `input`'s read (the
    # entire predicate!) is invisible to legacy's parent flow_map. The IR
    # records it. 2b sourcing flags from the IR fixes this class.
    ir_usage = dict(_build_ir(program("is_it_even")).body.variable_usage)
    legacy = _legacy_flags(_legacy_flow(program("is_it_even")))

    assert ir_usage == {"input": "read"}
    assert legacy["input"] == (False, False)


def test_for_target_writes_and_iterable_reads_missed():
    # Pinned: sum_it_up's `i` is written every iteration (for target) but
    # legacy shows read-only — the write went through the deconstruct case
    # flows' interface, bypassing FlowRegister.readin. The iterable's
    # `start`/`end` reads landed on case flows as well and never merged.
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


def test_ref_read_is_legacy_write_is_ir_read():
    # Pinned: reading a Ref-typed variable — here `a.append(1)`, a dynamic
    # capture, with NO assignment anywhere — sets legacy flow_write
    # (FlowRegister.readout on a non-ValueAdapter), while the IR records
    # "read". 2b must not treat Ref-cell reads as writes when sourcing
    # wiring decisions from the IR, or Ref contexts will extend fresh
    # cells where legacy read through the shared one.
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

    ir, sink = build_ir_for(source)
    assert not sink.diagnostics
    legacy = _legacy_flags(_legacy_flow(ns["append_to"]))

    assert dict(ir.body.variable_usage) == {"a": "read"}
    assert legacy["a"] == (False, True)

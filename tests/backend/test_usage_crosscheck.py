"""IrBody.variable_usage semantics, pinned directly (§7.2b / §1).

Semantics (§1): usage classifies the EFFECT on the variable's cell, not
the syntactic position — "read" = independent copy (cell identity
preserved), "write" = linear use that advances the cell. Classification
is adapter-declared via adapters.read_independently over the wiring
type; deliberately VALUE-leaf-only while Par reads linearize (§6).

Historical note (pre-cutover): this suite cross-checked the IR against
the legacy compiler's flow_map flags, which were NOT a complete usage
analysis — if/while tests evaluated in a never-merged `new_test()` flow,
for-target writes bypassed FlowRegister.readin via the deconstruct case
flows, iterable captures landed on case flows. The IR adopted the
complete analysis (including Ref/Inverse read-as-write as intended
semantics: linear reads ARE writes), agreed with legacy everywhere
legacy was complete, and the differential harness was retired with the
legacy compiler at the cutover (CUTOVER.md §4). What remains are the
IR usage pins themselves.
"""

import linecache
import textwrap

from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.source import extract_source
from natsune.frontend.symbols import collect_symbols
from natsune.special_forms import Par, Ref
from tests.frontend.helpers import build_ir_for
from tests.frontend.programs import PROGRAMS


def program(name: str):
    return next(f for f in PROGRAMS if f.__name__ == name)


def _build_ir(func):
    """The genuine pipeline path, as in test_snapshots.compile_program."""
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


def test_if_test_reads_are_recorded():
    # is_it_even's entire predicate is a read of `input` — recorded in
    # the IR usage even though the test evaluates inside the branch
    # machinery (the pre-cutover legacy flow_map never saw it).
    ir_usage = dict(_build_ir(program("is_it_even")).body.variable_usage)

    assert ir_usage == {"input": "read"}


def test_for_target_writes_and_iterable_reads():
    # sum_it_up: `i` is written every iteration (for target) and
    # `total` accumulates; start/end are read once as the iterable's
    # bounds. (The pre-cutover legacy flow_map flagged `i` read-only —
    # the write went through the deconstruct case flows' interface — and
    # missed the iterable reads entirely.)
    ir_usage = dict(_build_ir(program("sum_it_up")).body.variable_usage)

    assert ir_usage == {
        "start": "read",
        "end": "read",
        "total": "write",
        "i": "write",
    }


def test_ref_read_normalizes_to_write():
    # Reading a Ref-typed variable — here `a.append(1)`, a dynamic
    # capture, with NO assignment anywhere — is a linear use: the IR
    # normalizes it to "write" (adapters.read_independently refuses on
    # REFERENCE wiring).
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

    assert dict(ir.body.variable_usage) == {"a": "write"}


def test_par_reads_linearize():
    # Par is the conservative edge (§6): even an all-copyable Par
    # linearizes on read, because reads go through a readout of the whole
    # Par with the neighbor elements closed. Post-cutover,
    # element-pass-through reads relax this to the recursive discipline
    # rule; this pin (and the doc marker) is what that improvement
    # updates.
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

    assert dict(ir.body.variable_usage) == {"p": "write"}

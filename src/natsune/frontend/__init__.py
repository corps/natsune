"""The new compiler frontend (COMPILER_REFACTOR.md, phases 0–6).

Plain data in, plain data out. Nothing here imports `natsune.compiler`, and
nothing creates registers, wires, or flows. The old `natsune.compiler` module
stays untouched and remains the live implementation until the paused cutover
(phases 7–9).
"""

from natsune.frontend.diagnostics import (
    CompileDiagnostic,
    DiagnosticSink,
    Position,
    Severity,
    SourceMap,
)
from natsune.frontend.link import (
    EvalResult,
    EvaluatedValue,
    EvaluationFailure,
    LinkedInet,
    LinkedValue,
    LinkNotFound,
    LinkResult,
    eval_annotation,
    eval_compile_time,
    link_name,
)
from natsune.frontend.signature import Signature, analyze_signature
from natsune.frontend.source import FunctionSource, extract_source

__all__ = [
    "CompileDiagnostic",
    "DiagnosticSink",
    "EvalResult",
    "EvaluatedValue",
    "EvaluationFailure",
    "FunctionSource",
    "LinkedInet",
    "LinkNotFound",
    "LinkResult",
    "LinkedValue",
    "Position",
    "Severity",
    "Signature",
    "SourceMap",
    "eval_annotation",
    "eval_compile_time",
    "analyze_signature",
    "extract_source",
    "link_name",
]

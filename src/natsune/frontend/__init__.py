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

__all__ = [
    "CompileDiagnostic",
    "DiagnosticSink",
    "Position",
    "Severity",
    "SourceMap",
]

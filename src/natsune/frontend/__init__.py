"""The new compiler frontend.

Plain data in, plain data out. Nothing here imports `natsune.compiler`, and
nothing creates registers, wires, or flows. The old `natsune.compiler` module
stays untouched and remains the live implementation until the backend/cutover
phase (see COMPILER_REFACTOR.md).
"""

from natsune.frontend.diagnostics import (
    CompileDiagnostic,
    DiagnosticSink,
    Position,
    SourceMap,
)
from natsune.frontend.infer import (
    ParSubscriptError,
    ParSubscriptResult,
    infer_adapter,
    par_subscript_index,
)
from natsune.frontend.ir import IrFunction, build_ir, render_function
from natsune.frontend.link import (
    EvalResult,
    EvaluatedValue,
    EvaluationFailure,
    LinkedInet,
    LinkedValue,
    LinkNotFound,
    LinkResult,
    collect_call_links,
    eval_annotation,
    eval_compile_time,
    link_name,
)
from natsune.frontend.signature import Signature, analyze_signature
from natsune.frontend.source import FunctionSource, extract_source
from natsune.frontend.symbols import SymbolsTable, collect_symbols
from natsune.frontend.unsupported import UNSUPPORTED_EXPR, UNSUPPORTED_STMT

__all__ = [
    "CompileDiagnostic",
    "DiagnosticSink",
    "EvalResult",
    "EvaluatedValue",
    "EvaluationFailure",
    "FunctionSource",
    "IrFunction",
    "LinkedInet",
    "LinkNotFound",
    "LinkResult",
    "LinkedValue",
    "Position",
    "Signature",
    "SourceMap",
    "SymbolsTable",
    "UNSUPPORTED_EXPR",
    "UNSUPPORTED_STMT",
    "eval_annotation",
    "eval_compile_time",
    "analyze_signature",
    "build_ir",
    "collect_call_links",
    "collect_symbols",
    "extract_source",
    "infer_adapter",
    "link_name",
    "render_function",
]

"""Phase 2 — signature analysis.

Absorbs the old compiler's `args`, `return_annot` and `args_adapter` cached
properties into one explicit pass producing a frozen `Signature`.

Behavior notes and divergences (see COMPILER_REFACTOR.md §10):

- The old rejection raised one positionless `SyntaxError`; here each rejected
  feature becomes a positioned diagnostic and analysis continues (best-effort
  `Signature`) — the phase boundary decides to raise. The `kw_defaults` check
  of the old code is subsumed by the kw-only check (every `kw_defaults` entry
  pairs with a kw-only arg).
- Positional defaults, which the old check silently accepted (and ignored —
  a defaulted param can't be supplied through the net interface), are now
  rejected (§10 row 8).
- Positional-only parameters, which the old code silently *dropped* from
  `args` (wrong arity), are now rejected (§10 row 9).
- `get_type_hints` failures (e.g. unresolvable annotation names — raw
  NameError in the old code) become positioned diagnostics; analysis
  continues with the unresolvable annotations treated as absent.

Signature annotations are resolved with `get_type_hints`, deliberately NOT
phase-3 `eval_annotation`: the old signature path used `get_type_hints`, and
the plan's build order has signature analysis precede linking.

None of this imports `natsune.compiler`.
"""

import ast
import dataclasses
from typing import get_type_hints

from natsune.adapters import Adapter, ParValueAdapter, TypeExpression, adapter_from_type
from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.source import FunctionSource


@dataclasses.dataclass(frozen=True, slots=True)
class Signature:
    """The function's interface: positional params plus resolved adapters."""

    args: tuple[tuple[str, TypeExpression | None], ...]
    return_type: TypeExpression | None
    args_adapter: ParValueAdapter
    return_adapter: Adapter

    @property
    def arity(self) -> int:
        return len(self.args)


def analyze_signature(source: FunctionSource, sink: DiagnosticSink) -> Signature:
    """Analyze `source`'s signature, reporting rejections via `sink`."""
    func_def = source.func_def
    arguments = func_def.args
    source_map = source.source_map

    def error_at(message: str, node: ast.AST) -> None:
        # Caller-owned fallback (phase 0 decision): the FunctionDef is the
        # parent context available at this callsite.
        position = source_map.resolve(node) or source_map.resolve(func_def)
        assert position is not None  # parsed nodes always carry locations
        sink.add_at(message, source_map, position)

    if arguments.vararg is not None:
        error_at("*args is not supported in inet functions", arguments.vararg)
    if arguments.kwarg is not None:
        error_at("**kwargs is not supported in inet functions", arguments.kwarg)
    if arguments.kwonlyargs:
        error_at(
            "Keyword-only arguments are not supported in inet functions",
            arguments.kwonlyargs[0],
        )
    if arguments.posonlyargs:
        error_at(
            "Positional-only parameters are not supported in inet functions",
            arguments.posonlyargs[0],
        )
    if arguments.defaults:
        first_default = len(arguments.args) - len(arguments.defaults)
        for arg in arguments.args[first_default:]:
            error_at("Default values are not supported in inet functions", arg)

    try:
        hints = get_type_hints(source.func)
    except Exception as e:
        error_at(f"Could not resolve annotations: {e}", func_def)
        hints = {}

    args = tuple((arg.arg, hints.get(arg.arg)) for arg in arguments.args)
    return_type = hints.get("return")
    return Signature(
        args=args,
        return_type=return_type,
        args_adapter=ParValueAdapter([adapter_from_type(te) for _, te in args]),
        return_adapter=adapter_from_type(return_type),
    )

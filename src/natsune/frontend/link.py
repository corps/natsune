"""Phase 3 — linking.

New concept centralizing the three scattered compile-time `eval` sites of the
old compiler:

- `lookup_inet` (call-name resolution)            -> `link_name`
- annotation eval in `visit_AnnAssign`           -> `eval_annotation`
- handler-type eval in `parse_try_handler`       -> `eval_compile_time`
  (with an explicit description)

All three `eval` against the function's live globals; here they share one
engine and one consistent error format (`EvaluationFailure.message`), instead
of three ad-hoc message strings and two different raise shapes.

Division of responsibility: deciding *whether* a name should be resolved as a
global at all — the old `used_as_globals` gate in `lookup_inet` — stays with
the symbols phase and the IR builder. `link_name` only resolves.

Recursion note (COMPILER_REFACTOR.md §3.2): linking another inet function
reaches the *old* compiler object behind `__inet__` for now. `LinkedInet.ref`
stores it opaquely, and the only reads of it happen here, once, to copy
metadata (arity, arg adapters, return adapter) into plain data — so IR
construction never touches old internals (§11). At cutover, `ref` becomes a
`CompiledFunction` and the copied fields stay as they are.

None of this imports `natsune.compiler`.
"""

import ast
import dataclasses
from typing import Any

from natsune.adapters import Adapter, adapter_from_type

# The object behind `__inet__`: the old `InetFunctionCompiler` until cutover.
# Opaque by design — do not call methods on it outside `link.py`/`from_ref`.
OpaqueInet = Any


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedInet:
    """The name resolved to an inet function; metadata copied at link time."""

    ref: OpaqueInet
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter

    @classmethod
    def from_ref(cls, ref: OpaqueInet) -> LinkedInet:
        """Copy the metadata the frontend needs off the opaque compiler ref.

        Reads only `args_adapter` and `return_annot` — the documented shape of
        the current compiler object. Assumes the old-compiler shape for now.
        """
        arg_adapters = tuple(ref.args_adapter.concurrent_items)
        return cls(
            ref=ref,
            arity=len(arg_adapters),
            arg_adapters=arg_adapters,
            return_adapter=adapter_from_type(ref.return_annot),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedValue:
    """The name resolved to a plain (non-inet) value."""

    value: Any


@dataclasses.dataclass(frozen=True, slots=True)
class LinkNotFound:
    """The name could not be resolved against globals.

    `reason` is None for plain absence (NameError) and the stringified
    exception when evaluation itself failed some other way. Callers decide
    whether that is a diagnostic (old `lookup_inet` raised
    "Could not evaluate global function invocation" here) or a reason to
    fall back to the dynamic path.
    """

    reason: str | None = None


LinkResult = LinkedInet | LinkedValue | LinkNotFound


def link_name(name: str, globals: dict[str, Any]) -> LinkResult:
    """Resolve a bare name against compile-time globals.

    Absorbs `lookup_inet`'s eval. Note that `eval` falls back to builtins for
    names absent from `globals` (old behavior preserved: a call to e.g. `len`
    links to the builtin as a plain value).
    """
    try:
        value = eval(name, globals)  # noqa: S307 — compile-time linking by design
    except NameError:
        return LinkNotFound()
    except Exception as e:
        return LinkNotFound(reason=str(e))
    if hasattr(value, "__inet__"):
        return LinkedInet.from_ref(getattr(value, "__inet__"))
    return LinkedValue(value)


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluatedValue:
    """A successful compile-time evaluation of an annotation-like expression."""

    value: Any
    source_text: str


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """A failed compile-time evaluation, in the one consistent error format.

    `message` is the single template used for every compile-time eval failure
    in the pipeline (annotations, exception-handler types, ...), replacing the
    old ad-hoc strings ("Could not evaluate annotation in inet", "Could not
    resolve exception handler: ..."). Callers turn this into a positioned
    diagnostic via the phase-0 sink.
    """

    description: str
    source_text: str
    reason: str

    @property
    def message(self) -> str:
        return (
            f"Could not evaluate {self.description}: {self.source_text} ({self.reason})"
        )


EvalResult = EvaluatedValue | EvaluationFailure


def eval_compile_time(
    expr: ast.expr, globals: dict[str, Any], description: str
) -> EvalResult:
    """Evaluate a compile-time expression (e.g. an exception-handler type)."""
    source_text = ast.unparse(expr)
    try:
        value = eval(
            source_text, globals
        )  # noqa: S307 — compile-time linking by design
    except Exception as e:
        return EvaluationFailure(
            description=description, source_text=source_text, reason=str(e)
        )
    return EvaluatedValue(value=value, source_text=source_text)


def eval_annotation(expr: ast.expr, globals: dict[str, Any]) -> EvalResult:
    """Evaluate an annotation expression (absorbs `visit_AnnAssign`'s eval).

    The caller converts the result's adapter (`adapter_from_type`) and turns
    `EvaluationFailure.message` into a positioned diagnostic.
    """
    return eval_compile_time(expr, globals, "annotation")

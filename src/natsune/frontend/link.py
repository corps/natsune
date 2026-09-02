import ast
import dataclasses
from collections.abc import Iterable
from typing import Any

from natsune.adapters import Adapter, adapter_from_type

# The object behind `__inet__`: the old `InetFunctionCompiler` until cutover.
# Opaque by design — do not call methods on it outside `link.py`/`from_ref`.
OpaqueInet = Any


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedInet:
    ref: OpaqueInet
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter

    @classmethod
    def from_ref(cls, ref: OpaqueInet) -> LinkedInet:
        arg_adapters = tuple(ref.args_adapter.concurrent_items)
        return cls(
            ref=ref,
            arity=len(arg_adapters),
            arg_adapters=arg_adapters,
            return_adapter=adapter_from_type(ref.return_annot),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedValue:
    value: Any


@dataclasses.dataclass(frozen=True, slots=True)
class LinkNotFound:
    reason: str | None = None


LinkResult = LinkedInet | LinkedValue | LinkNotFound


def collect_call_links(
    body: Iterable[ast.stmt], globals: dict[str, Any]
) -> dict[str, LinkResult]:
    links: dict[str, LinkResult] = {}
    module = ast.Module(body=list(body), type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            links.setdefault(node.func.id, link_name(node.func.id, globals))
    return links


def link_name(name: str, globals: dict[str, Any]) -> LinkResult:
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
    value: Any
    source_text: str


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationFailure:
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
    return eval_compile_time(expr, globals, "annotation")

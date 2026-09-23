import ast
import dataclasses
from collections.abc import Iterable
from typing import Any

from natsune.first_order.adapters import Adapter
from natsune.frontend.inet_ref import InetRef


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedInet:
    ref: InetRef
    arity: int
    arg_adapters: tuple[Adapter, ...]
    return_adapter: Adapter

    @classmethod
    def from_ref(cls, ref: InetRef) -> LinkedInet:
        arg_adapters = tuple(ref.args_adapter.concurrent_items)
        return cls(
            ref=ref,
            arity=len(arg_adapters),
            arg_adapters=arg_adapters,
            return_adapter=ref.return_adapter,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LinkedValue:
    value: Any


@dataclasses.dataclass(frozen=True, slots=True)
class LinkNotFound:
    reason: str | None = None


LinkResult = LinkedInet | LinkedValue | LinkNotFound


def collect_called_names(body: Iterable[ast.stmt]) -> list[str]:
    """Called names, in first-appearance walk order (deterministic: the
    driver's callee pass runs in this order and links dict order follows
    it)."""
    module = ast.Module(body=list(body), type_ignores=[])
    names: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in names:
                names.append(node.func.id)
    return names


def collect_call_links(
    body: Iterable[ast.stmt], globals: dict[str, Any]
) -> dict[str, LinkResult]:
    links: dict[str, LinkResult] = {}
    for name in collect_called_names(body):
        links.setdefault(name, link_name(name, globals))
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

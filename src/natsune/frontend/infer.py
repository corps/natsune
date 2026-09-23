import ast
import dataclasses
from collections.abc import Mapping

from natsune.first_order.adapters import VA, Adapter, ParValueAdapter
from natsune.frontend.link import LinkedInet, LinkResult


@dataclasses.dataclass(frozen=True, slots=True)
class ParSubscriptError:
    message: str


ParSubscriptResult = int | ParSubscriptError


def par_subscript_index(
    base: ParValueAdapter, slice_expr: ast.expr
) -> ParSubscriptResult:
    if not isinstance(slice_expr, ast.Constant) or not isinstance(
        slice_expr.value, int
    ):
        return ParSubscriptError("Subscript index of Par must be a constant integer")
    if slice_expr.value < 0 or slice_expr.value >= len(base.concurrent_items):
        return ParSubscriptError(
            "Subscript index of Par must be in range [0, par_size)"
        )
    return slice_expr.value


def infer_adapter(
    node: ast.expr,
    variables: Mapping[str, Adapter],
    links: Mapping[str, LinkResult],
) -> Adapter:
    match node:
        case ast.Call():
            if isinstance(node.func, ast.Name):
                result = links.get(node.func.id)
                if isinstance(result, LinkedInet):
                    return result.return_adapter
            return VA
        case ast.Name():
            return variables.get(node.id, VA)
        case ast.Tuple():
            return ParValueAdapter(
                [infer_adapter(element, variables, links) for element in node.elts]
            )
        case ast.Subscript():
            base = infer_adapter(node.value, variables, links)
            if isinstance(base, ParValueAdapter):
                index = par_subscript_index(base, node.slice)
                if isinstance(index, int):
                    return base.concurrent_items[index]
            return VA
        case _:
            return VA

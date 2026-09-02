"""Phase 5 — adapter inference.

Absorbs the old compiler's `infer_expression_adapter`, `evaluate_call_adapter`
and `evaluate_subscript` as flat, pure functions.

- `infer_adapter(node, variables, links)` — one `match` over the node kinds
  the old code specialized (Call / Name / Tuple / Subscript), default VA.
  Deviations from the plan's written `(node, symbols, links)`: only
  `symbols.variables` is ever read, so the mapping is taken directly (usable
  mid-collection in phase 4 and from IR construction alike), and `links` is a
  precomputed name → `LinkResult` mapping — pure by design, no compile-time
  `eval` during inference; callers pre-resolve with phase-3
  `collect_call_links`.
- Invalid Par subscripts yield VA silently here BY CONTRACT. The diagnostic
  decision belongs to IR construction; `par_subscript_index` distinguishes
  the old compiler's two failure modes via `ParSubscriptError` (carrying the
  exact old messages), which phase 6's builder handles by landing them in
  the sink (§3.3, §10 rows 5 and 13).
- Unlike the old `variables[node.id]` indexing (raw KeyError, §10 row 10),
  unknown names yield VA.

None of this imports `natsune.compiler`.
"""

import ast
import dataclasses
from collections.abc import Mapping

from natsune.adapters import VA, Adapter, ParValueAdapter
from natsune.frontend.link import LinkedInet, LinkResult


@dataclasses.dataclass(frozen=True, slots=True)
class ParSubscriptError:
    """Why a Par subscript could not be resolved (the old compiler's message).

    Reintroduced in phase 6: the IR builder is the handler that was missing
    when this was first sketched — it lands `message` in the diagnostics sink.
    """

    message: str


ParSubscriptResult = int | ParSubscriptError


def par_subscript_index(
    base: ParValueAdapter, slice_expr: ast.expr
) -> ParSubscriptResult:
    """Absorbs `evaluate_subscript`: constant-int requirement + range check.

    Quirks absorbed exactly: negative literals (`p[-1]`) are UnaryOp, hence
    "not constant" (§10 row 13); `True` is an `int` subclass and indexes
    element 1.
    """
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
    """Infer the adapter of an expression (old `infer_expression_adapter`)."""
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

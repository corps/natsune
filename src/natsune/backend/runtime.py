"""Pure runtime helpers executed inside compiled nets: the dynamic exec
fallback's eval/exec context builders (CUTOVER.md §1) and the exception
split for future try-star lowering. Everything here runs as an agent
function through `merge_invocation`/`split_invocation` — plain data in,
plain data out."""

from typing import Any


def construct_locals(locals_values: tuple, locals_keys: tuple) -> dict:
    return dict(zip(locals_keys, locals_values))


def exec_expression(expr_str: str, context: tuple[dict, dict]) -> None:
    globals, locals = context
    exec(expr_str, globals=globals, locals=locals)


def eval_expression(expr_str: str, context: tuple[dict, dict]) -> Any:
    globals, locals = context
    return eval(expr_str, globals=globals, locals=locals)


def match_exception_group(
    exceptions: list[Exception], handler_group: tuple | type | None
) -> tuple[list, list]:
    """Split raised exceptions against a try-star handler's type bundle.

    Parked until IrTry lowering lands; carried over verbatim from the
    legacy compiler so the future lowering keeps the same semantics.
    """
    if handler_group is None:
        return exceptions, []

    matches: list[Exception] = []
    remaining = [*exceptions]

    while remaining:
        e = remaining.pop(0)
        if isinstance(e, ExceptionGroup):
            matched_group, remaining_group = e.split(handler_group)
            if matched_group:
                matches.extend(matched_group.exceptions)
            if remaining_group:
                remaining.extend(remaining_group.exceptions)
        else:
            if isinstance(e, handler_group):
                matches.append(e)

    return matches, remaining

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from natsune.frontend.diagnostics import (
    CompileDiagnostic,
    Position,
    SourceMap,
)


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionSource:
    func: Callable[..., Any]
    filename: str
    base_lineno: int
    module: ast.Module
    func_def: ast.FunctionDef
    globals: dict[str, Any]

    @property
    def source_map(self) -> SourceMap:
        return SourceMap(filename=self.filename, base_lineno=self.base_lineno)


def extract_source(
    func: Callable[..., Any], *, globals: dict[str, Any], filename: str
) -> FunctionSource:
    lines, base_lineno = inspect.getsourcelines(func)
    source_text = textwrap.dedent("".join(lines))
    module = ast.parse(source_text)
    source_map = SourceMap(filename, base_lineno)

    if not module.body:
        raise CompileDiagnostic.at_position(
            "Extracted source defines no statements",
            source_map,
            Position(base_lineno, 0),
        ).raised()

    first = module.body[0]
    if not isinstance(first, ast.FunctionDef):
        message = f"Expected a function definition, got {type(first).__name__}"
        diagnostic = CompileDiagnostic.at(message, first, source_map) or (
            CompileDiagnostic.at_position(message, source_map, Position(base_lineno, 0))
        )
        raise diagnostic.raised()

    return FunctionSource(
        func=func,
        filename=filename,
        base_lineno=base_lineno,
        module=module,
        func_def=first,
        globals=globals,
    )

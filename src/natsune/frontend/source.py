"""Phase 1 — source extraction.

Absorbs the `InetFunctionCompiler.func_def` cached property
(`inspect.getsourcelines` + `ast.parse` + lineno capture) into an explicit,
eager step producing a plain frozen data class.

Characterized `inspect.getsourcelines` behavior this module relies on
(verified on CPython 3.14; see tests/frontend/test_source.py):

- `base_lineno` always equals `func.__code__.co_firstlineno`.
- For decorated functions, `co_firstlineno` points at the *first decorator*
  line, so the extracted block starts with the decorator lines and the parsed
  `FunctionDef` has `decorator_list` populated. `base_lineno` therefore points
  at the first decorator, not the `def`.
- The extracted block is NOT dedented by `inspect`; this module dedents it, so
  methods and nested functions work. The old compiler crashed on any indented
  definition (`ast.parse` of an indented block raises IndentationError) — a
  deliberate divergence, logged in COMPILER_REFACTOR.md §10.
- Consequence of dedenting: column offsets in the parsed AST are relative to
  the dedented text, so for indented definitions resolved columns understate
  the true file column by the stripped margin. (The old column-passthrough
  behavior is unaffected for module-level definitions, where nothing is
  stripped.)

`extract_source` takes the function, its globals dict, and the filename all
explicitly: the `inet` decorator's frame-walking (`sys._getframe` for the
caller's `__file__`) stays in the caller and out of this module
(COMPILER_REFACTOR.md §11, "globals identity").

Nothing here imports `natsune.compiler`.
"""

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
    """Everything the later phases need about where a function came from.

    `globals` is stored by reference: it is the function's live globals dict
    (the one compile-time `eval` of annotations and links will use in phases
    2–3), not a copy.
    """

    func: Callable[..., Any]
    filename: str
    base_lineno: int
    module: ast.Module
    func_def: ast.FunctionDef
    globals: dict[str, Any]

    @property
    def source_map(self) -> SourceMap:
        """The phase-0 position mapper for this function's extracted AST."""
        return SourceMap(filename=self.filename, base_lineno=self.base_lineno)


def extract_source(
    func: Callable[..., Any], *, globals: dict[str, Any], filename: str
) -> FunctionSource:
    """Extract and parse the source block that defines `func`.

    Raises immediately (via the phase-0 diagnostic shape) if the extracted
    block does not start with a function definition. Extraction is an input
    boundary, not an analysis phase: there is no meaningful continuation
    without a `FunctionDef`, so this replaces the old bare `assert` rather
    than accumulating a diagnostic. Position fallbacks are supplied here at
    this callsite (the start of the extracted block), per the phase-0
    no-up-front-fallback design.
    """
    lines, base_lineno = inspect.getsourcelines(func)
    source_text = textwrap.dedent("".join(lines))
    module = ast.parse(source_text)
    source_map = SourceMap(filename, base_lineno)

    if not module.body:
        # Caller-owned fallback: with no statements there is no node to point
        # at, so report the start of the extracted block, in file coordinates.
        raise CompileDiagnostic.at_position(
            "Extracted source defines no statements",
            source_map,
            Position(base_lineno, 0),
        ).raised()

    first = module.body[0]
    if not isinstance(first, ast.FunctionDef):
        message = f"Expected a function definition, got {type(first).__name__}"
        # `first` comes from ast.parse and always carries a location; the
        # at_position fallback is defense for exotic hand-built ASTs.
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

"""Phase 0 — diagnostics.

Absorbs `InetFunctionCompiler.syntax_error` and the scattered
`raise SyntaxError(...)` sites of the old compiler into one explicit
mechanism:

- `SourceMap` owns the position bookkeeping, including the base-lineno offset
  arithmetic that used to live inline (`lineno + self.lineno - 1`).
- `CompileDiagnostic` is a plain record of one program error or warning.
- `DiagnosticSink` accumulates diagnostics; analysis phases add to it and
  phase boundaries decide whether to raise or continue. An immediate-raise
  path (`DiagnosticSink.fail`) is kept for states the validated-by-construction
  IR should make unreachable — defense, not policy.

Nothing here imports `natsune.compiler`.
"""

import ast
import dataclasses
from collections.abc import Sequence
from enum import Enum
from typing import NoReturn


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclasses.dataclass(frozen=True, slots=True)
class Position:
    """A position in the original source file."""

    lineno: int
    col_offset: int


@dataclasses.dataclass(frozen=True, slots=True)
class SourceMap:
    """Translates positions in an extracted function AST to file positions.

    `inspect.getsourcelines` returns a *dedented* snippet whose AST numbers
    lines from 1; `base_lineno` is the original-file line the snippet's first
    line corresponds to. Resolution applies `lineno + base_lineno - 1` — the
    exact arithmetic the old `syntax_error` performed inline — to nodes and to
    the fallback alike. Column offsets pass through untouched, pinning current
    behavior (they can be off for indented definitions; deliberately
    unchanged, see COMPILER_REFACTOR.md §11).
    """

    filename: str
    base_lineno: int
    fallback: Position

    def resolve(self, node: ast.AST) -> Position:
        """Resolve `node`'s position, falling back for locationless nodes.

        Mirrors the old `hasattr(node, "lineno")` checks: some AST helpers
        (e.g. `ast.Load`) carry no location info.
        """
        lineno = getattr(node, "lineno", self.fallback.lineno)
        col_offset = getattr(node, "col_offset", self.fallback.col_offset)
        return Position(lineno + self.base_lineno - 1, col_offset)


@dataclasses.dataclass(frozen=True, slots=True)
class CompileDiagnostic:
    """One program-level diagnostic with its resolved source position."""

    message: str
    filename: str
    lineno: int
    col_offset: int
    severity: Severity = Severity.ERROR

    @classmethod
    def at(
        cls,
        message: str,
        node: ast.AST,
        source: SourceMap,
        severity: Severity = Severity.ERROR,
    ) -> CompileDiagnostic:
        """Build a diagnostic for `node`, resolving position via `source`."""
        position = source.resolve(node)
        return cls(
            message=message,
            filename=source.filename,
            lineno=position.lineno,
            col_offset=position.col_offset,
            severity=severity,
        )

    def raised(self) -> SyntaxError:
        """The exact `SyntaxError` shape the old compiler raised.

        Kept for cutover parity: the `(message, (filename, lineno, col_offset,
        None))` form, no source text.
        """
        return SyntaxError(
            self.message, (self.filename, self.lineno, self.col_offset, None)
        )

    def __str__(self) -> str:
        return (
            f"{self.filename}:{self.lineno}:{self.col_offset}"
            f": {self.severity.value}: {self.message}"
        )


class DiagnosticSink:
    """Accumulates `CompileDiagnostic`s.

    Policy: analysis phases accumulate into the sink; phase boundaries decide
    to raise (`raise_if_errors`) or continue. Insertion order is preserved, so
    the first error is deterministic and matches the one the old compiler
    would have raised first.
    """

    def __init__(self) -> None:
        self._diagnostics: list[CompileDiagnostic] = []

    def add(self, diagnostic: CompileDiagnostic) -> None:
        self._diagnostics.append(diagnostic)

    def error(
        self, message: str, node: ast.AST, source: SourceMap
    ) -> CompileDiagnostic:
        diagnostic = CompileDiagnostic.at(message, node, source, Severity.ERROR)
        self.add(diagnostic)
        return diagnostic

    def warning(
        self, message: str, node: ast.AST, source: SourceMap
    ) -> CompileDiagnostic:
        diagnostic = CompileDiagnostic.at(message, node, source, Severity.WARNING)
        self.add(diagnostic)
        return diagnostic

    def fail(self, message: str, node: ast.AST, source: SourceMap) -> NoReturn:
        """Raise immediately instead of accumulating.

        For states that the validated-by-construction pipeline should make
        unreachable. Nothing is recorded: there is no continuation after a
        defense-path failure.
        """
        raise CompileDiagnostic.at(message, node, source).raised()

    @property
    def diagnostics(self) -> Sequence[CompileDiagnostic]:
        return tuple(self._diagnostics)

    @property
    def errors(self) -> Sequence[CompileDiagnostic]:
        return tuple(d for d in self._diagnostics if d.severity is Severity.ERROR)

    @property
    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self._diagnostics)

    def raise_if_errors(self) -> None:
        """Phase-boundary policy hook: raise the first error, if any.

        Warnings never block. The raised exception is the first error's
        `raised()` shape, identical to what the old compiler threw.
        """
        for diagnostic in self._diagnostics:
            if diagnostic.severity is Severity.ERROR:
                raise diagnostic.raised()

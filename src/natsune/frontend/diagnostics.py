"""Phase 0 — diagnostics.

Absorbs `InetFunctionCompiler.syntax_error` and the scattered
`raise SyntaxError(...)` sites of the old compiler into one explicit
mechanism:

- `SourceMap` owns the filename and the base-lineno offset arithmetic that
  used to live inline (`lineno + self.lineno - 1`). It has deliberately no
  fallback position: `resolve` returns None for nodes without location info.
- `CompileDiagnostic` is a plain record of one program error or warning, and
  always carries a concrete position. `at` returns None when the node's
  position can't be determined.
- Fallbacks are therefore the CALLER's responsibility: a callsite that must
  not lose a diagnostic resolves an alternative itself — e.g. the parent AST
  node available at the callsite, or the start of the extracted block — and
  hands the resolved `Position` to `at_position` / `DiagnosticSink.add_at`.
- `DiagnosticSink` accumulates diagnostics; analysis phases add to it and
  phase boundaries decide whether to raise or continue. An immediate-raise
  path (`DiagnosticSink.fail`) is kept for states the validated-by-
  construction IR should make unreachable — defense, not policy.

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

    `inspect.getsourcelines` returns a snippet (dedented by `extract_source`)
    whose AST numbers lines from 1; `base_lineno` is the original-file line
    the snippet's first line corresponds to. Resolution applies
    `lineno + base_lineno - 1` — the exact arithmetic the old `syntax_error`
    performed inline. Column offsets pass through untouched, pinning current
    behavior (they can be off for indented definitions; deliberately
    unchanged, see COMPILER_REFACTOR.md §11).
    """

    filename: str
    base_lineno: int

    def resolve(self, node: ast.AST) -> Position | None:
        """Resolve `node`'s position, or None if it carries no location.

        All-or-nothing: every node produced by `ast.parse` carries both
        `lineno` and `col_offset`; hand-built nodes (e.g. `ast.Load`)
        typically carry neither.
        """
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is None or col_offset is None:
            return None
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
    ) -> CompileDiagnostic | None:
        """Build a diagnostic for `node`, resolving its position via `source`.

        Returns None — not an error — when `node` carries no location info.
        Callsites that must not lose the diagnostic resolve a fallback
        themselves (the parent node at the callsite, the function def, the
        start of the block) and use `at_position` instead.
        """
        position = source.resolve(node)
        if position is None:
            return None
        return cls.at_position(message, source, position, severity)

    @classmethod
    def at_position(
        cls,
        message: str,
        source: SourceMap,
        position: Position,
        severity: Severity = Severity.ERROR,
    ) -> CompileDiagnostic:
        """Build a diagnostic at a caller-determined (file-resolved) position."""
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
        """Record a diagnostic, dropping exact duplicates.

        The full pipeline runs overlapping validations (the collector and the
        IR builder both check, e.g., list lvalues by design — §3.3); an
        identical message at an identical position with identical severity is
        one finding, not two.
        """
        if diagnostic not in self._diagnostics:
            self._diagnostics.append(diagnostic)

    def add_at(
        self,
        message: str,
        source: SourceMap,
        position: Position,
        *,
        severity: Severity = Severity.ERROR,
    ) -> CompileDiagnostic:
        """Record a diagnostic at a caller-determined position.

        For callsites that own a fallback: the position is used verbatim, so
        the diagnostic can never be lost for lack of location info.
        """
        diagnostic = CompileDiagnostic.at_position(message, source, position, severity)
        self.add(diagnostic)
        return diagnostic

    def error(
        self, message: str, node: ast.AST, source: SourceMap
    ) -> CompileDiagnostic | None:
        """Record an error at `node`'s resolved position.

        Returns the recorded diagnostic, or None — recording nothing — when
        `node` carries no location info. Use `add_at` with a caller-resolved
        position when the diagnostic must not be lost.
        """
        diagnostic = CompileDiagnostic.at(message, node, source)
        if diagnostic is not None:
            self.add(diagnostic)
        return diagnostic

    def warning(
        self, message: str, node: ast.AST, source: SourceMap
    ) -> CompileDiagnostic | None:
        """Record a warning at `node`'s resolved position (see `error`)."""
        diagnostic = CompileDiagnostic.at(message, node, source, Severity.WARNING)
        if diagnostic is not None:
            self.add(diagnostic)
        return diagnostic

    def fail(self, message: str, node: ast.AST, source: SourceMap) -> NoReturn:
        """Raise immediately instead of accumulating.

        For states that the validated-by-construction pipeline should make
        unreachable. Nothing is recorded: there is no continuation after a
        defense-path failure. If `node` carries no location info, raises the
        positionless legacy shape `SyntaxError(message)`.
        """
        diagnostic = CompileDiagnostic.at(message, node, source)
        if diagnostic is None:
            raise SyntaxError(message)
        raise diagnostic.raised()

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

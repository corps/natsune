import ast
import dataclasses
from typing import NoReturn


@dataclasses.dataclass(frozen=True, slots=True)
class Position:
    lineno: int
    col_offset: int


@dataclasses.dataclass(frozen=True, slots=True)
class SourceMap:
    filename: str
    base_lineno: int

    def resolve(self, node: ast.AST) -> Position | None:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is None or col_offset is None:
            return None
        return Position(lineno + self.base_lineno - 1, col_offset)


@dataclasses.dataclass(frozen=True, slots=True)
class CompileDiagnostic:
    message: str
    filename: str
    lineno: int
    col_offset: int

    @classmethod
    def at(
        cls,
        message: str,
        node: ast.AST,
        source: SourceMap,
    ) -> CompileDiagnostic | None:
        position = source.resolve(node)
        if position is None:
            return None
        return cls.at_position(message, source, position)

    @classmethod
    def at_position(
        cls,
        message: str,
        source: SourceMap,
        position: Position,
    ) -> CompileDiagnostic:
        return cls(
            message=message,
            filename=source.filename,
            lineno=position.lineno,
            col_offset=position.col_offset,
        )

    def raised(self) -> SyntaxError:
        return SyntaxError(
            self.message, (self.filename, self.lineno, self.col_offset, None)
        )

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}:{self.col_offset}" f": {self.message}"


@dataclasses.dataclass(slots=True, frozen=True)
class DiagnosticSink:
    diagnostics: set[CompileDiagnostic] = dataclasses.field(default_factory=set)

    def add(self, diagnostic: CompileDiagnostic) -> None:
        self.diagnostics.add(diagnostic)

    def add_at(
        self,
        message: str,
        source: SourceMap,
        position: Position,
    ) -> CompileDiagnostic:
        diagnostic = CompileDiagnostic.at_position(message, source, position)
        self.add(diagnostic)
        return diagnostic

    def error(
        self, message: str, node: ast.AST, source: SourceMap
    ) -> CompileDiagnostic | None:
        diagnostic = CompileDiagnostic.at(message, node, source)
        if diagnostic is not None:
            self.add(diagnostic)
        return diagnostic

    def fail(self, message: str, node: ast.AST, source: SourceMap) -> NoReturn:
        diagnostic = CompileDiagnostic.at(message, node, source)
        if diagnostic is None:
            raise SyntaxError(message)
        raise diagnostic.raised()

    def raise_if_errors(self, message: str) -> None:
        if self.diagnostics:
            raise ExceptionGroup(message, [d.raised() for d in self.diagnostics])

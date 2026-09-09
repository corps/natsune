"""Unsupported AST node types for the natsune compiler frontend.

These are the expression and statement types that are not yet supported
by the new compiler frontend. They are absorbed
from the old compiler's module-level tuples and re-validated during IR
construction.

`ast.TryStar` is new: the old collector rejected `try` but silently walked
`try/except*`.
"""

import ast

# Expression types that are not supported in the new frontend.
UNSUPPORTED_EXPR: tuple[type[ast.expr], ...] = (
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Starred,
    ast.Lambda,
    ast.NamedExpr,
)

# Statement types that are not supported in the new frontend.
UNSUPPORTED_STMT: tuple[type[ast.stmt], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Match,
    ast.Assert,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.TryStar,
    ast.TypeAlias,
    ast.Delete,
)

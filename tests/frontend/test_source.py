import ast
import importlib.util
import linecache

import pytest

from natsune.frontend.source import FunctionSource, extract_source
from tests.frontend.helpers import make_source


def _deco(f):
    return f


def _module_level(a: int) -> int:
    return a


@_deco
@_deco
def _decorated(a: int) -> int:
    return a


class _Container:
    def plain_method(self, a: int) -> int:
        return a

    @_deco
    def decorated_method(self, a: int) -> int:
        return a


def _outer():
    def inner(a: int) -> int:
        return a

    return inner


def _file_line(func, lineno: int) -> str:
    """The real line `lineno` (1-indexed) of the file that defines `func`."""
    return linecache.getline(func.__code__.co_filename, lineno).strip()


# --- AST shape and lineno bookkeeping ---------------------------------------


def test_module_level_extraction():
    source = extract_source(
        _module_level, globals=_module_level.__globals__, filename="prog.py"
    )

    assert source.func is _module_level
    assert source.filename == "prog.py"
    assert source.base_lineno == _module_level.__code__.co_firstlineno
    assert source.base_lineno > 1  # defined below the top of this module
    assert source.module.body[0] is source.func_def
    assert source.func_def.name == "_module_level"
    assert source.func_def.decorator_list == []
    assert source.globals is _module_level.__globals__


@pytest.mark.parametrize(
    "func",
    [
        _module_level,
        _decorated,
        _Container.plain_method,
        _Container.decorated_method,
        _outer(),
    ],
)
def test_base_lineno_equals_co_firstlineno(func):
    source = extract_source(func, globals={}, filename="prog.py")
    assert source.base_lineno == func.__code__.co_firstlineno


def test_decorated_function_block_includes_decorators():
    # Pin: co_firstlineno (and so base_lineno) points at the FIRST DECORATOR.
    source = extract_source(_decorated, globals={}, filename="prog.py")

    assert _file_line(_decorated, source.base_lineno).startswith("@_deco")
    assert len(source.func_def.decorator_list) == 2
    assert source.func_def.lineno == 3  # within the snippet: two decorators + def

    # ...so resolving the FunctionDef lands on the true `def` line in the file.
    pos = source.source_map.resolve(source.func_def)
    assert pos is not None
    assert _file_line(_decorated, pos.lineno).startswith("def _decorated")


def test_indented_definitions_are_dedented():
    # The old compiler crashed here: ast.parse of an indented block.
    source = extract_source(_Container.plain_method, globals={}, filename="prog.py")

    assert source.func_def.col_offset == 0  # dedented to the margin

    pos = source.source_map.resolve(source.func_def.body[0])
    assert pos is not None
    assert _file_line(_Container.plain_method, pos.lineno) == "return a"
    # Pin the known imprecision: the snippet column is relative to the
    # dedented text (4), while the true file column is 8.
    assert pos.col_offset == 4


def test_decorated_method_extraction():
    source = extract_source(_Container.decorated_method, globals={}, filename="prog.py")

    assert len(source.func_def.decorator_list) == 1
    assert source.func_def.lineno == 2  # decorator + def, within the snippet
    assert _file_line(_Container.decorated_method, source.base_lineno).startswith(
        "@_deco"
    )

    pos = source.source_map.resolve(source.func_def.body[0])
    assert pos is not None
    assert _file_line(_Container.decorated_method, pos.lineno) == "return a"


def test_nested_function_extraction():
    source = extract_source(_outer(), globals={}, filename="prog.py")

    assert source.func_def.name == "inner"
    pos = source.source_map.resolve(source.func_def.body[0])
    assert pos is not None
    assert pos.lineno > source.base_lineno
    assert _file_line(_outer, pos.lineno) == "return a"


def test_caller_owns_fallback_for_locationless_nodes():
    # SourceMap carries no fallback position: when a node has no location
    # (resolve returns None), the callsite supplies the alternative context —
    # here the FunctionDef node itself.
    source = extract_source(_module_level, globals={}, filename="prog.py")

    assert source.source_map.resolve(ast.Load()) is None
    position = source.source_map.resolve(ast.Load()) or source.source_map.resolve(
        source.func_def
    )
    assert position is not None
    assert _file_line(_module_level, position.lineno).startswith("def _module_level")


def test_globals_stored_by_reference():
    sentinel: dict = {}
    source = extract_source(_module_level, globals=sentinel, filename="prog.py")
    assert source.globals is sentinel


def test_make_source_globals_is_function_globals():
    source = make_source("""
        def f(a: int) -> int:
            return a
        """)
    assert getattr(source.func, "__globals__") is source.globals
    assert "f" in source.globals


def test_rejects_class_definition():
    with pytest.raises(SyntaxError, match="ClassDef") as excinfo:
        extract_source(_Container, globals={}, filename="prog.py")
    # Diagnostic position resolves to the real `class _Container` line.
    exc = excinfo.value
    assert exc.filename == "prog.py"
    assert exc.lineno is not None
    assert _file_line(_module_level, exc.lineno).startswith("class _Container")


def test_rejects_lambda_assignment():
    lam = lambda x: x  # noqa: E731

    with pytest.raises(SyntaxError, match="Assign"):
        extract_source(lam, globals={}, filename="prog.py")


def test_synthetic_module_layout(tmp_path):
    lines = [
        '"""Synthetic module."""',  # 1
        "",  # 2
        "import math",  # 3
        "",  # 4
        "# a comment",  # 5
        "",  # 6
        "class Widget:",  # 7
        "    FACTOR = 2",  # 8
        "",  # 9
        "    def scale(self, x: int) -> int:",  # 10
        "        y = x * self.FACTOR",  # 11
        "        return y",  # 12
        "",  # 13
        "",  # 14
        "def deco(f):",  # 15
        "    return f",  # 16
        "",  # 17
        "",  # 18
        "@deco",  # 19
        "@deco",  # 20
        "def add(a: int, b: int) -> int:",  # 21
        "    return a + b",  # 22
    ]
    path = tmp_path / "synthetic.py"
    path.write_text("\n".join(lines) + "\n")

    spec = importlib.util.spec_from_file_location("synthetic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def line_of(text: str) -> int:
        return lines.index(text) + 1

    # Indented method, after blank lines and a class attribute.
    scale = extract_source(module.Widget.scale, globals={}, filename=str(path))
    assert scale.base_lineno == line_of("    def scale(self, x: int) -> int:")
    ret_pos = scale.source_map.resolve(scale.func_def.body[-1])
    assert ret_pos is not None
    assert ret_pos.lineno == line_of("        return y")
    assert path.read_text().splitlines()[ret_pos.lineno - 1] == "        return y"

    # Decorated module-level function: base points at the first decorator.
    add = extract_source(module.add, globals={}, filename=str(path))
    assert add.base_lineno == line_of("@deco")
    assert add.func_def.lineno == 3  # @deco, @deco, def — within the snippet
    def_pos = add.source_map.resolve(add.func_def)
    assert def_pos is not None
    assert def_pos.lineno == line_of("def add(a: int, b: int) -> int:")
    ret_pos = add.source_map.resolve(add.func_def.body[0])
    assert ret_pos is not None
    assert ret_pos.lineno == line_of("    return a + b")


def test_make_source_basic():
    source = make_source("""
        def f(a: int) -> int:
            return a
        """)
    assert source.func_def.name == "f"
    assert source.base_lineno == 1
    assert source.func_def.body[0].lineno == 2


def test_make_source_namespace_seeding():
    sentinel = object()
    source = make_source(
        """
        def f(a: Target) -> int:
            return a
        """,
        namespace={"Target": sentinel},
    )
    assert source.globals["Target"] is sentinel
    # The annotation was evaluated at def time against the seeded namespace.
    assert source.func.__annotations__["a"] is sentinel


def test_make_source_ambiguous_snippet_requires_name():
    with pytest.raises(ValueError, match="exactly one"):
        make_source("""
            def f():
                return 1

            def g():
                return 2
            """)

    source = make_source(
        """
        def f():
            return 1

        def g():
            return 2
        """,
        name="g",
    )
    assert source.func_def.name == "g"


def test_make_source_filenames_are_unique():
    first = make_source("def f():\n    return 1")
    second = make_source("def f():\n    return 2")
    assert first.filename != second.filename

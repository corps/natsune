"""Phase 3 tests — linking.

Fake globals dicts only — no real compilation, no live net. The plan's named
error paths (absent name, eval raises, object without `__inet__`) all return
plain data; nothing here raises.
"""

import ast
import os
from types import SimpleNamespace

from natsune.adapters import VA, ParValueAdapter, adapter_from_type
from natsune.frontend.link import (
    EvaluatedValue,
    EvaluationFailure,
    LinkedInet,
    LinkedValue,
    LinkNotFound,
    eval_annotation,
    eval_compile_time,
    link_name,
)


class _RaisingGlobals(dict):
    """A globals dict whose lookups fail with something other than NameError."""

    def __getitem__(self, key):
        raise RuntimeError("boom")


def _inet_function(arg_types=(int, str), return_type=int):
    """A stand-in for the old shape: object with `__inet__` -> compiler-like."""
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_annot=return_type,
    )
    return SimpleNamespace(__inet__=compiler), compiler


def _expression(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


# --- link_name ---------------------------------------------------------------


def test_links_plain_value():
    assert link_name("x", {"x": 42}) == LinkedValue(42)


def test_absent_name_is_link_not_found():
    result = link_name("missing", {})
    assert result == LinkNotFound(reason=None)


def test_eval_failure_is_link_not_found_with_reason():
    # The name exists, but resolving it explodes — distinguishable from plain
    # absence via `reason`, so callers can phrase the diagnostic precisely.
    result = link_name("x", _RaisingGlobals())
    assert isinstance(result, LinkNotFound)
    assert result.reason is not None and "boom" in result.reason


def test_object_without_inet_is_plain_value():
    class NotInet:
        pass

    result = link_name("NotInet", {"NotInet": NotInet})
    assert isinstance(result, LinkedValue)
    assert result.value is NotInet


def test_links_inet_function_and_copies_metadata():
    func, compiler = _inet_function()
    result = link_name("f", {"f": func})

    assert result == LinkedInet(
        ref=compiler,
        arity=2,
        arg_adapters=(adapter_from_type(int), adapter_from_type(str)),
        return_adapter=adapter_from_type(int),
    )


def test_inet_return_without_annotation_maps_to_va():
    func, _ = _inet_function(return_type=None)
    result = link_name("f", {"f": func})
    assert isinstance(result, LinkedInet)
    assert result.return_adapter is VA


def test_metadata_is_copied_not_referenced():
    # Mutating the fake compiler's args after linking must not leak into the
    # already-copied tuple (the frontend froze its view at link time).
    compiler_args = [adapter_from_type(int)]
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter(compiler_args), return_annot=int
    )
    func = SimpleNamespace(__inet__=compiler)
    result = link_name("f", {"f": func})
    assert isinstance(result, LinkedInet)

    compiler_args.append(adapter_from_type(str))
    assert result.arg_adapters == (adapter_from_type(int),)
    assert result.arity == 1


def test_names_fall_back_to_builtins_like_the_old_eval():
    # `eval(name, globals)` consults builtins — old `lookup_inet` inherited
    # the same semantics, so a call to e.g. `len` links as a plain value.
    result = link_name("len", {})
    assert isinstance(result, LinkedValue)
    assert result.value is len


# --- eval_annotation / eval_compile_time -------------------------------------


def test_eval_annotation_success():
    result = eval_annotation(_expression("int"), {})
    assert result == EvaluatedValue(value=int, source_text="int")


def test_eval_annotation_subscript_expression():
    result = eval_annotation(_expression("Vec[int]"), {"Vec": list})
    assert isinstance(result, EvaluatedValue)
    assert result.value == list[int]


def test_eval_annotation_failure_uses_consistent_format():
    result = eval_annotation(_expression("Missing"), {})
    assert result == EvaluationFailure(
        description="annotation",
        source_text="Missing",
        reason="name 'Missing' is not defined",
    )
    assert result.message == (
        "Could not evaluate annotation: Missing (name 'Missing' is not defined)"
    )


def test_eval_annotation_never_raises_on_runtime_failure():
    result = eval_annotation(_expression("1 / 0"), {})
    assert isinstance(result, EvaluationFailure)
    assert result.source_text == "1 / 0"
    assert result.reason == "division by zero"
    assert result.message == ("Could not evaluate annotation: 1 / 0 (division by zero)")


def test_eval_compile_time_descriptions_share_one_format():
    # The same engine serves the future exception-handler-type site with a
    # different description; the message template stays uniform.
    result = eval_compile_time(_expression("MissingType"), {}, "exception handler")
    assert isinstance(result, EvaluationFailure)
    assert result.message == (
        "Could not evaluate exception handler: MissingType"
        " (name 'MissingType' is not defined)"
    )


def test_eval_compile_time_success():
    result = eval_compile_time(_expression("int"), {}, "exception handler")
    assert result == EvaluatedValue(value=int, source_text="int")


# --- architecture guard ------------------------------------------------------


def test_frontend_never_imports_the_old_compiler():
    # COMPILER_REFACTOR.md §5: "none of them import from `natsune.compiler`."
    # Checks actual import statements (docstrings mention the name on purpose).
    import natsune.frontend

    frontend_dir = os.path.dirname(natsune.frontend.__file__)
    scanned = 0
    for root, dirs, files in os.walk(frontend_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            scanned += 1
            path = os.path.join(root, filename)
            with open(path) as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    assert module != "natsune.compiler" and not module.startswith(
                        "natsune.compiler."
                    ), f"{filename} imports the old compiler"
    assert scanned >= 3  # the guard itself must actually scan the modules

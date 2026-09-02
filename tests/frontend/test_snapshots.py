"""Snapshot tests over the example programs (COMPILER_REFACTOR.md §8, step 8).

Each program in `tests/frontend/programs.py` is run through the full new
pipeline and its rendered IR — plus any diagnostics — is compared against the
stored snapshot in `tests/frontend/snapshots/<name>.ir`.

Workflows:
- Check:            uv run pytest tests/frontend/test_snapshots.py
- Gather / update:  NATSUNE_UPDATE_SNAPSHOTS=1 uv run pytest tests/frontend/test_snapshots.py

Adding a program: append it to `PROGRAMS`, run the update command, review the
new snapshot by eye (this is the "read the IR" checkpoint), and commit it.
"""

import difflib
import os

import pytest

from natsune.frontend.diagnostics import DiagnosticSink
from natsune.frontend.ir import build_ir, render_function
from natsune.frontend.link import collect_call_links
from natsune.frontend.signature import analyze_signature
from natsune.frontend.symbols import collect_symbols
from tests.frontend.helpers import make_source
from tests.frontend.programs import PROGRAMS, Program

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
UPDATE_ENV_VAR = "NATSUNE_UPDATE_SNAPSHOTS"


def snapshot_path(name: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{name}.ir")


def compile_program(program: Program) -> str:
    """Full pipeline (§9) → snapshot text: diagnostics section + rendered IR.

    The filename is pinned to `<name>.py` so diagnostic positions are stable
    across sessions regardless of the shared snippet counter.
    """
    source = make_source(
        program.snippet,
        namespace=program.full_namespace(),
        filename=f"{program.name}.py",
    )
    signature = analyze_signature(source, DiagnosticSink())
    symbols = collect_symbols(source, signature, DiagnosticSink())
    links = collect_call_links(source.func_def.body, source.globals)
    sink = DiagnosticSink()
    ir = build_ir(source, signature, symbols, links, sink)

    lines = ["# diagnostics"]
    if sink.diagnostics:
        lines.extend(f"- {diagnostic}" for diagnostic in sink.diagnostics)
    else:
        lines.append("# (none)")
    lines.append("")
    lines.append("# ir")
    lines.append(render_function(ir))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda program: program.name)
def test_program_snapshot(program: Program) -> None:
    actual = compile_program(program)
    path = snapshot_path(program.name)

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(actual)
        return

    if not os.path.exists(path):
        pytest.fail(
            f"no snapshot for program {program.name!r}; run with\n"
            f"  {UPDATE_ENV_VAR}=1 uv run pytest tests/frontend/test_snapshots.py\n"
            "to gather it, then review the file by eye before committing."
        )

    with open(path) as f:
        expected = f.read()
    if expected != actual:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"{path} (stored)",
                tofile=f"{program.name} (rendered)",
                lineterm="",
            )
        )
        pytest.fail(f"IR snapshot mismatch for {program.name!r}:\n{diff}")


def test_snapshots_dir_has_no_orphans() -> None:
    expected = {f"{program.name}.ir" for program in PROGRAMS}
    if not os.path.isdir(SNAPSHOT_DIR):
        pytest.fail(
            f"snapshot directory {SNAPSHOT_DIR} does not exist; run with\n"
            f"  {UPDATE_ENV_VAR}=1 uv run pytest tests/frontend/test_snapshots.py"
        )
    actual = {name for name in os.listdir(SNAPSHOT_DIR) if name.endswith(".ir")}
    orphans = actual - expected
    missing = expected - actual
    assert not orphans, (
        f"snapshots without programs (delete them or restore their programs):"
        f" {sorted(orphans)}"
    )
    assert not missing, f"programs without snapshots: {sorted(missing)}"

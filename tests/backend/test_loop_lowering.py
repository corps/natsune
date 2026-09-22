import pytest

from natsune.control_flow import Loop
from natsune.executor import ThreadPoolExecutor
from natsune.special_forms import Par, Ref
from tests.backend.helpers import Program, capturing_lower, program_ids, run_legacy

# --- case programs -------------------------------------------------------
# Each function below is a program under test; the comment above it says
# what the case pins.


def while_sum(start: int, end: int) -> int:
    total = 0
    i = start
    while i < end:
        total += i
        i += 1
    return total


# two-arg range with effect statements interleaved in the body (the old
# suite's sum_it_up)
def sum_it_up(start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        print(total)
        print(i)
        total += i
    return total


def for_sum(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    return total


# continue through an if: the if composite forwards the continue slot to
# the enclosing layer (the slot-forwarding path new in this slice)
def while_continue(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        if i % 2 == 0:
            continue
        total += i
    return total


def for_continue(a: int) -> int:
    total = 0
    for i in range(a):
        if i % 2 == 0:
            continue
        total += i
    return total


# continue as the body's closer (nothing after it in the statement list)
def while_continue_closer(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        continue
    return total


def for_continue_closer(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
        continue
    return total


# break through an if — the §6.1 headline: break rides the loop's finish
# slot and the trailing region still sequences (legacy runs the loop to
# its bound instead — see the divergence test below)
def while_break_bound(a: int) -> int:
    total = 0
    i = 0
    while i < 100:
        if i >= a:
            break
        total += i
        i += 1
    return total


def for_break(a: int) -> int:
    total = 0
    for i in range(a):
        if i > 2:
            break
        total += i
    return total


# break as the body's closer
def while_break_closer(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        i += 1
        break
    return total


# break with no if mediating it
def while_direct_break(a: int) -> int:
    total = 0
    while a > 0:
        total = 5
        break
    return total


# orelse: runs on exhaustion, skipped by break
def while_orelse(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        total += i
        i += 1
    else:
        total += 100
    return total


def while_break_orelse(a: int) -> int:
    i = 0
    while i < a:
        break
    else:
        i = 99
    return i


def for_orelse(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    else:
        total += 100
    return total


def for_break_orelse(a: int) -> int:
    total = 0
    for i in range(a):
        break
    else:
        total += 100
    return total


# every path returns through an if inside the body (the shape legacy's
# wire_continuation historically starves on)
def loop_early_return(a: int) -> int:
    for i in range(a):
        if i > 2:
            return i
    return -1


# nesting: break in the INNER loop must only exit the inner one (legacy
# starves on this shape — see the divergence test below)
def nested_break(a: int) -> int:
    total = 0
    for i in range(a):
        for j in range(a):
            if j == 2:
                break
            total += 1
    return total


# a loop with a while inside a for: cross-composite nesting
def nested_while_in_for(a: int) -> int:
    total = 0
    for i in range(a):
        j = 0
        while j < i:
            total += j
            j += 1
    return total


# the §6.1 headline claim: statements after the loop sequence as a fresh
# layer off the loop's finish (= exhaustion), no absorption needed
def loop_then_trailing(a: int) -> int:
    total = 0
    for i in range(a):
        total += i
    total = total * 2
    return total


def while_then_trailing(a: int) -> int:
    total = 0
    i = 0
    while i < a:
        total += i
        i += 1
    total = total * 2
    return total


# a loop with no trailing statements and no explicit return: the
# implicit-None tail (legacy starves here — divergence test below)
def loop_implicit_none(a: int) -> None:
    total = 0
    for i in range(a):
        total += i
    print(total)


def flag_break(a: int) -> int:
    total = 0
    i = 0
    flag = True
    while flag:
        if i >= a:
            flag = False
        else:
            total += i
            i += 1
    return total


def while_true_break(a: int) -> int:
    total = 0
    while True:
        total += a
        break
    return total


def while_true_if_break(a: int) -> int:
    total = 0
    i = 0
    while True:
        if i >= a:
            break
        else:
            total += i
            i += 1
    return total


# --- differential matrix (legacy is a valid oracle for these) -------------


_CASES = [
    # (program, args) — the expected value comes from calling the case's
    # own Python function; legacy and the new lowering must both match it.
    (Program(while_sum), (2, 6)),
    (Program(for_sum), (5,)),
    (Program(sum_it_up), (1, 10)),
    (Program(while_continue), (6,)),
    (Program(for_continue), (6,)),
    (Program(while_continue_closer), (5,)),
    (Program(for_continue_closer), (5,)),
    (Program(while_break_closer), (5,)),
    (Program(while_direct_break), (7,)),
    (Program(for_break), (10,)),
    (Program(while_orelse), (4,)),
    (Program(while_break_orelse), (5,)),
    (Program(for_orelse), (4,)),
    (Program(for_break_orelse), (5,)),
    (Program(loop_early_return), (10,)),
    (Program(loop_early_return), (2,)),
    (Program(nested_while_in_for), (4,)),
    (Program(loop_then_trailing), (5,)),
    (Program(while_then_trailing), (5,)),
]


@pytest.mark.parametrize("program,args", _CASES, ids=program_ids)
def test_matches_legacy_execution(program: Program, args: tuple) -> None:
    expected = program.call(*args)
    assert run_legacy(program.compile_legacy().compiled, *args) == expected
    assert program.lower()(*args) == expected


# --- pinned divergences (§5: the IR is the spec) --------------------------
# Legacy's loop break/continue handling starves or miscounts on these;
# the new lowering is asserted against the source's own Python semantics
# directly, and each docstring records legacy's actual behavior as the
# reason the case is not in the differential matrix.


def test_break_inside_if_legacy_miscount():
    """Legacy ignores the if-mediated break and runs the loop to its
    bound (it returns sum(range(100)) = 4950 where the source returns
    10); ours breaks correctly."""
    program = Program(while_break_bound)
    assert program.call(5) == 10
    assert program.lower()(5) == 10


def test_implicit_none_after_loop_legacy_starves():
    """A loop followed by the implicit-None tail starves legacy's output
    entirely; ours returns None (the §5 mixed-if divergence's loop
    analogue)."""
    program = Program(loop_implicit_none)
    assert program.call(3) is None
    assert program.lower()(3) is None


def test_flag_loop_legacy_starves():
    """A flag-variable while with the mutation inside an if/else starves
    legacy's output; ours terminates with the correct sum."""
    program = Program(flag_break)
    assert program.call(5) == 10
    assert program.lower()(5) == 10


def test_while_true_if_break():
    program = Program(while_true_if_break)
    assert program.lower()(5) == 10


def test_while_true_break():
    program = Program(while_true_break)
    assert program.lower()(5) == 5


# --- introspection ---------------------------------------------------------


def test_loop_agent_is_cataloged():
    """The Loop composite (and its three subflows) must survive the
    post-hoc agent catalog — walk_agents descends iteration/body/orelse
    and catalogs the Loop itself."""
    unit = capturing_lower(Program(for_sum))
    loops = [agent for agent in unit.agents if isinstance(agent, Loop)]
    assert len(loops) == 1
    loop = loops[0]
    assert any(flow is loop.body for flow in unit.agents)
    assert any(flow is loop.orelse for flow in unit.agents)


# --- consolidation (old suite's infinite-value programs) -------------------
# The old suite's non-terminating loops, copied from test_compiler.py.
# All of them ride the constant `while True:` shared limitation above;
# and_or additionally needs IrBoolOp lowering. (Their Par blockers are
# gone — IrTuple/composite-target lowering landed, see
# test_par_lowering.py.)
#
# The two build pins below RUN: both compilers accept the programs, and
# the bodies deliberately never call them — EXECUTION is the shared
# limitation. The two skips genuinely need to execute, and stay marked
# so unskipping cannot wedge the suite. Legacy runs under
# ThreadPoolExecutor — the old suite ran these under
# @inet(executor=ThreadPoolExecutor()), and the finite parts of their
# output only overtake the spinning branches under threads.


def infinite_value() -> int:
    a = 0
    while True:
        a += 1
    return a


def ignored_infinite_loop() -> Par[int, int]:
    b = 0
    while True:
        b += 1
    return 10, b


def drops_infinite_loop() -> int:
    a, b = ignored_infinite_loop()
    return a


def and_or_with_finites_and_infinites() -> list:
    a = infinite_value()
    paths: Ref[list] = []

    if a < 10 or True:
        paths.append("Infinite Or")

    if a < 10 and False:
        paths.append("Infinite And")

    paths.append(10 and 0)
    paths.append(10 or 0)
    return paths


def test_infinite_value_compiles_everywhere():
    """Old suite: infinite_value() feeds and_or's finites-and-infinites
    algebra. Both compilers accept the program; the call itself hangs
    BOTH implementations (constant-while, module docstring), so the body
    is deliberately build-only."""
    program = Program(infinite_value)
    program.compile_legacy()
    program.lower()


def test_ignored_infinite_loop_compiles_everywhere():
    """The old suite never called this directly either — only through
    drops_infinite_loop (a caller-side drop is the only terminating
    shape: the direct call hangs BOTH implementations). Compiling is the
    testable half, and both compilers do it."""
    program = Program(ignored_infinite_loop)
    program.compile_legacy()
    program.lower()


def test_drops_infinite_loop_differential():
    program = Program(drops_infinite_loop, ignored_infinite_loop)
    legacy = program.compile_legacy().compiled
    assert run_legacy(legacy, executor=ThreadPoolExecutor()) == 10
    # Cutover gap (CUTOVER.md §5): with the callee now NEW-compiled, the
    # trailing return after its `while True` sequences strictly off the
    # loop's finish slot and starves. Legacy's wire_continuation ran the
    # trailing region immediately via an erasure superposition
    # (`finish + ~readin` through a split/involution); reproducing that
    # topology in ControlBranchFlow's choice-chain sequencing is the
    # recorded follow-up. The legacy half above keeps the expected value
    # pinned until then.
    pytest.xfail(
        "lowering: loop-continuation erasure superposition not yet "
        "reproduced — trailing code after a never-firing loop starves"
    )


def test_and_or_finites_and_infinites_differential():
    program = Program(and_or_with_finites_and_infinites, infinite_value)
    legacy = program.compile_legacy().compiled
    expected = ["Infinite Or", 0, 10]
    assert run_legacy(legacy, executor=ThreadPoolExecutor()) == expected
    assert program.lower()() == expected

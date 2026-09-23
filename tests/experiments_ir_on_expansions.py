"""Prototype check: lowered expansions carry their IrNodes.

Lower a function with a loop containing an if, then walk the post-hoc
agent catalog and read the IR off the expansions — including exits the
composites consult for their own (and their subcomponents') shortcuts.
"""

from natsune.backend.control_flow import IfThenElseStatement, Loop, VariablesFlow
from natsune.backend.lowering import lower_function
from natsune.backend.protocol import LoweredUnit
from natsune.backend.python_backend import PythonBackend
from tests.backend.helpers import Program


def classify(n: int) -> int:
    total = 0
    for i in range(n):
        if i % 2 == 0:
            continue
        if i > 10:
            break
        total += i
    else:
        total = -1
    return total


def orelse_returns(n: int) -> int:
    # The orelse ALWAYS returns: its IrBody.exits is RETURN with no
    # FALLTHROUGH, so false_case's finish channel is provably dead and the
    # mask shortcut annihilates it (loop finish stays live via true_case).
    total = 0
    for i in range(n):
        if i == 2:
            break
        total = i
    else:
        return 99
    return total


class ExposingBackend(PythonBackend):
    def finish(self, artifact: LoweredUnit):
        loops = [a for a in artifact.agents if isinstance(a, Loop)]
        ifs = [a for a in artifact.agents if isinstance(a, IfThenElseStatement)]
        bodies = [
            a for a in artifact.agents if isinstance(a, VariablesFlow) and a.ir
        ]

        for loop in loops:
            node = loop.ir
            assert node is not None, "lowered Loop carries its IrFor/IrWhile"
            print(f"Loop ir={type(node).__name__} exits={node.exits!r}")
            print(f"  via node only: body.exits={node.body.exits!r} "
                  f"orelse.exits={node.orelse.exits!r}")
            print(f"  subcomponent flows carry their own bodies: "
                  f"body.ir={type(loop.body.ir).__name__} "
                  f"orelse.ir={type(loop.orelse.ir).__name__} "
                  f"iteration.ir={loop.iteration.ir}")

        for stmt in ifs:
            node = stmt.ir
            assert node is not None, "lowered statement carries its IrIf"
            print(f"IfThenElseStatement ir={type(node).__name__} exits={node.exits!r}")
            print(f"  then.ir={type(stmt.true_case.ir).__name__} "
                  f"else.ir={type(stmt.false_case.ir).__name__}")

        for body in bodies:
            assert body.ir is not None
            print(f"body flow: ir={type(body.ir).__name__} exits={body.ir.exits!r}")
        return super().finish(artifact)


program = Program(classify)
entry = program.lower(ExposingBackend())
print(f"execution: classify(7) = {entry(7)} (no break -> for-else fires -> -1)")
print(f"execution: classify(20) = {entry(20)} (break at i=11 -> 1 + 3 + 5 + 7 + 9 = 25)")
print(f"execution: classify(0) = {entry(0)} (empty loop -> orelse -> -1)")

program = Program(orelse_returns)
entry = program.lower(ExposingBackend())
print(f"execution: orelse_returns(5) = {entry(5)} (break at i=2 -> total=1)")
print(f"execution: orelse_returns(2) = {entry(2)} (exhausts -> orelse returns 99)")
print(f"execution: orelse_returns(0) = {entry(0)} (exhausts -> orelse returns 99)")

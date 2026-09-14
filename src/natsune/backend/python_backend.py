import ast
import dataclasses
import threading
from collections.abc import Mapping
from typing import Any, Callable, Collection

from natsune.adapters import Adapter
from natsune.backend.agents import AgentImpl, callee_invocation
from natsune.compiler import construct_locals, eval_expression
from natsune.connector import Connector, FrozenExpansion
from natsune.executor import Executor, ThreadPoolExecutor
from natsune.frontend import IrFunction
from natsune.invocations import filter_invocation, merge_invocation, send_parameters
from natsune.registers import (
    FromRegister,
    as_constant_register,
    borrow_registers,
    send_value,
    serialize_values,
)


@dataclasses.dataclass
class PythonBackend:
    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister:
        """The Python resolution of the dynamic fallback (§8.4): eval the
        rewritten source against a (globals, locals) context whose locals
        are the serialized captured values — the old
        evaluate_from_expression/construct_context, verbatim. CppBackend
        refuses this method (diagnostic per §4(2)).

        ``adapter`` is accepted for signature symmetry with the IR but the
        eval result carries VA, as in legacy; ``node`` goes unused here (an
        emitter backend is the consumer that pattern-matches it).
        ``eval_expression``/``construct_locals`` are legacy ext fns until
        they become Primitive/table entries (§4(1)); try-machinery wrapping
        (collect_exceptions) stays out — try is rejected outright."""

        (text_in, context_in), result = merge_invocation(eval_expression, connector)
        result_a, result_b = result.duplicate("share")
        send_value(as_constant_register(source_text, connector), text_in)

        value_readout = borrow_registers(list(captures.values()), result_b)
        # Evaluation order mirrors old construct_context exactly —
        # serialize2, then globals, then the construct_locals merge (which
        # evaluates serialize_n, then the keys constant). Invocation helpers
        # connect pairs eagerly, so creation order is net structure.
        context = send_parameters(
            serialize_values(connector, 2),
            (
                as_constant_register({}, connector),
                send_parameters(
                    merge_invocation(construct_locals, connector),
                    (
                        send_parameters(
                            serialize_values(connector, len(captures)),
                            value_readout,
                        ),
                        as_constant_register(tuple(captures.keys()), connector),
                    ),
                ),
            ),
        )
        send_value(context, context_in)
        return result_a

    def finish(
        self,
        func: IrFunction,
        flow: FrozenExpansion,
        agents: Collection[AgentImpl],
        *,
        name: str = "main",
    ) -> Any:
        return name, func, flow, agents


def as_callable(
    func: IrFunction, flow: FrozenExpansion, executor: Executor | None = None
) -> Callable[..., Any]:
    def impl(*args: Any) -> Any:
        outputs: list = []
        if executor is not None:
            exec_to_use = executor
        else:
            exec_to_use = ThreadPoolExecutor()

        inputs, output = callee_invocation(flow, len(func.params), exec_to_use)
        for to_register, arg in zip(inputs, args, strict=True):
            send_value(as_constant_register(arg, exec_to_use), to_register)

        end_event = threading.Event()

        def output_callback(x):
            outputs.append(x)
            end_event.set()

        to_register, from_register = filter_invocation(output_callback, exec_to_use)
        from_register.close()
        send_value(output, to_register)

        exec_to_use.run(end_event)

        if not outputs:
            raise ValueError("No output produced by the function")

        return outputs[0]

    return impl

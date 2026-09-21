import ast
import dataclasses
import threading
from collections.abc import Mapping
from typing import Any, Callable, Collection

from natsune.adapters import VA, Adapter, ValueAdapter
from natsune.backend.agents import AgentImpl, callee_invocation
from natsune.backend.protocol import LoweredUnit
from natsune.compiler import construct_locals, eval_expression, exec_expression
from natsune.connector import Connector, FrozenExpansion
from natsune.executor import Executor, ThreadPoolExecutor
from natsune.frontend import IrFunction
from natsune.invocations import (
    catch,
    filter_invocation,
    merge_invocation,
    send_parameters,
)
from natsune.ports import Erasure, Graft, Wire
from natsune.registers import (
    FromRegister,
    ToRegister,
    as_constant_register,
    as_from_register,
    as_to_register,
    borrow_registers,
    send_value,
    serialize_values,
)


@dataclasses.dataclass
class PythonBackend:
    executor: Executor | None = None

    def materialize_dynamic(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> FromRegister:
        (text_in, context_in), result = merge_invocation(eval_expression, connector)
        result_a, result_b = result.duplicate("share")
        send_value(as_constant_register(source_text, connector), text_in)

        value_readout = borrow_registers(list(captures.values()), result_b)
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

    def materialize_dynamic_to(
        self,
        node: ast.expr,
        source_text: str,
        captures: Mapping[str, FromRegister],
        adapter: Adapter,
        connector: Connector,
    ) -> ToRegister:
        capture_token = "___from_capture___"

        source_text = f"{source_text} = {capture_token}"
        (text_in, context_in), result = merge_invocation(exec_expression, connector)
        result_a, result_b = result.duplicate("share")
        send_value(as_constant_register(source_text, connector), text_in)

        x1, x2 = Wire.as_interface()
        captures = {capture_token: as_from_register(x2, adapter, connector), **captures}

        value_readout = borrow_registers([*captures.values()], result_b)
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
        # TODO: Revisit when we handle exceptions.
        result_a.close()
        return as_to_register(x1, adapter, connector)

    def finish(
        self,
        artifact: LoweredUnit,
    ) -> Any:
        return as_callable(artifact.func, artifact.main, self.executor)


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
        send_value(
            from_register,
            as_to_register(Graft(catch(output_callback), []), VA, exec_to_use),
        )
        send_value(output, to_register)

        exec_to_use.run(end_event)

        if not outputs:
            raise ValueError("No output produced by the function")

        if isinstance(outputs[0], Erasure):
            if isinstance(outputs[0].value, Exception):
                raise outputs[0].value
            raise RuntimeError("Unexpected runtime error: " + str(outputs[0].value))

        return outputs[0]

    return impl

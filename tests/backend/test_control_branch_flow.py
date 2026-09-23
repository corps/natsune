from typing import Any, Collection, cast

import pytest

from natsune.first_order.adapters import ValueAdapter, Variables
from natsune.backend.control_branch_flow import ControlBranchFlow
from tests.calculus import Calculus
from natsune.backend.control_flow import VariablesFlow
from natsune.frontend import IrFunction
from natsune.frontend.ir import Exits, IrBody, VariableUsage
from natsune.backend.registers import as_constant_register, send_value


def create_flow(
    variable_names: Collection[str],
    exits: Exits = Exits.FALLTHROUGH,
    variables_usage: dict[str, VariableUsage] | None = None,
) -> ControlBranchFlow:
    return ControlBranchFlow(
        VariablesFlow(
            variables=Variables({k: ValueAdapter() for k in variable_names}),
            return_adapter=ValueAdapter(),
        ),
        (
            variables_usage
            if variables_usage is not None
            else {k: "write" for k in variable_names}
        ),
        exits,
    )


def _ir_func(flow: ControlBranchFlow, arity: int) -> IrFunction:
    return IrFunction(
        position=cast(Any, None),
        name="test",
        params=tuple(
            (k, a)
            for _, (k, a) in zip(range(arity), flow.containing_flow.variables.items())
        ),
        return_adapter=flow.containing_flow.return_adapter,
        body=IrBody(
            variable_usage=flow.variables_usage,
            disjunctives=(),
            closer=None,
            statements=(),
            position=cast(Any, None),
        ),
    )


def test_empty_fallthrough() -> None:
    flow = create_flow(("a", "b"))
    flow.close()

    c = _send_variables(flow, (None, 1, 2))
    assert c.reduce_to_value(0) == (None, 1, 2)
    c.assert_erased(1)
    c.assert_erased(2)
    c.assert_erased(3)


def test_application_still_flows() -> None:
    flow = create_flow(("a", "b"))
    layer = flow.apply_new_layer()

    send_value(
        layer.variables_readout(),
        layer.control_output.finish_variables.readin(),
    )
    layer.close()
    flow.close()

    c = _send_variables(flow, (None, 1, 2))
    assert c.reduce_to_value(0) == (None, 1, 2)
    c.assert_erased(1)
    c.assert_erased(2)
    c.assert_erased(3)


def test_exits_prempty_close() -> None:
    flow = create_flow(("a", "b"), Exits.FALLTHROUGH | Exits.RETURN)
    # Intentionally do not close this.
    layer = flow.apply_new_layer()
    send_value(
        layer.variables_readout(),
        layer.control_output.finish_variables.readin(),
    )
    flow.close()

    c = _send_variables(flow, (None, 1, 2))

    # These are open because they counted as exits.
    c.assert_open(0)
    c.assert_open(1)

    # These are closed because they were not.
    c.assert_erased(2)
    c.assert_erased(3)


@pytest.mark.parametrize(
    "usage",
    [
        {"a": "read", "b": "write"},
        {"b": "write"},
    ],
)
def test_layer_bundle_forwards_verbatim(usage: dict) -> None:
    """The layer's finish bundle forwards verbatim at close(), whatever
    the boundary usage classification — the layer receives the full live
    bundle (see ControlBranchFlow.__post_init__). Real lowering never
    writes a read-classified register (the IR proves the body never
    writes it, so the forwarded value equals the original there); this
    test writes one deliberately to pin the verbatim rule."""
    flow = create_flow(("a", "b"), variables_usage=usage)
    layer = flow.apply_new_layer()

    send_value(as_constant_register(20, layer), layer.variable_registers["a"].readin())
    send_value(as_constant_register(30, layer), layer.variable_registers["b"].readin())

    send_value(
        layer.variables_readout(),
        layer.control_output.finish_variables.readin(),
    )
    layer.close()
    flow.close()

    c = _send_variables(flow, (None, 1, 2))
    assert c.reduce_to_value(0) == (None, 20, 30)


def test_return_values() -> None:
    flow = create_flow(
        (), Exits.FALLTHROUGH | Exits.RETURN | Exits.CONTINUE | Exits.BREAK
    )
    layer = flow.apply_new_layer()

    send_value(
        as_constant_register(20, layer), layer.control_output.return_value.readin()
    )

    layer.close()
    flow.close()

    c = _send_variables(flow, (None,))
    assert c.reduce_to_value(1) == 20


def _send_variables(
    flow: ControlBranchFlow,
    variables_input: tuple,
) -> Calculus:
    c = Calculus()
    with flow.containing_flow.invocation(c.executor, internal=False) as invocation:
        send_value(
            c.const(variables_input),
            invocation.port.variables.readin(),
        )

        send_value(
            invocation.wire.finish_variables.readout(), c.to_key(0, ValueAdapter())
        )

        send_value(invocation.wire.return_value.readout(), c.to_key(1, ValueAdapter()))

        send_value(
            invocation.wire.continue_variables.readout(), c.to_key(2, ValueAdapter())
        )

        send_value(
            invocation.wire.break_variables.readout(), c.to_key(3, ValueAdapter())
        )
    return c

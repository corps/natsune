"""The inet function interface, post-cutover.

`@inet` attaches an artifact (natsune.inet.InetFunction, wrapped by the
decorator's entry point) to the decorated function as `__inet__`. The
frontend resolves calls to such functions through the structural
`InetRef` protocol — it never imports the artifact type, and the backend
narrows `expansion` to `FrozenExpansion | PromiseExpansion` where it
wires. This module is a leaf: adapters/ports only."""

from typing import Protocol, runtime_checkable

from natsune.first_order.adapters import Adapter, ParValueAdapter
from natsune.first_order.ports import Expansion


@runtime_checkable
class InetRef(Protocol):
    """What the compiler needs to know about a marked inet function.

    `args_adapter`/`return_adapter` are the signature metadata the
    frontend checks calls against; `expansion` is the compiled net —
    None until compilation starts, promise-or-frozen from then on
    (see CUTOVER.md §2). `ensure_compiled` compiles on first use; a
    same-thread re-entrant call during the owner's compilation returns
    immediately with `expansion` still the outstanding promise.
    """

    args_adapter: ParValueAdapter
    return_adapter: Adapter
    expansion: Expansion | None

    def ensure_compiled(self) -> None: ...

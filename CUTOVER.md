# Cutover plan: replacing `natsune.compiler` with the new pipeline

STATUS: **COMPLETE**. `natsune/compiler.py` and the legacy differential
harness are deleted; the deferred driver (`natsune/inet.py`), the
frontend, and the backend are the only compiler. `@inet` is re-exported
as `from natsune import inet`. The two `test_drops` pins (§5) are the
recorded follow-up. This document is kept as the record of the cutover
design (the § numbering is referenced from code comments and tests).

This continues COMPILER_REFACTOR.md (frontend §0–§7 and the backend lowering are
done and green). What remains is the cutover: the `inet` decorator stops using
the legacy eager compiler, `natsune/compiler.py` and `tests/test_compiler.py`
are deleted, and the new approach becomes the only compiler.

## 0. Where things stand

**The goal of the cutover:** `@inet` marks a function for the compiler. After
the cutover:

- An inet function **called from an inet function** wires through the compiled
  net (`FrozenExpansion` grafted at the call site) — the compiler calling
  convention.
- An inet function **called from standard Python** goes through the entry
  point: push args as constant registers, drive an executor, block for the
  output (`as_callable` in `python_backend.py`).

**What `natsune/compiler.py` (1101 lines) contains, and where each piece
stands:**

| Legacy item | Status |
|---|---|
| `unsupported_expr` / `unsupported_stmt` | superseded by `frontend/unsupported.py` |
| `ReplaceWithSerializedVariables` | superseded by `_DynamicRewriter` (`frontend/ir/builder.py`) |
| `InetFunctionCompiler` | superseded by the frontend pipeline + `_FunctionLowering` |
| `InetBranchCompiler` | superseded by `_FunctionLowering` + `ControlBranchFlow` |
| `InetVariablesEvaluator` | superseded by `frontend/symbols.py` |
| `_try_iter` | already moved → `backend/agents.py: try_iter` |
| `construct_locals`, `exec_expression`, `eval_expression` | **must survive** — the exec-fallback runtime; `python_backend.py` imports them from `compiler` today |
| `_match_exception_group` | **must survive** — needed when `IrTry`/`TryStar` lowering lands |
| `inet` (decorator) | **must survive** — reimplemented on the new pipeline, deferred |

**The bridge today:** `legacy.py` defines `LegacyInetInterface`
(`.compiled` / `.args_adapter` / `.return_annot`). `frontend/link.py`
(`link_name` → `LinkedInet.from_ref` via `hasattr(value, "__inet__")`),
`frontend/ir/nodes.py` (`IrCallInet.ref`), and `backend/agents.py`
(`callee_invocation`'s `LegacyInetInterface` branch) all speak it.

**Known duplication to resolve:** the entry-point logic exists twice — the old
decorator's inner `impl` (`compiler.py`) and `as_callable`
(`python_backend.py`). They differ in two ways: the old one defaults to
`DeterministicSerialExecutor` (the new one defaults `ThreadPoolExecutor`), and
the new one unwraps `Erasure` and re-raises wrapped exceptions (an
improvement — keep it). Cutover keeps one entry point.

## 1. Moving the surviving compiler utilities

Mechanical, no behavior change; do this first so `compiler.py` shrinks to
exactly the pieces that will be deleted.

1. New module `natsune/backend/runtime.py` holding the pure exec-fallback
   helpers:
   - `construct_locals`, `exec_expression`, `eval_expression` (from
     `compiler.py`; update `python_backend.py` to import from here; compiler.py
     imports from here too until it dies — one definition throughout).
   - `_match_exception_group` (renamed `match_exception_group`) — parked here
     unused by the new pipeline for now, documented as the future `IrTry`
     split agent (the legacy compiler still calls its copy).
   - `try_iter` can move here from `agents.py` (or stay — taste; it is already
     out of compiler.py).
2. `compiler.py` re-points its internal uses at the moved copies and
   otherwise stays untouched: its superseded pieces (table above) are still
   the live machinery for the legacy differential tests, so they are deleted
   in Phase C with the module, not now.
3. Green check: full suite passes unchanged.

## 2. Deferred compilation

### Why the current eager model must go

Eager (today): decorating A compiles A immediately, and compiling A requires
every callee B to (a) already be decorated — `lookup_inet` evals the name in
globals at decoration time — and (b) already be *compiled*, because the call
site wires B's net at A-compile time. So callees must be defined and
decorated above the caller, in the same import sweep. This breaks:

- forward references (callee defined below the caller),
- mutually referential functions across modules (A's module imports B's,
  B's imports A's),
- any "co-function" arrangement where definition/import order does not match
  call-graph topologically.

It also means compile diagnostics fire at import time, whether or not the
function is ever used.

### The deferred model

`@inet` becomes a **pure marker**. Decoration does no compiling and no
signature analysis — it attaches a small artifact object as `__inet__`:

```python
# natsune/inet.py  (new module — the composition root / public API)
class InetFunction:
    func: Callable                  # the decorated python function
    default_executor: Executor | None

    # filled by ensure_compiled(), memoized:
    args_adapter: ParValueAdapter | None
    return_adapter: Adapter | None
    expansion: FrozenExpansion | None

    def ensure_compiled(self) -> InetFunction: ...
    def __call__(self, *args, executor=None) -> Any: ...  # entry point
```

**"Placed after all imports are in scope":** compilation triggers are exactly
the points at which the program is guaranteed (or declares) that all imports
have run:

1. **First entry-point call.** `InetFunction.__call__` runs
   `ensure_compiled()` before invoking the net. A call from standard Python
   can only happen after the importing module finished executing, so every
   name the function body references is in `globals()` by then. This is the
   default trigger and needs no user action.
2. **Explicit eager compile**, for users who want import-time diagnostics or
   pre-warming: `inet.compile(func)` / `func.__inet__.ensure_compiled()`,
   and optionally an `inet.compile_module(module)` convenience that walks a
   module's marked functions. Plain sugar over (1); not required for
   correctness.

### The compile driver

`compile_function(func, *, executor=None)` in `natsune/inet.py` is the
pipeline the tests already exercise piecemeal
(`tests/backend/helpers.py: build_ir_for_function`):

1. `extract_source(func, globals=func.__globals__, filename=...)`
2. `analyze_signature` → fills `args_adapter` / `return_adapter` on the
   artifact. Doing this at compile time (not decoration time) also makes
   annotation resolution lazier: `get_type_hints` runs after imports, so
   annotations naming later-imported classes now work (the old eager compiler
   would have failed at decoration).
3. `collect_symbols` — kept on the artifact: recursive callees need it
   (see below). Then `collect_call_links`.
4. **Expose the net slot:** after symbols, `artifact.expansion` is a
   `PromiseExpansion` (created lazily, see below). From here until the end
   of the compile it is promise-or-frozen, never `None` — lowering grafts
   `ref.expansion` uniformly.
5. **Callee pass:** for every `LinkedInet` in the links map, call
   `ref.ensure_compiled()` — depth-first, memoized; a re-entrant call (the
   ref is mid-compile on this thread) returns the ref's promise instead of
   recursing. This is what makes compilation order-independent: by the time
   the caller's net needs to wire a call, the callee is compiled **or
   promised** regardless of who was decorated first.
6. `build_ir` → `lower_function(ir, PythonBackend(executor=...))` →
   `LoweredUnit` → freeze → fill the promise and swap `expansion` to the
   `FrozenExpansion`.

### Recursion and co-functions: the `PromiseExpansion`

Deferral fully solves *mutual reference* (A↔B across modules, forward refs):
order no longer matters because nothing compiles until after import, and the
callee pass compiles on demand.

True *cycles* (A calls A, or A→B→A) need the in-flight function's own net
before it exists — lowering wires call sites eagerly. The legacy compiler
could not do this at all (a self-call hits `eval` on a not-yet-bound module
global during decoration and dies with a confusing `SyntaxError`). The
cutover adds a **promise** — DECIDED: this lands in the cutover, not later.

```python
# backend/agents.py
class PromiseExpansion:
    """Late-bound callee net: stands in for a FrozenExpansion that is
    still being compiled (recursive / mutually recursive inet calls).
    Grafts baked with the promise materialize through it once filled.
    By-reference mutation of the promise is the accepted Python-target
    semantics (CUTOVER.md, open decisions)."""

    input_adapter: Adapter
    output_adapter: Adapter
    name: str
    target: FrozenExpansion | None = None

    def __copy__(self):
        return self  # every baked graft shares the promise; fills are global

    def __call__(self, executor, port, wires):
        assert self.target is not None, (
            f"net for {self.name} used before compilation finished"
        )
        self.target(executor, port, wires)
```

Mechanics:

- **Adapters at promise-creation time.** `callee_invocation` wires the
  call site against the callee's `input_adapter`/`output_adapter`, and a
  compiled `VariablesFlow`'s adapters are built from its FULL variable set
  (params + locals, `RA_VA` slot first, statement order after that) — not
  from the signature. The in-flight artifact has exactly that set: its
  `SymbolsTable` (driver step 3, which precedes any possible re-entry).
  So the promise is built with
  `Variables(dict(symbols.variables))` through the same
  `flow_input_adapter` / `flow_control_adapter` constructors the real net
  will use, and — because promises can only be requested once the owner is
  past step 3 — the data is always there. A fill-time assertion pins the
  invariant (`promise.input_adapter == frozen.input_adapter` and likewise
  for output): if symbols collection and lowering's `collect_variables`
  ever diverge, compilation fails loudly instead of corrupting a net.
- **Creation is lazy, per artifact.** `artifact._promise()` creates the
  singleton on first re-entrant request and sets `expansion` to it.
- **Fill at freeze.** End of the driver:
  `promise.target = frozen`; `artifact.expansion = frozen`.
- **Why fills always land before any net runs.** Promises are created
  during the *callee pass* of the compiling function, and a nested
  `ensure_compiled` runs to completion (including its fill) before the
  caller's own `build_ir`/lowering even starts. So every net that embeds a
  graft of a promise was built after that promise was already filled —
  nets only run post-compile, from entry points. No wake-up machinery
  needed.
- **Locking.** One module-global compile lock serializes all compilations
  (compiles are milliseconds; this also kills the AB-BA deadlock two
  threads would otherwise hit first-calling a mutually recursive pair
  concurrently). It is an **RLock**, because the driver's callee pass
  compiles callees while holding it — same-thread nesting is the
  depth-first compilation itself. Per-artifact state detects cycles
  *before* the lock is touched: `_compiling` (a mid-compile owner's
  `ensure_compiled` returns its promise) and `args_adapter` (already
  filled for a mid-compile owner's `ensure_signature`). Cross-thread
  second callers block, then find the artifact compiled — they never see
  a promise.
- **Runtime shape.** A self-call graft materializes another copy of the
  function's own net — exactly how interaction nets express recursion
  (growth bounded by the base case at runtime). Infinite recursion behaves
  like the already-supported infinite loops: unbounded net growth.
- **Agent bookkeeping.** `AgentImpl` gains `PromiseExpansion`, and
  `walk_agents` catalogs it (a `Graft` whose `execute` is a promise), so
  `LoweredUnit.agents` tells backends the unit references a pending net.

Edge that stays a hard error: calling an entry point for an artifact that
is mid-compile **on the same thread** (possible only via a dynamic exec
fallback invoking it during its own compilation) — `__call__` asserts
non-promise. Legacy had no better behavior here.

### Two-pass callee resolution (implementation note)

Symbol collection types calls against callee return adapters
(`infer_adapter` reads the links), so callees must have SIGNATURES before
the caller's symbols are collected — but a cycle's promise needs the
owner's symbols. The driver therefore runs two passes over the resolved
markers (`_resolve_markers`): an `ensure_signature` pass (cheap
extract+analyze, no compile) before `collect_symbols`, then the full
`ensure_compiled` pass after — by which point a mid-compile owner has its
symbols and can materialize its promise. `ensure_signature` shares the
compile lock (RLock: the callee pass holds it while nesting) and its
`args_adapter` pre-check makes cyclic re-entry a no-op.

## 5. Exposed during the cutover: loop-continuation starvation

The differential suite caught exactly one semantic gap when callees
switched from legacy-built to new-built nets (test_drops —
`ignored_infinite_loop`'s trailing `return` after `while True`, drained
by the caller): **legacy's `wire_continuation` superposed the trailing
region's variables with erasures** (`finish + ~readin`, through a
split/involution), so code after a never-firing loop ran immediately
with erasure-backed cells. The restructured `ControlBranchFlow`
sequences the next layer strictly off the composite's finish slot via a
choice-chain — which starves forever when no slot ever fires.

Both pinned tests (`tests/test_inet.py::test_drops_infinite_loop`,
`tests/backend/test_loop_lowering.py::test_drops_infinite_loop_differential`)
keep the program, the expected value (10), and the legacy leg so the fix
lands against the real case. The follow-up: reproduce the erasure
superposition in the restructured sequencing (a naive `finish + ~readin`
into the layer's readin breaks the algebra — the split/involution
topology must be ported properly), or model "never finishes" explicitly
in `Exits` and let ControlBranchFlow shortcut the trailing variables.

### Entry point

`InetFunction.__call__` delegates to `as_callable` (moved/kept in
`python_backend.py`) so there is exactly one entry-point implementation.
Pin the semantics deliberately:

- default executor: **`DeterministicSerialExecutor`** (old decorator parity —
  e.g. `drops_infinite_loop` only terminates under an explicit
  `ThreadPoolExecutor`, and the suite is written against the deterministic
  default);
- `executor=` per-call override and `@inet(executor=...)` default override
  both preserved;
- keep `as_callable`'s `Erasure` unwrap-and-reraise (the old `impl` would
  return the `Erasure` object).

## 3. How the frontend finds and resolves deferred targets

Nothing about *detection* changes: `link_name` still keys on
`hasattr(value, "__inet__")`. What changes is what the marker is and what
happens at link time.

1. **The protocol.** Retire `legacy.py`'s `LegacyInetInterface`
   (`.compiled`-shaped). New leaf protocol (typing only, no runtime
   `isinstance`), e.g. `natsune/interface.py` or inside `frontend/link.py`:

   ```python
   class InetRef(Protocol):
       args_adapter: ParValueAdapter
       return_adapter: Adapter
       # Structurally typed so the frontend never imports the backend:
       # FrozenExpansion or PromiseExpansion, None until compile starts,
       # promise-or-frozen once compiling.
       expansion: Expansion | None
       def ensure_compiled(self) -> InetRef: ...
   ```

   Import direction stays clean: `interface.py` imports only adapters/
   ports (typing); `frontend` and `backend` both see it; `inet.py`
   implements it and the backend narrows `expansion` to
   `FrozenExpansion | PromiseExpansion`. (`IrCallInet.ref` and
   `LinkedInet.ref` change type to this; `LinkedInet.from_ref` reads
   `return_adapter` directly instead of
   `adapter_from_type(ref.return_annot)`.)

2. **Who triggers compilation — keep the frontend pure.** The frontend stays
   "plain data in, plain data out" (see `frontend/__init__.py` and
   `tests/frontend/test_link.py`'s import guard). `link_name` does **not**
   compile; it just classifies. The *driver* (step 4 of the compile driver
   above) walks the `collect_call_links` results and calls
   `ensure_compiled()` on every `LinkedInet` ref **before** `build_ir`, so by
   the time `build_ir` stamps `IrCallInet.ref` and the backend lowers
   `callee_invocation(ref.expansion, arity, flow)`, `ref.expansion` is
   always a frozen net.

3. **Backend simplification.** `callee_invocation` drops the
   `LegacyInetInterface` branch: it takes `FrozenExpansion |
   PromiseExpansion` (both structurally `ExpansionWithAdapters`) only, and
   the `IrCallInet` lowering passes `expr.ref.expansion` — which is
   promise-or-frozen per the driver, so recursive and non-recursive call
   sites lower identically. `FrozenExpansion` already carries
   `input_adapter`/`output_adapter`, so the backend never needs the link
   metadata at all — `args_adapter`/`return_adapter` on the artifact exist
   for the frontend's arity/adapter checks and for render/`infer_adapter`.

4. **Test fakes.** `tests/frontend/programs.py` and the `SimpleNamespace`
   fakes in `test_link.py` / `test_symbols.py` / `test_infer.py` /
   `test_ir_builder.py` attach `__inet__` objects shaped like the old
   compiler; they only feed `from_ref`/`infer_adapter`, so they update
   mechanically to the new attribute names (`return_adapter`,
   `expansion=None`).

## 4. Landing: deleting `compiler.py` and `test_compiler.py`

### 4a. The acceptance suite moves first

`tests/test_compiler.py` is the behavioral contract for the decorator. Port
it verbatim to `tests/test_inet.py` against the new decorator (same cases,
same assertions — `basic`, `invoke_an_inet`, `use_references`, `sum_it_up`,
inverses/Ref, `delayed_inverse`, infinite loops, `and_or...`). Add the
deferred-compilation cases:

- forward reference: caller decorated above its callee;
- two modules that import each other, each calling the other's inet
  functions (the co-function case), exercised via first call;
- plain-Python call before any inet call (trigger 1) and explicit
  `inet.compile()` (trigger 2);
- `executor=` passthrough (per-call and decorator-level);
- double-call memoization (second call reuses the net) and a two-thread
  race on first call (lock correctness);
- recursion, positive cases (the `PromiseExpansion` path):
  - self-recursion with a base case (`fact(5) == 120`) through the entry
    point;
  - mutual recursion (`is_even`/`is_odd`) entered from either side;
  - a three-function cycle (A→B→C→A);
  - recursive calls nested in `if`/loop bodies;
  - the recursive function also invoked as a plain callee of a third inet
    function;
  - two threads first-calling a mutually recursive pair concurrently (the
    global compile lock: no deadlock, no torn promise).

### 4b. Backend test infrastructure

`tests/backend/helpers.py`:

- `_inet_attached` / `Program.compile_legacy` / `run_legacy` exist to make
  the legacy compiler the *differential oracle*. On deletion day the oracle
  legs go away. The differential suites
  (`test_if_lowering`, `test_loop_lowering`, `test_par_lowering`,
  `test_dynamic_target_lowering`, `test_inverse_lowering`,
  `test_lowering_oracle`, `test_usage_crosscheck`) already assert against
  the source's own plain-Python semantics (`Program.call`) or pinned values;
  conversion is mechanical: delete the `run_legacy(program.compile_legacy(), ...)`
  halves and the now-dead helpers, keep the new-pipeline halves
  (`capturing_lower`, direct expectations). Where a test exists *only* to pin
  a legacy divergence (e.g. some `test_if_lowering` notes), keep the
  expected-value assertion as the IR spec and drop the legacy comparison.
- `test_usage_crosscheck.py` reconstructs legacy flow-maps via
  `InetFunctionCompiler` — retire the legacy half, keep the unconditional
  IR-usage assertions.
- `test_recorder_split.py` and `tests/experiments.py` just switch
  `from natsune.compiler import inet` → `from natsune.inet import inet`.

### 4c. The flip and the sweep

Single landing (C + D can be one PR if 4a is already in):

1. `natsune/inet.py` lands (decorator, `InetFunction`, `compile_function`).
2. `frontend/link.py` + `frontend/ir/nodes.py` speak `InetRef` (new shape
   only — no transitional dual protocol, since 4a/4b already migrated every
   `__inet__` producer: decorator, backend helpers, test fakes).
3. `backend/agents.py`: `callee_invocation` drops the legacy branch.
4. `backend/python_backend.py`: entry point stays `as_callable`; imports
   runtime helpers from `backend/runtime.py`.
5. Delete `natsune/compiler.py` and `natsune/legacy.py`.
6. Delete `tests/test_compiler.py` (superseded by `tests/test_inet.py`).
7. Sweep: `grep -rn "natsune.compiler\|natsune\.compiler" src tests` → zero
   hits (the `test_link.py` frontend-import guard then passes vacuously).
   Update the `frontend/__init__.py` docstring (it still says "the old
   `natsune.compiler` module stays untouched").
8. Optionally re-export the public API from the empty
   `natsune/__init__.py` (`from natsune.inet import inet`) so user code
   reads `from natsune import inet`.

## Sequencing (suite green after each)

| Phase | Contents |
|---|---|
| A | §1: `backend/runtime.py`, shrink `compiler.py` |
| B | §2 + §3: `inet.py` (marker + deferred driver + entry point), `PromiseExpansion` agent, `InetRef` protocol, `tests/test_inet.py` incl. deferral + recursion cases; `link.py`/`nodes.py`/`agents.py` flip; fakes + backend helpers migrate. Old compiler still present but unreferenced except by its own tests |
| C | §4: delete `compiler.py`, `legacy.py`, `test_compiler.py`, legacy test legs; sweep |

(B and C may land together; the split exists so the end-to-end decorator
suite can be reviewed against the old suite while both are runnable.)

## Decisions (resolved at the cutover)

1. **Recursion:** RESOLVED — the `PromiseExpansion` lazy graft landed in
   the cutover (§2). By-reference mutation of the promise (fill-at-freeze)
   is the accepted Python-target semantics; generalizing agents to a labeled
   form for non-Python targets remains future work, and canonicalizing
   compiled interfaces to signature-only adapters (instead of today's
   params+locals bundles) is the natural companion cleanup — it would make
   promise adapters derivable from the signature alone.
2. **Entry-point default executor:** RESOLVED — `DeterministicSerialExecutor`
   (legacy parity); per-call and decorator-level `executor=` overrides
   preserved, `as_callable`'s Erasure unwrap-and-reraise kept.
3. **Marker name:** RESOLVED — kept `__inet__`.
4. **Public surface:** RESOLVED — `natsune/__init__.py` re-exports `inet`
   (`from natsune import inet`); the module lives at `natsune.inet`.
5. **Explicit eager API:** RESOLVED — `InetFunction.ensure_compiled()` is
   public; a `compile_module` convenience was deferred until needed.

# LOWERING — handoff for the backend phase (IR → nets)

This document replaces the phases 0–6 refactor plan (completed; archived in git
history). Its job is to carry forward exactly the context needed to design and
build the lowering phase: the runtime stack as it exists, the seam where target
abstraction belongs, the points where that stack is entangled with Python, the
proposed `Backend` protocol, and the open questions to settle.

## 1. Where things stand

- The new pipeline is complete through the IR:
  `extract_source` → `analyze_signature` → `collect_symbols` →
  `collect_call_links` → `build_ir` → `render_function`, all under
  `src/natsune/frontend/` (plain data in/out; the import ban against
  `natsune.compiler` is enforced inside `tests/frontend/test_link.py`).
- The old `src/natsune/compiler.py` is byte-identical to its pre-refactor state
  and remains the live implementation (`inet` decorator, AST → `VariablesFlow`).
- Tests: 256 passing total — 171 frontend + 46 legacy
  (`tests/test_compiler.py` etc. exercise only the old code) + 39 backend
  (declaration layer, recorder split, agent invocation, lowering oracle;
  `tests/backend/`).
- Snapshot workflow: `make snapshots-check` /
  `make snapshots-update` (env var `NATSUNE_UPDATE_SNAPSHOTS=1` on
  `tests/frontend/test_snapshots.py`; review generated `.ir` by eye).
- Recent IR additions since the phase-6 pause (all snapshot-tested):
  - Sum types are PEP 695 aliases (`type IrExpr = …`, `IrStmt`, `IrTarget`,
    `IrBodyExit`, `VariableUsage`).
  - `IrTargetDynamic.ast_node` / `IrDynamic.ast_node` — the original
    `ast.expr` kept on the node (`compare=False, repr=False`) so consumers
    never re-parse `source_text`; `materialize_dynamic` passes both to the
    backend, which uses whichever is easier.
  - `IrBody` carries derived **fields** (not properties), computed bottom-up at
    build time by `analyze_ir_body(statements)` in `frontend/ir/nodes.py`,
    called from `_build_body` in `frontend/ir/builder.py`:
    - `variable_usage: Mapping[str, "read" | "write"]` — non-global usages,
      child bodies' own usages merged into parents (write wins over read).
    - `disjunctives` — the `IrIf | IrWhile | IrFor` statements before the exit
      with at least one own body whose exit is None (returns flow to the list).
    - `exit: IrBodyExit | None` — first statement that never returns flow
      (bare return/continue/break, or a disjunctive whose bodies all exit).
    - Invalid post-exit structures raise `IrStructureError` **at build time**.
  - The renderer prints `(usage a :write b :read)` per body and marks
    statements with ` *` (disjunctive) / ` !` (exit) after the tag.

Open semantic leftovers inherited from the old plan's §10 that are now
*lowering* decisions: AugAssign Ref/InPlace semantics (keep `IrAugAssign`
un-desugared; decided for now: lowering rebinds in all cases, matching the
old compiler — see §8.6), try/except\* machinery (still rejected outright),
and chained-assignment aliasing semantics.

## 2. The runtime stack the backend must live in

| Layer | File(s) | What it is |
|---|---|---|
| Ports / wires | `ports.py` | The net substrate: `Wire`, `WirePort`, `ValuePort`, `ConstantValuePort(value: Any)`, `Erasure`, `CombPort`, `ExtMergeFuncPort`/`ExtSplitFuncPort` (hold raw `fn: Callable`), `Graft(execute: Expansion, wires)` where `Expansion = (executor, port, wires) -> None`. Note `type Target = Wire \| Port` already exists here — do **not** reuse the name `Target` for the backend protocol. |
| Connector | `connector.py` | Primitives: `connect`, `annihilate`, `duplicate`, `sequenced_tuplate_from`, … `ExpansionBuilder(Connector)` accumulates `active_pairs` — a *net template* that is later copied into a live executor per invocation (`flow.invocation(connector)`). Also `serialize_wire`/`new_wires_cache` — existing graph→data machinery. |
| Adapters | `adapters.py` | Declarative wiring discipline: `Adapter` protocol (`initialize`, `close`, `produce_egression`, `produce_ingression`, `unpack`, `repack`, `__iter__`, `adapter_wiring_type`); `ValueAdapter`, `ParValueAdapter`, `ReferenceAdapter`, `InverseAdapter`. They only issue Connector ops — structural, target-independent by construction. |
| Registers | `registers.py` | Builder ergonomics over Connector+Adapter: `FromRegister`/`ToRegister` (one-shot, `split`/`duplicate`/`\|`/`&`/`+`/invert), `InterfaceRegister`s, `FlowRegister` (multi-extension variable cells) + `FlowRegisterUsage(flow_read, flow_write)`, `send_value`/`send_values` (adapter-aware, walks unpack/repack chains), `parallelize_value`/`serialize_values`/`borrow_registers`. |
| Flows / agents | `control_flow.py`, `invocations.py`, `calculus.py` | `VariablesFlow(ExpansionBuilder)` — the function-level template: `Variables` (name → Adapter), `return_adapter`, one `FlowRegister` per variable, exceptions flow, `FlowInput`/`FlowControl` adapters. Agent library: `IfThenElse`, `Loop`, `Tracer`, `CloseAfterContingent`, `MergeInputTo/OutputInto`, split/merge/filter invocations (`expansion_invocation` grafts an `ExpansionBuilder` as a `Graft`). |
| Old lowering | `compiler.py` | `InetBranchCompiler` — AST → the above. To be replaced by the new backend; its behaviors are the reference/oracle, not the spec. |

## 3. The seam: it's `Connector`

Everything Python-entangled in the *structure* layers funnels through one
protocol. Adapters take `connector: Connector`; registers are ergonomics over
Connector; `ExpansionBuilder` *is* a Connector. So the target abstraction is
not "re-implement registers and adapters" — it is:

> **Reuse adapters, registers, invocations, and `VariablesFlow` unchanged;
> drive them against whatever object implements the Connector primitive set.**

- Python backend → today's executor-backed `Connector` (zero rewrite).
- C++ backend → an *emitter* implementing the same primitive set, recording the
  net template and emitting agent definitions as source text. Templates are
  already data-shaped (`active_pairs`, `serialize_wire`), so an emitter can
  walk them; the only opaque things it meets are `Expansion` closures (§4).

## 4. Where Python is entangled (the complete short list)

1. **`Expansion` closures inside `Graft`s.** Not only control agents
   (`IfThenElse`, `Tracer`, …): `ExtMergeFuncPort`/`ExtSplitFuncPort` hold raw
   callables, and our own code passes builtins/user functions as ext fns
   (`filter_invocation(iter, …)`, `merge_invocation(eval_expression, …)`,
   `fold_split(unroll, …)`).
2. **Eval/exec fallbacks.** `eval(ast.unparse(...))` for globals/handler types
   (compiler.py:223, 641) and `exec_expression`/`eval_expression` as merge
   functions — the ancestors of `IrDynamic`/`IrTargetDynamic`.
3. **Python objects as constants.** `ConstantValuePort(value: Any)`.

The fix for (1) is the core design move: **grafts become declarations, not
closures** — an agent registry `(name, ports, semantics)` that is literally the
interaction-net symbol table:

```python
@dataclasses.dataclass(frozen=True)
class AgentDef:
    name: str
    input_adapter: Adapter   # ParValueAdapter when multiple: the adapter
    #                          lattice already has the product type — one
    #                          spelling per interface
    output_adapter: Adapter
    impl: InetCallable | NetTemplate | Primitive   # tagged union
```

Lowering then only ever handles wires, ports, and agent references.
`PythonBackend` resolves declarations to closures (today's behavior);
`CppBackend` resolves them to generated functions. Anything undeclarable
(eval/exec fallback) must go through an explicit per-target hook and be gated:
Python evaluates, C++ refuses or requires the callee be natsune-compiled.

## 5. The proposed Backend protocol

```python
# natsune/backend/
class Backend(Protocol):
    # Compile-time only for now: VariablesFlow-level templating + agent
    # declarations. Runtime — the live per-invocation substrate a net is
    # instantiated into — is deferred until this lands (see §8.1).

    def declare_agent(self, name: str, defn: AgentDef) -> AgentRef: ...
    # Target-neutral declaration; only impl RESOLUTION differs per target.
    # PythonBackend: NetTemplate → instantiate_template closure (today's
    # copy-per-invocation), InetCallable → duck-type the runtime object
    # (live Expansions as themselves; legacy compilers via
    # callee_invocation). CppBackend: NetTemplate → emitted function,
    # Primitive → intrinsic, foreign InetCallable → refuse.

    def resolve_call(self, ref: Any) -> AgentRef: ...      # IrCallInet.ref
    def agent_def(self, ref: AgentRef) -> AgentDef: ...    # registry hop:
    # refs are by-value handles; lowering needs the impl back to resolve.
    def materialize_dynamic(self, node: ast.expr, source_text: str,
                            captures: Mapping[str, FromRegister],
                            adapter: Adapter, connector: Connector) -> FromRegister: ...
    # Receives BOTH the original ast node and its unparsed text; backends
    # use whichever is easier (Python evals the text, an emitter can
    # pattern-match the node). node is always a real ast.expr — synthesized
    # augassign dynamics carry a synthesized BinOp (the old compiler.py:833
    # construction), and text derives from the node via ast.unparse so the
    # two can never disagree. Mirrors IrDynamic/IrTargetDynamic.ast_node.
    # captures is the ALREADY-LOWERED form: IrDynamic.captures is
    # tuple[tuple[str, IrExpr], ...] at the IR, but lowering resolves each
    # IrExpr to a FromRegister before calling — the backend never sees
    # IrExpr. FromRegister (not Port) because the eval context needs
    # register identity (serialize_values/borrow_registers); connector is
    # the ambient flow (captures may be empty); returns FromRegister
    # (dynamics inline eagerly, not deferred grafts).
    def finish(self, flow: VariablesFlow) -> Artifact: ...
    # Declares the function's own body as an AgentDef (NetTemplate extracted
    # from the flow's active_pairs). What the artifact *is* — runnable vs.
    # text — is §8.1, deferred with the runtime protocol.

def lower(function: IrFunction, backend: Backend) -> ...: ...
```

Naming decisions (settled in discussion — don't re-litigate without cause):

- **`Backend`**, not `Target`: `ports.Target` is taken (`Wire | Port`) and
  load-bearing. Precedent: rustc's `Backend` trait
  (`rustc_codegen_llvm`/`_cranelift`). Package `natsune/backend/`, mirroring
  `frontend/` (the empty package skeleton already exists).
- Verb is **`lower`/`lowering`** — matches the refactor doc's own vocabulary
  (the old `InetBranchCompiler` was "the actual lowering") and avoids
  overloading "compile", which means the whole pipeline here.
- `Target` as a concept means the *destination flavor* (`PythonBackend`,
  `CppBackend` instances) — fine as a doc word, not as a type name.

### 5.1 VariablesFlow over the recorder split

`ExpansionBuilder` conflates two roles: the target-neutral **recorder**
(adapters, interface registers, `active_pairs`, `close`/`optimize` —
`connector.py:227`) and a Python-only **closure** (`__call__(exec, port,
wires)`, copying pairs into a live executor — the `Expansion` a `Graft`
holds). `VariablesFlow` itself uses only Connector-level operations
(`FlowRegister(adapter, self)`, `send_value(..., self)`, interface wiring);
it subclasses `ExpansionBuilder` purely to *be* a graftable template.

1. **Extract the recorder.** `NetTemplateBuilder(Connector)` owns adapters,
   interface registers, `active_pairs`, `close`+`optimize`,
   `serialize_active_pairs`. `ExpansionBuilder(NetTemplateBuilder)` keeps
   only `__call__`. Behavior-preserving; oracle-checked.
2. **Liberate the closure.** `__call__`'s body becomes
   `instantiate_template(template, exec, port, wires)`;
   `ExpansionBuilder.__call__` delegates. "Use this template here" is now a
   library function the Python backend owns, not an intrinsic method of
   templates. Re-base `VariablesFlow(NetTemplateBuilder)` — behaviorally
   identical.
3. **Grafts carry references.** `expansion_invocation` becomes
   `agent_invocation(ref, ...)`: `Graft` records `(AgentRef, port, wires)`
   instead of an `Expansion` closure. The registry resolves: `impl =
   NetTemplate | InetCallable | Primitive` (§4's union), with
   `NetTemplate` extracted from a finished builder's `active_pairs` (already
   data via `serialize_active_pairs`).
4. **`lower()` through the backend.** `lower` builds `VariablesFlow`s as
   templates (each recording into itself), declares them via
   `declare_agent`, `finish(flow)` exports the body template as an
   AgentDef. That is the whole protocol for now — runtime (live substrate,
   per-call instantiation) is deferred until templating + lowering land
   (§8.1).

Steps 1–2 are behavior-preserving refactors and can land before any backend
code (migration step 0). Nuance kept on purpose: `VariablesFlow` remains *a*
connector — its registers record into it, and bodies must stay re-invocable
templates (a flat net can't loop). Templates staying self-contained is what
keeps the golden-net oracle meaningful (a `VariablesFlow` diffs the same
regardless of eventual substrate) and is what lets the runtime side — the
live per-invocation substrate — be deferred wholesale without touching the
middle stack. `declare_agent` is one API on both targets — registration is
declaration, resolution is per-target (see the sketch comments).

**Status: steps 1–2 landed.** `connector.py` now has `NetTemplateBuilder`
(recorder) and `instantiate_template` (the liberated closure);
`ExpansionBuilder(NetTemplateBuilder)` delegates `__call__` to it.
`VariablesFlow` is re-based on `NetTemplateBuilder` with a delegating
`__call__` — Expansion-compatible until step 3 makes grafts carry
AgentRefs. `backend/types.py#net_template_of` is the canonical
`NetTemplate` producer (used by `PythonBackend.finish`). Oracle: full
legacy suite green.

**Step 3: landed.** `AgentRef` moved to
`ports.py` (it is net-level currency); `Graft` carries an optional
`agent: AgentRef | None`, and `serialize_port` renders tagged grafts as
`graft:<name>` (untagged legacy grafts render exactly as before).
`backend/runtime.py#resolve_impl` resolves declarations to Expansions
(InetCallable wrapping a live Expansion only — NetTemplate/Primitive
refuse until §8.1; legacy compilers refuse here, they resolve through
callee_invocation) and
`agent_invocation` is the declarative counterpart of
`expansion_invocation`: identical wiring, adapters from the declaration,
graft tagged with the ref. The legacy agents still call
`expansion_invocation` directly — deliberate, not deferred debt: sweeping
them would change legacy serialization output (untagged `graft`
renderings that legacy tests assert verbatim), and those classes die at
cutover anyway. `agent_invocation` is the lowering-facing entry; new
lowering code consumes it exclusively, starting with the composite
prototype (§7.2b).

## 6. Practical notes

- **`FlowRegisterUsage` ↔ `IrBody.variable_usage`.** `registers.py` already
  tracks per-variable `flow_read`/`flow_write`, and
  `variables_readout(flow_map)` branches on it. The IR's build-time
  `variable_usage` is the same analysis one level up — the new lowering can
  feed `IrBody.variable_usage` (and per-body `disjunctives`/`exit`) directly
  into flow-register and continuation decisions. Cross-check these two
  analyses against each other; disagreement is a bug in one of them.
- **Golden-net oracle.** `serialize_wire` + `new_wires_cache` can render any
  `VariablesFlow` to data. While `compiler.py` lives, the new lowering can be
  diffed against the old compiler's flows graph-for-graph. There is also
  prior art for emitting code from this stack:
  `karakuri.codegen_buffer.generate` is already used in `invocations.py`
  (`make codegen` regenerates types). **First recorded divergence:** the IR
  constant-folds foldable unary ops (`return -1` → `IrConst`) where legacy
  emitted a dynamic eval net — the IR is the spec; asserted deliberately in
  `tests/backend/test_lowering_oracle.py`.
- **Opaque callee refs.** `IrCallInet.ref` currently holds old-style
  `__inet__` compiler objects; the frontend treats them as opaque and copies
  metadata (arity, adapters). Realized in §7.2a: `resolve_call` mints
  `call_N` declarations keyed by `id(ref)` (the compilers are unhashable
  dataclasses) and `agent_def(ref)` is the hop back; at cutover the ref
  type becomes `CompiledFunction`. Known approximation: when a legacy
  callee lacks a `return_adapter` attribute, the declared output adapter
  falls back to VA (this describes the call interface; the resolved flow
  interface that `callee_invocation` wires is taken from the callee flow
  itself).
- **Nondeterminism.** The old `random_identifier` must not leak into backend
  agent names — cross-target/cross-module linking needs stable names (the IR
  already uses deterministic `__natsune_N__` placeholders).
- **Exceptions.** Exception capture (`ExceptionSink`, `collect_exceptions`,
  try machinery) remains deferred; `VariablesFlow.exceptions` is a
  `FlowRegister` threaded through every template today, so the backend will
  need at least a stub story for it.

## 7. Migration path

0. **Recorder split** (§5.1): extract `NetTemplateBuilder` from
   `ExpansionBuilder`, move the copy-closure into `instantiate_template`,
   re-base `VariablesFlow` — behavior-preserving, oracle-checked, no
   backend code required. **Landed.**
1. `natsune/backend/` skeleton + `PythonBackend` as a thin shell over the
   existing executor/eval machinery (behavior-preserving by construction).
   **Landed:** `backend/types.py` (AgentRef, InetCallable, Primitive,
   NetTemplate, AgentDef, LoweredUnit, net_template_of), `protocol.py`
   (compile-time subset per §5), `python_backend.py`
   (declare/resolve/materialize_dynamic/finish — materialize_dynamic is
   implemented, see §8.4), `agents.py` (the survey: primitives as static
   AgentDefs; composites classified per-call-site — the flow_map coupling
   is the load-bearing note), `runtime.py` (resolve_impl +
   agent_invocation, §5.1 step 3). The runtime half of `finish` is still
   open (§8.1).
2. Write the new Ir-lowering against the protocol; diff its `VariablesFlow`s
   against the old compiler's using the serializer as the oracle (all 18
   snapshot programs exist for this).
   **First slice landed:** `backend/lowering.py#lower_function` lowers the
   straight-line subset (Const/Var/Dynamic expressions; Assign with the
   legacy target-to-target chain, AugAssign rebind per §8.6, Return
   stop-at-first-return, ExprStmt, implicit-None tail) with exact
   `serialize_active_pairs` equality against legacy (oracle programs in
   `tests/backend/test_lowering_oracle.py`; see the divergence note in §6).
   Remaining lowering work, in order:

   a. **IrCallInet — landed.** `lowering.py#_from_expr` routes IrCallInet
      through `backend.resolve_call` + `backend.agent_def(ref)` +
      `runtime.callee_invocation(defn, connector)`: manual wiring (a
      `Graft(flow, [Wire()], agent=ref)` built by hand), because the
      legacy `InetFunctionCompiler.invocation` creates its own untagged
      graft deep inside `VariablesFlow.invocation`, out of reach for
      tagging. `callee_invocation` duplicates the legacy register-sorting
      dance (slot 0 = callee exceptions, closed; one register per
      parameter; locals erased) and the `with closer(...)` is
      load-bearing: its exit annihilates the interface registers'
      dangling state wire-ends, which serialization is sensitive to.
      Oracle: three call cases in `tests/backend/test_lowering_oracle.py`,
      exact equality after stripping declaration-layer `graft:<name>`
      tags (the tag is registry identity, not net structure — §5.1
      step 3). Discoveries recorded along the way:

      - **`PythonCallable` renamed to `InetCallable`** (field `fn` →
        `ref`): the union member means "a callable net object in the
        runtime environment" — today the legacy `__inet__` compiler, at
        cutover `CompiledFunction` — resolved per-target by duck-typing:
        live Expansions resolve as themselves (`resolve_impl`); legacy
        compilers via `callee_invocation`'s manual wiring. Not
        Python-specific: a C++ backend resolves same-unit callees to
        emitted functions and refuses foreign ones.
      - **`resolve_call` keys by `id(ref)`.** `InetFunctionCompiler` is
        an unhashable dataclass, so object-keying crashed on first real
        use (the old test's dummy was hashable and hid it).
      - **Legacy flow_map constraint — not inherited.** Legacy
        `FlowVariableMap.update` (control_flow.py:483) crashes unless
        every callee variable name already exists in the caller's map;
        legacy only compiles calls when names coincide (all legacy-suite
        call sites happen to). `callee_invocation` skips the flow_map
        side effect by design (it moves into lowering, fed by
        `IrBody.variable_usage`), so the new lowering has no such
        constraint; the oracle call cases name variables to keep the
        legacy side compilable for comparison.
      - **Exceptions gate:** `should_capture_exceptions` is False for
        now — IrTry doesn't exist, so no body can request capture; when
        it lands, its presence in the body decides the gate (see §6
        exceptions bullet).

   b. **IrIf.** Lower branches as VariablesFlows; declare a per-site
      composite AgentDef (impl=NetTemplate embedding branch AgentRefs) and
      invoke via `agent_invocation`. Replace IfThenElseStatement's
      flow_map side effect with IrBody.variable_usage; requires the
      wire_continuation + mapped_variables_readin machinery in
      control_flow.py. Hardest piece — study the old If branch
      (compiler.py:886–908) first.
   c. **IrWhile/IrFor.** Loop composite; same threading plus recursion
      (an AgentRef appearing in its own template is just a cycle).
   d. **IrTuple/IrTargetTuple/IrParIndex/IrBoolOp.** Par packing and
      element-wise typing; mostly mechanical.
   e. **Full 18-snapshot oracle.** Point the oracle at all of
      `tests/frontend/programs.py` (delayed_inverse and the infinite-loop
      programs may need special handling).
3. Only then `CppBackend`: Connector-as-emitter + agent registry → C++ source.
4. Cutover (delete `compiler.py`, switch the `inet` decorator) stays a
   separate, last step.

## 8. Open questions to settle (in rough priority order)

1. **Python target output kind** — lower to in-memory net objects through the
   existing `invocation` machinery, or generate Python source text? This
   decides what `Backend.finish` returns and whether backends need a common
   wrapper (e.g. `LoweredUnit`) or target-specific artifacts. `connector()`
   was removed from the protocol until this is settled; the first milestone
   is templates + declarations only (compile-time protocol, §5).
2. **Agent taxonomy** — what exactly goes in `AgentDef.impl`'s union; how the
   `control_flow.py` agent library (`IfThenElse`, `Loop`, `Tracer`, …) is
   declared; whether ext fns (`iter`, `unroll`, `join`, `eval_expression`)
   become primitives per target or a target-provided table.
3. **Constants** — `Any`-typed `ConstantValuePort` vs. serializable constants
   for C++; where the adapter-driven value discipline constrains them.
   Constants enter nets solely as recorded `ConstantValuePort`s inside
   templates (lowering uses `as_constant_register` directly; there is no
   `constant()` factory on the protocol — dropped as unused), so this
   question is purely about how an emitter serializes what's already
   recorded.
4. **Dynamic fallback gating** — is `materialize_dynamic` Python-only by
   contract, with a diagnostic for C++? (Likely yes; matches `IrDynamic`
   being the escape hatch.) **Resolved (first lowering slice):** the
   signature is `captures: Mapping[str, FromRegister]` — bare Ports lacked
   the register identity `serialize_values`/`borrow_registers` consume —
   plus an explicit `connector` (captures may be empty) and a
   `FromRegister` return (dynamics inline eagerly, not deferred grafts);
   `node` is a strict `ast.expr` — augassign lowering synthesizes the same
   BinOp the old compiler built and derives `source_text` from it via
   `ast.unparse`, so node and text can never disagree. The eval
   fallback lives in `PythonBackend.materialize_dynamic` verbatim
   (old construct_context); lowering routes through the protocol and is
   target-agnostic for dynamics.
5. **`IrStructureError` at build** — currently raised out of `build_ir`; should
   it become a `DiagnosticSink` diagnostic instead (analysis-style) now that it
   fires during parsing? (Cutover-consistency question, not urgent.)
6. **AugAssign Ref/InPlace semantics** — provisionally decided: lowering uses
   the rebind interpretation (`target = target op value`) in all cases. No
   `__iadd__` type dispatch (type logic stays limited to annotations) and no
   runtime dual-path (keeps lowering simple). For Ref-typed targets the
   rebind still egresses linearly through the reference adapter, so aliasing
   observers see the shared cell update — "mutation" in the reference sense
   only, never `__iadd__` dispatch. This is the existing compiler's behavior
   (AugAssign lowers to read-modify-write through the target register), so
   the golden-net oracle should agree. Python-style in-place methods remain
   reachable by calling `__iadd__`/`__isub__`/… directly as ordinary calls.
   Revisit only if a true in-place update primitive proves necessary.
   **try/except\*** remains rejected outright and is still blocked on lowering
   design (both are inherited §10 leftovers).

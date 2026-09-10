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
- Tests: 287 passing total — 174 frontend + 49 legacy
  (`tests/test_compiler.py` etc. exercise only the old code) + 64 backend
  (declaration layer, recorder split, agent invocation + call lowering,
  lowering oracle, usage cross-check, if lowering + differential
  execution; `tests/backend/`).
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
      Usage classifies the EFFECT on the variable's cell, not the syntactic
      position: "read" = independent copy (cell identity preserved);
      "write" = linear use that advances the cell. Assignments are
      writes; a read is a write whenever the variable's wiring type is
      not the VALUE leaf — Reference/Inverse read linearly, and ALL Par
      reads linearize today (a Par read is a whole-Par readout with the
      neighbor elements closed). Classification is adapter-declared via
      `adapters.read_independently` over the wiring type, with
      SymbolsTable.variables as the env. The Par conservatism is marked
      for post-cutover relaxation (element-pass-through reads → the
      recursive discipline rule: a Par is a read iff every constituent
      is) — see the §6 marker.
    - `disjunctives` — the `IrIf | IrWhile | IrFor` statements before the
      closer that can fall through (`Exits.FALLTHROUGH` set on their
      `exits`; returns flow to the list).
    - `closer: IrBodyExit | None` — first statement that never returns flow
      (bare return/continue/break, or a disjunctive with no fall-through
      path).
    - Invalid post-close structures raise `IrStructureError` **at build
      time**.
  - The renderer prints `(usage a :write b :read)` per body and marks
    statements with ` *` (disjunctive) / ` !` (closer) after the tag.

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
  feed `IrBody.variable_usage` (and per-body `disjunctives`/`closer`) directly
  into flow-register and continuation decisions.
  **Cross-checked (tests/backend/test_usage_crosscheck.py).** Semantics
  settled: usage classifies the effect on the cell — "read" = independent
  copy, "write" = linear use (advances the cell). Ref/Inverse read-as-write
  is INTENDED semantics (linear reads are writes), adopted by the IR via
  `adapters.read_independently`; the analyses now agree everywhere legacy
  is complete, and the soundness invariant is unconditional (legacy flag
  ⇒ IR entry; legacy write ⇒ IR write). Legacy's flags remain a partial
  record — fragment flows whose flags never reach the parent: if/while
  tests evaluate in a `new_test()` flow that is closed and grafted, never
  merged (test reads invisible — is_it_even's `input` is (False, False)
  in legacy, "read" in IR); for-target writes bypass `FlowRegister.readin`
  via the deconstruct case flows' interface (sum_it_up's `i` is
  read-flagged only); iterable captures land on case flows too
  (`start`/`end` unflagged). Consequences for 2b: (a) source wiring flags
  from the IR — it is the complete analysis; (b) the golden-net oracle
  for composites must still reproduce legacy's *decisions*, which were
  made on the partial flags — expect deliberate divergences around
  if-test reads and for-target cells, asserted as the IR being the spec
  (as with unary folding).
  **MARKED post-cutover improvement:** Par reads currently linearize —
  `read_independently` is VALUE-leaf-only because a Par read is a
  whole-Par readout with the neighbor elements closed (even a copyable
  element advances the cell). Element-pass-through reads (readout of the
  item, pass-through of the rest) relax the rule to the recursive
  discipline (VALUE recursively — a Par is a read iff every constituent
  is); this pin lives in test_par_reads_linearize + the read_independently
  docstring. Relatedly, `FlowRegister.readout`'s TODO ("adapter
  responsibility") resolves to this same predicate; defer the flip to
  cutover — the flags are legacy-internal and the new lowering does not
  consult them.
- **`flow_map` control-output flags (`finish`/`continue`/`break`/
  `return_output`).** **MARKED post-cutover removal:** the lowering
  mirrors legacy's dynamic marking (compiler.py:818/927/933) at the same
  statement positions, but nothing in the new path consumes the flags —
  `agent_invocation` skips the flow_map merge, `_lower_if` shortcuts the
  composite's control slots unconditionally (slice-1 branches cannot
  exit), and `finish` ignores the top flow's map. They are fully
  derivable from the IR (`IrBody.exits`, or the `closed` result
  `_lower_statements` already returns). When slice 2 hands shortcut
  decisions to lowering (per the §5.1 agents note), compute the maps
  from IR facts instead of mutating them during statement lowering, and
  drop the mirroring marks (search "§6 flow_map bullet").
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

   b. **IrIf — slice 1 landed (per-branch scheme).** The variable_usage
      union is never formed — it is an either-or, not a union: the parent
      wires the full variables bundle into the composite context (every
      cell extended — current value out, fresh state continues), each
      branch lowers as a self-contained flow (the legacy
      `new_branch().parse_statement_body` shape) whose fall-through emits
      every variable's final state, and the taken branch's bundle is the
      only live one — the composite dispatches. The continuation is the
      parent flow itself: branch outputs feed the extended cells, and
      result control slots the branches cannot emit are shortcut
      (mirroring wire_continuation). Branches are declared per-site
      (`if_N`, impl=InetCallable wrapping the IfThenElseStatement
      composite — a deviation from the older impl=NetTemplate plan: the
      dispatch is primitive machinery; embedding branch AgentRefs in a
      data template lands with §8.2). Validation: branch bodies remain
      pair-comparable with legacy (the reference is reconstructed via
      InetBranchCompiler — the compiled parent net's optimize pass
      inlines branch bodies and dissolves the composite, so the parent
      net is not a usable oracle here); composite wiring is IR-is-spec
      wholesale (the adopted divergence — legacy classified its context
      from the union and erased unflagged cells, we forward and let the
      branch decide); the decisive check is differential execution — the
      same program through the legacy compiler and the new lowering
      (tests/backend/test_if_lowering.py; its driver is the §8.1 runtime
      in miniature). Slice 1 scope: falling-through branches only
      (exiting branches raise; closer/disjunctive machinery is slice 2);
      tests limited to the supported expression set.

      **Slice 2 — next: exiting branches (unlocking is_it_even).** Scope
      is IrReturn inside branches; break/continue stay shortcut until
      §7.2c (they cannot occur without loops). Design, settled under the
      per-branch scheme:

      - **Branch side — mostly free.** Branch flows carry the full
        return adapter already; the existing IrReturn path in
        `_lower_statements` routes to the branch's own
        `control_output.return_value` and returns the closed-signal, so
        `_lower_branch` just skips the finish tail when closed. The
        composite routes the branch's whole FlowControl output to its
        result (`IfThenElseBase.__call__` sends
        `invocation.wire.readout() → conditional.result.readin()`), so
        no new branch machinery is needed. Narrow the
        `Exits.FALLTHROUGH`-only guard to permit RETURN; keep
        BREAK/CONTINUE raised (the Exits machinery —
        `_analyze_body_flow`/`IrBody.exits` — already classifies all of
        this at build time).
      - **Parent side — conditional slot consumption.** return_value is
        consumed (→ `flow.control_output.return_value.readin()`) iff
        either branch's exits carry RETURN; finish_variables is consumed
        (→ the extended cells, as in slice 1) iff either body falls
        through; otherwise shortcut. Legacy reference for the return
        merge: `wire_continuation`'s `b | a` (composite return |
        continuation return → parent return readin, compiler.py:758-766)
        — under our scheme there is no continuation flow, so it is just
        the composite's return into the parent's return readin.
      - **Both-branches-return:** the IrIf is the body's closer
        (`IrBody.closer` — already classified at build time);
        `_lower_body` must skip the implicit-None tail (`closed or
        self.ir.body.closer is not None`), and the context-extended
        cells' give-sides are never fed by a finish bundle — decide
        during implementation how to retire them (annihilate-on-shortcut;
        legacy's `shortcut()` is extend+annihilate, so mirroring that
        shape is the safe default).
      - **Mixed return/fall-through:** both slots live; the returning
        branch's internal finish readin dangles — legacy has exactly
        this shape and runs it (is_it_even), so no action expected, but
        watch the oracle.
      - **Validation:** differential execution extends directly — the
        driver in tests/backend/test_if_lowering.py is the harness; add
        is_it_even (`10 → True`, `11 → False`), mixed return/fall-through
        in both orders, and both-branches-return (the if as closer; the
        implicit-None tail must not fire). Branch-body pair-equality
        carries over unchanged — returning branch bodies reconstruct via
        the same InetBranchCompiler harness (parse stops at the return).
   c. **IrWhile/IrFor.** Loop composite; same threading plus recursion
      (an AgentRef appearing in its own template is just a cycle). Run
      AFTER slice 2 — continue/break control slots become live here and
      want the return-slot plumbing from exiting branches in place
      first.
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

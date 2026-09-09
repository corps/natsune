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
- Tests: 217 passing total — 171 frontend + 46 legacy
  (`tests/test_compiler.py` etc. exercise only the old code).
- Snapshot workflow: `make snapshots-check` /
  `make snapshots-update` (env var `NATSUNE_UPDATE_SNAPSHOTS=1` on
  `tests/frontend/test_snapshots.py`; review generated `.ir` by eye).
- Recent IR additions since the phase-6 pause (all snapshot-tested):
  - Sum types are PEP 695 aliases (`type IrExpr = …`, `IrStmt`, `IrTarget`,
    `IrBodyExit`, `VariableUsage`).
  - `IrTargetDynamic.ast_node` — the original `ast.expr` kept on the node
    (`compare=False, repr=False`) so consumers never re-parse `source_text`.
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
und desugared — decide at lowering), try/except\* machinery (still rejected
outright), and chained-assignment aliasing semantics.

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
    input_adapters: tuple[Adapter, ...]
    output_adapter: Adapter
    impl: PythonCallable | NetTemplate | Primitive   # tagged union
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
    def connector(self) -> Connector: ...
    def declare_agent(self, name: str, defn: AgentDef) -> AgentRef: ...
    def resolve_call(self, ref: Any) -> AgentRef: ...      # IrCallInet.ref
    def constant(self, value: Any, adapter: Adapter) -> Port: ...
    def materialize_dynamic(self, source_text: str,
                            captures: ..., adapter: Adapter) -> Graft: ...
    def finish(self, flow: VariablesFlow) -> Artifact: ... # runnable vs. C++ text

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
  (`make codegen` regenerates types).
- **Opaque callee refs.** `IrCallInet.ref` currently holds old-style
  `__inet__` compiler objects; the frontend treats them as opaque and copies
  metadata (arity, adapters). In the backend, `resolve_call`/`declare_agent`
  is where they become `AgentDef`s; at cutover the ref type becomes
  `CompiledFunction`.
- **Nondeterminism.** The old `random_identifier` must not leak into backend
  agent names — cross-target/cross-module linking needs stable names (the IR
  already uses deterministic `__natsune_N__` placeholders).
- **Exceptions.** Exception capture (`ExceptionSink`, `collect_exceptions`,
  try machinery) remains deferred; `VariablesFlow.exceptions` is a
  `FlowRegister` threaded through every template today, so the backend will
  need at least a stub story for it.

## 7. Migration path

1. `natsune/backend/` skeleton + `PythonBackend` as a thin shell over the
   existing executor/eval machinery (behavior-preserving by construction).
2. Write the new Ir-lowering against the protocol; diff its `VariablesFlow`s
   against the old compiler's using the serializer as the oracle (all 18
   snapshot programs exist for this).
3. Only then `CppBackend`: Connector-as-emitter + agent registry → C++ source.
4. Cutover (delete `compiler.py`, switch the `inet` decorator) stays a
   separate, last step.

## 8. Open questions to settle (in rough priority order)

1. **Python target output kind** — lower to in-memory net objects through the
   existing `invocation` machinery, or generate Python source text? This
   decides what `Backend.finish` returns and whether backends need a common
   wrapper (e.g. `LoweredUnit`) or target-specific artifacts.
2. **Agent taxonomy** — what exactly goes in `AgentDef.impl`'s union; how the
   `control_flow.py` agent library (`IfThenElse`, `Loop`, `Tracer`, …) is
   declared; whether ext fns (`iter`, `unroll`, `join`, `eval_expression`)
   become primitives per target or a target-provided table.
3. **Constants** — `Any`-typed `ConstantValuePort` vs. serializable constants
   for C++; where the adapter-driven value discipline constrains them.
4. **Dynamic fallback gating** — is `materialize_dynamic` Python-only by
   contract, with a diagnostic for C++? (Likely yes; matches `IrDynamic`
   being the escape hatch.)
5. **`IrStructureError` at build** — currently raised out of `build_ir`; should
   it become a `DiagnosticSink` diagnostic instead (analysis-style) now that it
   fires during parsing? (Cutover-consistency question, not urgent.)
6. **AugAssign Ref/InPlace semantics** and **try/except\*** — inherited §10
   leftovers, both blocking on lowering design.

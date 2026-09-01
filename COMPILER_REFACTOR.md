# COMPILER_REFACTOR — Plan for restructuring `src/natsune/compiler.py`

Goal: turn the single-file Python-AST → interaction-net compiler into a pipeline of
separate passes, using plain data classes and flat functions instead of interlinked
compiler objects, with intermediate results that can be constructed and inspected in
tests without an executor or live net.

**Revised scope (this revision of the plan):**

- We build **phases 0–6 only**, ending at an **IR** (intermediate representation) as
  the deliverable. The IR replaces the old "phase 6 = wire the dynamic fallback"
  idea: everything up to and including phase 6 is pure analysis + IR construction;
  no net objects are ever created in the new system at this stage.
- **Phases 7–9 (net lowering, statement lowering, packaging/cutover) are paused.**
  They are sketched at the end of this document but not built yet. After phase 6 we
  stop and evaluate (e.g. by rendering IRs for real programs) before proceeding.
- **The old implementation is untouched.** `src/natsune/compiler.py` stays exactly
  where it is and keeps working. All new work goes into **new files**
  (`src/natsune/frontend/`). Consequence: we cannot name the new package
  `natsune/compiler/` while `compiler.py` exists (Python can't have both), so the
  new code lives under `natsune/frontend/` and can be renamed/absorbed into
  `natsune/compiler/` at cutover time.
- **No characterization tests for the old compiler.** Some current behaviors are
  believed wrong (see §10). New tests target the *new* system and encode *intended*
  behavior. Where the new system deliberately diverges from the old one, the
  divergence is logged in §10 so cutover has a checklist.

---

## 1. Current state and entanglements

What lives in `compiler.py` today:

| Component | Role |
|---|---|
| `inet` decorator | Entry point; constructs `InetFunctionCompiler`, wraps call-time glue |
| `InetFunctionCompiler` | Source extraction, signature/annotations, symbol table owner, adapter inference, inet linking (`lookup_inet`), error construction, drives body compilation, provides `invocation()` |
| `InetVariablesEvaluator` | Pre-pass over AST: collects declared variables, marks global names, rejects unsupported nodes |
| `InetBranchCompiler` | The actual lowering: expressions (`evaluate_from_expression`, `evaluate_to_expression`, `evaluate_special_form_from_expression`), statements (`parse_statement_body`), control flow (`parse_test`, `parse_deconstruct_iter`, `parse_try_*`, `wire_continuation`), exception capture |
| `ReplaceWithSerializedVariables` | `NodeTransformer` that finds "special form" subexpressions, replaces them with fresh names, and builds the serialized locals context for `eval`/`exec` fallback |
| Runtime helpers | `_try_iter`, `_match_exception_group`, `construct_locals`, `exec_expression`, `eval_expression` |

Entanglements to break (these motivate the phase boundaries below):

1. **Circular references.** `InetBranchCompiler` holds `function_compiler` and calls
   back into it (`syntax_error`, `lookup_inet`, `infer_expression_adapter`,
   `used_as_globals`); `InetFunctionCompiler` creates branches and runs the
   evaluator. Two objects, one job, mutable shared state with an implicit ordering
   dependency: evaluator must run before any lowering.
2. **Analysis fused with effect.** `ReplaceWithSerializedVariables.generic_visit`
   calls `evaluate_special_form_from_expression` *during* AST traversal — the
   "which subexpressions are special?" question (pure analysis) is answered by a
   function that also wires nets (effect).
3. **`parse_statement_body` does everything.** ~150 lines: statement dispatch,
   per-statement lowering, control-flow continuation wiring, `flow_map` flag
   mutation, and implicit state — the live `body_iter` is captured by loop/if/try
   handlers to compile "the rest of the body", so the remainder is invisible
   state, not a value.
4. **Errors thrown from the depths.** `syntax_error` raises immediately from
   anywhere in the traversal, mid-net-construction. No diagnostic accumulation.
5. **Ad-hoc compile-time `eval`.** `lookup_inet`, `parse_try_handler`
   (handler type), and `visit_AnnAssign` (annotations) each `eval` against globals
   inline, with no single "link" concept.
6. **Non-determinism.** `random_identifier` makes rewritten source strings
   unpredictable, blocking snapshot tests.
7. **Hidden control flow.** Compilation is triggered by `__post_init__` +
   `cached_property`, so reading an attribute can run the whole compiler.

Non-goals: changes to the net substrate (`registers`, `ports`, `connector`,
`invocations`, `control_flow`, `interactions`, `optimizer`) are out of scope. The
public `inet` decorator API is preserved (cutover is a paused phase).

---

## 2. Design principles for the refactor

- **Phases with data contracts.** Each phase is a function: plain data in, plain
  data out. **Nothing in phases 0–6 creates a register, wire, or flow.**
- **Flat functions over methods.** Handlers are module-level functions; shared
  per-function state lives in frozen data classes passed explicitly.
- **Dispatch tables over if-chains.** Per-node-type handler tables replace the
  isinstance cascades.
- **Diagnostics before effects.** Every program error that *can* be detected is
  detected by phase 6 and lands in a diagnostic list. The IR is "validated by
  construction": anything that survives phase 6 should lower without program
  errors. (Note: this *moves* several checks that today fire during lowering —
  inet-call arity/keywords, Par subscript bounds, list-lvalue rejection — earlier.
  That is a deliberate, logged divergence: fail before any net construction.)
- **Determinism by injection.** Fresh-name generation is an injected factory so
  rewrites are snapshot-stable.
- **The IR is the inspectable layer.** Debugging, testing, and future
  optimization all read the IR, not the net graph.

---

## 3. The IR (the phase-6 deliverable)

### 3.1 Why a *structured tree* IR (and not a CFG)

The source language is fully structured (if/loops/try, no goto), and the net-level
target is likewise structured: `IfThenElse`, `Loop`, and `VariablesFlow`
combinators consume nested sub-flows, and variables themselves flow through these
combinators as first-class wires rather than living in an implicit store. A CFG
would force us to invent phi-like nodes for variable-state merge points; the
structured IR gets them for free from tree nesting, and each IR node maps ~1:1
onto a lowering handler later. If optimization over a flat graph is ever wanted,
tree → CFG is a mechanical, add-later transformation.

A crucial payoff of the tree: **"the rest of the body" becomes structure.**
Today, `parse_statement_body` passes a live `body_iter` into loop/if/try handlers
to compile the continuation — invisible shared state. In the IR, an `if`'s
continuation is simply *the sibling statements after the `if` in the enclosing
body*. Lowering later walks the tree and always knows what follows.

### 3.2 Node inventory (sketch)

All nodes are frozen dataclasses; every node carries its source location
(filename, lineno, col) for diagnostics, and every expression node carries its
`Adapter` (filled during construction from the inference phase).

**Function level**

- `IrFunction` — name, params (name + adapter each), body: `IrBody`,
  return adapter.
- `IrBody` — ordered list of `IrStmt`.

**Statements**

- `IrAssign` — targets (one or more; chained assignment modeled directly),
  value. Tuple-destructuring targets are `IrTarget` patterns (name or tuple of
  names), not re-parsed expressions.
- `IrAugAssign` — target, operator, value. **Kept as its own node, not desugared**
  to read-modify-write (see §10, suspect 1: Ref/InPlace semantics may differ).
- `IrIf` — test, then-body, else-body. `elif` chains nest as an `IrIf` in the
  else-body.
- `IrFor` — target pattern, iterable expression, body, orelse.
- `IrWhile` — test expression, body, orelse.
- `IrTryStar` — body, handlers (each: exception-type expr as *linked value*,
  optional bound name, handler body), orelse.
- `IrReturn` — value expr or None. `IrBreak`, `IrContinue` — leaf terminals.
- `IrExprStmt` — expression whose value is discarded.
- `pass` produces no node (dropped at construction).

**Expressions**

- `IrVar` — a resolved local variable: name + adapter (from the symbol table),
  plus an `is_global` flag for names routed through the dynamic path.
- `IrConst` — literal value.
- `IrTuple` — Par construction; element exprs.
- `IrParIndex` — base expr + constant integer index; only exists when the base's
  adapter is a `ParValueAdapter` and the index is a valid constant int.
- `IrBoolOp` — `and`/`or` with a flattened, n-ary operand list.
- `IrCallInet` — a call to a linked inet function. Holds an **opaque callee
  reference** (today that's the old `InetFunctionCompiler` behind `__inet__`; the
  frontend treats it as opaque) plus *copied* metadata: arity, per-arg adapters,
  return adapter. Copying metadata keeps IR construction independent of old
  internals; at cutover the reference type becomes `CompiledFunction`.
- `IrDynamic` — the eval/exec escape hatch as a first-class node: the rewritten
  source string (special-form subexpressions replaced by placeholder names), plus
  an ordered capture list mapping placeholder name → captured sub-expression **as
  IR**. Because captures are IR exprs, nesting is uniform and lowering recurses.
  This node is the direct product of the old `ReplaceWithSerializedVariables`
  scan, minus all wiring.

Notes on the inventory:

- `IrDynamic` is the catch-all: the builder tries to type an expression as one of
  the nodes above; if none applies (unknown calls, exotic subscripts, anything the
  old system sent to `eval_expression`), it becomes `IrDynamic` via the scan.
- There is deliberately no IR node for lvalues: assignment targets are
  `IrTarget` patterns on the statement nodes, since variables are the only
  first-class store (the old "lvalue via exec fallback" case shows up as
  `is_global` targets handled at lowering).
- The exception *capture* concept (old `collect_exceptions`) is a lowering-time
  concern (it wraps registers with `ExceptionSink`) and lives in a paused phase;
  the IR carries only the data lowering needs (e.g. `IrCallInet`'s adapter) to
  apply it.

### 3.3 Construction rules (what desugars, what validates)

Built in `frontend/ir/builder.py` as flat functions over the phase 0–5 outputs.
Decisions made once, here, instead of being smeared across lowering:

- Desugar: `AnnAssign` with value → `IrAssign` (adapter already registered by the
  symbols phase); `BoolOp` chains flattened; `elif` nesting normalized; multi-target
  `Assign` kept explicit; `pass` dropped.
- Validate (diagnostics, not exceptions): unsupported statement/expression types
  (the old `unsupported_stmt`/`unsupported_expr` tuples), inet-call keyword args
  and arity mismatch, Par subscript type/range, list-lvalue rejection, annotation
  and link failures (surfaced from earlier phases), tuple-pattern arity vs Par
  arity mismatch (§10, suspect 4: today this silently degrades to `VA`).
- Scan for dynamic fallback: `scan_dynamic(expr) -> (rewritten_source,
  captures)` where captures are AST subexpressions in deterministic order,
  placeholders allocated by an injected name factory. The builder wraps this into
  `IrDynamic` with captures re-built as IR (recursion through `build_expr`).
  Register-creation ordering — the old code's traversal order — becomes irrelevant
  at this stage since nothing is created; it becomes a concern of the paused
  lowering phase, which will walk the finished IR.

### 3.4 Rendering

`frontend/ir/render.py`: `render_function(IrFunction) -> str`, a deterministic
indented/S-expression-ish printer. This is the primary evaluation tool at the
pause: run it on real functions, read the IR, decide what's wrong before lowering
is written. It is also the backbone of golden tests.

---

## 4. Layout (new files only; `compiler.py` untouched)

```
src/natsune/frontend/
  __init__.py      # public surface: build_ir, IrFunction, render_function, ...
  diagnostics.py   # Phase 0
  source.py        # Phase 1
  signature.py     # Phase 2
  link.py          # Phase 3
  symbols.py       # Phase 4
  infer.py         # Phase 5
  ir/
    __init__.py    # re-exports node types
    nodes.py       # Phase 6a: the dataclasses from §3.2
    builder.py     # Phase 6b: AST -> IR (desugar, validate, scan_dynamic)
    render.py      # Phase 6c: renderer

tests/frontend/    # new-system tests only
  test_diagnostics.py
  test_source.py
  test_signature.py
  test_link.py
  test_symbols.py
  test_infer.py
  test_ir_builder.py
  test_ir_render.py
```

One module per phase is the starting point; merging small ones later is cheap,
splitting a monolith is not. Naming is bikesheddable (`frontend` vs `pyast` vs
`lower`); the constraint is only that it must not be `natsune/compiler` until the
old file goes away.

---

## 5. Phases 0–6: contracts and units of work

Every phase takes the previous phases' outputs plus a diagnostics sink; none of
them import from `natsune.compiler`. "Absorbs X" means "reimplements the logic
currently in old-compiler X, into the new file" — the old code itself is not
modified.

### Phase 0 — Diagnostics (`diagnostics.py`)

Absorbs `InetFunctionCompiler.syntax_error` and the scattered
`raise SyntaxError(...)` sites.

Units of work:
- `CompileDiagnostic` data class: message, filename, lineno, col_offset, severity;
  a `raised()` accessor producing the current `SyntaxError` shape (kept for
  cutover parity).
- Position resolution including the base-lineno offset arithmetic that currently
  lives inline (`lineno + self.lineno - 1`); implemented and tested here once.
- A collector/sink threaded through the context; policy: analysis phases
  accumulate, phase boundaries decide to raise or continue.
- Decision to settle: hard errors during *paused lowering phases* should be
  unreachable (validated-by-construction IR); keep an immediate-raise path for
  defense.

Tests: position formatting for nodes with and without location info; base-lineno
offsets for functions defined below the top of a module.

### Phase 1 — Source extraction (`source.py`)

Absorbs the `func_def` cached property (inspect.getsourcelines + ast.parse) and
lineno capture.

Units of work:
- `FunctionSource` frozen data class: func, filename, base_lineno, module AST,
  `FunctionDef` node, globals dict reference.
- `extract_source(func, globals) -> FunctionSource`; explicit validation that
  module body[0] is a `FunctionDef` (currently a bare assert).

Tests: synthetic nested/indented function definitions (the
`inspect.getsourcelines` behavior is subtle); AST shape and lineno bookkeeping.

### Phase 2 — Signature (`signature.py`)

Absorbs `args`, `return_annot`, `args_adapter`, and the kw-only/vararg/default
rejection.

Units of work:
- `Signature` data class: positional args as `(name, TypeExpression | None)`
  pairs, resolved `args_adapter` (Par), resolved `return_adapter`.
- `analyze_signature(source) -> Signature`.

Tests: 0/1/n args, missing annotations, rejection of kw-only/vararg/default.

### Phase 3 — Linking (`link.py`)

New concept centralizing the three scattered compile-time `eval` sites.

Units of work:
- `LinkResult`: a name resolves to an inet function (opaque reference + copied
  metadata: arity, arg adapters, return adapter), a plain value, or nothing.
- `link_name(name, globals) -> LinkResult` (absorbs `lookup_inet`).
- `eval_annotation(expr, globals)` (absorbs annotation eval from
  `visit_AnnAssign` and handler-type eval from `parse_try_handler`), with one
  consistent error format.
- Recursion note: linking other inet functions reaches *old* compiler objects via
  `__inet__` for now; the frontend stores them opaquely (§3.2, `IrCallInet`).

Tests: fake globals dicts — no real compilation needed. Error paths: absent name,
eval raises, object without `__inet__`.

### Phase 4 — Symbol collection (`symbols.py`)

Rework of `InetVariablesEvaluator` from a `NodeVisitor` into flat walk functions
producing data.

Units of work:
- `SymbolTable` frozen data class: `variables: dict[str, Adapter]`,
  `used_as_globals: frozenset[str]`.
- `collect_symbols(func_def, signature, globals, sink) -> SymbolTable`.
- Named sub-functions: `mark_target` (incl. global-target conflict diagnostic),
  assignment-target adapter inference (tuple/Par matching — revisit policy, §10
  suspect 4), annotated-assignment registration (simple-only rule),
  unsupported-node rejection.
- The `used_as_globals` rule (any `Name` not already a known local, including
  names assigned later) is re-decided here rather than copied blindly — §10
  suspect 2.

Tests: snippet in → `SymbolTable` out: simple/chained/tuple assignment, AugAssign
introducing names, AnnAssign with evaluated annotation, for-targets, global
reads. Diagnostics asserted from the sink.

### Phase 5 — Adapter inference (`infer.py`)

Absorbs `infer_expression_adapter`, `evaluate_call_adapter`,
`evaluate_subscript`.

Units of work:
- Flat `infer_adapter(node, symbols, links) -> Adapter` over Name / Call / Tuple
  / Subscript, default `VA`.
- Par subscript checking as its own function (constant-int requirement, range
  check), reused by IR construction.

Tests: in/out-of-range subscripts, non-constant slices, call adapters for linked
vs unlinked callees.

### Phase 6 — IR construction (`ir/nodes.py`, `ir/builder.py`, `ir/render.py`)

The new terminus of the pipeline. See §3 for the design; units of work:

1. `nodes.py` — the frozen dataclasses of §3.2.
2. `builder.py` — `build_ir(source, signature, symbols, links, sink) -> IrFunction`:
   statement dispatch table (per `ast.stmt` type), expression builder with
   typed-node-first / `IrDynamic`-fallback policy, desugaring rules (§3.3),
   validation (§3.3), `scan_dynamic` with injected name factory.
3. `render.py` — deterministic renderer.

Tests (the heart of the new test suite):
- Per-node construction: each statement/expression kind from a synthetic snippet,
  asserting the exact IR shape (adapters, resolved names, captured metadata).
- Desugaring: AnnAssign→Assign, elif nesting, BoolOp flattening, pass-dropping.
- `IrDynamic`: rewritten source string (deterministic placeholders), capture list
  contents and order, nested special forms inside dynamic expressions.
- Validation: every diagnostic path (arity, keywords, subscript bounds, list
  lvalues, unsupported nodes) asserted from the sink — and asserting that *no*
  exception was raised.
- Golden render tests for a battery of programs (the "read the IR" workflow).

---

## 6. Paused phases (sketch only — do not build yet)

Recorded so the IR can be checked against what lowering will need; revisit after
evaluation.

- **Phase 7 — expression lowering.** One handler per IR expression node;
  `IrDynamic` lowering reassembles the old wire-scan + eval-context construction
  (`construct_context`, `serialize_values`, `build_locals`). Ordering question to
  answer then: net-construction order (IR walk order vs the old traversal order).
- **Phase 8 — statement/body lowering.** Tree walk where "continuation" is the
  following siblings; per-statement handlers; `wire_continuation`-style helpers in
  a `backend/controlflow.py`; try-star machinery; exception capture. Outcome data
  classes may prove unnecessary — the tree already encodes termination
  structurally (a `body` whose last relevant statement is `IrReturn` vs one that
  falls off the end).
- **Phase 9 — packaging/cutover.** `compile_function` orchestrator, opaque
  callee references become `CompiledFunction`, `inet` decorator switched over,
  `natsune/frontend` renamed/absorbed into `natsune/compiler`, old file deleted.
  The §10 divergence log is the review checklist for this step.

---

## 7. Testing strategy

- **New-system tests only** (per current direction). No characterization tests
  are added for the old compiler; the existing `tests/test_compiler.py` stays as
  the untouched legacy gate (and remains green precisely because the old code is
  untouched).
- Per-phase unit tests, one file per module under `tests/frontend/`; every phase
  function runs on synthetic input (`ast.parse` a snippet) with no executor.
- Golden/snapshot tests anchored on the deterministic IR renderer and on
  `scan_dynamic`'s injected name factory.
- Intended behavior, not bug parity: when a new-system test and the old compiler
  disagree, that's expected — record it in §10.
- Fixture builders in `tests/frontend/helpers.py`: `make_source(snippet)`,
  `make_symbols(...)`, `build_ir_for(snippet)` one-liners.

---

## 8. Build order (within the paused scope)

1. Create `src/natsune/frontend/` skeleton + `tests/frontend/`; old
   `compiler.py` untouched (hence the `frontend` name — see §4).
2. `diagnostics.py` (Phase 0) with position tests.
3. `source.py`, `signature.py` (Phases 1–2).
4. `link.py` (Phase 3).
5. `symbols.py` (Phase 4) — making the §10 semantic decisions as we go.
6. `infer.py` (Phase 5).
7. `ir/nodes.py` → `ir/builder.py` → `ir/render.py` (Phase 6), with golden tests.
8. **Pause / evaluation checkpoint:** render IRs for the functions in
   `tests/test_compiler.py` and any real programs; read them; settle the §10
   suspects; only then plan phases 7–9.

Steps 2–6 are pure-data and low risk; step 7 is where the semantic decisions
concentrate — one commit per module, each fully tested.

---

## 9. Definition of done for this scope

- `natsune.frontend.build_ir` turns an `@inet`-compatible function into an
  `IrFunction` without creating any net objects and without importing
  `natsune.compiler`.
- A failing program produces diagnostics, never a mid-flight exception.
- `render_function` output is deterministic and readable enough to review real
  programs.
- All new tests pass; all old tests pass; `compiler.py` is byte-identical.

---

## 10. Semantic suspects to re-decide (divergence log)

Behaviors of the current system that look wrong or accidental. The new system
should decide each deliberately, implement the decision, and log it here — this
log becomes the cutover checklist in phase 9.

| # | Suspect | Where today | Candidate decision |
|---|---|---|---|
| 1 | `AugAssign` desugars to read-modify-write, which may not preserve mutation-through-reference semantics for `Ref`/`Inverse`-typed variables (cf. the `take_reference` test where `a += 10` mutates in place) | `parse_statement_body` AugAssign branch | Keep `IrAugAssign` as its own IR node; give it explicit Ref-aware lowering semantics later |
| 2 | Any name not already a local is treated as a global — including names assigned *later* in the function (read-before-assignment silently reads a global) | `InetVariablesEvaluator.visit_Name` | Decide: diagnostic on use-before-assignment vs preserve Python's global fallback |
| 3 | For-loop targets are force-marked with adapter `VA`; `ast.walk` over targets silently ignores non-Name nodes | `visit_For` | Type loop targets from the iterable where inferable; diagnose non-Name target nodes |
| 4 | Tuple-assignment targets silently get adapter `VA` when the value's Par arity doesn't match | `visit_Assign` | Make the mismatch a diagnostic |
| 5 | Several checks (inet-call arity/keywords, Par subscript bounds, list lvalues) fire during *lowering*, after nets are partially constructed | `evaluate_special_form_from_expression`, `evaluate_subscript`, `evaluate_to_expression` | Move all validation to IR construction (already the plan, §3.3) — a logged divergence by construction |
| 6 | Chained assignment `a = b = expr` wires `b` from `a`'s register rather than re-evaluating — subtle and untested | `parse_statement_body` Assign branch | Model explicitly in `IrAssign`; decide semantics once |

---

## 11. Risks and gotchas

- **Cutover-time regression risk is deferred, not eliminated.** Because the old
  compiler is untouched, nothing regresses now; but phases 7–9 will re-wire
  lowering with intentional divergences (§10). The renderer + divergence log are
  the safety instruments for that moment — invest in renderer quality now.
- **Opaque callee references.** Until cutover, `IrCallInet` holds old-style
  compiler objects behind `__inet__`. Don't let frontend modules call methods on
  them beyond the metadata copied at link time, or we reintroduce the coupling
  this refactor removes.
- **`inspect.getsourcelines` edge cases** (indented/class-level definitions) —
  currently untested anywhere; characterize behavior in Phase 1 tests before
  relying on it.
- **Base-lineno offset arithmetic** (`lineno + self.lineno - 1`) appears correct
  but untested; pin current behavior in Phase 0 tests, then decide if it's right.
- **Globals identity.** `func.__globals__` vs caller-frame filename capture in the
  `inet` decorator has frame-depth subtleties; Phase 1's `extract_source` should
  take both explicitly so the frame-walking stays isolated (and paused-phase
  reviewable).

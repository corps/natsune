# COMPILER_REFACTOR — Plan for restructuring `src/natsune/compiler.py`

Goal: turn the single-file Python-AST → interaction-net compiler into a pipeline of
separate passes, using plain data classes and flat functions instead of interlinked
compiler objects, with intermediate results that can be constructed and inspected in
tests without an executor or live net.

## Status (handoff summary — read me first)

**Phases 0–6 are built, tested, and paused before lowering.** The old
`src/natsune/compiler.py` is byte-identical to its pre-refactor state and
remains the live implementation; everything new lives under
`src/natsune/frontend/`, tested by `tests/frontend/` (219 tests total across
both suites, all green; run with `uv run pytest`).

What exists:

- The pipeline: `extract_source` → `analyze_signature` → `collect_symbols` →
  `collect_call_links` → `build_ir` → `render_function`. Plain data in and
  out; no net objects anywhere; no imports of `natsune.compiler` (enforced by
  an architecture-guard test).
- Phase 0 diagnostics: positioned, accumulate-then-decide via `DiagnosticSink`
  (exact-duplicate suppression for the pipeline's overlapping validations,
  plus an immediate-raise defense path). Fallback positions are owned by
  callsites, not the source map — see the Phase 0 decision note in §5.
- Golden IR snapshots for the 18 example programs copied from
  `tests/test_compiler.py` as real functions (`tests/frontend/programs.py`),
  stored under `tests/frontend/snapshots/`, gathered/checked via
  `make snapshots-update` / `make snapshots-check`. All 18 compile with zero
  diagnostics.

Reading order for a new maintainer: this status block, then §2 (principles,
including the settled `match`-over-dispatch-tables and no-up-front-fallbacks
decisions), the Phase 0–6 sections of §5 (each ends with its settled
Decisions block), §10 (the divergence log — semantic decisions plus the
old-compiler bugs found by probing: raw-KeyError crashes on assignment from
globals, silent positional-default/positional-only holes, the indented-
definition crash, the TryStar gap), and finally the snapshots themselves.

Conventions that differ from the original plan text (each recorded in its
phase section): statement dispatch uses `match`, not handler tables;
`SourceMap` carries no fallback position; `collect_symbols`/`build_ir` take
`FunctionSource` rather than bare `(func_def, globals)`; try blocks are
rejected outright (§10 row 12); phase-5 inference takes plain mappings and a
precomputed links dict.

What is NOT done: phases 7–9 (§6) — expression/statement lowering, packaging,
and cutover — and the §10 rows whose final form needs a lowering design
(rows 1 and 6 at minimum; row 2's AugAssign asymmetry is related). The
immediate next step is reading the snapshots closely, settling those rows,
and only then planning phases 7–9.

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
- **`match` over dispatch tables.** Per-node-type dispatch uses structural
  pattern matching (`match node: case ast.Assign(): ...`) rather than handler
  dicts or isinstance cascades — settled during Phase 4; handlers keep their
  concrete node types instead of `Any`.
- **Diagnostics before effects.** Every program error that *can* be detected is
  detected by phase 6 and lands in a diagnostic list. The IR is "validated by
  construction": anything that survives phase 6 should lower without program
  errors. (Note: this *moves* several checks that today fire during lowering —
  inet-call arity/keywords, Par subscript bounds, list-lvalue rejection — earlier.
  That is a deliberate, logged divergence: fail before any net construction.)
- **No up-front fallbacks.** Position resolution returns `None` when a node
  carries no location, rather than silently substituting a context-free default
  recorded before the real context was known. Whether an unlocatable node is
  worth reporting — and against which surrounding node — is a decision for the
  callsite (see the Phase 0 decision note).
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

## 4. Layout (built; `compiler.py` untouched)

```
src/natsune/frontend/
  __init__.py      # public surface: build_ir, IrFunction, render_function, ...
  diagnostics.py   # Phase 0
  source.py        # Phase 1
  signature.py     # Phase 2
  link.py          # Phase 3 (+ collect_call_links)
  symbols.py       # Phase 4 (incl. the absorbed UNSUPPORTED_EXPR/STMT tuples)
  infer.py         # Phase 5 (infer_adapter, par_subscript_index)
  ir/
    __init__.py    # re-exports node types + build_ir/render_function
    nodes.py       # Phase 6a: the dataclasses from §3.2
    builder.py     # Phase 6b: AST -> IR (desugar, validate, scan_dynamic)
    render.py      # Phase 6c: renderer

tests/frontend/    # new-system tests only (173 tests, all green)
  helpers.py           # make_source / parse_function / analyze_for / build_ir_for
  programs.py          # the 18 example programs as real functions (+ PROGRAMS)
  snapshots/*.ir       # stored golden IRs, one per program (18 files)
  test_diagnostics.py  # 20 tests
  test_source.py       # 20 tests
  test_signature.py    # 12 tests
  test_link.py         # 15 tests
  test_symbols.py      # 27 tests
  test_infer.py        # 19 tests
  test_ir_builder.py   # 35 tests
  test_ir_render.py    # 6 tests
  test_snapshots.py    # 19 tests (18 golden IRs + orphan check)
```

One module per phase held up; nothing needed merging or splitting. Naming is
bikesheddable (`frontend` vs `pyast` vs
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
- Decision (settled during Phase 1): position fallbacks are **caller-owned**.
  A fixed fallback `Position` stored on `SourceMap` at extraction time presumes
  knowledge about future diagnostic sites that the extractor cannot have — the
  right context (a parent AST node, the enclosing statement, the start of the
  block) exists at each callsite, not up front. So `SourceMap` carries only
  `filename` and `base_lineno`; `resolve(node)` and `CompileDiagnostic.at(...)`
  return `None` for nodes without `lineno`/`col_offset` (all-or-nothing: every
  `ast.parse` node has both, hand-built nodes usually neither). Callsites that
  must not lose a diagnostic resolve their own alternative — typically
  `source.resolve(node) or source.resolve(parent_node)` — and pass it to
  `CompileDiagnostic.at_position` / `DiagnosticSink.add_at`, which take the
  resolved position verbatim. The node-based sink sugar (`error`/`warning`)
  returns `CompileDiagnostic | None` and records nothing when the position is
  undeterminable: the obligation is visible in the signature. The
  immediate-raise defense path `fail` degrades to the positionless legacy
  bare-raise shape `SyntaxError(message)` — which the old compiler also had
  (e.g. the unsupported-args raise) — when even it has no position.
- A collector/sink threaded through the context; policy: analysis phases
  accumulate, phase boundaries decide to raise or continue.
- Decision to settle: hard errors during *paused lowering phases* should be
  unreachable (validated-by-construction IR); keep an immediate-raise path for
  defense.

Tests: position formatting for nodes with and without location info; base-lineno
offsets for functions defined below the top of a module; caller-owned fallback
chains (`resolve(node) or resolve(parent)`, `add_at`) and the None-return
contract of `resolve`/`at`/`error`.

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
Phase 1 is also where the Phase 0 fallback question was settled — extraction
has no basis to guess a default position, so fallbacks moved to callsites (see
the Phase 0 decision note); `extract_source` supplies the start of the
extracted block (in file coordinates) as its own callsite-level fallback.

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

Decisions (settled during Phase 4; details in §10 rows 2–4 and new rows 10–12):
- `collect_symbols(source, signature, sink)` takes the phase-1 `FunctionSource`
  rather than bare `(func_def, globals)` — it bundles them with the `SourceMap`
  the diagnostics need.
- Suspect 2: global-fallback rule preserved (probe-verified: read-then-assign
  already errors via the conflict diagnostic); self-referential first binding
  is now diagnosed; AugAssign-introduced names stay.
- Suspect 3: loop targets keep VA; exotic targets diagnosed.
- Suspect 4: tuple mismatch is a diagnostic.
- Try blocks (`Try` and `TryStar`) are rejected outright for now — owner call:
  try parsing is not trusted; revisit with the paused lowering phases.
- Probing the old compiler exposed three crash paths (rows 10–11: assignment
  RHS globals; row 12: TryStar), all fixed by the new collector.

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

Decisions (settled during Phase 5):
- `infer_adapter(node, variables, links)` takes the variables *mapping* rather
  than the whole `SymbolTable` (only `.variables` is ever read — keeps it
  usable mid-collection in phase 4 and from IR construction alike), and a
  precomputed `links` mapping (pure: no compile-time `eval` during inference;
  phase 3's new `collect_call_links` pre-resolves a body's call targets once).
- Invalid Par subscripts yield VA silently by contract; the diagnostic belongs
  to IR construction. `par_subscript_index` collapses both failure modes to
  `int | None`; the old compiler's two distinct messages are reintroduced by
  phase 6's builder when it wires the sink.
- Phase 4's placeholder inference in `symbols.py` is superseded: the collector
  pre-resolves call links and delegates to `infer_adapter`.

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

Decisions (settled during Phase 6):
- `IrTryStar` is deferred with the try machinery (owner decision, §10 row 12):
  the builder rejects `try`/`except*` statements; the node returns if/when try
  parsing is trusted and lowering is designed.
- Assignment targets get a third shape beyond the sketch's two:
  `IrTargetDynamic(source_text, captures)` — the old lowering routed non-Name
  and non-Tuple lvalues (`x[0] = v`, `obj.a = v`) through the exec fallback,
  and the IR must carry what that fallback consumes (same shape as `IrDynamic`).
- `IrNode.position` (phase-0 `Position`) is on every node but excluded from
  equality — node identity is structural, so tests compare shape regardless of
  snippet placement.
- The dynamic scan transforms the ORIGINAL AST (as the old rewriter did), not a
  re-parse: diagnostics fired inside the scan keep true positions and dedupe
  against the ones fired while typing the same node.
- Placeholder names default to the deterministic `__natsune_N__` sequence,
  skipping the function's variables and globals (§2, determinism by injection);
  a custom factory can be injected into `build_ir`.
- Capture set = the old special-form set exactly (local reads, linked inet
  calls, tuples, valid Par indexes). Constants and boolean expressions stay in
  the rewritten source; `IrBoolOp` exists for expressions built at top level.
- The full pipeline runs overlapping validations (collector and builder both
  check unsupported nodes, list lvalues, tuple mismatch — §3.3), so
  `DiagnosticSink.add` drops exact duplicates (same message, position,
  severity): one finding, reported once.
- Constant folding: `ast.UnaryOp` over a literal (`-1`, `not True`, `~2` —
  with CPython's operand-type rules; found via the `delayed_inverse`
  snapshot, where `b: Inverse[int] = -1` otherwise degraded to an
  `IrDynamic` with no captures) folds to `IrConst`. Unfoldable operands stay
  dynamic. Constants are never captured inside dynamic scans, so folded
  literals still appear verbatim in rewritten source text.

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

1. ✅ Create `src/natsune/frontend/` skeleton + `tests/frontend/`; old
   `compiler.py` untouched (hence the `frontend` name — see §4).
2. ✅ `diagnostics.py` (Phase 0) with position tests.
3. ✅ `source.py`, `signature.py` (Phases 1–2).
4. ✅ `link.py` (Phase 3).
5. ✅ `symbols.py` (Phase 4) — making the §10 semantic decisions as we go.
6. ✅ `infer.py` (Phase 5).
7. ✅ `ir/nodes.py` → `ir/builder.py` → `ir/render.py` (Phase 6), with golden
   tests.
8. **Pause / evaluation checkpoint:** render IRs for the functions in
   `tests/test_compiler.py` and any real programs; read them; settle the §10
   suspects; only then plan phases 7–9.

   Status: DONE. All 18 example programs from `tests/test_compiler.py` live in
   `tests/frontend/programs.py` as real, undecorated functions (cross-program
   calls carry fake `__inet__` compilers, mirroring the paused decorator) and
   are extracted through the genuine `extract_source` path. They render under
   `tests/frontend/snapshots/<name>.ir` via `tests/frontend/test_snapshots.py`
   (golden snapshot tests; `make snapshots-update` regathers, `make
   snapshots-check` compares). All 18 programs compile through the pipeline
   with zero diagnostics. Reading the snapshots is the input to settling the
   remaining §10 rows and planning phases 7–9.

Steps 2–6 were pure-data and low risk as predicted; step 7 is where the
semantic decisions concentrated — one module per step, each fully tested.
The pipeline is complete through the IR; phases 7–9 below remain the
unfinished work.

---

## 9. Definition of done for this scope — MET

- ✅ `natsune.frontend.build_ir` turns an `@inet`-compatible function into an
  `IrFunction` without creating any net objects and without importing
  `natsune.compiler` (the import ban is enforced by a test).
- ✅ A failing program produces diagnostics, never a mid-flight exception.
- ✅ `render_function` output is deterministic and readable enough to review
  real programs (golden-tested over all 18 example programs).
- ✅ All new tests pass (173); all old tests pass (46); `compiler.py` is
  byte-identical (checked via `git status` at every step of the work).

---

## 10. Semantic suspects to re-decide (divergence log)

Behaviors of the current system that look wrong or accidental. The new system
should decide each deliberately, implement the decision, and log it here — this
log becomes the cutover checklist in phase 9.

| # | Suspect | Where today | Candidate decision |
|---|---|---|---|
| 1 | `AugAssign` desugars to read-modify-write, which may not preserve mutation-through-reference semantics for `Ref`/`Inverse`-typed variables (cf. the `take_reference` test where `a += 10` mutates in place) | `parse_statement_body` AugAssign branch | **Half-done (Phase 6):** `IrAugAssign` is kept as its own IR node, not desugared; the explicit Ref-aware lowering semantics are a phase-8 decision |
| 2 | Any name not already a local is treated as a global — including names assigned *later* in the function (read-before-assignment silently reads a global) | `InetVariablesEvaluator.visit_Name` | **Decided (Phase 4, probe-verified):** preserve the order-dependent global fallback — a read preceding a normal assignment already trips the global-target conflict diagnostic. The truly silent hole is self-referential first binding (`a = a + 1` with `a` unknown: old = silent uninitialized local; Python = UnboundLocalError), now flagged with "Read of variable before assignment". AugAssign-introduced names stay silent by design (documented feature) |
| 3 | For-loop targets are force-marked with adapter `VA`; `ast.walk` over targets silently ignores non-Name nodes | `visit_For` | **Decided (Phase 4):** targets keep VA for now (typing from the iterable awaits settled lowering semantics); tuple targets recurse; exotic leaves (`for x[0] in ...`) are diagnosed instead of silently marking the root name as a local |
| 4 | Tuple-assignment targets silently get adapter `VA` when the value's Par arity doesn't match | `visit_Assign` | **Decided (Phase 4):** mismatch is a diagnostic; matching Par values type element-wise (nested tuples recurse into nested Par items) |
| 5 | Several checks (inet-call arity/keywords, Par subscript bounds, list lvalues) fire during *lowering*, after nets are partially constructed | `evaluate_special_form_from_expression`, `evaluate_subscript`, `evaluate_to_expression` | **Done (Phase 6):** inet-call keywords/arity and Par subscript bounds are validated in the builder (diagnostics via the sink; an invalid expression falls back to `IrDynamic`); list lvalues and tuple mismatch are additionally validated at collection (§3.3) |
| 6 | Chained assignment `a = b = expr` wires `b` from `a`'s register rather than re-evaluating — subtle and untested | `parse_statement_body` Assign branch | Model explicitly in `IrAssign`; decide semantics once |
| 7 | Indented definitions (methods, nested functions) crash the old compiler: `inspect.getsourcelines` returns an undented block, so `ast.parse` raises IndentationError before compilation even starts | `InetFunctionCompiler.func_def` (found while characterizing Phase 1) | `extract_source` dedents the extracted block; columns stay snippet-relative (understate by the stripped margin for indented defs) |
| 8 | Positional parameter defaults are silently accepted and ignored: the old check rejects only `kw_defaults`/`kwonlyargs`/`kwarg`/`vararg`, so `def f(a, b=5)` compiles to a 2-arity net while Python callers may invoke it with one argument | `InetFunctionCompiler.args` (found during Phase 2) | Reject defaults at signature analysis — a defaulted param cannot be supplied through the net interface |
| 9 | Positional-only parameters (`def f(a, /, b)`) are neither rejected nor included in `args` — silently dropped, producing a wrong-arity net | `InetFunctionCompiler.args` (found during Phase 2) | Reject at signature analysis; supporting them later means including them in `args` |
| 10 | Assigning from a global crashes with a raw `KeyError`: `visit_Assign` infers the value's adapter via `variables[id]` indexing (`a = b_global` with `b_global` a module global) | `InetVariablesEvaluator.visit_Assign` → `infer_expression_adapter` (probe-verified, Phase 4) | Unknown names yield VA and the RHS name is collected into `used_as_globals` (dynamic path) |
| 11 | `AnnAssign`/`AugAssign` RHS are never walked by the collector, so globals referenced there are never marked — `a: int = b_global` passes collection and crashes at *lowering* with a raw KeyError | `visit_AnnAssign`, `visit_AugAssign` (probe-verified, Phase 4) | Both RHS expressions are walked by the new collector |
| 12 | `try` is rejected as unsupported, but `try/except*` (`ast.TryStar`) is silently walked by the collector, and handler bound names are never declared | unsupported_stmt tuple / `generic_visit` (probe-verified, Phase 4) | **Owner decision:** skip all try blocks for now ("don't trust my own implementation for parsing try blocks") — the new collector rejects both `Try` and `TryStar`; revisit with phases 6+ and the paused lowering |
| 13 | Negative Par subscript literals (`p[-1]`) are rejected as "must be a constant integer", because the unary minus makes them `UnaryOp`, not `Constant` — while `p[True]` is accepted (bool is an int subclass) and indexes element 1 | `evaluate_subscript` (characterized Phase 5) | Absorbed exactly: `par_subscript_index` discriminates the two failure modes via `ParSubscriptError` (old messages verbatim), and the Phase 6 builder lands them in the sink while falling back to `IrDynamic` |

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

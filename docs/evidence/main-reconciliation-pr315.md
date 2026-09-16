# Main through PR #315

I reconcile main `5c358000` with my integration history after `b37136cc`.
The nine incoming commits cover nested evaluator arrays, relative paths,
write/close failures, native void locals, array return kinds, opcode
classification, uncalled record layouts, projected array fields and array growth.

## Decisions

- I retain my `Value`-backed nested evaluator arrays. Main's `Array **`
  conversion assumes a different representation. My existing nested fixture
  already includes the incoming shadow, and the compatibility suite passes.
- I retain growable relative-path buffers and one captured working directory.
  I add the incoming module shadows and filesystem assertions.
- I retain the shared writer helper. Three dedicated production test objects
  substitute stdio through a forced include; normal production objects do not.
  I remove GNU-only linker wrapping. The separate seven-wrapper probe still
  tests open failures, short writes, close failures and empty writes.
- I preserve my selective omission of unresolved uncalled record layouts,
  both entry and `__init__` roots, and representable native helper functions.
  I retain dynamic record shapes and import all seven incoming native tests.
- I preserve void-before-store for scalar locals using my existing tagged
  value representation. Integer, boolean and string parameters retain tags
  through direct calls and tail calls. Self-tail restarts reset non-parameters.
  Typed consumption checks the runtime tag; a void integer return traps.
- I replace the remaining fixed integer-array arena with owned capacity
  growth. String arrays already had that mechanism. I retain mutable handle
  identity, copy borrowed backing storage before growing it, check capacity
  arithmetic and release owned buffers and handles on normal completion.

I change two incoming growth fixtures to assert their boolean comparison and
return an integer, matching their declared result type. I replace void-local
emitted-text assertions with runtime tag assertions. These tests exercise the
contract without requiring main's older internal representation.

## Evidence

- Focused interpreter, transpiler, VM-builtin and filesystem suites pass on
  Darwin. VM builtins report 23 passing cases.
- Native translation reports 1,712 passing checks and 1,073 shape checks.
- Fresh ASan/UBSan translator and shape objects pass the same counts; the
  sanitizer driver verifies instrumentation. This is not a leak-freedom claim.
- The combined native compiler, writer, path and nested-array matrix passes
  33 methods. New cases compare void-local behavior with NanoVM and exercise
  70,000-element integer growth, literals, aliases, borrowed buffers, ten
  allocation/size/storage failures and zero tracked allocations after teardown.
- The C-seed filesystem program compiles with dependency shadows and executes
  the new relative-path assertions successfully.
- The final native compiler suite passes all 24 methods, including added
  record-frame counts 8, 9, 10, 18 and 100 under generated-code ASan/UBSan.
  These protect my existing frame implementation against PR #317's octal-size
  defect; they do not mark that external branch fixed or merged.

Bootstrap Stage 3 passes, including self-hosted execution without the C seed.
`make -j1 test-quick` passes: 17 core-language programs, 242 eligible bytecode
examples, 33 affine methods and the remaining runtime/Forth checks. Six example
exclusions are checked rather than silently omitted. The SDL Forth IDE builds;
graphical initialization is explicitly skipped without `xvfb-run` and `timeout`.
I do not count that skip as interactive graphics acceptance.

## Boundaries

Aggregate void-before-store support remains work. My scalar tests do not
establish aggregate optional storage or complete VM/native equivalence.
Record-array capacity, full ownership checking, callbacks, effects, graphical
packaging and the remaining branch/release gates remain in my roadmap.
This merge does not publish 5.0, enable Horde SSO or clear any fleet hold.

## Refreshed release inventory

My September 16 refresh finds 22 open PRs and no open GitHub issues. I inspect
their titles, bodies, labels and milestones; none explicitly carries the 5.0
label/milestone scope, but my creator's all-branches request still applies.
Main has advanced to `4936f8cf` (PR #316), beyond this merge checkpoint.
Its match-arm return typing must not overwrite my 5.0 return contract.

I review PR #317's full diff and reproduce its `r[008]` C compilation failure.
Its `r[010]` would allocate eight records, not ten. I post the finding at
https://github.com/jordanhubbard/nanolang/pull/317#issuecomment-5697117488 and
file `task_7a307efeea0e4ea5b2476904b2cbdfea`. I do not close either new PR or
claim that the refreshed inventory is reconciled.

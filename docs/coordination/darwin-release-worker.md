# NanoLang worker coordination

Updated 2026-10-08 by the Darwin release worker, Codex session
01a118d7-e6a1-79d1-ba73-c1c266da7eb0. I coordinate through Git at the user's
request. The remote worker has not acknowledged this handoff.

## Published checkpoints

My implementation branch is `release/5.1-completion-20261007`, PR #974.
Its current pushed pin is `d87fb4a12`. The full compiler-product gate at
`1321b8bdf` passes all 109 methods. HEAD and tracked sources stay unchanged;
only the user's untracked test remains, with an unchanged hash.

My separate `fix/5.1-retire-legacy-globals-20261008` branch is at `221c200d7`.
The checker correction at `b76b33a53` passes a clean raw bootstrap: Stage1
and Stage2 are byte-identical, 492,432 bytes, SHA256
`a729c1d528ecd8dc29621583fa2da855df6a4ff0f3433817512c03657213642c`.
All 72 installed C-seed/Stage1/Stage2 product checks also pass. I integrated
the three checker commits into the release branch through `e4bd644a2`.
The temporary checkout remains because its binaries reference host-library paths.

The full source-snapshot rerun at clean `f0f6a0c62` terminated with exit 120
and explicit OSError 28; HEAD, sources and probe stayed unchanged. I archived
its complete available log and manifest. Other failed subcases remain
unclassified because unittest could not finish printing diagnostics.
I recovered capacity by excluding redundant historical evidence from completed
temporary clones with Git sparse checkout, retaining their source and binaries.
The receipt records 52.072 GiB free; a fresh full rerun at the same pin is live.

## Current ownership

I retain the previously announced closure/parser/checker/native translator,
shape solver, compiler-phase schema/generated callers, bootstrap component,
Makefile and compiler-product test ownership. I also own `src/module_builder.c`
and its source-snapshot/link-response tests for the Darwin linker correction.

I am migrating `tests/selfhost_shadow_emitter.nano` and
`tests/test_native_shadow_emitter.py` onto NanoISA emission and native/VM
execution, with related edits in `src_nano/compiler/nanoisa_codegen.nano` and
Make prerequisites. The first run exposes missing opaque-null lowering. My
correction retains the original behavioral corpus and rejects nonzero or wrong
opaque arguments; the full migrated suite passes 12 methods. Added forwarding
and driver publication controls also pass. These migration changes are pushed
at `a055b57c1`. The core gate terminated with 90 methods and two obsolete
refusal expectations for callable globals and function arguments. I retain the
original failures and replace those expectations with VM/native execution,
including an additional global-call result. The affected method passes.
This correction is pushed at `e50ea978b`. Its full core gate executes 84 passing
methods, but six shadow-emitter methods cannot initialize because the fixture
compiler hits its ten-second shadow deadline. I retain the full failure and
unchanged-source receipt at `d87fb4a12`. A bounded retry of that class passes
all six methods in 69.493 seconds with the same deadline. The original timeout
remains unexplained; final candidate qualification is still required. I also
own `tests/test_nanoisa_flat_records.py` for this change.
This is not full release acceptance.

The legacy emitter still has other live callers and tests requiring migration.
I have not reduced the release scope or claimed release readiness.

## Checkout preservation

`/Users/jordanh/Src/nanolang` remains the primary checkout. Preserve
`tests/user_guide/refresh_language_pure_function.nano` as user work. Cleanup
recovery remains under `.git/worktree-cleanup/20261007-171119`.
My temporary sparse worktree `/private/tmp/nanolang-retirement-20261008` retains
bootstrap binaries and their absolute host-library paths. Do not remove it or
other qualification artifacts while they are still referenced.

## Remote worker handoff

Please reply on PR #974 or add your own handoff file here with machine/session,
branch and source pin, task/file ownership, tests and intended integration
target. Keep implementation on separate branches and coordinate overlapping
files before integration. Use a normal fast-forward push on this shared branch.
Linux qualification of a published exact pin would help, but this is a proposal
until acknowledged, not an assignment of an unknown worker's task.

## Lexical acceptance and parser retirement checkpoint

I pushed `7d9b59def` on `fix/5.1-lexical-scope-acceptance-20261008`. This branch
uses the existing temporary retirement worktree. I own `src/typechecker.c`,
`tests/test_genenv_scope.py`, and its `tests/test_one_ir_compiler.py` integration.
Executed tests expose a C checker defect: a local integer does not hide a
same-named function. I now resolve lexical callees first and retain their own
full signatures. Three methods pass: four positive VM/native products and
twelve exact-status output-preserving refusals. Direct C typechecker/environment
gates and ten lexical-boundary methods also pass. The separate full bootstrap
remains live in Stage2; the direct C gate explicitly omits its stage1 prerequisite
and is not a substitute for that bootstrap. I have not integrated this branch.

I also own `tests/file_service_parser.nano.in` on the primary release worktree
for `task_4c50868184bd42f2aa4fd5f81e4bfd4a`. I migrate its legacy transpiler
helper calls to NanoISA program emission/refusal/recovery, retaining publisher
metadata and parser/checker assertions. The C-seed fixture compiles and executes
successfully; installed Stage1/Stage2 fixture qualification is running. These
fixture edits remain uncommitted pending that result. Full paired parser
acceptance and final fresh candidate qualification remain required.

After tool handles were lost, I verified the bootstrap and snapshot PIDs still
exist; snapshot logs continue advancing. I have not restarted those live runs.

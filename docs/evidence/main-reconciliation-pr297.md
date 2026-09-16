# My current-main reconciliation

I reconcile integration parent `a80a4a85` with main parent `b37136cc` (PR #297).
I preserve both histories, not only their patches.

## My conflict decisions

- I retain SDL's native linkage anchor and add upstream header priorities.
  I retain the image adapter's dynamic-array declaration and use the explicit
  SDL2 include paths.
- I retain recursive array type information and my existing nominal-name
  fallback. I consult recursive expression metadata before qualified and
  unqualified array-producing function signatures.
- I keep the self-hosted runner's Stage 1 default. Its explicit
  `NANOLANG_SELFHOST_COMPILER` override takes precedence over `NANOC`.
  I retain upstream per-case compilation logs and quote the compiler path.
- I retain both roadmap histories. Upstream Stage 2 failure counts describe
  that branch's earlier run, not a fresh result for this integration.

## My verified boundaries

On this macOS checkout, `make test-c-backend` passes all seven cases.
`make test-one-ir-compiler` passes all 21 methods, including fresh full-compiler
bytecode, native translation, strict C compilation and compiling/executing
the hello program with the resulting compiler.

`python3 -m unittest tests.test_array_adapter_boundaries
tests.test_sdl_image_array_boundary` passes all four tests. My C seed also
compiles and executes `tests/selfhost/test_nested_array_indexing.nano`.
These checks do not test interactive SDL startup or deployed packaging.

My freshly rebuilt Stage 2 passes
`NANOLANG_SELFHOST_COMPILER=./bin/nanoc_stage2 sh tests/selfhost/run_selfhost_tests.sh`:
20 checks pass, none fail, including the import-path and CLI suites. This
supersedes the incoming branch's earlier 11-of-16 result. The build still
reports different native bootstrap binaries; I do not claim a fixed point.

My six `tests.test_selfhost_array_compatibility` methods pass, including nested
values and scalar/nested rejection boundaries.

`make test-quick` does **not** pass: its rebuilt native compiler passes 16
language cases and fails `tests/nl_functions_filter.nano`. Direct float and
boolean array literals select the integer `nl_filter` helper, so C rejects
their callback signatures. I track the repair as
`task_ac94d5cc420e481a896d5f1a2d37f595`. Later quick-gate targets are not run.
The C-backend and native-compiler acceptance results above come from a separate
successful invocation, not from the command chain stopped by this failure.

My logs are `/tmp/nano-main-reconcile-gates.log`,
`/tmp/nano-main-reconcile-focused.log`, `/tmp/nano-main-reconcile-selfhost.log`,
`/tmp/nano-main-reconcile-arrays.log` and `/tmp/nano-main-reconcile-sdl.log`.
They are local diagnostics, not durable release artifacts.

My September 16 GitHub inventory has no open issues and 12 open PRs:
#266, #267, #269, #273, #274, #281, #282, #283, #284, #285, #286 and #287.
I inspect their titles, bodies, labels, milestones and head SHAs. Every head
is an ancestor of integration parent `a80a4a85`; that does not mean it is
landed on main. I leave those PRs open until the main integration is landed.

MAC still rejects release-parent claims with `agent_status_unavailable`.
I do not infer successful task ownership or completion from recorded evidence.
This merge is not a tag, release, canonical bootstrap fixed point or complete
backend-equivalence claim.

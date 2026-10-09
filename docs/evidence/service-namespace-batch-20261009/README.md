# My service namespace integration batch

I retain source identity before nominal type checking and execution authority.
My C loader now collects the full service-bearing graph before building its
namespace. My independent Nano resolver receives the parsed merged graph before
ordinary name rewriting. Both collect declarations before aliases, retain the
original target through selective and qualified imports and re-exports, and
reject collisions and unresolved selections. A new failing C invocation clears
previous namespace facts. Anonymous and lexical functions are not module names.

My Nano driver discovers imports through its actual parser and retains every
source line, including module headers and adjacent declarations. I parse legacy
bare imports and re-exports, detect cycles, and keep malformed sources on the
companion-protection diagnostic path. Introspection facts are retained by source
path; an ambiguous name-based introspection operation refuses before output.

## Checked boundaries

- [Final graph gate](final-graph-tests.log): fourteen graphs through the actual
  C loader plus C seed, NanoVirt and the freshly C-produced Nano driver. I cover
  bare, qualified and selective imports, re-exports, repeated symlink imports,
  distinct same-basename origins, missing selections, late/own/alias collisions,
  multiline imports, comments, adjacent declarations and anonymous functions.
  Prior output remains intact. Valid service graphs still reach my execution
  guard; these are namespace acceptance checks, not File execution.
- [Paired identity gate](paired-tests.log): my independent Nano namespace fixture
  is compiled by NanoVirt and the updated Nano driver, then run in NanoVM and
  native AOT with ASan/UBSan. Aliases preserve identity, separate origins remain
  distinct, invalid ownership/target maps and swapped or missing snapshots
  refuse. The catalog's two exact four-integer foreign signatures work in both
  producers and both engines. This gate preceded the final bare-import parser
  parity edit; the final driver build and graph gate were repeated afterward.
- [Selected C sanitizers](namespace-sanitizer.log): I instrument the C namespace
  builder and runner with LLVM ASan/UBSan and leak detection, including every
  builder allocation-failure prefix and unchanged output on failure. Shared
  parser, loader, snapshot, plan and runtime objects remain ordinary here.
- [Acquisition preservation](acquisition-preservation.log): all seven actual
  Nano-driver companion/refusal cases pass, including a parse error whose
  diagnostic destination aliases a companion file.
- [Module neighbors](module-neighbors.log): ordinary same-basename function
  bindings, four repository metadata programs and ambiguous-introspection
  output preservation pass through the updated Nano driver.
- [Bootstrap control tests](bootstrap-control-tests.log) pass with the exact
  file-source catalog host manifest added to the allowed closure. I did not run
  a full bootstrap for this namespace batch. [Final driver compilation](final-driver-build.log)
  runs selected dependency shadows through NanoVirt. Shadow policy and diff
  whitespace checks pass.

My namespace bounds are 5,000 modules, sixteen service requests, thirteen
catalog names per request, 64 aliases, 256 ordinary names and 1 MiB copied
namespace text. These are explicit limits, not whole-compiler memory accounting.
My C namespace owns copied paths/names and borrows AST nodes; my compiler
retains the parsed graph through its checking lifetime.

I retain the [first paired failures](first-paired-failures.log) and
[first acquisition regression](first-acquisition-regression.log): physical path
expectations, premature same-basename refusal, missing direct Nano imports and
premature parse-error refusal were corrected. I do not count failed attempts
as qualification.

## Evaluator CI repair

CI [37974454367](https://github.com/jordanhubbard/nanolang/actions/runs/37974454367)
at 0ee20e47e passed the previously blocked emitter stage and failed in evaluator
shadow tracing. Its [Linux sanitizer terminal](linux-sanitizer-terminal.log)
shows a NULL stderr stream. The evaluator test harness nested suppression through
its coroutine test and run_ctx_init, then restored NULL. The exact traced run
[reproduces locally](evaluator-trace-before.log) and [passes after correction](evaluator-trace-after.log).
I balance suppression depth and assert restoration of the original stream.
This local result does not claim a corrected Linux sanitizer run.

## Remaining release work

This batch does not enable File execution. Nominal checking, independent C/Nano
File lowering, generated shadows, explicit grants, staged publication and full
whole-driver bounds remain open under #989. The full 5.1 scope remains
unchanged. Full bootstrap, Linux/platform qualification and release-candidate
acceptance must follow the complete implementation batch.

[source-sha256.json](source-sha256.json) records the final changed implementation
and test sources. I retain no generated binaries in this evidence directory.

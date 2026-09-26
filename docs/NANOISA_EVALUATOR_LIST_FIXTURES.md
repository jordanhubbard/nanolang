# My nominal list and evaluator lifetime fixture checkpoint

I prepare these controls under task_7b805000dfda4da386b55d4691e8c647 and its
required tuple, deferred-task and cache-generation children. My reviewed source
ends at d299aa517. I have not built a fixture, run a source program, or measured
acceptance at this checkpoint. Enum list parity remains required separately;
I explicitly refuse the intermediate implicit enum-list route.

## My observable boundaries

| Control | Actual path and required observation |
| --- | --- |
| List edits | Parsed record-list growth through 33 elements, alias reads, exact first/middle/end insert, element-returning remove/pop, set, retained strings, iteration and receiver/index/value trace 123 |
| Nominal provenance | Record and list locals, reassignment, globals, fields, direct/indirect arguments and returns; exact imported same-spelling declarations and legitimate aliases |
| Callable metadata | Nested array, tuple and callable nominal annotations; unresolved leaves refuse; conditional, if and match callable branches agree on the exact owner |
| Lookup precedence | Real lowercase mutation-named function and uppercase constructor-named function retain their ordinary calls |
| Pending values | Parsed direct and indirect calls capture earlier strings and nested records before a later argument replaces their global owner; tuple and record construction do the same |
| Escaping results | A public callable returns a tuple containing a record and string; I destroy Environment, AST, tokens and cache, then inspect and release the independent graph |
| Scheduler | Actual owned callbacks, early completion and error, active cancellation/release refusal, nested await, self-cycle error, first error, completed-result copy failure/recovery and use after task release |
| Bundles | Actual evaluator bundle functions and callable clone hook: checked failure at every measured allocation, failed enqueue into full/ID-exhausted scheduler, cancellation without invoking the queued target |
| Borrow staging | Direct calls to the actual private evaluator staging helper show shared/mutable formals preserve identity and ordinary formals copy before replacement; this is runtime helper coverage, not new source admission |
| Async evaluator | Parsed nested async record returns repeated 130 times, checking actual returned fields and automatic task release |
| Cache generations | Actual fresh load and cached hit, failed edge allocation before cached publication, successful recovery, exact leased cache/environment teardown refusal |
| Private compilation | Actual successful private object compilation, missing input, missing output directory and selected compiler option failure; modeled Environment creation, transpiler NULL and write/close reporting failures cover the remaining restoration branches |
| Original regression | Exact unchanged tests/token_value_bytes.nano emits and executes with all original insert/remove/pop, token byte-count and imported shadow assertions |

The source runner selects C seed, Stage1 and Stage2 explicitly and retains each
compiler's output. It also uses the actual C evaluator and canonical NanoISA
producer/VM. It never substitutes one producer for another after a failure.
Negative source controls preserve an existing output sentinel and reject C
compilation failures as evidence of a checker refusal. The evaluator must parse
and import those negative programs successfully, then observe type_check=false
before run_program. Compiler signals and sanitizer diagnostics never count as
an expected refusal. New source programs contain shadows; deliberately rejected
programs retain trivial shadows where an invalid value cannot have an executable
successful shadow. The original imported LexerToken graph is not reduced and its
existing internal ten-second shadow deadline is unchanged.

## My allocation domains

`test_evaluator_owned_lifetimes.c` includes the complete env.c owning translation
unit under malloc/calloc/realloc/strdup/free hooks. Separate fixture-owned eval.c
and module.c copies use the same hooks in that binary. The ordinary env.o,
eval.o and module.o providers are excluded from that link. All other prepared
providers remain explicit inputs. A second binary uses ordinary env.o and fresh
fixture-owned eval.c/module.c without allocation hooks for parsed programs and
real loader/compiler paths. Neither link silently loads a second definition of
an owning provider.

I enable failure injection only around the new checked graph/signature/bundle,
list and provider operations. A successful call first measures the exact number
of allocation attempts. I fail each measured attempt in both persistent-prefix
and single-transient modes, require false with unchanged outputs and old state,
require zero outstanding observed allocations after rollback/teardown, then run
a fresh successful control. My realloc hook retains the old allocation identity
on failure. The graph and callable signature copies include nested fields and
metadata, not just a scalar result. Limits include depth 128, malformed counts,
provider reference/lease overflow and dead generations.

Cache initialization is a separately fatal API. I measure its exact four
allocation sites and run each failure in a fresh process. Its atexit observer
requires every tracked partial cache allocation released, alongside the exact
status-1 diagnostic. This does not convert the cache initializer to a recoverable
API. Loader edge-registration failure is recoverable and checked in-process.
Modeled private write failures call the actual fclose once; modeled environment
or transpiler failure does not claim a real host allocator failure. Invalid
compiler options and nonexistent paths exercise real host failures.

I do not inject failure into legacy create_string, parser/checker metadata or
arbitrary compiler allocation and then advertise recovery. Those paths still
include fatal policies. Unsuppressed sanitizer failures in any executed scope
remain failures and enter my ledger; I do not hide them with a retry or label them
unrelated without evidence. My cumulative Environment snapshot arena remains
explicitly retained until Environment teardown, not expression-bounded cleanup.

## My runner and gate order

The Python fixture retains unique command argv, bounded process-group terminal
records and file-backed stdout/stderr before assertions. Cleanup uses bounded
TERM and KILL waits, and a surviving process group fails the command. Every
command snapshots retained products before and after to a content-addressed store,
including compiler temporaries under its private TMPDIR. Prepared provider hashes
must match before and after the class. External qualification additionally seals
tracked source, selected tools and complete actual prepared provider maps at phase
boundaries; I make no full transitive runtime claim from this fixture's local map.

After independent fixture review, my intended order is:

1. Fresh isolated provider build and full bootstrap with explicit supported native
   and sanitizer selectors on Linux and Darwin; retain first terminals.
2. Focused checked C ownership controls in ordinary and supported strict sanitizer
   configurations. Report exactly which owning TUs and prepared providers are
   instrumented. No missing configuration is inferred from another host.
3. Complete new source/evaluator matrix, including actual private compilation and
   unchanged LexerToken regression, on those frozen fresh tools.
4. Existing generated-list metadata poison, full evaluator, coroutine scheduler,
   environment scoping, module metadata/cache and callee-snapshot neighbors.
5. Unchanged full `make test`, including its original complete verifier corpus;
   a focused success does not close that full gate or unrelated U8 work.

The existing generated-list poison fixture previously registered an undeclared
MetadataItem. My reviewed source deliberately rejects that missing nominal
identity. I record its setup migration before editing: declare the record,
require rejection before declaration and success afterward, and add exact
registered identity assertions. Every old zero/NULL poison assertion remains.
No historical failure or successful rerun is claimed for this static migration.

## My first preparation prerequisite

Both b915 make-build attempts stopped before bootstrap and these fixtures at
compiler_contracts.nano:66. I preserve the complete first reports under the
paths in NANOISA_LIST_FORWARD_DECLARATIONS.md. The corrected source checkpoint
has nine methods: the added method inspects actual parsed pending/importer
extern declarations, both module orders, actual alias resolution and collision
refusals, plus direct registered ordinal/owner controls. It executes no foreign
ABI. I retain every earlier assertion and the unchanged full LexerToken gate.

My exact-root index supplement retains all nine Python methods and all existing
C controls. It adds measured initial/no-growth/12-root-growth publication sweeps,
exact old-table byte/alias preservation, same-Environment and fresh recovery,
valid-allocation collision probing, nested/type misses and independent teardown.
The mandatory first index allocation extends first-string publication from two
to three observed sites and first tuple retirement from one to two. I preserve
all original assertions and include every newly measured allocation in both modes.

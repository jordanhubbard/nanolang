# My evaluator shadow timing boundary

I retain the first cc606 build reports at
/tmp/nanolang-record-lists-cc606-linux-prepare and Puck
/private/tmp/nanolang-record-lists-cc606-puck-prepare. Both stopped at the original
ten-second parser shadow deadline, before bootstrap or the nine-method fixtures.
I track task_2deaad56f65c497f80546220aa1ca9d0.

I use a separate macro-only diagnostic source checkpoint. I rebuild env.c and
eval.c with that macro and link a fresh C seed against hash-verified unchanged
cc606 providers. I retain exact source, provider, tool, argv, output and binary
identities. I run the original parser_driver.nano compiler command once with its
complete imported shadows and unchanged ten-second supervision, inside a bounded
120-second external process group. No reduced graph or timing acceptance follows.

My stderr markers identify each shadow start/end, monotonic time, cumulative
checked snapshot allocation attempts, clone nodes, cumulative published arena roots, retirement attempts, borrowed-root
lookup calls and visited entries. I cap markers at 8192 and check counter overflow.
I measure cumulative clone and borrowed-root lookup nanoseconds separately; clone
timing is inclusive and not additive with overall shadow timing. Only successful
clock samples count; a diagnostic clock/overflow failure exits explicitly.
Allocator counters cover only the record/tuple/string graph include, not all
compiler allocations. Retired storage remains alive exactly as in production.

My counters and clocks add overhead. Completed marker intervals show work before
the last marker; the killed interval has no end sample. Cumulative arena cost is
a hypothesis until measured. I do not optimize or alter ownership in this build.

I count retirement attempts, including refused attempts, and cumulative roots
published by snapshot/retirement; roots is not a current-live or peak-memory
measurement. The actual allocator wrappers count attempts in env_record_lists.inc
only. Callable signature allocation is outside that counter. The disabled macro
adds no clock or counter calls and leaves the original clone/lookup bodies intact.

## My first launcher terminal

The first fresh diagnostic TUs and link succeeded. The external executable lacked
my bin/../src layout: resolve_project_root derived /tmp and early generated module
compilation failed to include runtime/nl_string.h. Status 1 after 1.021 seconds,
no shadow marker, is not a timing measurement. I retain log SHA256
dd6b17ab253c399b8d7b3ae867c7d498a7979c7493787c747f9218e2f868eeb3 and newly
generated obj/nano_modules files. No existing source/tool/provider bytes changed.

My corrected launcher places a freshly linked binary under an external bin
directory with src and module-tree symlinks to verified frozen sources. Its cwd
and new module products are external. I preserve the original full parser input,
all shadows and both deadlines; I do not repeat the failed Make builds.

The second launcher reached std module preparation but lacked the executable's
nano_as_capture.so companion (module_gcc_read_capture resolves /proc/self/exe
unless NANO_AS_CAPTURE_HELPER is set). I retain status 1 at 2.571 seconds and log
8c8b4d5ef283f81d54c47677a533141a0bfd5829d2765fd5a0c9fa21fc30ea9a, again with
no shadow marker. My next launcher selects the original archived helper explicitly.
I inspect main/module/module_builder/FFI root consumers together: runtime sources
and headers use bin/../src, generated-list fallback uses scripts relative to cwd,
and optional catalogs use argv0-relative share paths. I preserve those source
paths, explicit CC/helper/module/cache choices and unset competing compiler flags.

## My measured interval

The corrected 192ef launcher ran the original full parser graph once. It exited
1 after 19.934 seconds total, with my unchanged ten-second shadow deadline.
I retain 405 markers: 202 completed shadows, then parse_block_recursive without
an end marker. The first begin through that final begin spans 1.988659339 seconds.
Within that measured interval borrowed-root lookup consumed 1.633750426 seconds,
visiting 353,202,393 entries in 28,655 calls. Cloning consumed 0.032518385 seconds.
I observed 1,147,757 graph allocation attempts, 28,284 cumulative published roots
and 9,069 retirement attempts. I cannot attribute the remaining killed interval
from these markers. Instrumentation overhead remains part of these observations.

My diagnostic seal is docs/evidence/evaluator-lifetime-diagnostic/seal.json:
99 retained reports, 4,064 unique archived objects (757,097,052 bytes), and 11
equal command input pairs. Three diagnostic-launch commands exited 1: two before
shadows and one at the original deadline. All selected source/tool/provider bytes
match their before maps. The CAS is /tmp/nanolang-evaluator-lifetime-diagnostic-artifacts.
Large report copies are compressed with both raw and stored digests. I also retain
the four original Make logs/statuses; their complete preparation maps remain at
the original local/Puck roots and are not represented as fully copied by this seal.
Intermediate files deleted inside compiler/module helpers were not reconstructed;
my endpoint product maps retain surviving module sources, captures and binaries.

This supports an exact Environment-owned root index as a candidate under
task_c2e9d2f19f1b4a359e841edcceda4abd. It does not establish total timeout cause
or passing production acceptance. My original full graph and all gates remain.

## My next unchanged graph

At40870, both hosts compiled the parser and then stopped during typecheck.nano's
full shadow graph at the original ten-second deadline. Linux make-build returned2
in65.577s; Darwin returned2 in60.595s. Their source/tool maps agree and process
groups are gone. Minimum monitored free space was4,598,083,584bytes on Linux and
33,105,829,888bytes on Darwin; no disk guard fired. These are timeout observations,
not infrastructure attribution. Bootstrap and all13 fixture methods are unreached.

I retain the new terminal summaries/logs under
`docs/evidence/record-lists-40870-preparation/`. Complete44report/4056object Linux
preparation remains at `/tmp/nanolang-record-lists-40870-linux-prepare`; the
44report/4051object Darwin preparation remains at
`/private/tmp/nanolang-record-lists-40870-puck-prepare`. My new diagnostic starts
from the exact40870 production pin in a separate tree. I keep all shadows and
the10s child/120s outer limits. Any added counter or timing wrapper receives
source review before execution. No timing result qualifies production behavior.

## My729ce typechecker measurement

My separate729ce observer completed its fresh three-provider build and link,
then ran the unchanged full typecheck_driver.nano graph once. The compiler
returned1 at my original ten-second shadow deadline; the launcher's recorded
elapsed time was28.311s including its precommand inventory. All six command
input pairs agree, both process bounds held, and the process group is gone.

I retain985 markers and492 completed shadows. First begin through the last
unmatched check_expr_node begin spans9.730779906s. Within that interval I count
21,836,131 graph allocation attempts,19,497,224 clone nodes and641,226 cumulative
published roots. Snapshot cloning takes0.484468380s and740,342 borrowed-root
lookups take0.103279909s. Index work across lookup/publication/rehash visits
5,500,063 slots in2,168,004 calls. These scopes overlap and are not additive.

Nominal view calls/time, checked annotation allocations and legacy payload
allocations have zero delta during the measured shadow interval. Before the
first marker I observe10,862 view calls and33,667,967ns inclusive view time.
The two largest completed shadows are parse_block_recursive at2.020s and
parse_unsafe_block_recursive at1.983s. I cannot attribute the unmatched final
interval or the remaining measured time from the counters alone.

The seal is `docs/evidence/typecheck-lifetime-729ce/seal.json`:55reports,
4,087unique objects/781,009,958bytes and six equal command pairs. The CAS is
`/tmp/nanolang-typecheck-lifetime-729ce-artifacts`. I preserve scoped allocation,
observer overhead and deleted-intermediate limitations; this is not qualification.
My next authorized measurement targets actual record_list_find visits/time and
graph disposal/root publication, before selecting any further correction.

## My3f0c registry/disposal measurement

The separate3f0c diagnostic also returns1 at the original ten-second deadline,
with all six command input pairs equal and no capacity/outer timeout. I retain
991markers/495completed shadows; the last unmatched begin is
owned_pattern_complete. First begin through that point spans9.934263301s.
List lookup performs73,420calls/2,141,454visits in7,153,976ns. Root publication
performs666,514calls in200,703,059ns. Outermost graph disposal totals1,846,758ns,
with160,691recursive nodes counted. Snapshot clone time is497,871,112ns and
indexed borrowed-root lookup is107,943,618ns. These scopes and observer overhead
are not an additive attribution of the entire elapsed interval.

The measured list lookup cost does not support a registry index as a timeout
repair. I retain this diagnostic seal separately at
`docs/evidence/list-registry-3f0c/seal.json`, including complete source/provider
maps and the unchanged input. The remaining time is still unattributed.

My next existing-profiler preflight succeeds for perf --version but denies an
own-child task-clock event with status255 because perf_event_paranoid is4. I
preserve the commands/status/output at
`/tmp/nanolang-typecheck-profile-preflight`; I do not change host policy. The
approved fallback is private exclusive scope accounting for evaluator dispatch,
call/scope cleanup and Environment lookup/index maintenance. That measurement
must precede any production correction or attribution claim.


## My exclusive scope measurement and next correction boundary

I retain the single 77226ef6e full checker diagnostic in
`evidence/exclusive-77226/seal.json`: 64 reports, 4087 unique artifacts,
781743455 bytes and six equal command input pairs. The own-child perf preflight
was denied by the host's existing perf_event_paranoid=4 policy; I did not change
that policy. My fallback observer adds checked exclusive scope entry/exit clocks.
The run returns 1 at my original ten-second shadow deadline; the 120-second outer
bound and capacity guard do not fire. I retain 853 markers and 426 completed
shadows, with lookup_field_type_kind the last unmatched begin.

Between the first and last marker, all exclusive scope times sum exactly to
9999172551 ns. Symbol index synchronization accounts for 4748758758 ns over
3549172 calls; function lookup exclusive time accounts for 2942919683 ns over
1610387 calls. These intervals include instrumentation overhead, including
6223993 expression scope entries. They are diagnostic attribution, not original
production timing percentages or a qualification pass. The final unmatched
interval remains unmeasured. My list registry measurement remains too small to
justify a new registry index.

### My proposed scope-pop correction, before implementation

I found that eval_scope_release lowers symbol_count and then invalidates the
entire optional symbol index on every cleanup. My existing symbol_index_sync
already supports precisely this operation: it pops saved numeric links and
hashes without reading freed names, before indexing new slots. The ordinary
insertion path calls env_get_var_same_file before writing a reused slot, which
synchronizes the old count first. My existing test_env_symbol_index checks this
pop-before-insert behavior, reverse-scan equivalence, cross-file metadata and
allocation failure fallback.

I propose removing only the full-index invalidation from eval_scope_release.
The index remains Environment-owned; no borrowed name or Symbol pointer is
retained in it. Graph retirement, all frees, final symbol_count, function result
ownership, and error behavior remain unchanged. Module import's raw slot write
continues to invalidate explicitly. Destruction and allocation-failure fallback
continue to invalidate. This does not add a cache or alter declaration identity.

Before executing, I will add an actual evaluator cleanup regression proving
index identity survives an entered scope, shadowed bindings restore correctly,
freed names are not read, insertion after pop is correct, and explicit raw-slot
invalidation still works. I retain every old allocation/fallback control. A
fresh unchanged full checker graph must pass its existing deadline before I
claim timeout progress; complete bootstrap, thirteen methods, sanitizer scopes
and whole Make acceptance remain required. Function lookup's measured cost is
separate; I do not change its namespace/builtin/generated-list precedence in
this first correction.

## My full compiler graph boundary



My fresh 50de Make builds pass on Linux (82.773s) and Darwin (88.172s),
including unchanged component shadow graphs. Explicit bootstrap Stage 1 then
hits the original ten-second shadow deadline on both hosts (status2;
48.789s/65.949s command elapsed). No nominal import error occurs in these
terminals. Source/tool endpoint maps agree, groups are gone, and no outer
or capacity timeout fired. I retain terminal summaries under
`docs/evidence/record-lists-50de-preparation/`; full maps/CAS remain at the
original frozen Linux/Puck preparation roots. Fourteen methods remain unreached.

I reuse the reviewed 77226 observer only as diagnostic instrumentation. I
reapply its macro-only source delta onto exact 50de production and inspect the
result before execution. My compiler input is the original bootstrap
src_nano/nanoc_v06.nano, with the complete selected shadow graph. I rebuild all
three instrumented providers and freshly link against hash-verified 50de
providers; no old failed binary is replayed. The external argv0 layout, capture
helper, module roots, compiler selection and retained generated products remain
explicit. Exclusive stage sums remain distinct from inclusive graph counters.
No measured diagnostic result qualifies production or changes the deadline.

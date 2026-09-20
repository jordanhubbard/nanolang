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

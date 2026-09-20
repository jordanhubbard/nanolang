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

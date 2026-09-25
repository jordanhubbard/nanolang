# My raw C-seed hashmap ownership

I adopt generated native C-seed hashmaps into an explicit process-lifetime owner.
I preserve raw pointer identity across aliases, returns and separately compiled
modules. Explicit `map_free` first removes the registration, then releases the
map normally. At process exit I detach registrations and invoke their typed
cleanup adapters without holding the owner lock. This is actual cleanup, not a
claim of bounded retention in long-running programs or lexical reference counting.
My Wasm path retains its existing behavior; this checkpoint qualifies native C.

The owner is shared through `gc.c` but separate from reference-counted GC objects.
It consumes fresh raw pointers, calls the finalizer on registration-allocation
or exit-handler failure, accepts duplicate adoption without duplicate cleanup,
and requires quiescent callers for explicit bulk cleanup. Its finalizers may
forget their already-detached value and must not adopt new owned values.

All four ownership/capture Python methods pass with Homebrew LLVM ASan/UBSan,
leak and use-after-return checks. The owner harness directly compiles `gc.c`
and `gc_struct.c` with instrumentation; it checks 4,000 concurrent allocations,
2,000 explicit frees, duplicate registrations, both failure paths, counted bulk
cleanup and counted actual exit cleanup. The generated-program check exercises
string/int maps, returned maps, aliases, explicit free and a separate NanoLang
module. It disables conservative global/stack/register roots for LeakSanitizer;
its exit cannot pass merely because the owner registry holds a pointer.

A rebuilt instrumented Stage 1 compiler still fails its original hello smoke,
now at 18,724 bytes in 92 allocations. The previous capture-repaired checkpoint
reported 19,213 bytes in 95 allocations. I remove exactly the visited hashmap,
its entry table and its key (489 bytes total). Parser lists and owned path
strings remain under `task_02204077a4d34d4aa5ce20ef7054e113`; I do not claim
full instrumented bootstrap or installed-package qualification.

The complete neighboring gates pass: all 10 GC-struct tests, all 27 dynamic-array
tests and the ordinary bootstrap they require. The full transpiler gate also
passes its StringBuilder checks and both assertion-literal methods. I retained
the live tool sessions through a temporary observation stall; their final exit
codes and log contents establish completion, not a restarted command.

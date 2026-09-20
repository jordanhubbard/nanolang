# My generic-call match identity and VM fixture qualification

I repair two C-checker match paths so a direct call retains its checked concrete union return facts. I preserve the exact union-identity gate. I separately correct an existing VM underflow fixture: File service opcodes take the public service-refusal route before ordinary stack-underflow checks. My production repair is 8538; my VM fixture correction is c140. No passive-flow or shadow-timeout repair is included.

## My measured gates

| Gate | Linux | Darwin (puck) |
| --- | --- | --- |
| Initial component `make build` | passed 72.117s | passed 79.691s |
| Unchanged userguide compiler invocation | passed 3.776s | passed 2.975s |
| Totality and shared match policy |12 methods passed |12 methods passed |
| Actual fresh self-host bootstrap |passed 264.145s |passed 264.544s |
| Canonical guard corpus |6 methods passed 95.328s |6 methods passed 94.976s |
| Unchanged userguide corpus |40 snippets passed 171.690s |40 snippets passed 154.089s |
| Corrected VM fixture |274570 checks passed |274570 checks passed |

My first runner labeled a component `make build` phase `bootstrap`; that was not self-host bootstrap. Its next guard corpus ran all six methods: one passed and five errored because Stage1 was absent. I retained both original terminals, then ran actual `make bootstrap` and only the unreached/corrected phases. The C-seed binary stayed byte-identical through actual bootstrap. I do not relabel the initial failures or claim fail-fast unittest behavior.

The first full `make test` stopped at the old VM fixture: 271494 checks passed and one failed on both hosts. My corrected fixture preserves ordinary-underflow assertions, checks all four File local/caller-stack combinations, and asserts unchanged caller roots. Fresh corrected runs passed 274570checks and adjacent VM subtests. Their baseline build/provider copies were hash verified; the failing legacy test executable was not copied or replayed.

Darwin's next full suite reached reference transport but eight methods aborted because literal default `cc` selected Apple sanitizers while the fixture required LeakSanitizer. I retained that 473.474-second terminal. My external correction explicitly selected already inventoried Homebrew Clang through `NANO_NATIVE_TEST_CC`; no fixture or suppression changed. The unchanged eight-method transport corpus then passed in 7.598 seconds. Other compiler selectors remain recorded separately.

Linux's corrected full `make test` returned 2 after 870.355 seconds. Its 90-method flat-record corpus ran all methods:88 passed and two failed. Compiling my emitter reached the unchanged ten-second shadow deadline; unchanged passive-flow source was rejected for invalid passive eligibility metadata. PR910 c140 arm64 CI independently reached the same 90-method result and same two diagnostics. These are not green full-suite or release results.

I independently built canonical af8809 and repaired 8538 compilers from source in separate trees. Both reproduced both failures with fresh retained command/output evidence. That comparison is separately published at `docs/passive-shadow-baseline-attribution`, commit b951f2e0a. Tasks 4c107eb5835743dd9e8c0683eadd78ea and4931a66f39c04b1a96981a4e00d43646 remain open. The baseline timeout is observed, not explained.

## My evidence boundary

My phase reports preserve commands, stdout/stderr, statuses, process-group supervision outcomes, source maps, selected tools and providers, generated module products, and retained artifacts. Builds and userguide phases legitimately regenerate providers; I report their before/after differences rather than asserting one immutable toolchain across the complete sequence. Generated module/cache artifacts are separate products. My inventories cover selected executable tools and participating providers, not every transitive system library or executable.

My new totality harness and guard corpus retain temporary products before cleanup and overwrite. Older unchanged full-suite tests use their own TemporaryDirectory cleanup; their deleted failed module paths are not recovered historic artifacts. I retain their logs, commands and surviving build products and state that limit. No failed historical executable was replayed. Full compiler allocation coverage, every backend, and complete 5.1 release acceptance are not claims of this bounded repair.

Darwin's final full `make test` returned 2 after 1114.818 seconds. The same 90-method corpus ran in 689.274 seconds, with two failures and four subtest errors across two additional methods. Projected-record-array global store-first/main-first native executables and tuple seed/source native executables each exceeded 120 seconds. I preserve the exact timeout tracebacks without attributing them to LeakSanitizer, infrastructure, or a loop. Their separate task is `task_2f52721aac374ac592b61438315dc981`. The reference transport correction does not select the literal `cc` used by these tests.

My combined Linux seal retains 1042 reports, 5018 unique objects (1,058,211,965 bytes) and 13703 references. My combined Darwin seal retains 1181 reports, 4814 unique objects (871,041,943 bytes) and 17661 references. The per-host artifact-store/index manifests identify their archived bytes; actual report copies are tracked here. Qualification source pins remain 8538 and c140, not this later documentation head.

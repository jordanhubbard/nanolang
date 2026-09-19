# I qualify my private local-file service

I freeze source and tests at `4aad73a034f1b53d2072b2a1f9f88dbf25e2546b`.
My reviewed production remains `29bfceb2` capability lifecycle plus `9d333492`
file adapter; `d5bedefc` clarifies dispose versus destroy without changing code.
My contract is [NSI_LOCAL_FILE_SERVICE.md](../NSI_LOCAL_FILE_SERVICE.md), child
`task_f9ac5bb2adbf44198a5bdb8ed41309ff` under d03c. I keep that parent open.

| Frozen gate | Result | Outer seconds |
| --- | --- | ---: |
| Linux GCC, two private methods | PASS | 1.027 |
| Linux Clang23, two private methods | PASS | 1.035 |
| Linux six adjacent NSI targets | PASS | 4.890 |
| Darwin Homebrew Clang23, two private methods | PASS | 3.196 |
| Darwin first six adjacent targets, default cc | PASS | 5.048 |
| Darwin six adjacent targets, explicit actual Apple Clang | PASS | 5.128 |

Each instrumented run passes 2,634 assertions, performs 395 real temporary-file
opens and 395 real closes, counts 20 allocator calls, and leaves no live resource.
I test binary NUL bytes, zero-length read, both direction refusals and positioning
failure, partial read/write errors, ownership transfer and aliasing, full-table
rollback, stale/duplicate/cross-context identity, generation exhaustion,
320 bounded-live reuse iterations, and first-error preservation through disposal.
An independent sentinel remains live. Context/table allocation failures preserve
outputs; mint failure after real acquisition exercises rollback close, including
its secondary error. My context-address reuse control changes only private
identity on the same allocated context; I do not assume malloc reuses addresses.

The instrumented translation unit includes the actual production C sources after
private allocator/stdio interception. It can inspect private generation limits;
it is not an independently linked production binary. The second method separately
compiles and links unmodified `nsi_file.c` and `nsi_cap.c`, and tests real byte I/O,
direction guards, transfer, close and terminal disposal without private hooks.
Both methods retain strict C11 `-Wall -Wextra -Werror`, ASan/UBSan and leak detection.
Close-error interception closes the real stream before reporting the injected
error. I do not infer operating-system recovery from that controlled error.

I seal 53 reports in [report-sha256.json](local-file-service/report-sha256.json).
Each of the three inventories has 2,180 tracked input files equal before/after
and equal to my evidence-authoring tree. Linux has six actual fixed tools; Darwin
first entry has five, including the actual Homebrew compiler. The first Darwin
adjacent default-cc entry did not separately hash the underlying Apple compiler.
I retain that limitation and its passing log, then qualify only the affected six
adjacent targets with an explicit actual Apple Clang path and six-tool inventory.
I do not relabel the initial entry as fully compiler-identified.

My runner manifests retain exact commands, environments, statuses and log hashes.
The post-run artifact inventory hashes retained fixture executables and build/run
logs; these are post-run identities, not independent before/after executable
inventories. Linux frozen source remains at
`/home/jkh/Src/nanolang-local-file-qualified`; Darwin remains at
`/private/tmp/nanolang-local-file-qualified-4aad`. No failing gate was observed.

The adjacent targets are `test-nsi-cap`, `test-nsi-shm`, `test-nsi-fabric`,
`test-nsi-runtime`, `test-nsi-gen`, and `test-nsi`. These fresh C-only trees lack
an installed NanoLang compiler, so the conditional source-client check does not
run. I claim no bootstrap, generated binding, paired source, VM/AOT service-call,
Socket/GPU or full release acceptance. The private adapter remains unselected by
existing execution entry points; later public integration needs its own contract.

## My normal-suite integration

After the frozen4aad sanitizer qualification, I record the integration plan at
`70e3b8d7` and change only Makefile.gnu at `96d0f650`. My production and fixtures
remain byte-identical to4aad. `test-nsi-file` now compiles and runs both fixtures
with the selected `CC`, project `CFLAGS/LDFLAGS`, and explicit strict C11 flags;
`test-units` depends on it. Separate obj output paths retain failed artifacts.
`test-nsi-file-sanitizers` preserves the original Python qualification unchanged,
including explicit compiler selection and ASan/UBSan/leak policy.

Fresh isolated normal-target runs pass with actual Linux GCC (0.882 seconds)
and actual Apple Clang (1.802 seconds), each with the same 2,634 instrumented
checks and ordinary linked control. Both runs retain equal before/after 2,180
source inputs and actual compiler/build-tool inventories. Only Makefile.gnu
differs between the old4aad and new96d0 source inventories; old sanitizer claims
remain pinned to4aad. I inspect the direct `test-units` prerequisite and claim
only the selected normal target, not a new full test-units run. I do not repeat
unchanged sanitizer or adjacent gates.

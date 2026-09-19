# I qualify private CUDA buffer ownership on an actual GPU

I qualify task3c92 only: a private Linux LP64 adapter under d03c/ed702. Reviewed
production is7d51cfb3 after the bounded54eb accounting correction; frozen corrected
fixtures are52e6bf96. My [JSON seal](private-gpu-lifecycle.json) retains18 current
source/fixture inputs, complete before/after source inventories, actual compiler,
linker, driver library and CUDA header hashes, first/corrected manifests, logs,
archived input bytes and produced binaries. No public service binding, source
owner, NanoISA import, VM/native service dispatch or kernel execution is admitted.

My actual host is Linux aarch64 sparky. The separately linked adapter reports
CUDA GPU `NVIDIA GB10`, Driver API version13020 and device UUID
`80f79155371d354562f3c9d0e066444f`. My sealed loader file is
`/usr/lib/aarch64-linux-gnu/libcuda.so.595.84`; the runner retains the kernel driver
version text too. This is actual device allocation/copy/release, not OpenCL CPU
fallback, malloc simulation or a source integer handle. I do not infer another
GPU/device/driver/platform from this result.

| Frozen check | Steps | Result | Sum of command seconds |
| --- | ---: | --- | ---: |
| First5dd73 GCC run | 40 |39 pass, rollback fixture assertion fails |6.637 |
| Corrected52e6 GCC13.3 normal |56 |pass |11.701 |
| Corrected52e6 GCC13.3 ASan/UBSan/LSan |56 |pass |14.434 |
| Corrected52e6 Clang18.1.3 ASan/UBSan/LSan |56 |pass |14.270 |
| Corrected52e6 File/Socket/capability adjacency |23 |pass |7.110 |

Each corrected GPU configuration includes compiler identity, two strict C11
builds (`-Wall -Wextra -Werror -pedantic`), the separately linked real-device
fixture and52 instrumented cases, each in a new process. Sanitizers instrument
the adapter, capability table and fixture; their unmodified proprietary driver
is not sanitizer-instrumented. Leak detection is explicitly enabled. No warning
suppression, driver-error skip or failed-case retry changes the outcome.

The first rollback fixture injected current-query3, which is allocation's
restoration query. Its assertion expected rollback-entry query4. I retain that
first terminal and all original inputs/binaries/logs. Root reviewed the static
call sequence and the harness-only3→4 correction before the full corrected run.
Production and assertions remain unchanged; added diagnostics expose the actual
result fields. The corrected log records primary701, cleanup702, skipped702,
SKIPPED disposition, no free attempt and zero free calls. Context disposal later
reclaims the allocation without rewriting the skipped outcome. I do not relabel
39 earlier passes as a completed gate or infer missing first-run field values.

My ordinary instrumented case passes301 assertions, creates/destroys two adapter
contexts and allocates/frees163 real buffers, with zero fixture recovery. A
separate actual foreign CUDA context stays current across adapter operations and
disposal; the fixture owns and destroys that unrelated context itself while
holding its own loader reference. Full64-owner capacity, transfer rollback,
96 reuse iterations, both token-edge output overlaps, exact byte/NUL/subrange
transport, stale/duplicate/cross-context refusals and surviving independent buffer
content are checked. Another case fills all eight context records and refuses
ninth creation before loading or publishing output.

Fault cases distinguish missing loader/symbols, host allocation, driver queries,
context creation/current/push/pop, allocation before/after real success, copy and
synchronization, free before/after actual release and context destruction before/
after actual destruction. Read staging preserves caller bytes; failed write
contents refuse further data access. Per-entry diagnostics retain attempted,
error, skipped cause, disposition and later context-reclaimed facts. Generation
exhaustion refuses before driver allocation. Terminal faults latch future
acquisition without any reset or repeated raw-resource release.

In the injected destroy-before case, the adapter records unknown context release
and retains its bounded quarantine. The fixture observes its own still-live
context and destroys it directly once: `fixture_recovery_contexts=1`. That is
external fixture cleanup, not adapter success. In destroy-after, the real driver
already destroyed the context before the injected error; fixture recovery is0,
and adapter uncertainty remains. Other terminal cases retain their explicit
context/restoration/release flags and loader references. Static reachable
quarantine is an intentional bounded lifetime, not a claim of driver cleanup or
an absence of retained resources. Every fault runs in a fresh process.

All three manifests report unchanged source/tool/head identities and clean
tracked trees. Qualified binaries and original inputs remain in the sealed
`/tmp/nanolang-gpu-*` directories. Darwin GPU and unavailable-stub execution are
not measured here. Real other-platform GPU, public Device/Buffer/Result,
representative both-frontend/VM/native integration and legacy OpenCL identity87ca
remain required separate parent work. I close no full d03c/ed702 obligation.

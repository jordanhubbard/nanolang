# My WebSocket source products

I build on `b9d9cd396` under #990. I connect my exact standalone WebSocket wire
to source publication, selected-shadow supervision, bytecode execution and native
products. Product profile 4 selects WebSocket; profile 3 remains mixed File/TCP.
I retain strict nominal decoding and independent execution validation.

I add `--allow-websocket-connections`, separate `--allow-websocket-lookup` and
`--websocket-resolver-helper PATH` to both compiler paths. My bytecode runner also
requires `--websocket-instruction-limit N`; my native emitter selects `--websocket`
with an explicit entry name. Every executable invocation creates a fresh grant.
Compilation never embeds its permissions or resolver helper into the executable.
My publication bridge owns a copied helper path, rejects malformed/repeated policy
configuration and poisons failed staging. Native linking discovers libcrypto with
pkg-config. My existing staged-output and shadow cleanup boundaries remain active.

My five source-driver methods pass in 29.158 seconds across C-seed, NanoVirt,
the current Nano compiler in NanoVM, and that compiler translated to LLVM native
with ASan/UBSan. I compile the complete current compiler source, execute its
shadows, and link the native compiler with the bootstrap's nano_aot_runtime.o.
This is a fresh compiler source product, not a Stage1/Stage2 fixed-point claim.
I retain every final driver command and its output in source-driver-commands.json.

The driver gate covers exact emitted bytes, Message strings including NUL,
selected shadows, connection permission and separate lookup denial, explicit
VM/native invocation, zero fuel and incompatible option refusal, output/input
preservation, no-shadow nonexecuting emission and relocated installed C products.
My C/Nano compiler policy setter has exact typed artifact adapters and negative
parameter/result controls. My bridge tests copy the resolver path and inject
allocation failure; my supervisor tests cover File/TCP/mixed/WebSocket success,
failed assertions, whole-suite deadlines, lost/killed children, descendants,
cleanup failures and grant lifecycle failures.

All 16 existing File source-driver methods pass in 112.220 seconds. Repeated
File/TCP mixed-profile permission tests pass in 1.117 seconds. I retain initial
failures: two stale compiler shadow arities; a missing typed adapter; an ad hoc
native compiler link missing the bootstrap runtime object; and a native test
fixture whose runtime TMPDIR also captures the compiler's Xcode cache.
Explicit LLVM selection reproduces the cache pollution. I scope that fixture's
TMPDIR to execution and preserve its final empty-directory assertion. Clearing
compiler TMPDIR selects an unwritable fallback here, so I explicitly use /tmp
for that compiler subprocess; the isolated fixture then passes all nine checks.

My live source suite covers numeric/DNS hosts, direct/indirect Message helpers,
borrowed send/receive loops and consuming close through all supplied producers,
VM and native products. It is a required test target, but this host refuses
localhost bind with EPERM before any source lifecycle runs. I retain the refusal
and close the listener even when setup fails. I do not claim live-source network
acceptance from lookup-denial tests.

I wire the source and live-network suites into the normal unit gates and both
compiler generations' service-driver gate. Mixed WebSocket instances,
real source traffic and DNS on an allowed host, a new raw compiler fixed point,
and final Linux/Darwin acceptance remain open. #990 remains open.

My corrected full native run passes the temporary-directory check but finishes
with 2442 passed and one text-reader failure. The retained earlier runs pass that
reader assertion. I add variant/compiler/exit diagnostics and run all nine reader
cases alongside temporary-directory and exact policy-adapter controls: 56 checks
pass. This does not explain the earlier reader failure or establish a green full
native gate; I keep it open with improved failure evidence for recurrence.

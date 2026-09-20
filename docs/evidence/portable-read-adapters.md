# I qualify my private native read-text boundary

I retain production `f36010d82` unchanged. These are direct C and typed LLVM ABI
checks of my private adapter and copied managed STRING wrapper. I do not admit
NanoISA CALL_EXTERN, select a public LLVM/Wasm profile, qualify source programs,
or claim real Wasm hosts or installed linkage. Task b7ef and parent2d2 stay open.

## My actual matrix

Every configuration runs the same observed and unhooked C/LLVM O0/LLVM O2 routes.
Each observed route passes1741 checks, including four measured call allocations
and every persistent-prefix/single-transient failure with fresh recovery.

| Configuration | Source attribution | Observed / unhooked checks per route |
|---|---|---|
| Linux GCC13 ordinary | b9ba | 1741 / 531 |
| Linux GCC13 ASan/UBSan/LSan | b9ba | 1741 / 531 |
| Linux Clang23 ordinary | b9ba | 1741 / 531 |
| Linux Clang23 ASan/UBSan/LSan | b9ba | 1741 / 531 |
| puck Apple17 ordinary | b9ba | 1741 / 531 |
| puck Homebrew23 ordinary | b9ba observed; fresh7b66 unhooked | 1741 / 532 |
| puck Homebrew23 ASan/UBSan/LSan | fresh7b66 | 1741 / 532 |

The four Linux phases take3.077/5.476/3.323/5.827s; Apple ordinary4.706s.
Homebrew ordinary's original2.822s phase passes all three observed routes then
fails unhooked compilation. Its corrected three-route continuation takes2.101s.
The fresh sanitizer supervisor takes7.083s. Original successful runs are never
relabeled as the corrected fixture. All42 intended route executions pass across
these separately attributed records; no single all-green original matrix is claimed.

I rebuild all three production TUs (host, wrapper, managed_strings) in each
configuration's observed/unhooked mode. Observed mode hooks project malloc/
calloc/free in all three and SDK-safe I/O observers only in the host TU. Other
libc/system allocations are not part of that project tracker. Sanitizer flags
apply to these TUs and the C fixture; LLVM forwarding objects are independently
compiled with their recorded flags and carry no memory accesses. There are no
prebuilt compiler/provider objects. This is not whole-compiler instrumentation.
All sanitizer runs use detect_leaks=1, halt_on_error and empty LSAN_OPTIONS.
Apple ordinary does not imply Apple LSan acceptance.

The controls include real empty/one-byte/multibyte/partial-UTF8/NUL/1MiB/excess
files, a real multibyte filename,64 copied allowlist rows surviving source mutation
and free,65-row refusal,4096/4097 path boundaries, empty/absent/denied contexts,
pairwise buffer/context overlap and integer-wrap refusals before opens. Actual
closed descriptors and a live unrelated sentinel are checked. Modeled read-error
reporting follows a real one-byte read; modeled close-error follows a real close.
These are deliberately separate from actual OS failures. A prior LIMIT survives
modeled close failure. Active-context destruction refuses during observed fopen.

The wrapper borrows the argument; three retained references represent caller,
outside alias and a simulated global root. No generated-global behavior is
inferred. Full initial slots plus prepared collection storage force result-byte,
slot-table and workspace allocation paths. Every failure preserves original
roots/bytes, fresh recovery succeeds, and cleanup reaches zero tracked/managed
objects. Successful result bytes survive callback scratch destruction. Unknown,
unset, oversized or NUL callback output cannot publish a managed result.

## My retained failures and corrections

1. Original15069 Linux external discovery uses file-name queries for executable
   helpers. It stops before builds. The reviewed driver uses program-name for
   cc1/collect2, retaining file-name queries for actual libraries.
2. Corrected external setup reaches15069 Linux's fixture, whose unflagged target
   probe mixes a compiler selection warning into its exact triple check. No
   production build or adapter execution occurs.
3. Concurrent15069 puck stops on strict LLVM target mismatch before adapter
   execution. b9ba derives the exact target from retained compiler-generated IR,
   keeping strict warnings and the unchanged typed forwarding body.
4. Homebrew ordinary b9ba passes three observed routes, then rejects three
   set-but-unused observer globals in unhooked compilation. One assertion in7b66
   requires those unhooked observers remain untouched. No warning suppression,
   production change or weakened original assertion follows. The continuation
   verifies archived object/input hashes and uses an external corrected fixture;
   its cwd remains the frozen original root. Sanitizers rebuild in a fresh tree.

## My sealed scope

[The manifest](portable-read-adapters/report-sha256.json) hashes every retained
report/driver/map in the committed seal. The consolidated store at
`/tmp/nanolang-read-adapter-artifacts` contains647 unique artifacts,
434834313 bytes. Linux contributes538 raw report files and puck436; aggregate
manifests/drivers and explicit input identities are additional files.

I verify28 equal map pairs (15 Linux,13 puck). The actual maps retain10894 Linux
and9583 puck input/product references; these counts are not inferred from phases.
Current input checks verify24193 Linux and36286 puck distinct source/tool paths.
Sources include the full tracked source maps; runtime fixture inventories archive
participating C/header/include files and selected compilers/Python/runtime helpers.
This is endpoint equality, not proof of intermediate immutability or all
transitive tools/libraries. SDK selection and exact tool identities are recorded.

The downloaded puck seal archive is `/tmp/nanolang-read-adapter-puck-seal.tar.gz`,
11372752 bytes, SHA256
`1fc8be2a24753398afad5ee3e8e1ccfb00cf9bf2d0a7075c995b09fae082a6d5`.
I rehash each extracted object and all local artifacts. Original qualified
checkouts, failed reports and successful outputs remain preserved.

# My private real Wasm read-text acceptance

I qualify production9f99d4757 and corrected fixture9856fbad8, after independent
source and fixture review. Native PR903 remains the prerequisite. This is direct
private guest/Node/Wasmtime host acceptance, not NanoISA import admission, emitted
source acceptance, installed CLI/package acceptance or closure of b7ef/2d2.

| Host and scope | Measured result |
|---|---|
| Linux actual engines | Eight routes pass in8.433s supervised phase (8.360s unittest). |
| Puck actual engines | Eight routes pass in5.732s supervised phase. |
| Linux native/query neighbors | Three phases pass in25.226s. |
| Puck corrected native/query neighbors | Three phases pass in19.132s. |

Each host freshly compiles all three guest TUs at O0 and O2, ordinary and
NMS_TESTING. Each of those four guests runs under actual Node and separately
under the private official Wasmtime43 binding. Each ordinary route reports
336 checks, eight primary real file vectors and16 modeled groups; each observed
route reports362 checks, eight vectors and19 modeled groups. Thus each host
retains2792 successful assertions across eight routes, not an exhaustive proof.
Every actual-engine phase retains41 commands, generated LLVM IR, objects, Wasm,
malformed modules, private wheel/member bytes and host fixture products.

I exercise real copied file contents/returns and allowlists, Unicode filename,
partial UTF8/NUL legacy behavior, exact1MiB/excess limits, real memory growth and
maximum refusal, pre-effect spans/overlaps, actual engine rejection of malformed
order/body, trap-terminal handling and explicit host disposal. Filesystem errors,
host allocation/publication failures and callback growth/reentry are separately
labeled trusted hooks; guest three-site failure prefixes have their exact
NMS_TESTING domain. My engines and guest Wasm are not ASan/UBSan rebuilt. Node GC
reclamation timing, fatal engine OOM recovery and malicious hook containment are
not established. A trapping guest does not claim generated managed cleanup ran.

My native neighbors use unchanged managed core observed/production C controls
and module lifecycle/dispose-first controls, freshly built with GCC on Linux
and Homebrew Clang on puck under ASan/UBSan and unsuppressed leak detection.
They do not stand in for managed Wasm or emitted cleanup. Ordinary native
read-text adjacency rebuilds its three TUs for direct C and typed LLVM O0/O2:
1741 observed checks and532 unhooked checks on each route/host. The declaration
query rebuilds current providers and runs5994 instrumented allocation-control
checks plus267 linked controls on each host. Its query TUs/providers are ordinary
builds here, not newly sanitizer qualified. No query bytecode or host operation
executes. Full installed CLI/package linkage remains later under2d2; puck's
missing Wasmtime CLI is not replaced by a pretend CLI acceptance result.

## My selected environments and retained failures

Linux uses LLVM23git11af80c3 Clang/LLD, Node26.0.0 and Python3.14.5. Puck uses
Homebrew Clang/LLD23.1.1, Node26.9.0, Python3.14.7 and SDK26.2. Both bind exactly
the platform-specific official Wasmtime43.0.0 wheel recorded in the design
provenance, privately extracted and member-hashed. No package is globally installed.
Linux's native Clang adjacency explicitly selects GCC13 headers; Wasm compilation
uses its own wasm32 target and does not inherit native GCC selection flags.
The Linux clang --version log retains its informational GCC14-header warning;
actual strict Wasm and native compilation checks pass without warning suppression.

I preserve all first terminals. Both4ee gates return1 during linker version
preflight (Linux1.268s, puck0.450s): symlink resolution changed wasm-ld invocation
to generic lld. No guest compiled or ran. Precode1cae6132e and one-line9856 fixture
correction preserve selected executable spelling while retain() hashes actual
resolved bytes. The corrected fresh complete engine matrix above is separate.
Puck's first neighbor driver returns1 in0.048s before tests because non-login
SSH PATH omits installed Homebrew pkg-config. Precodea087bc66a retains that
terminal; the reviewed external driver explicitly selects the tool and sets
PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin. Only previously unrun puck
neighbors execute after that correction. Completed engine/Linux phases do not
repeat. All terminal process groups disappear and leaders are reaped; no timeout
or cleanup-error terminal is hidden. Original trees/reports remain intact.

## My immutable evidence boundary

[My report manifest](portable-read-wasm/report-sha256.json) seals report bytes;
[my summary](portable-read-wasm/qualification-summary.json) identifies each phase.
The merged artifact store contains881 unique objects totaling
564206210 bytes, with33065 archived references. I compare34 complete
before/after source/tool/input pairs; product changes are expected and separately
archived. Each engine phase hashes14997 tracked source paths at both endpoints.
The collectors then freshly rehash30117 current Linux paths and30116 puck paths,
covering both preserved source trees, selected providers and extracted bindings.
[My sixteen selected inputs](portable-read-wasm/qualified-inputs.json) are byte
identical to the qualified9856 tree at publication; later ledger text is separate.

Endpoint equality is not intermediate immutability or complete transitive system
toolchain coverage. I inventory selected executables/runtime files and actual
provider bytes, not every SDK/config/system-library dependency. Command products
are copied to content-addressed storage before fixture assertions. The puck
archive is37311586 bytes, SHA256
`5dae57f63f9808497d2746a6c6cc93a1c0dfd7e565eee836618d58ed6d9317fb`;
its report/object members are retained locally. Original and corrected evidence
remain separately attributed; no parent or full-release criterion is closed.

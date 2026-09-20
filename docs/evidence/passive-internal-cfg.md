# I validate internal passive short-circuit control flow

My production12a7cc3fe implements the [reviewed contract](../NANOISA_PASSIVE_INTERNAL_CFG.md).
Version2 nodes admit bounded, reachable forward control flow with exact targets,
equal stack heights at joins, exact static dependency/input sets and one final
result store. Version1 stays on its original validator. This changes neither
short-circuit source semantics nor passive wire encoding.

At corrected1f0db21e6, fresh Linux GCC13 and Darwin AppleClang17 builds pass all
12 measured phases per host. Each passive suite passes26 methods, including
four new CFG methods, and the direct instrumented fixture passes273 checks.
Both independent source producers emit matching flow/par bytecode metadata;
the unchanged flow program prints15 in VM and native code. Scoped passive-TU
allocation tests pass with GCC/HBClang ASan/UBSan, detect_leaks=1 and an empty
LSAN_OPTIONS. Other linked providers remain ordinary: this is not whole-program
sanitizer coverage. The two unchanged scalar/tagged string-to-float comparison
methods pass too; their existing Darwin runner disables leak detection locally,
which I do not conflate with the separate strict passive sanitizer result.

The first35957 Linux checks already pass. Its Darwin suite reaches273 C checks
and25/26 Python passes, then the old arctan native compile fails strict unused
function diagnostics for nparse_binary64. Corrected1f0db adds only a standard-C
function reference under exactly the existing wrapper-emission predicate,
matching nearby optional numeric helpers. It does not execute the parser,
change parsing, or suppress a warning. All strict arctan and conversion checks
then pass in fresh trees. No historical failed executable was replayed.

I retain the earlier Darwin external-driver preparation error separately:
its Git archive lacked the required tracked-file manifest. No build launched.
I supplied the manifest before the prepared run; I do not call that first
preparation successful or claim a nonexistent raw stderr file.

My [seal](passive-internal-cfg/report-sha256.json) contains388 reports,
865 content-addressed objects totaling391281422 bytes under
/tmp/nanolang-passive-internal-cfg-artifacts,14246 artifact references and76
equal source/tool endpoint pairs. Exact commands, compiler/actual Python and
resolved Darwin Xcode clang/ld hashes, SDK version and phase terminals remain.
The complete Darwin archive has SHA256
`d3d688f080976dc2fa93e08f5e2568d34e631d521bb007fcaf6125d07374c97e`.
I do not hash the full SDK or operating system. Original ordinary Python tests
clean their temporary products; I retain their logs rather than claiming those
missing files. Report-owned flow products and inventoried provider bytes remain.

Source/fixture review preceded execution. Independent seal audit and canonical
integration are recorded below; actual merge remains pending. The separate fullgraph emitter
shadow deadline, Darwin native executable timeouts and full5.1 release remain
open. This bounded passive repair does not establish complete compiler/bootstrap
fixed points or production correspondence to the formal model.

## My independent seal audit

I retain the [independent audit](passive-internal-cfg-independent-review.json)
with SHA256b1c2a8dbffc8550ceeeab655bd51bbbc942fd9559de58e426a69ecc18a4c1508.
It verifies every sealed Git report and CAS object,14246 references,76 matching
endpoint pairs, the complete Darwin archive and3798 selected source/fixture
Git identities. Both corrected hosts pass all12 recorded phases. It preserves
the original Darwin failure and all stated instrumentation/retention limits.
It does not rehash current remote endpoints or establish full-product acceptance.
Canonical integration is recorded below.

## My current-main integration

At `6ae98a219a9588ec554348dd55c3b2f127f3ea51`, I integrate canonical
`0061feed573ad7e4cac2a9e61d5ed38f0fb62440` with both additive Make and roadmap
histories intact. I rebuild fresh providers on Linux and puck. All twelve
phases pass on each host: 26 passive methods and 273 direct checks, unchanged
flow VM/native output 15, both independent source comparisons, scoped passive
allocation sanitizer checks, and the two existing float-conversion methods.
The original sanitizer instrumentation and temporary-product retention limits
above apply unchanged.

My [integration seal](passive-internal-cfg-integration/summary.json) records
239 reports, 493 retained artifact objects totaling 211112401 bytes, 9205
artifact references and 48 equal source/tool endpoint pairs. I retain the
Darwin archive identity in that seal and do not relabel earlier measurements.
This repairs the bounded passive gate; the compiler shadow deadline and separate
Darwin native timeouts still require their own evidence before release.

## My affine-union integration

At `933aa91eab7092b238805f772b4fd3564997ae19`, I integrate actual PR893
merge `0dc58835ffd2183f6c35f90a4d02bd2da18e4a19`. The passive implementation,
new fixtures and optional generated-parser reference remain unchanged; the
shared ownership/affine providers change, so I rebuild them in separate fresh
trees on Linux and puck. All twelve phases pass on each host with the same
26 methods, 273 direct checks, exact output and comparison assertions.

My [new supplement](passive-internal-cfg-union-integration/summary.json)
retains 239 reports, 493 artifact objects totaling 212243248 bytes, 9205
references and 48 equal endpoint pairs. Original evidence remains at its own
pins. The [preceding independent integration audit](passive-internal-cfg-integration-review.json)
verified every report, artifact, selected Git input and current endpoint for6ae;
it does not claim to cover this later933 integration. Full release gates remain
open, including the still-observed compiler shadow deadline.

I retain the [independent933 audit](passive-internal-cfg-union-integration-review.json):
every Git report, CAS object, artifact reference, endpoint pair and actual
terminal matches. Current source/tool/product maps also match on both hosts.
All fourteen incoming source/fixture paths match canonical0dc588; all2444 selected
production/schema/test inputs per host match frozen933. This is scoped test
evidence, not full-product acceptance or a formal proof.

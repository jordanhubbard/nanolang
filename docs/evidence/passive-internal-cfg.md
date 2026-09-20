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

Source/fixture review preceded execution. Independent seal audit, canonical
integration and actual merge remain pending. The separate fullgraph emitter
shadow deadline, Darwin native executable timeouts and full5.1 release remain
open. This bounded passive repair does not establish complete compiler/bootstrap
fixed points or production correspondence to the formal model.

# My selected-pattern and catalog checkpoint

I track this work under GitHub #976. Full 5.1 acceptance remains open.

## Selected patterns

I retain concrete selected payload types in my ordinary Nano emitter. In both
ownership emitters, I alias copyable selected union payloads without consuming
or renaming the original binding. Owned payloads retain destructive unpacking.
I preserve exact variant, complete field and distinct field checks.

My changed Nano emitter component builds with its mandatory shadows, including
an ordinary selected-payload reuse assertion. My 50 C-compiler selected/generic
pattern and ownership methods pass. Five explicit C NanoISA mixed-profile cases
cover integer/string reuse and wrong-variant, missing-field and duplicate-field
refusal with prior output preservation. Both accepted modules pass verification,
VM execution, and strict LLVM ASan/UBSan/leak native execution.

My fresh NanoISA-native compiler also passes all 50 regression methods,
all five mixed-profile VM/native cases, and the full shared ownership-effect
source matrix, including imported resource/effect identities and refusals.

## Catalog formatting

I memoize successful immutable catalog views and join ordered fragments with a
balanced tree instead of repeatedly copying the growing output prefix. Invalid
catalogs retain the original getter validation and empty result. I preserve all
catalog bounds and existing shadow assertions.

My focused cold/repeated three-catalog probe matches the independent C catalog
serializer byte for byte in VM and native execution. The output SHA256 is
`4104c938ba3e0d58fe0334e513b6fe66383427772aba3b5d4061024a75a1d5f4`.
Native generated code is compiled with strict warnings, ASan/UBSan, and leak
checking; the linked runtime archive retains its existing instrumentation.

| Probe | Allocated bytes | VM instructions | Foreign calls |
| --- | ---: | ---: | ---: |
| Original | 4,047,013 | 63,561 | 1,742 |
| Memoized | 2,031,109 | 32,068 | 873 |
| Memoized, balanced join | 248,177 | 66,904 | 873 |

Reduced allocation does not establish reduced full-compiler execution time.
The retained memoization-only full audit still failed the unchanged ten-second
VM shadow gate. I require a fresh complete audit for the final implementation.

## Evidence

`patterns/` retains original refusals, component builds, full C regression output,
and executable mixed-profile checks. `catalog/` retains the independent C
reference, source probe, exact-output report and instruction/allocation profiles.
The compiler audit records immutable source hashes and preserves its emitted
shadow module before invoking the ordinary deadline gate.

## Foreign dispatch boundary

My final catalog source still exceeded the full deadline. Its diagnostic profile
spent 9.389 seconds in foreign dispatch. Each ordinary VM foreign call scanned
all module code for pending service instructions twice: once at VM dispatch and
again in the delegated public bridge. I now share that completed check only
within one synchronous ordinary dispatch using a private helper. Public bridge,
callback and isolated entry points keep their checks; I retain failure-output
behavior and do not cache authorization across calls or foreign execution.

The exact retained module now passes the unchanged `--check-shadows` gate in
5.94 seconds. Paired profiles retire identical instructions and calls, with
foreign-dispatch time reduced to 4.854 seconds. All 28 FFI methods pass, including
retained callbacks. Both instrumented and linked File-opcode configurations pass
1,890 checks each, including foreign-call refusal and unchanged output. I retain
the first Make setup failure (LLVM `opt` absent from PATH) and the successful
explicit Homebrew LLVM rerun. My fresh full compiler product now passes its normal shadow gate, verification,
strict native translation/build and startup, then the full shared effect matrix.
The manifest confirms unchanged pinned inputs throughout all six steps.

## Fresh compiler product

`product/` retains the six-step audit, exact source/runtime hashes, artifact
hashes, all 50 selected-pattern regression results, and the mixed-profile cases.
The full source emission completes in 338.7 seconds with the ordinary shadow
gate enabled. I then verify the emitted module, translate it to C, compile with
strict warnings, start the compiler and execute the shared effect-source matrix.
The compiler product is generated with the previously qualified C-built Nano
seed; tested programs are emitted by the resulting NanoISA-native compiler.

This closes these selected-pattern and deadline defects. I do not claim a new
Stage1/Stage2 byte-identity bootstrap, Linux qualification or full 5.1 acceptance.
Broader resource collections, borrowed continuations and combined effect/global
flow remain separate requirements under #976.

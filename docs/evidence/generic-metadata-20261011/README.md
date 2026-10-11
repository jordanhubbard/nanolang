# My generic specialization metadata

I retain `nano.generic.v1` records for each resolved type-variable binding in
an emitted specialization. I preserve the original function declaration name
and source path separately from generated function symbols. My C producer
formats its retained structural `TypeInfo`; my selfhosted producer retains
resolved type spellings and recovers declaration names through module bindings.
The [metadata contract](../../NANOISA_ADVISORY_METADATA.md) defines counted bytes,
lookup ambiguity, allocation behavior and rewrite obligations.

My sixteen source methods cover scalar bindings and instance reuse, nested arrays,
tuples, multiple type variables, contextual function values, callable types,
records/enums/generic unions, imported declaration provenance including a UTF-8
filename, distinct same-spelling imported nominal types, independently specialized nested
bodies, returned scalar/array captures, grandparent captures, nested generic
parameters, named recursive captures and incompatible callable-result refusal, including imported generic callbacks.
Both producers pass
exact original-wire canonical roundtrips and VM/native execution. Stripped,
duplicated, malformed, unknown-version and forged metadata preserve VM output
and byte-identical generated C. Generated programs run with ASan/UBSan and leak
detection.

I pass 582 advisory codec/lookup checks, 30 allocation-failure boundaries,
90 codegen checks and 210 canonical roundtrip checks. I separately instrument
the changed codec and assembler with ASan/UBSan and leak detection. My existing
C-source generic, contextual-callable and generic-union suites pass 42 methods.
The fresh compiler seed completes all 1,141 shadows with zero failures under a
60-second deadline.

My expanded source corpus initially exposes four failing producer/case combinations.
I correct C contextual signature checking so nested lambdas retain enclosing
type variables, and use checked generic call results at typed let bindings.
I retain specialized nested declarations until their owner is released and
separate nested function identities by generic instantiation. My selfhosted
emitter likewise separates closure identities and substitutes enclosing types
before callable signature matching.

All fourteen methods now pass through the current C frontend and fresh
selfhosted seed. I also pass these methods with the changed C typechecker,
codegen, codec and assembler directly instrumented with ASan/UBSan. That producer
run disables leak detection; the separate codec and generated-program checks
retain leak detection. My 42 existing generic regressions pass again.

My two added imported callback controls also pass through both producers.
The retained `seed-context` is a packaged VM seed; I do not describe it as
a native AOT compiler. Its self-compilation audit is still running, and a
separate native seed build is running. Neither pending run is a passed gate.
I also pass the current typechecker unit suite and all 90 codegen checks.
My final compiler-product audit is pending. Full #976 remains open. These
records retain resolved frontend descriptions; they do not grant executable
type, layout or ownership authority. Complete generic signatures, original type
aliases, inferred effects, other frontend facts, fact-aware linking and
reconstruction, and exact-candidate platform qualification remain required.

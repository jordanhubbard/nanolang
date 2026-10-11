# My generic specialization metadata

I retain `nano.generic.v1` records for resolved type-variable bindings in each
emitted specialization. I preserve the original declaration name and source
path separately from generated symbols. My C producer formats checked
`TypeInfo`; my selfhosted producer retains resolved type spellings and recovers
original declaration names through module bindings. My [contract](../../NANOISA_ADVISORY_METADATA.md)
defines counted bytes, ambiguity, allocation behavior and rewrite obligations.

I also repair nested callable specialization. My retained failures show reused
nested identities, unspecialized callable results, and callbacks incorrectly
classified as independent generic declarations. I separate nested identities by
enclosing instantiation, retain substituted nested annotations for their owner's
lifetime, and compare complete specialized result signatures in both frontends.

My seventeen source methods cover scalar bindings and reuse, arrays, tuples,
multiple variables, contextual function values, callable types, nominal and
generic union identities, imported provenance including UTF-8 paths, independent
nested bodies, scalar/array/record captures, grandparent captures, generic nested
parameters, recursion, and direct/imported signature refusal. Two closures
sharing one record specialization retain distinct captured records.

Canonical assembly preserves original module bytes. Stripped, duplicated,
malformed, unknown-version and forged advisory records leave VM output and
generated C unchanged. Generated programs execute with ASan/UBSan and leak
detection in the product audit.

I pass 582 advisory codec checks, 30 allocation-failure boundaries, 210 canonical
roundtrip checks, the current typechecker unit suite, 90 codegen checks and 42
existing C generic/contextual/union methods. I directly instrument the changed
codec and assembler with ASan/UBSan and leak detection. I separately instrument
the changed C typechecker, codegen, codec and assembler: all fourteen initial
source methods pass with leak detection disabled for that producer run. The two
imported controls and record-capture control are later additions, tested through
the ordinary producers and resulting native compiler.

My eight-step [product audit](product-manifest.json) passes with every recorded
implementation input unchanged. The packaged VM seed compiles the compiler to
NanoISA in 895.9 seconds; verification, C translation and strict C11 linking pass.
The resulting native AOT compiler passes all seventeen source methods and 27
adjacent purity, generic-union and contextual-callable methods. The entry gate
predates the three added controls and reports fourteen methods; its resulting
product gate runs the complete seventeen-method corpus. I preserve the earlier
failed audit under `before-nested-fix/`.

My independently built native seed passes sixteen methods plus the later record
control. Its separate [eight-step audit](native/product-manifest.json) also passes
with unchanged inputs, seventeen resulting-product methods and 27 adjacent
methods. I retain separate manifests for native and packaged VM seed execution.
Both independently emitted compiler modules are byte-identical.

Full #976 remains open. These records describe frontend bindings and do not
authorize types, layouts or ownership. Complete generic signatures, original
type aliases, inferred effects, other frontend facts, fact-aware linking,
reconstruction and exact-candidate platform qualification remain required.

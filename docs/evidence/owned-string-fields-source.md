# I qualify retained STRING owner fields through source producers

I freeze source and harness at `366f77eb491479b2967a32f0fddc500e204904e0`,
after canonical `bfdde227` integration. My paired source production is
`2fd854a9`; the later comment precisely excludes FLOAT fields/signatures.
The runtime-only and initialization-join seals remain separate and unchanged.

| Phase | Observed result |
| --- | --- |
| Fresh bootstrap | PASS, 265.899 seconds |
| Explicit interpreter/NanoVirt/emitter/VM/native/metadata/shadow setup | PASS, 25.858 seconds |
| Three focused methods, GCC native sanitizer controls | PASS, 191.461 seconds |
| Same three focused methods, Clang native sanitizer controls | PASS, 191.904 seconds |
| First 58-method adjacent run | Exit 1, 604.558 seconds: five diagnostic subtest failures in one method; the other 57 methods passed |
| Fresh corrected affected method at `ff570537` | PASS, 174.458 seconds |

My [first source seal](owned-string-fields-source-first.json) retains 27 reports,
25 source-file identities and 405 final fixture/tool artifacts. All five failed
subtests had already checked positive refusal status, unchanged output bytes and
absence of parser errors; the existing `supported scalar operator` diagnostic
was missing from the migrated STRING-order case's regex. I recorded that finding
before the case-specific harness correction. My [corrected seal](owned-string-fields-source.json)
retains 11 reports, 20 implementation/harness files, 11 actual bin tools and 14
new fixture artifacts. The implementation and bin tools did not change. I do not
relabel the first aggregate run or claim a second full 58-method run.

The new focused harness extracts the original PREFIX and Bundle test text
verbatim. It runs the interpreter and C-seed/Stage1/Stage2 normal compilation,
both canonical source paths and fresh emitters, actual VM/native execution,
and all selected close/main shadows. Additional cases preserve direct observed
STRING aliases after shell consumption, reversed destructuring, mutable local
replacement, empty strings, nested owned calls/results and source order.
Concatenation, ordering, standalone STRING results, FLOAT owner fields and
borrowed STRING roots retain checked output-preserving refusal. The old mixed
STRING/INT comparison and unconsumed-owner controls remain negative.

Generated native programs use GCC or Clang 18, strict C11 warnings and
AddressSanitizer/UndefinedBehaviorSanitizer with Linux leak detection. This is
not full sanitizer instrumentation of every compiler/VM object. The fresh
producer tools are recorded at class teardown; unchanged bin-tool maps bracket
each fixture phase. Final fixture files are retained, while reused per-case
native output paths are final snapshots rather than every intermediate binary.
Twelve qualified bin/helper files were copied and hash-verified before later
integration. Source inventories, commands, statuses and compiler hashes are in
the seals.

I qualify Linux at these exact pins. Darwin source acceptance, subsequent
canonical integration and whole-product/release gates remain distinct; this
source evidence alone does not close the full platform task or its parents.

## My canonical integration

I preserve the source-qualified tree and its tools, then merge canonical
`9821e3901a407f392cf5fb60ab23222809181dc7` in a separate tree at
`c54af77e`. Both source producers remain byte-identical to the reviewed source
seal. Make and roadmap conflicts preserve both additive branches; the shared
affine-state addition includes the canonical private mixed Samples query.

My [integration seal](owned-string-fields-integrated.json) records all six
phases passing: GCC setup8.959/fixture30.768/frozen13.412 seconds and Clang
setup0.343/fixture19.587/frozen15.987 seconds. Sources and frozen tools/objects
remain unchanged. STRING fields, initialization joins, print/failure recovery,
nested results, borrowed references, private mixed layout/FLOAT proofs, the
112,454-check mixed Samples composition suite and1,239 owned binary64 controls
pass. This native-only integration does not repeat source bootstrap.

The private composed query always constructs the FLOAT proof first and returns
on its failure. Existing direct STRING-owner proof refusals plus the integrated
composition suite preserve that boundary. I did not separately execute a
composed-STRING control. The mixed query target uses its normal strict C11 O1
flags; generated native STRING/nested controls retain their sanitizer flags.
Exact commands and object reuse are recorded, not described as a fully
sanitizer-instrumented runtime. Darwin source acceptance remains open.

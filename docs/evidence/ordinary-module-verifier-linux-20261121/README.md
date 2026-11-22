# My ordinary unlinked whole-module verifier

I qualify `6a2fb2e750a29a3b8b993e40165265d99c6ffc64` on Linux ARM64 under #1142.
I reuse my checked V2 module/conversion facts and ordinary function-body helper.
I preserve the actual public loader's positive declared-depth pass before its
full ordinary unlinked module pass. All function metadata precedes every body;
service policy, signatures/locals, empty results, overlapping spans and foreign
argument limits retain their original priority.

`verify_declared` preserves zero-depth absence and checks only selected bodies
after the required global metadata. Its success has complete=false. `verify`
returns complete=true only after the ordinary unlinked pipeline succeeds.
Transport/conversion refusals, exact ordinary messages/depths, known malformed
body refusals and deferred affine/ownership/private profiles remain distinct.
Every refusal discards partial depths. Inert ownership metadata with no functions
and no required ownership walk follows the C module's ordinary path.

My actual C oracle calls public V2 deserialization/conversion,
nvm_verify_declared_max_stacks, nvm_verify and nvm_verify_function_max_stack.
I normalize only the shared decoder's exact public refusal as D and the existing
affine refusal family as A, retaining their original public diagnostics in the
recorded command stderr. I do not replace those APIs with a copied verifier.

My complete Linux gate exits0 in1444.113s (method1443.739s):7132 modules in each
of two modes through five routes,126/122 independent expectations and six fresh
actual source products per mode. I use direct C-seed native, C-produced VM and
ASan/UBSan native, and the actual qualified NanoLang Stage1's VM and ASan/UBSan
native products. Every complete report matches its declared ordinary C outcome
or explicit private boundary, and all five pure routes agree exactly.

| My mode | Pass | Exact ordinary refusal | Transport/conversion refusal | Known malformed body | Explicit deferral |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full ordinary |1646|4254|368|674|190|
| Declared only |6515|244|368|2|3|

I materialize6472 existing ordinary body contexts as actual V2 modules, with
valid noncurrent bodies so an artificial empty helper cannot mask the target.
I add core module, independent metadata/declaration-priority, multi-function and
producer controls. I keep463 legacy contexts separate:6 nonmodule contexts,
229 nonexpressible signatures,2 nonexpressible imports,32 legacy count/pool
contexts not materialized by this corpus and194 linked-graph contexts. These
are not silently counted as whole-module tests or implementation bounds; the
previous direct helpers retain their own coverage. Full linked graphs remain
required work.

The actual VM --verify-only command agrees on131 independent ordinary consumer
controls. Targeted ASan/UBSan instrumentation of the actual public verifier,
types, decoder, ISA, V2 module/reverse converter and test oracle matches all7132
reports in both modes plus the six source products. I use no sanitizer environment
overrides; other linked providers retain ordinary instrumentation. My original
identifier gate passes3309 fragments with zero reserved names, retaining135
lexer-refused fragments with their executable tests.

I preserve the first native preflight's four over-deferrals of inert ownership
metadata, its source/binary and every report. I correct precisely the no-function,
no-required-walk case; nonempty ownership execution remains deferred. The
corrected preflight and full five-route gate preserve all other diagnostics.

My source producer is the original frozen raw-equal bdffa4ccc native Stage1:
SHA256 `03bbebe611897c01d01db780a459ddff537fbc477bcfeb7a289f40caa7c811ec`,
receipt SHA256 `c76d723871bd3f935b1ea40afb677ffd0e0f159f5db043416b3673c769900ebc`.
I select the new development source root explicitly and preserve all original
source/tool/host/artifact/context guards. All3564 guarded current inputs and
current test tools remain unchanged. This is not a new raw compiler bootstrap
or installed-prefix-isolation result.

My lossless archive retains31728 logical records,290649434 raw bytes and11306
unique gzip artifacts totalling60064516 bytes. A separate reader verifies all
raw/compressed hashes and all3564 exact source Git blobs. Build caches are
omitted; actual inputs, products, command stdout/stderr, tools and original
provider receipts are retained. The publication audit records that independent
check separately from the main archive counts.

**Paired Darwin qualification remains pending.** Puck owns the serial Darwin
capture/source reservation and #1157/#1158/#1159/#1171; I start no competing
Darwin gate. No executable selector uses this component. Complete linked,
affine/owned/private-profile verification, the pure C emitter, final-source paired
bootstrap and the full #976 release scope remain open.

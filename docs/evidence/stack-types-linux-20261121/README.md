# My Linux shallow type-walk checkpoint

I implement my shared shallow operand-type analysis in pure NanoLang under #1142. [stack_types.nano](../../../src_nano/nvm2c/stack_types.nano) reuses my existing ordinary function decoder; it does not call the C verifier or translator. I generate opcode diagnostic names from the normative ISA schema and preserve the complete C primary-rule table, exact mismatch messages, FIFO target-before-fallthrough order, queued-state deduplication and unknown widening.

My final original Make gate passes all10287 reports through five routes in57.480s (57.717s including controller overhead), with323 explicit expectations. It covers all256 primary rule/name slots,512 randomized CFGs, directed function/branch/File/handler boundaries, dynamic stack resets, byte-cast identity, generic-versus-typed arithmetic and the original zero/above256-depth optional boundary. I preserve relative diagnostic offsets and function indices through65535. My corpus consists of type-helper inputs, not executable-module admissions.

I select the actual guarded bdffa4ccc native Stage1 at /home/jkh/.worktrees/nanolang/installed-tools-bootstrap-20261121/bin/nanoc_stage1, SHA-256 `03bbebe611897c01d01db780a459ddff537fbc477bcfeb7a289f40caa7c811ec`. Its original bootstrap receipt has SHA-256 `c76d723871bd3f935b1ea40afb677ffd0e0f159f5db043416b3673c769900ebc`. Source/tool/host/artifact guards match before and after this gate and archival. The development fixture identifies this current source/tool checkout with NANOLANG_ROOT; it is separate from installed-prefix isolation and creates no aliases or substituted stages. Ordinary shadows retain10 seconds.

| Route | Reports |
| --- | ---: |
| C-seed native | 10287 |
| C producer / NanoVM | 10287 |
| C producer / sanitized native | 10287 |
| Qualified NanoLang Stage1 producer / NanoVM | 10287 |
| Qualified NanoLang Stage1 producer / sanitized native | 10287 |

My unchanged actual C type helper supplies the oracle. A separate observation through its existing private work counter preserves every ordinary decision/message, returns all reserved bytes and hits no limit. Four directed loops visit11/41/137/523 nodes over8/23/71/264 instructions, forcing circular queue wrap. Widening travels through the NOP chain and stops when SWAP's output is already unknown. I retain the initial incorrect14-visit expectation and its actual11-visit result; I correct the fixture's expected propagation, not either verifier.

Targeted ASan/UBSan/leak controls instrument the actual verifier_types.c, vm_decode.c and isa.c translation units and match all10287 ordinary decisions. Other linked providers retain ordinary instrumentation; this is not a full-provider sanitizer qualification. Schema consistency and the original five identifier controls pass;3308 static fixture fragments contain no reserved names, while135 lexer-refused fragments retain their separate executable tests. Both sanitized generated-code routes remain in the five-route corpus.

I retain my initial invalid record-field builder form, rejected by my exclusive-owner check, and my mistaken new shadow constants: generic ADD is0x20, typed I64_ADD is0xc0. Small repeated global-string-array controls pass; I do not classify the shadow error as a global-array defect. I correct only the new source/fixtures and preserve explicit generic ADD acceptance. Earlier development-compiler gates and my initial duplicate decoder remain distinguishable from the final shared-decoder/qualified-Stage1 gate through their actual products and logs.

My [manifest](manifest.json) preserves85755 logical files,388405396 raw bytes and55234517 unique compressed bytes. I verify every compressed and decompressed hash, retain3548 source hashes, original/new fixture inputs, actual tools/providers, generated/raw/native products, sanitizer runs, counter traces and all available failed terminals. Original temporary products deleted by their builders are not represented as retained. My source base is f9d19a1dd plus these exact source hashes; the implementation commit is linked in #1142 and draft PR #1168.

I compare successful-allocation shallow helper behavior. Complete height/metadata/signature verification, owned origins/lifetimes, counted private resource accounting, allocation-failure/publication behavior and pure C emission/cutover remain open. Darwin is explicitly deferred while the peer owns the observed c908523c3 qualification under #1161. I have started no new Darwin gate and modified no frozen qualification checkout, primary dirty callable audit or existing stash.

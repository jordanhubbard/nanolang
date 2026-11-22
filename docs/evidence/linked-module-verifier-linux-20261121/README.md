# My ordinary linked-module and flat-graph verifier

I qualify source `76cd20f2a4c68f858389fcbd6995a6f6d3694819` on Linux ARM64 under #1142. I add `verify_linked` for the ordinary root query and `verify_graph` for every present target against the same flat table. I load root/target V2 bytes and positive declared-depth obligations in source order, then preserve ordinary C graph policy, metadata/body order, call shapes, diagnostics and depths. No executable selector uses these APIs.

A successful root query does not certify an unused target body. My graph wrapper checks those bodies in slot order. Graph index0 names the root, index+1 names a failed target, and minus1 reports success with the root depths. Every refusal discards partial depths. I ignore unresolved-slot payloads; unused unresolved slots remain permitted. I do not discover dependencies or grant host authority. Owned/affine/private admission and the pure C emitter remain required.

My actual public C oracle calls V2 deserialization/conversion and declared-stack verification for each present source, then `nvm_verify_linked`. The complete-graph oracle calls that same public API on every present target. Successful root depths come from the public unlinked maximum query after linked shapes have passed. I normalize only the existing exact shared-decoder refusal as D and affine deferral as A, retaining actual C diagnostics separately.

My complete linked gate passes in732.272s (method731.915s):304 carriers in both modes through five routes,78/75 independent expectations and six fresh original producer targets per mode. I retain C-seed native, C-produced VM and sanitized native, and original qualified NanoLang Stage1-produced VM and sanitized native products.

| My pure mode | Valid | Exact ordinary refusal | Transport refusal | Malformed body | Private deferral | Harness refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linked root |43|248|2|2|4|5|
| Complete present graph |12|251|2|30|4|5|

I compare295 ordinary V2 decisions exactly with C. Five carriers exercise only the test envelope, including an invalid legacy resolved flag; its64-slot limit is a harness bound. Four explicit private controls retain C14/C13 outcomes separately from my A deferral. Of194 earlier linked contexts, I retain192 here (191 V2 contexts and one invalid carrier); two target signatures exceed the V2 arity/result representation and remain counted apart. I preserve every ordinary signature, loader, service/ownership policy, unresolved-target and malformed/unused-target control.

After the shared preparation/context change, my original7132-module gate passes both full/declared modes through all five routes in1352.980s (method1352.595s), including its original126/122 independent expectations and six fresh original source products per mode. All original outcome counts remain unchanged: full1646 valid/4254 ordinary/368 transport/674 malformed/190 deferred; declared6515/244/368/2/3. Declaration-only success retains complete=false.

Targeted ASan/UBSan instrumentation of the actual C verifier/types/decoder/ISA/V2 module/reverse-converter and both oracles passes every linked/graph and ordinary full/declared report, plus all fresh original products. I use no sanitizer environment overrides; other linked providers retain ordinary instrumentation. The original identifier gate passes3309 static fragments, zero reserved names and135 separately retained lexer refusals. All3567 guarded source/test/specification/Make inputs, tools and frozen producer guards match.

I retain the first linked preflight's355.261s terminal failure, source and all three-route reports. C and every pure route agree, but one handwritten expected underflow string has different wording. I correct only that expected string, then expand the corpus and qualify all five routes. Initial compile/shadow setup refusals and expanded native preflights remain in the archive. I change no deadline or assertion to admit a product failure.

My development source producer remains the frozen raw-equal bdffa4ccc native Stage1, SHA256 `03bbebe611897c01d01db780a459ddff537fbc477bcfeb7a289f40caa7c811ec`, receipt `c76d723871bd3f935b1ea40afb677ffd0e0f159f5db043416b3673c769900ebc`. Ordinary shadows remain10s, native compilation600s and each guest comparison batch600s. I preserve1800/30 bootstrap limits and start no new raw bootstrap.

My lossless archive retains18526 logical records,371027175 raw bytes and11838 unique gzip payloads totalling73440423 bytes. An independent reader checks every raw/compressed hash and all3567 exact source Git blobs. The publication audit retains its completed report separately from these counts. Build caches are omitted. The retained capture review separately checks seven exact0419 controls and787 peer x86_64 archive records; it grants no product admission and does not execute x86_64 products on this ARM64 host.

My [same-source Darwin checkpoint](../linked-module-verifier-darwin-20261122/README.md) now passes both complete gates, targeted C sanitizers and identifiers. Every corpus byte and all48 normalized report files match Linux;3567 source inputs and the three original example inputs match exact Git blobs on both hosts. The old failed qualification roots, original compiler providers, primary dirty audit and stashes remain outside this work. Full #1142/#976 acceptance, private/affine verification, pure C emission and final-source paired bootstrap remain open.

I separately retain the three original producer example inputs in `publication-audit/original-example-inputs`. Their bytes match both the qualification-base and implementation Git blobs; they are separate from my3567 core/test/Make guard count.

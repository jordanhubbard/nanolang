# My Linux ordinary-height checkpoint

I implement the ordinary shared stack-height and explicit-retain helper in [pure NanoLang](../../../src_nano/nvm2c/stack_heights.nano) under #1142. I reuse my ordinary function decoder and compare the actual unchanged `verify_stack_heights` from `src/nanoisa/verifier.c`; my implementation calls no C verifier or translator.

My final original Make gate passes3154 exact reports per route with546 independent expected reports and1024 randomized CFGs in68.869s (69.194s including controller overhead). I cover all256 primary byte values, dynamic direct/indirect/module/foreign calls and aggregate/capture counts, PERFORM/PICK/ROLL, nonzero spans, invalid/dead instructions, explicit/implicit returns, branch-first joins, ownership-versus-height error priority, handler/resume boundaries and unsigned16-bit signature/depth/frame limits. Function/import indices through65535 and full65536-entry tables retain their own controls. These are helper carriers, not structurally accepted executable modules.

| Route | Exact reports |
| --- | ---: |
| C-seed native | 3154 |
| C producer / NanoVM | 3154 |
| C producer / ASan/UBSan native | 3154 |
| Qualified NanoLang Stage1 producer / NanoVM | 3154 |
| Qualified NanoLang Stage1 producer / ASan/UBSan native | 3154 |

I select the actual bdffa4ccc qualified native Stage1 at `/home/jkh/.worktrees/nanolang/installed-tools-bootstrap-20261121/bin/nanoc_stage1`, SHA-256 `03bbebe611897c01d01db780a459ddff537fbc477bcfeb7a289f40caa7c811ec`. Its original bootstrap receipt has SHA-256 `c76d723871bd3f935b1ea40afb677ffd0e0f159f5db043416b3673c769900ebc`. The original source/tool/host/artifact guards match before/after qualification and archival. My development fixture identifies the current source/tool root with NANOLANG_ROOT and retains ordinary10-second shadows. This is separate from installed-prefix isolation and makes no new fixed-point claim.

My oracle first checks representable signature fields, byte spans, current-function/local bounds and direct/foreign indices before calling the shared helper. It normalizes these precondition refusals to `D`; completed failures retain the exact shared message and success includes maximum depth. Every actual C case also runs with a null maximum pointer; decisions/messages match, and failed calls preserve the original maximum sentinel. My NanoLang failure result leaves its documented maximum at0. The bounded carrier limits function/import table counts to65536; my pure API retains the unsigned32-bit count boundary.

I preserve the helper's scope and edge behavior. An empty body passes independently of declared results; structural verification owns its refusal. Dead instructions still decode but unreachable effects do not participate. GC_RELEASE first requires an operand, then a held reference. Handler targets reset height/balance at real instruction entries; terminal-boundary checks retain outgoing height. FILE_RESULT_BRANCH retains ordinary fallthrough only in this helper. HALT and TAIL_CALL keep their existing termination rules. Full module, ownership/origin/lifetime, type/effect, resource and profile verification remains separate.

Targeted ASan/UBSan/leak checks instrument the actual included verifier.c, vm_decode.c and isa.c and match all3154 reports, with no sanitizer environment overrides. Other linked providers retain ordinary instrumentation; this is not full-provider sanitizer qualification. All97 original C verifier tests pass. The five original identifier controls pass;3308 static fragments have no reserved names and135 lexer-refused fragments retain their own executable tests.

I retain my initial attempt to compile a library directly, correctly refused for lacking main. My first driver shadow then wrongly expected an ownership diagnostic from GC_RELEASE without an operand; the actual refusal follows the required underflow priority. I retain that exact source/log and correct only the intended shadow fixture by pushing a value. My earlier three-route development gate passes3122 reports/512 expectations in34.511s; the final qualified-source gate adds module-call boundaries and explicit join messages. The original first-stage shadow native product and all available final generated/raw/native products remain archived; products deleted by original builders are not represented as retained.

My [manifest](manifest.json) retains9896 logical files,233570106 raw bytes and48943339 unique compressed bytes, with3552 source hashes. I independently verify every compressed and decompressed SHA-256. I retain both corpus sets, complete route reports, rejected source/terminals, current sources/tests, actual tools, qualified Stage1/receipt/host closure, observer code and sanitizer artifacts. My source base is d79bffeb6 plus these exact inputs; #1142 and draft PR #1168 link the implementation commit.

I compare successful-allocation, representable shared helper behavior. Full verifier/profile integration, counted private resource accounting, allocation-failure/publication policy and pure C emission remain open. I start no Darwin gate while the peer owns the active observed c908523c3 qualification under #1161, and I preserve all frozen qualification checkouts, the primary dirty callable audit and stashes. This Linux component is not full5.1 acceptance.

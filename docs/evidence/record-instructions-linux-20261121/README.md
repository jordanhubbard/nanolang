# My copied ARRAY/MAP instruction recipes

Under #1142 I implement record_instructions.nano over my pure function decoder.
I copy the actual C planner's94 ordinary recipes and six MAP extensions, exact
runtime obligation bits, dynamic literal/call/return stack counts, global-slot
increment and relative successors. CALL keeps its continuation and separate
callee; TAIL_CALL and RET terminate their local CFG. I preserve integer/F64
operand halves and inspect unreachable instructions. Partial results are cleared
on refusal. These descriptive facts grant no profile or execution authority.

My thin C fixture includes the actual managed_array_shapes.c translation unit
and calls its existing private instruction copier, with analysis deliberately
separate. I do not clone that copier into an oracle. This is not full origin,
type/lifetime, stack-flow or budget analysis. Inferred MAP key/value/result tags
also remain for that analysis. The existing unsigned global-index increment,
including UINT32 wrap, remains visible; full analysis separately rejects invalid
globals before execution. No result here enables a product route or cutover.

All2014 reports match through five Linux ARM64 routes in94.456 seconds: direct
C-seed native execution, both producers in NanoVM, and both producers through
sanitized generated C AOT. I retain994 explicit validity expectations, all256
primary opcode values per profile, dynamic counts through256/257, maximum
signatures/globals, conditional/terminal/interior branches, invalid callees,
unreachable refusals, signed extrema/F64 NaN bits and randomized streams.
Reports include relative descriptor fields and operand halves for functions
up to1024 instructions. The65536/65537 instruction-boundary controls return
validity/count summaries after complete copying. Both source producers execute
mandatory useful shadows under the existing component deadline.

The actual C copier/fixture translation unit also matches all2014 reports with
ASan/UBSan and leak detection. Other linked providers in that diagnostic retain
their original ordinary objects; I do not call it all-provider instrumentation.
My unchanged identifier gate passes five controls and3308 static fragments,
retaining135 lexer-refused fragments with their executable tests.

I retain the actual seven derived fixture products, all inputs/reports, source
and tool identities, complete commands/terminals and the actual C oracle images.
The genuine retained native Stage1 artifact is the NanoLang source provider.
Its original bootstrap receipt later has a preserved cached-link refusal; that
receipt is not current resumable authority. The fresh c908523c3 bootstrap is a
separate qualification run. My manifest covers2440 logical inputs,161945337 raw
bytes and37931964 unique compressed bytes, with checked raw/compressed hashes.

Darwin, complete checked ARRAY/MAP profiles, ordinary verification, the pure
emitter, installed lifecycle and release acceptance remain open. I keep the
paired roadmap row unchecked.

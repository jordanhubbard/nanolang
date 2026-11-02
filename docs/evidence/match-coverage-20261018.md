# My bounded match-coverage checkpoint

I continue #998 on `fix/5.1-match-facts` from aggregate `5f2f7c0b3`.
My existing C and self-hosted coverage checkers discard their successful
results. I retain that result in UNKNOWN-initialized AST fields and emit
`nano.match.v1` at the actual match lowering site. I generate the self-hosted
AST shape from my schema, preserving the schema as its source of truth.

I retain checked total integer or named-union coverage separately from the
scrutinee domain. I preserve the existing diagnostics for missing coverage,
conditional-only wildcards and arms after an unconditional wildcard. Ordinary
imported matches retain source ownership; self-hosted positions explicitly
identify flattened-input coordinates. Active generic instances stay UNKNOWN
until per-instance retained facts are qualified. Affine-use and the complete
frontend-fact requirement remain open.

My first C-hosted build refused field mutation without an exclusive borrowed
owner. I preserve that failure in
`/private/tmp/nano-998-match-native/build.log`. I replace the complete stored
AST record through its existing list setter instead of weakening borrowing.
My next build refused the helper's private cross-module call
(`build-corrected.log`); I explicitly export the parser update helper and
retain its reset/preservation shadow. The corrected intermediate compiler
build passes in `build-exported.log`.

My first C raw-codegen test assumed that parsing followed by code generation
never checks types. Return lowering actually calls the real expression
checker before emitting the match. I retain the failure in
`/private/tmp/nano-998-match-c-gates.log` and now assert parser UNKNOWN followed
by the checker-backed positive fact. My unchecked self-hosted case-emitter
shadow remains the actual UNKNOWN control.

My final C codegen gate passes 91 cases. My advisory codec gate passes 825
checks and 42 allocation boundaries, including malformed lengths, indexes,
reserved bytes, paths, duplicate sites, unknown versions and atomic metadata
append failures with prior entries preserved. I retain the terminal at
`/private/tmp/nano-998-match-c-final.log`.

My three seed source methods pass in 1.193 seconds at
`/private/tmp/nano-998-match-seed-final.log`. I exercise ordinary and imported
integer/union matches, actual two-type generic instantiation with UNKNOWN
facts, four refusal controls preserving prior output, byte-for-byte canonical
transport, VM execution and ASan/UBSan native execution. Stripped, forged and
duplicate records leave emitted native C unchanged.

My final changed C-hosted producer builds successfully with dependency/root
shadows retained and a bounded 300-second shadow deadline. I retain the build
terminal at `/private/tmp/nano-998-match-native/build-final.log` and the product
at `/private/tmp/nano-998-match-native/nanoc-selfhost`. All three source methods
pass through the C seed and this changed producer in 2.142 seconds
(`/private/tmp/nano-998-match-paired.log`). All twelve neighboring declared and
inferred purity/effect methods pass through both producers in 7.299 seconds
(`/private/tmp/nano-998-match-neighbors.log`).

My serial strict gate passes 222 Clang and 222 GCC compilations plus the relink
fixture (`/private/tmp/nano-998-match-fast.log`). Regenerating all schema
artifacts leaves their contents unchanged.

Both freshly bootstrapped NanoISA generations and exact Linux/Darwin aggregate
qualification remain separate pending gates. I do not call the C-hosted
compiler a fresh NanoISA stage. This isolated work does not launch a second
full bootstrap or close #998.

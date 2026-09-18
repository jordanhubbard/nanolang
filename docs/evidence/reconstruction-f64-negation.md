# My typed binary64 negation reconstruction evidence

I tested production `131c23cf` with frozen harness `be39b777` on main base
`a0a8fbcf`, after contract `29ea05f3`. Neither production nor harness changed
while any gate ran. I admit only exact FLOAT F64_NEG and render unary minus.

Four focused methods passed with GCC in 62.632 seconds and Clang in 62.555
seconds. Nineteen patterns include signed zeros, subnormal/normal and finite
boundaries, infinities, signed quiet/signaling NaNs and varied payloads. For
every input I observed negated bits, unchanged original bits and double-negated
bits through exact signed integer observers. I used no float-equality or text
oracle. All three canonical producers emitted F64_NEG as checked in their dumps.

Helper calls, snapshots before later stores, discarded results, both branch
arms and a bounded pure loop passed. Test-only instrumentation counted exactly
two helper evaluations across standalone C and all Nano producer routes. I do
not admit global opcodes by instrumenting these observers. Exact other-tag and
underflow analyzer controls passed. Remaining float operations preserved prior
outputs on refusal. Five adjacent methods passed in 0.467 seconds, covering
exact binary64 facts, updated unsupported F64_ADD output-preservation controls,
94 old operator refusals, and entry/mixed-local/nonempty-join boundaries.

Original modules ran through the current VM and native translator. Standalone
reconstructed C and generated native C used GCC/Clang O2 strict warnings and
ASan/UBSan with nonrecovering errors. Reconstructed Nano used qualified PR720
C-seed, Stage1 and Stage2 executable compilation, plus NanoVirt/Stage1/Stage2
canonical emission followed by VM/native execution. I do not claim identical
sanitizer/compiler flags for all legacy producer-selected subcommands.

Qualified producer source remains `f87267c21a356424ce101060521ce841a5eb17bb`
in `/home/jkh/Src/nanolang-binary64-bit-transport`. Its four compiler binaries
and recorded library hashes matched before and after gates. Current local
assembler/facts/VM/native tool identities, frozen test hashes and log hashes
are retained in [my manifest](reconstruction-f64-negation.json). This is not a
fresh bootstrap or hermetic relocation claim.

I retained logs under `/tmp/nanolang-reconstruction-f64-negation-` with suffixes
`build.log`, `gcc.log`, `clang.log` and `adjacent.log`. Fresh per-command inputs,
modules, generated sources and logs remain under `/tmp/nano-reconstruct-f64-*`.
No gate failed in this child. I did not replay historical PR679 artifacts.

This qualifies the named tools and ordinary default floating environment; it
does not establish every host or floating trap mode. Binary arithmetic, casts,
float truthiness, generic FLOAT operations, heap reconstruction and full
reconstruction remain open. Task `task_945d1fdcaaee499daddf1e0187e70a2b`
awaits canonical integration before ledger closure.

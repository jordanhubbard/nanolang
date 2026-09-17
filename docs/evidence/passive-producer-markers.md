# My passive producer markers

I resolve `.par_begin`, `.par_node result-local [external-parameter ...]`, and
`.par_end` using my assembler's authoritative function and instruction offsets.
I write the existing version-2 passive record. I do not duplicate instruction
sizing in my self-hosted frontend or invent eligibility facts in the assembler.
My ordinary passive verifier still checks every generated claim.

Markers are scoped to one function, cannot nest, need at least one node, and
cannot mix with raw `.passive` chunks. Canonical disassembly still emits exact
hexadecimal records. My dynamic record buffer checks size growth and retains
its owned allocation if growth fails.

Nine passive methods pass. New cases cover exact equality with the previously
hand-encoded four-scalar guarded record, multiple blocks in two executed
functions, canonical byte roundtrips, and eleven incomplete/mixed marker forms.
Both positive products run in NanoVM and strict native C output. The retained
249 metadata checks, 96 verifier cases and 210 canonical roundtrip checks pass.
The assembler allocation harness also passes ASan/UBSan with initial allocation
failure, failed buffer growth, retained content, recovery and cleanup checks.

This completes `task_6798838c50b74d698505602f06b70194`. Frontend retention,
dependency/purity rejection and emission remain
`task_e0c6fd18cfbb49c78e4c509d8444417e`; no frontend acceptance is claimed here.

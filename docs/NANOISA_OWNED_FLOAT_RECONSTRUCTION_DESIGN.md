# I retain binary64 locals alongside exact owned resources

I record a design-only slice of #999 at fbd503d59. Rows9538/10409 remain the parent scope; I do not close them or claim implementation. I wait for the STRING component's integrated facts interface before touching shared production files.

## I preserve the unchanged acceptance

I retain `tests/test_owned_record_patterns.py::test_unsafe_pattern_keeps_outer_shadow` with its complete PREFIX: Handle has only integer fd; close consumes Handle and returns int, with its original shadow. Main binds outer float fd=2.5; unsafe destructures Handle{fd:42} into an inner integer fd and asserts42; afterward outer fd remains2.5, then main returns0. I retain main's original shadow. `test_source_borrow_emission.py::test_float_local_unsafe_original_pattern` already sends this exact source through the paired producer gate.

I first retain ordinary modules and selected-shadow modules separately from each canonical producer. I run original verification and VM execution before reconstruction; then independently verify generated Nano/C behavior and generated consistency shadows. I do not claim to recover the original shadow source from bytecode. Numeric local slots, not reused descriptive fd spelling, preserve lexical identity. Unsafe adds no foreign, pointer or ownership authority.

## I keep admission local-only

I propose exact nonparameter FLOAT local declarations with NO_INDEX and existing mode-zero rules in the consuming resource profile. Existing INT/BOOL/resource signatures, resource leaves/results and ownership identities remain unchanged. FLOAT parameters, results, resource fields, mixed implicit conversions and foreign operations remain refusals. I preserve verifier-first admission, definite initialization, exact owner cleanup, finite function/instruction/local limits and atomic output refusal.

The facts local whitelist currently rejects FLOAT. Its instruction serializer also lacks raw F64 operands: widening the tag guard alone would lose literal bits. I expose PUSH_F64 using the established scalar facts `f64_bits` representation: exactly sixteen hexadecimal digits obtained from the decoded operand's raw binary64 bytes, without decimal conversion. I introduce no opcode or wire format. Malformed/missing/wrong-width facts fail before output. Exact parameter-versus-local checks remain independent.

I reuse existing scalar Analyze F64 expressions and Emit helpers only after auditing owned overrides: declaration/type maps, scalar stack suffix isolation, opcode guards, temporary materialization, C helper inclusion and local initialization must explicitly accommodate FLOAT locals. A FLOAT value must never be mistaken for a staged owner. Existing resource field predicates stay narrow.

## I preserve binary64 semantics

I reuse authoritative binary64 arithmetic helpers and memcpy bit transport, including signed zero, infinities and NaN payload transport. I do not promise arithmetic preserves a NaN payload beyond existing runtime policy. Typed F64 predicates retain unordered behavior: NE is true for NaN, ordered comparisons and EQ are false. Generic three-way comparisons are not substitutes. I retain once-only operand evaluation and existing arithmetic failure/rounding policy; no fast-math or decimal literal approximation.

Initial scope is the unchanged straight-line unsafe fixture. I then qualify copies, overwrites, conditional joins and current supported pretests as individually admitted. The larger existing consuming worker fixture includes break/continue; the newly implemented exit grammar is single-exclusive borrowed-only. I do not label that consuming composition supported or silently widen it here. Borrowed FLOAT-local control is also a distinct existing source case needing later explicit disposition.

## I require finite acceptance before integration

- I retain initial facts refusal and exact unchanged source/module/provider hashes.
- I prove the ordinary and selected-shadow anchor through source, canonical assembly/binary transport, original verifier/VM, reconstructed Nano and strict native C execution, and generated consistency shadows.
- I exercise signed-zero and representative finite/subnormal/infinite/NaN bit transport, typed NaN predicates, copied/overwritten locals, lexical slot shadowing and zero-path definite initialization.
- I preserve refusal for unsupported FLOAT fields/signatures, uninitialized/mistagged locals, malformed bit facts and unsupported control, with prior output intact.
- I compare unchanged old INT/BOOL/resource output and retain existing local limits; I measure generated locals rather than assuming room.
- I qualify both current seed and fresh stages plus exact Linux/Darwin candidate gates separately. No design or scalar-only success substitutes for owned qualification.

I retain `NANOISA_OWNED_FLOAT_LOCALS_UNSAFE.md` as producer/runtime evidence: it records completed bounded local qualification and explicitly excludes fields/signatures. This proposal concerns reconstruction, not new executable source admission.

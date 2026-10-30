# My computed contextual byte parity checkpoint

I track this work under #1004. My seed gained deliberate modulo-256 computed
integer conversion in #963 (`b6b332d57`). The earlier reconstruction fixture's
blanket computed-integer refusal was stale for that producer; the self-hosted
producer still lacked the conversion.

I retain exact literal range checks: a byte literal must be 0 through 255.
For a checked computed integer, my shared contextual-type hint admits the
conversion, and I emit the operand once and then `CAST_U8`.
Existing byte values retain their tags. Definite noninteger values remain
refused. I route computed byte returns before tail-call selection so direct,
generic, and indirect calls cannot bypass a pending result-tag conversion.

My focused fixture actually calls bindings, assignments, byte parameters, and
direct/indirect/generic returns, plus byte-array literals, push, and set. It
checks a mutable call counter after each
operand, negative and positive boundaries including signed integer extrema,
exact U8 function signatures, and the conversion opcode. It runs the verified
module and emitted native C under ASan/UBSan. Separate negative controls retain
prior output for out-of-range literals and definite wrong types.

My first focused self-hosted run failed before emission: the shared
`apply_value_type_hint` accepted only literal integers. I preserve that failure
at `/private/tmp/nano-1004-paired-focused.log`. The corrected helper accepts
checked nonliteral INT only; its shadows preserve out-of-range literal and
BOOL/FLOAT refusals. An initial byte-array fixture also needed whitespace
between its generic closing `>` and `=` to avoid the existing `>=` token.

My seed and a newly C-hosted compiler built directly from the changed
`src_nano/nanoc_v06.nano` pass both focused test methods in 1.207 seconds,
including all five literal/type refusal controls and ASan/UBSan native runs.
My build retains dependency/root shadow checks with a bounded 300-second
shadow deadline. I retain the focused terminal at
`/private/tmp/nano-1004-paired-checked.log` and build terminal at
`/private/tmp/nano-1004-native/build-checker.log`.

My strict gate also passes 222 Clang and 222 GCC compiles plus the relink fixture.

Both fresh installed NanoISA bootstrap stages still require aggregate
qualification. I do not treat the C-hosted compiler as that evidence, and
#1004 remains open until the combined candidate passes those gates.

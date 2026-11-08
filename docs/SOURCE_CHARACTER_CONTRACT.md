# My character builtin source contract

I retain the original complete `tests/unit/test_stdlib_comprehensive.nano` fixture and its shadows. My existing character operations classify ASCII codes, extract decimal digit values and convert ASCII letter case. Classification returns BOOL; digit extraction and case conversion return INT. My host wire contract takes one INT operand.

I preserve the VM's existing integer conversion and ASCII rules in native adapters, including the distinct four-character `is_whitespace` policy. I do not replace these contracts with locale-dependent classification. I resolve bound function values before builtin dispatch and preserve the existing reserved-definition policy.

I require exact arity/type refusals before publication, original whole-fixture acceptance, byte-range and boundary controls, once-only evaluation and VM/native agreement. A component result leaves fresh Stage 1/2 and integrated platform qualification open.

I expose six predicates (`is_digit`, `is_alpha`, `is_alnum`, `is_whitespace`, `is_upper`, `is_lower`) and three INT results (`digit_value`, `char_to_lower`, `char_to_upper`) in this correction. The first, second, third, fifth and sixth predicates retain the VM's existing C `int` conversion. `is_whitespace` compares the complete input against space, tab, newline and carriage return. Digit extraction and case conversion also use the complete INT input; case conversion returns it unchanged outside ASCII letter ranges. My native case adapters must not narrow that value.

I test every byte value against independent ASCII membership and byte-case tables, then check negative, non-ASCII and full-width integer boundaries under native `-O0` and `-O2`. Explicit foreign declarations keep their identity; I do not rename them into character builtins or admit an arbitrary library through the bounded native host table.

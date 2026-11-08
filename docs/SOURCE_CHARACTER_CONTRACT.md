# My character builtin source contract

I retain the original complete `tests/unit/test_stdlib_comprehensive.nano` fixture and its shadows. My existing character operations classify ASCII codes, extract decimal digit values and convert ASCII letter case. Classification returns BOOL; digit extraction and case conversion return INT. My host wire contract takes one INT operand.

I preserve the VM's existing integer conversion and ASCII rules in native adapters, including the distinct four-character `is_whitespace` policy. I do not replace these contracts with locale-dependent classification. I resolve bound function values before builtin dispatch and preserve the existing reserved-definition policy.

I require exact arity/type refusals before publication, original whole-fixture acceptance, byte-range and boundary controls, once-only evaluation and VM/native agreement. A component result leaves fresh Stage 1/2 and integrated platform qualification open.

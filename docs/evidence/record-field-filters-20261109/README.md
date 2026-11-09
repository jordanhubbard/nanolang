# My measured record-field filter candidate

I follow the actual f184 Apple CPU sample: ordinary field access and byte-field
contextualization account for 35/32 string-comparison snapshots. I preserve exact
first-match names, field/declaration order and all value/ownership rules.

I first move the unchanged lookup/context bodies into a private include shared
by the actual evaluator and my native unit. The unit compiles the actual helpers
and current environment/GC source with ASan/UBSan; it does not mock STRING or GC
ownership. I test reordered packed metadata, duplicate/empty names, shared and
non-ASCII prefixes, seven extreme byte inputs, non-integer refusals and null/
empty definitions. I also require a real owned STRING copy to remain valid after
its source is released, plus unchanged non-STRING value identity and flags.
My wide-record observation performs 2,000 actual lookups with 128,000 strcmp
calls before filtering. This is a call-count control, not a timing comparison.

My first unit build refuses the deliberately concatenated UTF-8 C test literal
under -Werror. I retain that initial refusal, parenthesize the concatenation
explicitly and pass the unchanged sanitizer target. I retain commands, source
before/after hashes, actual post-run Mach-O image observation and the baseline
image locally under .test_output/record-field-filter-units/linear/.

I hold full local provider builds for disk pressure. These bounded native helper
controls do not qualify the complete evaluator, C provider or source compiler.
Whole-source fixed paired measurements and all original component/platform,
fresh/corpus, budget, NativeSDK/lifecycle, security and release gates remain
required. Frame-carrier lifetime remains separate under #1116.

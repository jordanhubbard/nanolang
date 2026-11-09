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


I now reject a field only when its initial byte differs, before the same full
strcmp equality check. I retain first-match order even for duplicate/empty and
non-ASCII C metadata; no index cache or mutable ownership fact is introduced.
The same actual native sanitizer controls pass after filtering. The wide control
performs 2,000 comparisons for the same 2,000 lookups, while all expected refusal
text is byte-identical. I retain both actual native unit images locally and hash
the post-run bytes. This reduced comparison count is not a measured source-
creation speedup. Syntax-only compilation of the full actual evaluator with the
same compiler/header flags also passes; no local full-provider build is claimed.

I merge the completed sampler PR #1125 into its parent OR branch at 48016fec4,
preserve real ancestry and synchronize that merge into this candidate. The
whole-source comparison baseline is that pre-filter commit 48016fec4. I require
unchanged NanoLang/factory inputs and the original six-attempt order and ten-
second shadow supervisor, then all original component/platform gates, before
retaining this production candidate. The unchecked roadmap item remains open.

# I map a bounded borrowed FLOAT caller-local extension

## I retain required source acceptance

Required scope9538/10409 and issue999 preserve verified ownership and structured source reconstruction. The existing producer test at test_source_borrow_emission.py:1809 accepts the exact source substitution retained in the [baseline](evidence/borrowed-float-local-baseline-20261019/README.md). Three ordinary producers and an independent original-shadow input verify/execute with two functions24/3locals. Caller-only literal2.0 and equality assertion currently trigger facts refusal. This is distinct from consuming FLOAT break/continue, helper FLOAT, and resource FLOAT fields/signatures.

## I propose exact guard boundaries before code

A prospective `borrowed_float_caller` discriminator requires the existing ownership-v2/two-function single-exclusive helper profile: function0 has no parameters and scalar INT result, function1 has exactly one EXCLUSIVE resource formal, existing path/layout/depth guards hold. Only function0 nonparameter local declarations may contain FLOAT with NO_INDEX. Helper local/formal/result declarations, resource fields and caller/helper signatures keep their present checks. No new ownership kind, format, native runtime or verifier permission is involved.

In hl_owned_facts.inc, the FLOAT-local alternative is extended only for that exact function0 discriminator, alongside the existing consuming-only alternative. Existing final parameter-mode/layout authority validation still applies. Raw16hex F64 extraction is reused unchanged. In OwnedAnalyze the same exact discriminator augments caller scalar_tags and FLOAT-local checks; existing declared() signature/field checks remain. Both simple-instruction handling sites must agree so an incremental route cannot accidentally widen the contract.

The accepted anchor requires body PUSH_F64, typed F64_NEG, FLOAT local store/load and F64_EQ only. I propose explicitly bounding new borrowed FLOAT operations to that demonstrated literal/equality subset initially; arbitrary arithmetic, conversion or FLOAT pretests are not implicitly admitted by reusing SIMPLE. Typed NEG closes regeneration of negative finite raw literals, including signed zero; no general arithmetic is added. Existing FLOAT pretest filters, resource hold restoration, helper totality proofs and actual witness selection remain unchanged. The helper performs only existing INT/BOOL behavior, so no new witness FLOAT snapshot is needed.

I reuse existing raw-bit C literals and finite Nano literal rendering rather than a new representation. Nonfinite Nano literal output retains its existing atomic refusal. Actual generated counts are unknown until the first unchanged ordinary/original-shadow publication and seed verifier/VM measurement, including independently selected generated consistency shadow. The existing128 cap is fixed; no planner or scratch route expands to fit a hypothesis.

## I qualify only after measured feasibility

I require both existing owners and exact control flow/assertions to survive; caller-local FLOAT values cannot enter resource layouts, borrows, signatures or helper witness facts. After first actual counts, focused evidence should cover all three producers, O0/O2 sanitizer native parity, finite literal rawbits/BOOL equality, the unchanged branch and original shadow, forbidden helper/field/signature/FLOAT-pretest cases, atomic output failures and old-profile byte equality. First malformed/opcode/type/provenance controls must fail before publication. Existing consuming FLOAT and borrowed INT/BOOL paths remain byte-identical.

This commit is contract/baseline only. Exact production discriminator/opcode changes require review; no admission edit or build follows from this document.

## I retain the first rendering refusal and narrow planner route

The first changed facts/analyzer gate succeeds, but unchanged ordinary and original-shadow Nano output refuses the fixed cap at 131 caller locals and 21 helper locals. C output publishes; Nano preserves the sentinel. I retain the first refusal under `/private/tmp/borrowed-float-local-first-20261020`. Existing caller-only finite planning estimates 98/21, covers all 104/107 statements, and preserves all 13 assertions, five path calls, and FLOAT expression order. This estimate is not regenerated evidence.

I enroll only an actually FLOAT-bearing caller validated by the exact discriminator, after existing immutable helper/witness proofs. I reuse existing INT/BOOL patterns on the caller; helper stays unplanned, no scratch bank is added, and FLOAT expressions are not folded or recomputed. I explicitly refuse composition with helper-local ownership, owner reassignment, or exit-aware loops. Existing non-FLOAT profiles retain their rendering. Actual regenerated counts and VM/native gates remain required.

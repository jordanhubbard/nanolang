# My grouped union or-pattern source candidate

I preserve one C-compatible OR:A:B:C pattern per match arm. I retain duplicate
alternatives, one guard/body and once-only scrutinee/guard effects. An underscore
inside a group is an exact variant name, not a wildcard. I validate every
alternative against the existing exact source union domain and mark coverage
only for unconditional groups. Existing integer, wildcard, ordinary binding
and zero-field/discard rules remain intact.

I emit short-circuit tag checks that join before the single guard and body.
The original scrutinee remains rooted across false guards. A discarded group
retains the whole union type in its hidden local rather than inventing a payload
binding for an encoded group. I change no AST layout, opcode, affine/borrow/effect
provenance check or foreign ownership admission.

My [local checks](local-checks.json) pass the existing actual checker component,
including the new parser and coverage shadows, and all four C-created control
methods. The checker does not import the emitter shadows; those still require
hosted source creation. My [additional refusal/lexer checks](expanded-refusals.json)
pass after adding duplicate-only missing coverage and source-only ownership
controls. I do not claim the source-only controls execute locally.

I retain [actual local products](local-products.json), [every C-created command
and diagnostic](local-texts.json.gz), the [checker image](local-checker-product.json)
and [checker text](local-checker-texts.json.gz). Eight native images across initial
and corrected attempts match actual Mach-O bytes; six belong to the corrected
three positive controls. Existing local providers are pinned binaries without
current C-source provenance acceptance. Source-created, fresh and paired platform
execution remains open.

My first authored positive fixtures were invalid. Runtime gate(true) did not
establish unconditional coverage; I add the required fallback without removing
ordering assertions or the separate non-exhaustive refusal. Ordinary C constructor
lookahead requires uppercase variant names; I exercise the valid explicit generic
construction path for the underscore variant. I retain both initial refusals and
their source/product/command observations under initial-fixture-refusals/.
The original tests/unit/test_match_orpatterns.nano stays unchanged.

The common C/source controls retain the original fixture, duplicate and overlapping
groups, effect order, primitive expression results and nine syntax/type/coverage
refusals. Source-only controls retain the already stricter exact-name refusals
and require resource-payload ownership to stay closed. I do not assert the C
checker already provides the source checker's stricter unknown-name diagnostics.
All original source selections, deadlines, fresh Stage 1/2, full corpus, budgets,
security, native lifecycle and release obligations remain required.

My final checker rerun in `final-checker.json` passes after I preserve direct
ordinary-pattern lookups and decode only grouped alternatives. Its current
parser/typechecker/provider hashes stay unchanged across the actual native
checker run; I retain its image observation and command text in
`final-checker-product.json` and `final-checker-texts.json.gz`. I do not count
these checker controls as source-created emitter or fresh-stage acceptance.


## My exact 328 hosted results and block-arm correction

I independently retain all three intermediate attempt artifacts, CRC-check their
ZIP contents and compare every retained source-creation NanoLang input with exact
Git 328442f1f. My audits under source-328-{clang,gcc,apple}-attempt/ verify
actual grouped-pattern product hashes, stable inputs and complete command text.
Clang starts 1,059 shadows and completes 1,058 before the ten-second deadline;
GCC completes all 1,286 but still reaches the deadline before publication.
Both have zero recorded completed-body assertion failures and no published
source compiler. Their started selections are ordered prefixes of Apple's
complete selection. Each has six actual C-created native images, three modules
and nine previous-output refusal controls.

Apple creates the actual source compiler after all 1,286 shadows complete and
passes 32 of 33 component methods. I verify its retained compiler image, ten
C/source-created grouped-pattern native images, five modules and 21 preserved
output refusals. The source-only resource control reports the intended E0036
ownership diagnostic; I strengthen it to require that category explicitly.
The unchanged original fixture refuses emission with `I require a statement
block in each match arm`. This is a correctness failure, not a source-creation
deadline. I retain the exact positive-fixture failure separately from refusals.

I correct the grouped parser branch to store PNODE_BLOCK explicitly for block
arms, matching existing ordinary/wildcard branches. parse_block returns the
block ID while its last-expression type can name the final inner statement.
My added parser shadow checks the block tag, one statement and its return kind;
my emitter shadow requires statement-block lowering. I keep the original fixture
unchanged. My actual local checker passes the parser shadow and native checker
entry; emitter execution and source-created/fresh/platform acceptance remain
hosted requirements. I retain initial extra-parenthesis and enum/int cond-type
editing refusals under block-tag-initial-* and their corrected final run in
block-tag-checker.*. I qualify no compiler from those initial refusals.

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

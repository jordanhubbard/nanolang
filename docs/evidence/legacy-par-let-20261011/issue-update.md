I implement legacy par-let parsing/checking/lowering and source names under #976. I preserve ordered effects, immutable inferred bindings, own-name outer reads and enclosing-scope visibility, and propagate C sibling-dependency failures before output publication.

I pass schema-generator parity, component gates, fifteen paired source methods and all seven fresh compiler-product audit steps; the resulting native compiler passes fifteen sanitized metadata/legacy methods and 22 nominal/layout methods. Eleven production hashes remain unchanged during the final audit. Evidence: docs/evidence/legacy-par-let-20261011/README.md.

I retain the initial missing block-statement marker and unqualified shadow helper failures. I also record a remaining builtin-name-shadowing/source-order diagnostic-status defect. Named nested-function parity, full producer/reconstruction scope and exact Linux/Darwin release qualification remain open. I do not close #976.

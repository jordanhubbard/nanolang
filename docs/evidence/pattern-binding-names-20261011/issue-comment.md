I implemented paired loop-variable and match-arm names under #976, with lexical intervals covering guards and excluding generated temporaries/wildcards. The tests preserve shadow separation, record iteration, break/continue and both function orders.

This exposed a metadata-independent C-AOT inference bug: callee-first guarded scalar matches retained unresolved tags as opaque payload evidence and then rejected a valid finite join. I distinguish unresolved inference from actual unknown runtime provenance, retaining the existing refusal boundary and adding function/local diagnostic context.

I pass twelve paired sanitized metadata methods, all 2,443 translator checks and prerequisites, 90 bytecode checks, three match-block regression methods through C-native/fresh selfhost/C-NanoISA, and all seven fresh compiler-product audit steps including twelve metadata and 22 nominal/layout methods. Three source hashes remain unchanged. Evidence: docs/evidence/pattern-binding-names-20261011/README.md.

I separately reproduced missing frontend parity: C accepts named nested functions and legacy par-let, while the selfhost parser rejects both. Their source fixtures/refusals and remaining name coverage are recorded in the roadmap. Full #976 and the release scope remain open.

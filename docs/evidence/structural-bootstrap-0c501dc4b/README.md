# My integrated structural generic bootstrap

I complete all 17 Darwin bootstrap steps with compiler inputs from 0c501dc4b in
`obj/bootstrap-nanoisa/run-7e8e9gef`. Source, tool and immutable host-library
checks pass; no competing native module builds change my host closure. Both
native compiler stages compile and execute hello-world products. Final seed,
Stage 1 and Stage 2 module verification passes. Raw Stage 1 and Stage 2 modules
are byte-identical, with SHA256
`c9d2594227cd76e226687aae6a488d89b6a3e130093d069683cf5e9309a5274f`.
I retain the full receipt and per-step logs. This run takes 1,237.62 seconds;
that is one measured bootstrap, not a general latency guarantee.

All 21 shared generic methods pass through the fresh full seed and each installed
native compiler stage. They include structural arrays, callback parameters and
locals, qualified/selective imports, mismatched shapes, uninferred results,
prior-output preservation and previous generic controls. Successful products
execute mandatory shadows, NanoVM and sanitized generated C. All 28 imported
global methods also pass through both newly installed stages.

My first seed run passed 20 methods and failed only a diagnostic assertion:
the self-hosted checker rejected array<E> where array<int> was required, while
the C checker named the missing concrete binding. I retain that baseline and
accept either specific error for this refusal, retaining positive failure status
and exact prior-output preservation. The C refusal control still passes.

This qualifies the tested integrated source corpus on this Darwin host, not the
full 5.1 release or reproducibility across clean hosts. C enum/tuple metadata,
module-owned enum identities in both producers, other implementation rows and
exact-candidate Linux/target/release gates remain open. My isolated enum candidate
is outside these bootstrap inputs and is not covered by this receipt.

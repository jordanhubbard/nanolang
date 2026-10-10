# My integrated structural generic seed

I run a clean sequential bootstrap at compiler source 0c501dc4b in
`obj/bootstrap-nanoisa/run-7e8e9gef`, without overlapping native module builds.
My fresh compiler seed compiles and verifies. All 21 shared generic methods
pass through that exact seed module, including structural arrays, callback
parameters and locals, qualified/selective imported functions, mismatched shapes,
uninferred results, output preservation and previous generic controls. Successful
programs execute mandatory shadows, NanoVM and sanitized generated C.

My first shared run passed 20 methods and failed only a diagnostic assertion:
the self-hosted checker rejected array<E> where array<int> was required, while
the C checker named the missing concrete binding. I retain that baseline and
accept either specific error for this refusal, retaining positive failure status
and exact prior-output preservation. The C refusal control still passes.

I pin the seed hash and verify the immutable host-library hashes remain unchanged.
Stage 1 generation is live when I record this checkpoint; Stage 2 equality,
installed native stages and final release qualification remain unverified.

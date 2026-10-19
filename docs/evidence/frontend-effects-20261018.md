# My bounded frontend effect metadata checkpoint

I track this first slice under #998. I retain `nano.effects.v1` independently
of declared purity and reuse my existing reachable-body purity analysis.
My mask distinguishes possible I/O, mutable reads, writes, unknown analysis,
unsafe operations, and resources. It grants no execution authority.

I deliberately bound advisory analysis to 64 source function bodies and 4096
AST visits. My self-hosted path also bounds scratch slots and leaves globals
without symbol context unknown. I keep declared-pure validation unchanged.
I do not complete affine-use, match-site exhaustiveness, imported/global,
generic, or nested-function precision in this checkpoint.

My local Darwin C seed passes 754 codec checks and 34 allocation boundaries.
My strict gate passes 222 Clang and 222 GCC compiles plus the relink fixture.
My four inferred-summary methods exercise actual emitted records for direct
calls, recursion, I/O, mutation, unknown callbacks/loops, and both advisory
budget boundaries. They check byte-identical canonical assembly, VM/native
execution, and unchanged execution/native code after stripping, duplicating,
forging, or changing the version of metadata. These native checks use ordinary
host compilation; I do not describe them as sanitizer evidence.

I retain the first failures: a 4100-local fixture exceeded the backend's local
limit, so I use many assertions to exhaust AST work without that unrelated
limit. My first Stage 1 run omitted unused functions, so the corrected fixtures
call every function whose emitted summary I inspect. Adding a second advisory
family exposed a pooled-string assumption in the old purity forgery test; I
now select or append the replacement payload and update only its metadata
reference. I retain valid-forgery and prior-output assertions.

My seed and both fresh self-hosted stages pass all four inferred-summary
methods and all eight declared-purity methods in the final combined run
(12 methods, each selecting all three producers, 12.524 seconds). My complete
fresh bootstrap passes native hello programs, final verification, unchanged
source/tool inputs, and raw Stage 1/Stage 2 byte identity. Both raw modules have
SHA-256 `38bbdcfbd5d294d5ea283f336cb1bcab25937768704a74b1b287fd9baefd9252`.
I retain local bootstrap logs in `obj/bootstrap-nanoisa/run-wn3twokp` and the
combined test terminal at `/private/tmp/nano-998-all-producers.log`.
Hosted Linux/Darwin qualification of the combined release candidate remains
required; this local checkpoint is not release readiness.

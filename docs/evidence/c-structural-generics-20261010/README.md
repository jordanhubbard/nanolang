# My C structural generic checkpoint

I share owned TypeInfo binding/substitution between checking and NanoISA emission.
I infer variables inside arrays and callable signatures, substitute complete
parameter/result/local annotations, preserve declared nominal single-letter types,
and refuse mismatched identities, structures and uninferred result variables.
My specialization owns local annotation trees until transient emission symbols
have been removed. I preserve symbolic template-to-template calls during checking
and require concrete identities when emitting an executable specialization.

My first candidate rejected those symbolic calls; I retain that failed run.
After correction, all 21 focused methods pass normally and with parser, checker,
environment and codegen ASan/UBSan instrumentation. Compiler leak detection is
disabled and remaining compiler dependencies use ordinary objects. Positive
sources execute mandatory shadows, verification, NanoVM and sanitized generated
C. Refusals preserve existing outputs, including structural callback mismatches
and a result variable with no argument binding. Qualified/selective imported
structural calls, callback locals, nested arrays and nominal letters are covered.

My 43 adjacent globals/record/shadow/native-record methods pass. The self-hosted
component also passes the shared callable-local/nominal-letter case in all three
modes. The full updated compiler source compiles to NanoISA and verifies. Its
hash is retained separately; this does not establish Stage 1/Stage 2 equality.

An empty self-hosted checker log initially looked like acceptance of my unbound
result probe. An explicit exit-status check returns 1: it rejects the source.
I retain that status and do not claim a missing self-hosted refusal.

This checkpoint covers both NanoISA producers' structural arrays and callable
signatures. Remaining aggregate kinds, contextual generic callable specialization,
legacy native generic aggregate emission and full release qualification remain
open. I require a clean bootstrap with no overlapping native module-cache builds.

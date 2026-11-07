# Profile plan composition

I compare the exact command-vector sets planned by `7db768cc675b3487342823351710bff685f5a3b1` and candidate `61fb56d58`, using my current unchanged Make rules and the same private include root on Darwin. I execute no compiler in this comparison. My [result](comparison.json) retains the revisions, counts and newly selected source paths.

I preserve all 1,254 prior Clang test-profile vectors and all 1,246 prior GCC vectors. I add 114 vectors per compiler, including the five new binding/isolation/snapshot C fixtures and their actual companion providers, ordinary/instrumented hooks and private generated-header paths. I remove zero prior vectors. These host-specific plan counts differ from the Linux audit and exclude the separate product compile set.

My executed binding result remains in `docs/evidence/binding-compile-profiles-20261107/`; the earlier complete executed Make/private result remains in `docs/evidence/compiler-profile-hosted-20261107/`. This plan comparison does not issue a native receipt, prove full Python-profile coverage or qualify the combined runtime budget.

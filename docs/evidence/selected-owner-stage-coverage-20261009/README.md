# My selected-owner qualification coverage

The earlier 0273c40fd installed array stages reproduce all eighteen reported
failures in the unchanged selected/generic ownership suites (36 methods,
2.380 seconds). This establishes that d1b138fbe did not introduce that boundary.
It does not satisfy ownership acceptance.

My generic suite called the checker on a detached TestCase. Its first failing
compiler assertion escaped and prevented later compilers from running. My
same-source control reports one failure before correction and two after
calling the helper on the active test instance. I retain both exact outputs.
The assertions, programs and output-preservation checks are unchanged.

Source diagnosis locates the full dependency: nb_union_supported admits scalar
payloads only; nb_union_type_identity cannot identify resource record arguments;
nb_union_layout_bytes has no nested payload layout; union layout ownership flags
are emitted as zero. The C ownership decoder also requires scalar fields without
nested indices, and affine_bytecode's AGG_PACK rejects owned payloads. I must
carry selected-variant ownership through module metadata, verifier state,
construction, moves, destructive projection, branch joins, calls, VM and native
cleanup before admitting these programs. Removing only the frontend refusal
would not implement that contract. Full ownership remains required for 5.1.

The corrected 36-method suite runs in 23.342 seconds and reports 28 failures:
eight selected-owner stage subcases and twenty generic-owner stage subcases.
The ten additional failures are the previously skipped Stage2 runs, not new
product regressions. C-seed positives and checked refusal cases continue to pass.

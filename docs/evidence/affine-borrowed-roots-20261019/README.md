# I qualify borrowed-root observations

I base this isolated slice on `7c7f3f5af`. I record ownership before implementation in [#998](https://github.com/jordanhubbard/nanolang/issues/998#issuecomment-6261783849). My parent independently reviewed the production change without a blocking finding.

I retain copied C tools from `/Users/jkh/Src/nanolang-51-affine-use-facts/bin`, with hashes in `providers.json`. I build `bin/nanoc_borrowed` from the changed Nano sources using that retained `nanoc_c`; its imported/root shadows pass. This is a current-source C-hosted Nano compiler, not fresh Stage1/Stage2 bootstrap evidence.

My commands, from this worktree, are:

```sh
NANO_SHADOW_TIMEOUT_SECONDS=300 bin/nanoc_c src_nano/nanoc_v06.nano -o bin/nanoc_borrowed
NANOLANG_AFFINE_USE_COMPILERS="$PWD/bin/nano_virt:$PWD/bin/nanoc_borrowed" NANO_SHADOW_TIMEOUT_SECONDS=300 python3 -m unittest -v tests.test_affine_use_metadata
NANOLANG_AFFINE_COMPILERS="$PWD/bin/nano_virt:$PWD/bin/nanoc_borrowed" python3 -m unittest -v tests.test_function_affine_metadata.FunctionAffineMetadata.test_called_resource_routes tests.test_function_affine_metadata.FunctionAffineMetadata.test_no_resource_pass_is_unknown tests.test_function_affine_metadata.FunctionAffineMetadata.test_checked_generic_instances_stay_unknown tests.test_function_affine_metadata.FunctionAffineMetadata.test_resource_refusals_preserve_output
```

I pass all five affine-use methods in 71.367 seconds and four function-completion neighbors in 3.632 seconds. C and changed Nano emit exactly one shared `view` OBSERVE and three mutable `view` OBSERVEs for the real helper fixtures, with no borrowed-root CONSUME. Both paths pass original VM and O2 ASan/UBSan native execution; removing advisory facts leaves generated C identical and VM behavior unchanged. Existing owned OBSERVE/CONSUME, advisory forgery/duplication, generic UNKNOWN, move/borrow refusal and prior-output preservation controls pass. The checker fixture runs through the C evaluator's actual shadow execution and compiled native execution, checking raw UNKNOWN, equal-epoch different owners, incomplete functions and stale snapshots after rechecking for both reference modes. Helper shadows reject unknown, plain, scalar, nested and malformed targets.

I preserve the earlier failed run in `paired-source-and-checker-unused-helper-failure.log`. Its two new emitted-fact assertions used a synthetic main that never called the borrowed helper. The ownership emitter retains actual direct-call dependencies, so those helper facts were correctly absent. I corrected the fixture to call each helper through its real qualifier and consume its owner; I did not change emission or checker gates. The preceding direct checker owner assertions and other source tests had passed. The original C/Nano asymmetry log is also retained.

I change only successful resource field-root observation classification. Owned-value classification, consuming uses, exact owner/current/completed epoch gates, verifier admission and executable behavior remain unchanged. Full #998, broader paths/projections, imported/generic coverage, fresh aggregate generations and Linux/Darwin qualification remain open. I run no full bootstrap in this slice.

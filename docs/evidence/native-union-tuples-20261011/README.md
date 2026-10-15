# Native union tuple layouts and generic identities

I repair tuple dependency ordering for ordinary and specialized union payloads.
I encode complete tuple and callback argument trees in native generic names,
including nested arguments and canonical nominal owners. I retain List type
names and use signature emission for pointer-backed collection payloads.

I reproduce the incomplete tuple field in `baseline.log` and the distinct
arguments collapsing to `Box_unknown` in `distinct-baseline.log`.
My new `tests.test_native_union_tuples` has eight methods; together with the
ten shared aggregate payload methods, I execute tuple order, distinct tuple,
callback and primitive List arguments, nested callback signatures, imported
record owners and ordinary/generic record List payloads.

## Checked routes

- `native-final.log`: 18 methods through current `bin/nanoc_c`, including shadows and execution.
- `nanoisa-final.log`: 18 methods through current `bin/nano_virt`, verification, VM and sanitized C AOT execution.
- `product-payload.log`: 18 methods through the retained self-hosted product `/private/tmp/nl51-list-record/compiler-final`, verification, VM and sanitized C AOT.
- `nominal-current-c.log`: 16 layout methods with `NANOLANG_NOMINAL_COMPILERS=nanoc_c`.
- `seed-build.log`: current C compiler builds `src_nano/nanoc_v06.nano` successfully. The fresh seed passes 33 payload/generic/module methods (`fresh-seed-payload.log`) and all 16 nominal-layout methods (`fresh-seed-nominal.log`).
- `component-gates.log`: `make test-typechecker test-transpiler` succeeds.
- `instrumented-payload.log`: 18 methods through a separately linked compiler with the changed typechecker and transpiler objects instrumented by Homebrew Clang ASan/UBSan. I disable leak detection for this compiler check; it is not a whole-compiler leak qualification.

My adjacent runs each execute 54 methods but fail 22 subcases in the 16-method
nominal suite because that suite independently defaults to the old installed
`nanoc_stage1,nanoc_stage2`. The other 38 methods pass (generic union parameters,
module union owners and generic functions). I retain both failed runs and the
explicit current-compiler nominal result. I do not treat these runs as an
installed-stage qualification or remove the existing release requirement.

I have not changed NanoLang compiler sources or the AOT translator in this
checkpoint. I keep #976 and the full release scope open. My GitHub API status
request failed as recorded in `issue-status.err`; Git transport is separate.

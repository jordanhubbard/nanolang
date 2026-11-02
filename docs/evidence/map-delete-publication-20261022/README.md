# Exact STRING→INT DELETE publisher qualification

I qualified existing paired publishers against the DELETE consumer ancestry 879db48ac and the preserved f83927078 test additions (merge 537e7a68d). No publisher production code changed. The new positive source is the unchanged tests/unit/test_map_field_operations.nano with exactly one HashMap<string,string> replaced by HashMap<string,int> and two "kept" literals replaced by 42. All calls, DELETE operations, assertions and shadows remain. STRING-valued maps remain outside this selected profile.

The mixed omission oracle retains DELETE and every previous assertion, adding the already-qualified unsupported str_trim assertion; both optional buffers still disappear together. The integer-only older operation expectation now requires authority. Other STRING-value cases retain their old omission expectation. The f839 raw ordinary/shadow, map-only/multiple/forward-record and declaration-mismatch coverage is preserved.

## Actual products and gates

Current isolated C seed and independent C shadow harness link the exact retained aggregate closure with five current consumer translation units. Direct C controls pass 100 checks. Current isolated public nanoisa/nano_vm/nvm2c tools verify and execute both C outputs. Installed tools are unchanged.

One fresh C-hosted canonical compiler and one fresh C-hosted dual raw parser driver were built serially with current module query sources and private cache: 141.997s and 77.323s, each with an explicit effective 900s timeout. Complete commands, source/object/archive hashes and selected current VM are retained. These products are not raw Stage1/Stage2 or bootstrap evidence. The canonical VM hash was captured during the build, not falsely labeled a prelaunch sample; source/object inputs have before/after pins. The raw build includes the VM in its before/after input pins.

The full eight-method publisher corpus passed in 58.459s: 42 actual modules across C ordinary/C original shadow, checked Nano ordinary/captured checked shadow and raw Nano ordinary/original shadow. All 42 pass public verification, VM execution and transport roundtrip; 84 O0/O2 native executions pass with ASan/UBSan. Thirty-three selected modules force native collection; nine intentionally omitted modules retain their legacy path. Exact wire endpoint/binding comparisons are independent across producers. Raw parsed declaration evidence is not semantic-checker provenance.

The actual two older map-operation methods passed in 8.837s using explicit test-owned ROOT/tool symlinks to current isolated providers and the fresh canonical compiler. Their assertions are unchanged except the reviewed integer authority expectation. This is not evidence for default installed-tool or fresh-stage workflow paths.

The new permanent integer Holder allocation method passed in 389.984s with 723 publisher and 508 query positions, each transient/persistent denial plus independent valid retry (2462 denied invocations). Existing hard-error/prior-output checks and assembler/save growth controls remain unchanged. This measures publication/query allocations, not every frontend allocation. Exact instrumented providers, flags and source hashes are retained; the raw per-position logs remain at the artifact directory in allocation-artifact-summary.json.

Clang and GCC16 strict syntax pass for the changed C include harness; Python syntax and git diff checks pass. Eight old scalar/flat/depth2/depth3 source outputs are byte-identical between retained and current C/C-hosted Nano products. A separate read-only previous publisher-corpus comparison records 20 exact Nano module matches; 10 C dumps differ only in their retained source-directory strings (binary checksums consequently differ). Two malformed-control artifacts were excluded from that compatibility comparison. No normalization is claimed as byte equality.

Publisher and Nano lowering production sources are unchanged from f83927078. The six exact old proof budgets and 80 strict consumer configurations are the already-qualified consumer receipt, not a newly repeated gate.

## Retained setup failures

The first operation-script launch used the wrong working directory and failed import before tests. Only workdir was corrected. The first allocation harness linked Apple clang against Homebrew LLVM sanitizer objects and failed with __asan_version_mismatch_check_v8. The retry selects the exact /opt/homebrew/opt/llvm/bin/clang used for the current objects; no policy, assertions, or sanitizers were weakened.

## Boundaries and handoff

The required test-map-field-publication target includes the new positive and all eight methods; test-publication-allocation includes its own independently measured integer Holder method. Existing test-map-field-operations retains both actual methods. Default targets still require fresh stages. The scoped aggregate.mk uses explicitly retained C-hosted canonical/raw selectors and actual aggregate C/public providers without claiming fresh bootstrap.

Consumer proof/recipe/execution/native gates, six exact legacy budget rows, old sizes and retained first failures are in ../map-delete-consumer-20261022. No consumer production delta followed that receipt. Unsupported STRING families, other map pairs and remaining authority profiles remain separate scope. Fresh-stage/platform acceptance is still pending.

The bounded aggregate handoff is `make -f docs/evidence/map-delete-publication-20261022/aggregate.mk -n map-delete-publication-scoped`, then the same reviewed target without `-n`. It rebuilds actual aggregate C/public tools and links both real C harnesses; only Nano canonical/raw products use explicit retained selectors. The actual operation-test tool mapping is in operations-config.json and operations.py. The allocation environment and exact selected object paths are in allocation.json. No installed provider was replaced.

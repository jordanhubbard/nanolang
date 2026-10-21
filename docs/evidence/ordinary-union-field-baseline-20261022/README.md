# Ordinary union record-field baseline

I measured the unchanged `Outer.boxed:Box<int>` source on aggregate `f53539b9c`, before any union-field authority changes. Ownership: [#1001 comment](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6326717133).

All six independently emitted modules passed current public verification, ordinary VM execution, native translation, and native O0/O2 ASan/UBSan execution (12 native runs). All contain `.types 1 0 1`, no layouts/ownership sections, and return 0 from the pinned host `nl_nanoisa_array_publication` query. Query 0 is the current ordinary fallback/omission result, not selected union proof. No failed command occurred. Provider/source hashes before and after match.

| Module | Bytes | Functions | Instructions |
|---|---:|---:|---:|
| C ordinary | 1100 | 1 | 40 |
| C original shadow | 1288 | 3 | 48 |
| C-hosted canonical ordinary | 876 | 1 | 35 |
| C-hosted checked shadow | 1140 | 3 | 44 |
| Raw parser ordinary | 812 | 1 | 35 |
| Raw parser original shadow | 1076 | 3 | 44 |

`unchanged-direct.nano` is copied without edits from the retained generic-record-field-context source and matches `test_direct_field_match_executes` in `tests/test_native_generic_record_fields.py`. The checked shadow capture wrapper copies the actual staged module and executes the pinned current VM with unchanged arguments/status. Raw parser outputs do not establish semantic-checker provenance. The qualified canonical/raw products are retained C-hosted products, not fresh raw Stage1/Stage2 evidence.

`baseline.py` and `status.json` retain exact commands, timings and outcomes; `counts.json` retains module hashes; the pin manifests retain source/compiler/consumer/host-query identities. Dump files retain actual original instructions. Binary/native output artifacts remain at `/private/tmp/nanolang-union-field-baseline-20261022`; no frontend/bootstrap build or installed-product mutation was performed. Native runs establish legacy behavior only: canonical selected union representation, forced-GC lifetime and allocation failure remain unqualified. Fresh-stage/platform acceptance remains open.

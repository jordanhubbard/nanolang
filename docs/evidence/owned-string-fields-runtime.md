# My retained STRING field runtime evidence

I qualify runtime-only source/harness `6b0888c5`, integrated with canonical
PR804/805 at `47ad402f`. Root and independent review approved production
`bf50c4e7`; integration preserves the nonparameter FLOAT-local guard and keeps
FLOAT fields/signatures refused. Both source producers remain unchanged.
My [contract](../NANOISA_OWNED_STRING_FIELDS.md) records all fixture migrations
before execution. My [seal](owned-string-fields-runtime.json) records exact
files, tools, commands, logs and generated artifacts.

| Phase | Seconds | Outcome |
| --- | ---: | --- |
| GCC tool preparation | 9.091 | pass |
| GCC fixture preparation | 30.745 | pass |
| GCC frozen runtime/adjacent gates | 12.199 | pass |
| Clang tool preparation | 0.353 | pass |
| Clang fixture preparation | 18.995 | pass |
| Clang frozen runtime/adjacent gates | 15.079 | pass |

My outer runner exits zero. All 1,666 tracked source/harness inputs remain
unchanged. Each frozen phase preserves its complete tool/object map; the final
map contains 305 entries. I archive 16 bin/test files for each compiler before
changing compiler preparation. The measured maps cover all objects; the
archives do not claim to preserve every historical object.

Both compiler configurations pass:

- 4,394 new VM STRING-field allocation checks, 216 budgets and 196 actual
  injected failures, with phase/case/API/fault diagnostics and recovery;
- 710 new field/lifecycle checks through four VM APIs, four emitted native
  cases and binary/text transport; empty/nonempty branches, nested returned
  owners, aliases surviving shell destruction, local overwrite, direct STRING
  parameters, EQ/NE, copies/discards/permutation and assertion cleanup;
- strict generated-native GCC/Clang allocation budgets, zero live roots,
  result preservation and immediate recovery, plus defensive helper equality,
  checked size and reference-count refusal;
- 600 prior owned-string allocation checks, 150 invocation-proof checks and
  prior print execution; every native failure retains an exact output prefix,
  and recovery retains complete expected output with zero roots;
- 207 nested-result query/DAG/depth/allocation checks, including new STRING
  leaves and unchanged query output on allocation refusal;
- 3,188 prior nested-result allocation checks with 280 injected failures,
  714 prior result/lifecycle checks, 75 reference-place checks,
  138 mixed-layout descriptor checks, 339 private mixed-FLOAT proof checks and
  1,239 owned binary64 checks.

My new refusal controls retain ARRAY/FLOAT fields, standalone STRING results,
STRING ordering and borrowed STRING-bearing roots. All three closed target
profiles and both private mixed-layout/proof queries refuse the STRING owner
fixture. These are checked refusals, not rejected-module executions.

My generated native harnesses use `-std=c11 -Wall -Wextra -Werror`,
ASan/UBSan and Linux leak detection. The VM fixture executables link ordinary
non-sanitized common/VM objects; I do not claim a fully instrumented VM build.
Clang fixture preparation rebuilds its selected fixture units and relinks
executables; shared production objects may retain GCC compilation. My logs
retain exact commands rather than implying a complete Clang rebuild.

I preserve all reports under `/tmp/nanolang-owned-string-runtime-6b0888`.
There is no failed command in this qualification. I have not run source
bootstrap, enabled source STRING fields, qualified Darwin, or completed the
original Bundle source acceptance. Those next stages remain required, including
canonical inline construction and complete selected shadows. Runtime success
alone closes neither this child nor any ownership/managed/release parent.

## My additional initialization-join qualification

Root review identified that my original fixture branches before STRING locals
are created. I preserve that full seal and tree. A separate tree at frozen
`80c8830b` adds direct query tests for all four initialization combinations,
two valid modules executing each initialized branch, and two one-arm missing
initialization modules that are refused without execution. Query failure leaves
tag/mode output sentinels unchanged; repeated meets are idempotent.

My [additional seal](owned-string-initialization-joins.json) records GCC and
Clang runs, each passing377 checks across four VM APIs and both native branch
cases at strict C11/O2 with ASan/UBSan/Linux leak detection. Frozen execution
phases take0.427 and0.440seconds; the outer runner exits0 with source and
binary/object maps unchanged. Preparation is separately measured. No
production changes, source admission or Darwin claim follows from these tests.

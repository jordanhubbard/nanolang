# I retain optional FLOAT join storage

I base this isolated slice on `7c7f3f5af` plus borrowed-root commit `fce2fcdf4`. I record ownership and the accepted-source defect before changing production in [#976](https://github.com/jordanhubbard/nanolang/issues/976#issuecomment-6261993452). I separately retain the U8 self-hosted List profile boundary in [#1001](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6262049160).

## My observed defect and boundary

My original FLOAT present/empty/unchecked programs pass actual C and retained Stage1/Stage2 source compilation, original verification and VM execution. The snapshotted aggregate translator refuses OPTIONAL-to-FLOAT conversion. BOOL and STRING controls pass VM and O0/O2 ASan/UBSan native execution. The JSON probe bundles retain exact source, provider hashes, commands and diagnostics. Initial lexical `List<Fact>=` fixture errors are separately retained; I corrected the spacing without changing the semantic shape.

Correctly typed U8 fixtures pass through C into VM/native execution but retained Nano stages refuse `unsupported local type List<Fact>`. An untyped U8 fallback is a source cond-type error; a typed fallback with an INT comparison produces a separate native U8-to-INT refusal. I retain these as separate boundaries and do not modify U8 handling.

## My bounded change and checks

I add only FLOAT to the existing OPTIONAL-to-inferred-scalar storage widening branch. The `conversion_kind` guard still rejects exact destinations; payload matching remains exact. No verifier, source admission, join classifier, or emitted extraction changes are part of this diff.

My graph test first flows an exact scalar into destination storage, then introduces an OPTIONAL producer. INT/BOOL/STRING/FLOAT cover inferred success, exact destination refusal, mismatched payload refusal, producer preservation and repeated solve stability. The initial test incorrectly queried a poisoned graph after expected failure; I retain that failed run and corrected the harness. The corrected pre-fix graph run has exactly one failure: FLOAT inferred widening. After the fix, all 3262 shape checks pass.

I build only isolated C translator dependencies and the shape unit:

```sh
make -f Makefile.gnu -j1 test-nvm2c-shapes nvm2c
NANO_OPTIONAL_RECORD_JOIN_COMPILERS="$PWD/bin/nano_virt /Users/jkh/Src/nanolang-51-owned-reconstruction/bin/nanoc_stage1 /Users/jkh/Src/nanolang-51-owned-reconstruction/bin/nanoc_stage2" python3 -m unittest -v tests.test_optional_record_join
```

My source matrix retains original INT tests and adds present/empty/unchecked/raw-missing FLOAT, BOOL and STRING cases. Each accepted module passes original verification and runs through VM and O0/O2 ASan/UBSan native output. Missing reads must fail, with FLOAT reporting `F64_EQ requires two floats`, INT reporting integer operand failure, and BOOL/STRING reporting failed assertions in the VM. Native execution must stop at its checked invariant without sanitizer findings. The first expanded run incorrectly expected a generic assertion for FLOAT; I preserve that failure and correct only its expected diagnostic.

I retain test artifacts at the path printed in each log and bundle final commands/status/output plus source and hashes here. These are retained producer qualifications, not fresh bootstrap generations. I do not edit the aggregate worktree or run full bootstrap. Full #976/#1001 and fresh platform/generation obligations remain open.

My final reviewed fixture passes all three test methods in 48.822 seconds: 54 verified modules and 108 sanitized native runs. The scalar update shadow now makes a real checked=0 helper call and checks its fallback; main is documented as a runtime matrix control because raw-missing is intentionally failing. Independent production/graph review found no blocker.

# My aggregate-formatting ownership checkpoint

I found a reference-compiler ownership defect while preparing canonical aggregate conversion. My instrumented baseline `case.nano` formats an integer array, a record and a nested union; `run.log` reports 1,024 leaked bytes in four scratch buffers. The generated runtime builder returned its malloc-backed buffer without giving it an owner.

I now copy completed bytes into GC string storage, free the scratch allocation and reset the builder. My regression checks retained result aliases, nested record/union formatting, repeated formatting, string quoting/Unicode, empty arrays, buffer growth beyond 256 bytes and operand evaluation exactly once. These tests qualify the reference ownership repair, not canonical aggregate formatting.

## Baselines and remaining compatibility

`reference-ownership-before.log` passes both focused ownership methods ordinarily. With generated-product ASan/UBSan/leak/UAR, `instrumented-ownership-before.log` fails both: 2,560 bytes in three allocations and 27,392 bytes in 107 allocations.

The exploratory tests also revealed independent compatibility failures. `compatibility-baselines.json` and the individual `.nano` reproducers establish:

- Each enum-field, whole-float-array and nested-array assertion fails in C-seed shadow evaluation but passes in the generated native program. Native-only diagnostic controls replace only the shadow invocation; they retain production assertions and are not acceptance passes for the original programs.
- The simple integer-array conversion passes the C seed and is refused by both canonical stages.
- `reference-before.log` also retains the valid local name `long` becoming invalid generated C. The separate keyword-binding task is recorded in `keyword-task.json` and the roadmap.

`exploratory-tests.py` retains the original expanded assertions. `nested-array-shadow-repro.py` retains the remaining nested-array failure after nominal/whole-float cases were separated. `reference-focused-before.log` includes an intermediate fixture-edit error (`"saved".Loud`); it is not a product defect. Focused ownership fixtures omit the independently failing semantic cases; the full formatting parent remains open and those reproducers remain obligations.

The parent is `task_fe7abd6028d14ae387e70e0e83b8885e`; the ownership repair is `task_1d21d60b31c84be795fbdd629a180997`. I have not implemented canonical aggregate conversion in this checkpoint. 
## Post-repair checks

`ordinary-after.log` rebuilds the C seed and passes both ownership methods in 2.464 seconds. `instrumented-after.log` passes all 16 formatting, scalar-string and aggregate-global methods in 44.025 seconds with generated-product ASan/UBSan, leak detection and stack-use-after-return detection. The two formerly leaking programs pass unchanged. Compiler and VM internals are not instrumented by these environment flags.

`bootstrap-transpiler.log` passes fresh Stage 1/Stage 2 bootstrap, compiler smoke checks, StringBuilder boundary tests and both assert-literal methods. The instrumented comparison above preceded this bootstrap; its C seed already contained the repair, and its canonical stages used unchanged frontend/translator sources. The rebuilt stages receive a final ordinary comparison below. `hosted-snapshot.json` records the successful Backend Matrix run for the preceding pushed aggregate-global commit `2f4516367`; it does not qualify this unpushed ownership change or full CI. `ci-snapshot.json` records main CI still queued at the observation time.

`final-neighbors.log` passes all 16 ordinary methods through the rebuilt products. My source/product hashes are retained in `checkpoint.json`. These checks repair reference ownership; they do not establish final hosted/full-platform qualification or canonical aggregate formatting.

MAC rejects the direct open-to-completed transition (`ownership-close.txt`). I attached the evidence and retained the task lifecycle without forcing completion.

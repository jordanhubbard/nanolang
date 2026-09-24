# Canonical aggregate formatting — incomplete checkpoint

I lower ordinary array, record and union string conversion into shared NanoISA operations. I evaluate the input once, preserve float and declared enum formatting, and build nested strings through the existing managed string operations.

My three source methods pass through the C seed. Before this change, both canonical stages refuse all three in native and VM modes (`pr522-aggregate-lowering-before.log`, twelve failures). With the rebuilt emitter, all three generated modules execute in NanoVM. Native translation and execution pass for scalar/nested arrays and retained record strings, including ASan/UBSan with leak and use-after-return checks. Native translation still refuses the record/string/unit union method. I retain that assertion in `test-canonical-aggregate-formatting`, which participates in `test-units`; this checkpoint is not a passing acceptance gate.

`direct-before` retains the first implementation's output. `direct-corrected` retains the revised guarded variant projection pattern and its unchanged native failure. `direct-instrumented` records each exact command and result. The native failure occurs before C emission and is not a sanitizer finding. The instruments apply to the two native products that were emitted, not to every compiler/tool binary.

`union-control` removes formatting entirely. The original stages still reject the alternating record/string payload while their VM routes and the C seed execute it. This native storage defect is tracked by `task_065a1a2c9858b8968fd3851135b48c47`. The finite scalar/integer-array variant field facts cannot currently join a record payload; simply suppressing the kind conflict would lose the nested layout contract. I require constructor-aware payload shapes and managed storage before claiming this case supported.

I retain the aggregate parent `task_fe7abd6028d14ae387e70e0e83b8885e` as open. Generic/resource formatting, broader source compatibility, full platform qualification, and refreshed final-source fixed points are not established here. `run-direct.py` records the original scratch runner and paths; the checked-in test is the portable source regression.

Fresh `make bootstrap test-transpiler` passes. The rebuilt Stage 1/Stage 2 source matrix runs five methods in 8.909 seconds: every C-seed and VM route passes; six native subtests fail across the three union methods. Two methods pass on every route. The added differently shaped record variants fail with a `string to int` shape conversion, confirming that adding a record tag to the scalar payload bitset alone would not solve the representation problem. I retain both storage controls without formatting.

All eight canonical scalar string-conversion neighbors pass through the rebuilt stages (7.801 seconds).

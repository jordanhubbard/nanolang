# Deferred constructor payload selection

I add `nvm_shape_select_variant(source, tag, target)` to the private native shape graph. The solver copies only that constructor's payload into a separate view, revisiting selections alongside directed storage conversions until they converge. This allows caller facts to arrive after a guarded projection and allows selected payloads to contain another selected union.

The caller must establish a tag guard on the same unchanged runtime value. This API does not prove control flow, value identity, nominal identity, ownership or field bounds. An absent tag contributes no source facts and creates no member; it is not by itself permission to execute a projection. Unknown producer kinds and unknown selected payload shapes are refused at final convergence. Incompatible selected layouts remain errors. Cyclic selections retain the recursive graph without equating distinct consumer views.

The added known/unknown caller regression failed in both producer orders: one known constructor made the selected view resolve while an unresolved caller silently supplied no constraints. I now reject unresolved inputs to constructor-indexed storage at final convergence. I strengthen the previous graph test that permitted an unknown input into a declared variant destination: it still verifies that the consumer does not invent producer members, then requires refusal. I do not weaken source acceptance assertions. `unknown-caller-before.log` preserves the four failed checks.

The additive tests cover both caller orders, deferred copies after selection, nested selections registered before their enclosing producer, missing tags without source mutation, same-tag layout conflicts, invalid source IDs/tags, graph poisoning and cleanup, cyclic selections, and unknown caller evidence. All previous scalar and integer-array variant constraints remain intact.

## Remaining integration

My compiler does not yet emit these selection constraints. Native classification must carry constructor-indexed field facts through locals, globals, stack joins, calls and returns, and preserve the relation between a tag test and its source value on each control-flow edge. A shape ID describes storage, not value identity; reassignment must invalidate old guards. Flat field vectors and shared record projections cannot substitute for per-constructor facts. Runtime projection checks and managed lifetime behavior must agree with the selected view.

The original six native union failures remain acceptance failures under `task_065a1a2c9858b8968fd3851135b48c47`. This graph prerequisite does not unblock PR522 by itself.

## Qualification

Final ordinary and fresh private ASan/UBSan/leak/UAR runs each pass all 2,657 translator assertions and 1,820 shape checks. The sanitizer driver verifies instrumentation in the actual translator and shape objects. `final-native.log` and `final-native-instrumented.log` are the final gates.

Earlier `shapes.log`, `instrumented.log` and `native.log` precede the unknown-caller regression/correction. `final-shapes.log` and `final-instrumented.log` include that correction but precede the four cyclic-selection assertions; the final full gates include those assertions. I retain these build boundaries rather than treating earlier passes as final-source evidence.

`source.log` retains the unchanged five-method source matrix: six native failures in 11.240 seconds; every C-seed and canonical VM route passes, as do the native array and retained-record formatting methods. The source-level union requirement is still unmet.

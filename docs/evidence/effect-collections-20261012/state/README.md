# My typed array local ownership state

I retain checked element-table facts in shared immutable affine declarations.
Array equality follows element structure and exact nominal leaves, including
equivalent duplicate table rows. Resource classification follows nested arrays
to their leaf; arrays with resource leaves require consuming transfers. Copyable
arrays support initialization, replacement and intersection at joins. Resource
obligations must agree at joins and cannot disappear through overwrite.

My handler-state transitions admit typed array parameters and results. Captured
resource arrays remain inaccessible and cannot be reinitialized. Copyable captures
retain initialization. Owned parameters must move before resumption, with exact
result type and no remaining local obligations; lexical result checks use the
same structural equality. Failure preserves the original state and outputs.

I guard record field/packing, borrowed-place and owned-record-parameter APIs
against treating element-table indexes as record-layout indexes. Collection
borrowing and element operations remain unimplemented here. The bytecode analyzer
explicitly refuses any module using the typed element table, so these local-state
facts grant no new executable permission.

The final state suite passes 563 ordinary and 634 allocation-fault checks. The
fault sweep covers state creation, array-fact storage and handler cloning, with
unchanged parent state and successful recovery after each failure. The same 634
checks pass with affine-state code, ownership reader and test harness instrumented
by ASan/UBSan/leak detection; other linked objects retain their instrumentation.
Tests include nested resource arrays, distinct nominal record leaves, copyable
record leaves, structural duplicates, initialization joins, explicit moves,
resumed/lexical results, captured-owner refusal and record/borrow API isolation.
The ownership contract suite additionally checks the 256-row nested-array bound
through local type comparison and preserves transport/refusal controls.

All 1,747 ordinary and 2,555 fault/capacity bytecode checks pass. The shared C-source
effect suite and 26 VM/native runtime/fault fixtures pass against the changed
backend. Production code stayed unchanged during these regression runs; later
additions expanded only tests and comments. I claim no fresh Nano compiler product
or executable array-effect source admission at this checkpoint.

Next I connect exact element facts to bytecode stack/array operations and handler
control flow, then both producers and matched VM/native cleanup. Resource payload
layouts, other collection kinds, borrowed continuations and full 5.1 remain open.

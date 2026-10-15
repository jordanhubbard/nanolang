# My typed collection slots

I bind ARRAY function results, parameters and locals to the exact checked EFFECTS
revision-2 element table. While parsing declarations I collect the largest required
index without interpreting it as a record layout. I resolve that bound only after
the complete extension suffix and every element row have passed validation.
A typed ARRAY slot cannot borrow, and a revision-2 module cannot mix typed and
NO_INDEX ARRAY slots. Revision-1 untyped transport remains accepted as before.

My combined contract suite passes 1,876 checks normally and with the ownership
reader and test harness instrumented by ASan/UBSan/leak detection. Other linked
objects retain their existing instrumentation. The new cases include local,
parameter and result bindings; array indexes beyond the record-layout count;
all truncated prefixes; missing, out-of-range and near-UINT32_MAX references;
both borrow modes; revision mismatch; unchanged query outputs; and exact binary
and unverified assembly round-trips. Ordinary assembly, affine-state construction,
verification and native emission refuse the new typed-slot fixture.

Existing affine bytecode checks pass 1,747 ordinary and 2,555 fault/capacity
assertions. The shared C-source effect suite and 26 runtime/fault fixtures pass.
I retain their logs and the exact sanitizer commands. I claim no fresh Nano
compiler build or source array-effect execution at this step.

The next integration is affine local state: exact type comparison, resource
classification through nested arrays, copyable initialization, owned moves,
handler captures and resumed/lexical exits. Collection operations, producer
lowering, runtime cleanup, other collection kinds and borrowed continuations
remain part of the full release work.

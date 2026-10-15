# My array identity and contextual literal checkpoint

I retain #979 failures and their component corrections. My native translator
now compares exact array handles with generic equality across all eight storage
kinds. Eleven focused methods pass; the full native gate passes 2435 execution,
3098 shape and 379 callable checks. I preserve the original independent-array
fixture while replacing its obsolete expected emission refusal with execution.

My self-hosted checker retains byte and nested literal context and widens byte
returns to integer destinations. The next full all-kinds fixture exposed array
handles reaching I64_EQ. My emitter now selects generic EQ/NE for matching
supported array types. The rebuilt checked compiler component passes all four
methods in 3.674 seconds: six byte destinations, nested literals, the complete
slice bit/bounds/identity fixture, and twenty output-preserving refusals. Each
positive case runs verified NanoVM and strict LLVM C11 ASan/UBSan native code
with leak detection enabled. I retain generated source and command logs.

These are component results, not fresh installed-stage or Linux qualification.
The migrated slice gate still requires a fresh bootstrap and full rerun. My
null-boundary test checks the actual generated NanoISA helper's invariant trap;
it does not assert the retired DynArray helper's null-as-empty behavior.

My empty-main GNU sanitizer control times out with detect_leaks=1 on this Darwin
host; detect_leaks=0 and the default configuration return zero. Subsequent GNU
runs must record their explicit configuration; LLVM leak checking remains on.
The earlier full slice diagnostic retains its original deadline and failure.

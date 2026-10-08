# My native function-array checkpoint

I give arrays of named functions a distinct native storage kind. Their owned
word-array allocation holds non-owning target indices; reads restore function
tags and preserve missing elements as void. Exact and dynamic writes check the
function tag. I trace array ownership through locals, globals and record fields,
without treating target indices as pointers. Deferred array-read constraints
preserve optional function results.

My retained C-seed record/function-array module now executes successfully in
NanoVM and strict native C11 with ASan/UBSan and leak detection. My new controls
check empty constructors and literals, nonempty literals, push/set, mutation
through aliases, global and record storage, returned array elements, target zero,
missing indices, invalid write tags, VM-matching printing and at least two
observed collections. The combined callable suite passes 28 methods.

I also retain two comparison failures: projected unequal functions compared
equal through null text pointers, and exact function operands were refused.
My corrected equality and ordering compare target IDs with preserved tags.
Controls cover all nine direct/global/array operand pairings and distinguish a
function from an integer with the same payload. These are module-local named
functions; I do not claim captured or imported callable identity support.

My broader native gate passes 2,431 structured-C checks, 2,553 shape checks and
204 callable-analysis checks. A fresh LLVM shape binary passes all 2,553 checks
with ASan/UBSan and leak detection. The manifest records source hashes, commands,
terminal statuses, the prepared source compiler and log hashes.

My self-hosted producer still refuses the retained source's record type. Complete
source-producer container admission, captured closures, current raw bootstrap,
platform qualification and the remaining 5.1 release requirements remain open.
The separately archived clean 89-method gate at f701198ad predates this repair.

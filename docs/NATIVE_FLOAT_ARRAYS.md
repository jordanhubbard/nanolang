# My native float-array contract

I record this contract before implementation under `task_ef670a36b03a4df388594c8f1500bd85`, a bounded prerequisite of task2578.

I admit exact `array<float>` source declarations and NanoISA TAG_FLOAT array constructors. I preserve float element shapes separately from int and bool, including empty arrays. My native carrier must preserve IEEE double values and signed zero without numeric conversion or incompatible pointer aliasing. I reuse existing aggregate lifetime accounting, stable handles and root closure rather than changing collection policy.

I preserve exact float writes, consumed operand evaluation order, alias-visible mutation, helper returns and root lifetimes. Raw missing array reads remain void; concrete float consumers retain exact checked tags. Integer/bool arrays never become float arrays by inference. Unsupported nested arrays, new host array ABI kinds and LLVM/Wasm heap admission are outside this child.

I require ordinary literal/empty/filled/read/write/append/length and call/return controls through the source emitter and same-module VM/native execution, including signed zero, fractional values, alias growth and sanitizer cleanup. I retain refusal controls for wrong element types. I record actual gates before claiming completion.

The canonical emitter also lacks functional filter/map/reduce lowering. Existing task_c83543611db54102b1f75f7a94f9e93d owns that separate dependency. Passing first_float alone does not complete parent2578 or the core filter example.

## My representation

I use native storage kind 12 for float arrays and retain an ARRAY edge to the exact FLOAT element shape. Kind 12 is not a NanoISA tag. My existing `narr_t` stores 64-bit words and retains stable handles across growth; float literals and writes use `nvalue_from_float` to copy double bits into those words. Reads return tagged float words and consumers use `nvalue_require_float` to copy them back. I perform no integer arithmetic on float payload words and do not cast an integer data pointer to a double pointer.

My storage predicate is named `word_array_storage` because shared physical allocation does not mean shared element type. Existing exact literal/write/call shape checks distinguish int, bool and float arrays. My root traversal recognizes kind 12 both in active frames and in record/boxed edges, then marks the same owner before the unchanged sweep.

## My current checks

At implementation checkpoint `57e23817`, ten focused and adjacent methods pass with GCC in 7.021 seconds and Clang in 8.666 seconds. Generated native programs use ASan, UBSan and leak detection. The four new methods exercise signed zero through a typed helper return, fractional literals and filled arrays, empty append, mutation aliases, missing-read tags, and a record-retained array across allocation churn. Both C-seed and canonical source emitter outputs execute in VM/native; wrong source element declarations preserve previous output.

I retain initial development evidence separately: the first native check identified an omitted float constructor whitelist; the first source build identified an omitted type-admission branch; a source fixture incorrectly assigned the void result of `array_set`; and the expanded churn test initially lacked a Python concatenation operator. Corrected checks above pass. These observations do not characterize the historical product compiler failures.

Full native/shape and emitter gates are pending. Inferred float-array literal admission also needs its explicit source branch and positive control before readiness. I do not close this child or parent2578 from the current focused results.

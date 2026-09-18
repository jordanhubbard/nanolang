# My native float-array contract

I record this contract before implementation under `task_ef670a36b03a4df388594c8f1500bd85`, a bounded prerequisite of task2578.

I admit exact `array<float>` source declarations and NanoISA TAG_FLOAT array constructors. I preserve float element shapes separately from int and bool, including empty arrays. My native carrier must preserve IEEE double values and signed zero without numeric conversion or incompatible pointer aliasing. I reuse existing aggregate lifetime accounting, stable handles and root closure rather than changing collection policy.

I preserve exact float writes, consumed operand evaluation order, alias-visible mutation, helper returns and root lifetimes. Raw missing array reads remain void; concrete float consumers retain exact checked tags. Integer/bool arrays never become float arrays by inference. Unsupported nested arrays, new host array ABI kinds and LLVM/Wasm heap admission are outside this child.

I require ordinary literal/empty/filled/read/write/append/length and call/return controls through the source emitter and same-module VM/native execution, including signed zero, fractional values, alias growth and sanitizer cleanup. I retain refusal controls for wrong element types. I record actual gates before claiming completion.

The canonical emitter also lacks functional filter/map/reduce lowering. Existing task_c83543611db54102b1f75f7a94f9e93d owns that separate dependency. Passing first_float alone does not complete parent2578 or the core filter example.

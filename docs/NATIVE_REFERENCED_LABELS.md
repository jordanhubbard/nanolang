# My referenced native branch labels

I record `task_3d5492ef63924b4d91a11364a80fde83` before implementation.
The unchanged range fixture at source `553a57bb` reaches strict GCC unused
labels in generated `nl_bounds`; `/tmp/nanolang-range-arity-paired.log`
retains the refusal before native binary execution.

The emitter prescan marks targets of every decoded jump, while terminated
unreached instructions are later skipped. A target with only a skipped
incoming jump may still be visited by fallthrough and emit an unused label.
I preserve the existing target/stack-join/termination algorithm. During body
emission I retain byte offsets and lengths of generated label prefixes, and
mark destinations of every actually emitted PC goto, including conditional,
forward, backward and end-of-body edges. After emission and before declaration
insertion, I blank only unreferenced prefixes, retaining their statement and
newline. Offsets, not pointers, survive output-buffer growth.

I add no false jumps, warning suppression or new control-flow reachability
rules. Every allocation is checked and released on success/refusal. I require
ordinary zero-trip/range, live backward/conditional joins, skipped terminal
following JMP controls, strict GCC/Clang execution and adjacent native/shape
checks. Historical compiler crashes and their artifacts remain excluded.

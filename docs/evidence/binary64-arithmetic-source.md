# My bounded source scalar arithmetic qualification

I apply my shared scalar binary64 policy to the main interpreter, both optimized
callback evaluators, and both legacy C emitters. I snapshot binary operands once,
in source order. I generate the selfhost runtime text from the same checked C
header. Exact FLOAT arithmetic globals use ordered startup; local function
prototypes precede C-seed startup calls. I leave transport, negation, comparison,
array operators and external math unchanged.

I tested production through `32119b8f` and frozen harness `fb160ece` on Linux
AArch64 with GCC13.3 and Clang23. My fresh source bootstrap at `fe1d376b` passed
both stages, ordinary hello and installed-compiler checks. This is not a compiler
fixed-point or full product acceptance claim. [My raw evidence inventory](binary64-arithmetic-source.json)
records source/tool pins, retained logs and actual typed-opcode assembly hashes.

## My route boundaries

| Source route | Exact scalar and ordered globals | Float callback source |
| --- | --- | --- |
| Main interpreter | Passed | Passed literal arrays; separate dynamic-array test reaches optimized map/reduce |
| C-seed native legacy C | Passed | Passed |
| Selfhost Stage1/Stage2 legacy C | Passed | **Blocked**: retained generated-shadow array-kind failure from int-only reduce selection |
| C-seed canonical VM | Passed | Passed |
| C-seed canonical native C | Passed | Explicit FUNCREF refusal, previous output preserved |
| Selfhost Stage1/Stage2 canonical VM/native C | Passed | Passed with explicitly typed reduced local; direct reduce-result bit observer remains checked refusal |
| All three canonical producers through LLVM/O2 LLVM/Wasmtime/Node | Passed scalar-only source | Not claimed |
| Separate C-seed public `--target c` | **Blocked**: raw operators and unresolved bit calls in retained emitted text | Not claimed |

I verify canonical scalar modules contain all four typed F64 arithmetic opcodes
and bit observers. My observers compare integers, not floating equality. I cover
signaling/quiet NaNs, signed-zero division precedence, input/negation bits,
rounding, subnormals and a contraction-sensitive chain. Ordinary f64_* functions
and temporary-like source names remain usable. Globals cover immutable/mutable
bindings, earlier references, unary/nested arithmetic, and distinguishable 12/45
operand-order traces.

## My completed bounded gates

I passed five supported-route methods with GCC in 52.929s and Clang in 37.711s.
Legacy generated programs use O2, contraction enabled and UBSan; generated
native NanoISA C uses O2, strict warnings, ASan and UBSan. I passed five adjacent
helper/source-bit methods in 10.092s, all 120 existing interpreter cases, and a
separate direct dynamic-array test with 24 optimized callback result checks and
8 unchanged input patterns. Interpreter objects are the normal build; I do not
claim whole-interpreter sanitizer instrumentation. I compare the actual emitted
runtime header byte-for-byte for native C-seed `--keep-c` and both selfhost
`--target c` products. I verify frozen source/tools after gates and retain the
unchanged historical PR720 tool/library identities.

## My retained failures and corrections

My initial bootstrap refused a private cross-module runtime provider; I exported
it and retained both logs. Initial source fixtures used `array<float>=` (lexed as
GE) and a nonexistent float_to_int name; I corrected their whitespace and
operand signature. The identity test initially selected C-seed's separate
`--target c` backend; I recorded that missing public route rather than count it
as the native legacy emitter.

Adding a Make test target caused a test-eval prerequisite to rebuild Stage1.
I retained that attempt, waited for completion, captured new hashes, and repeated
affected source gates. Final interpreter checks invoke the exact captured test
compile commands directly, avoiding incidental compiler rebuilds.

The corrected global test exposed a C-seed declaration-order compilation error;
I recorded task74b6 before moving existing prototypes ahead of startup emission
and performed a fresh bootstrap. The later aggregate source gate ran five
methods in 65.319s with two selfhost legacy callback failures. A float callback
was passed to integer-only nl_reduce and its shadow failed before publication.
I retain that gate and do not replay those artifacts. My separately named
supported-route test excludes that blocked route explicitly; its passing result
is not full callback acceptance.

Parent `task_50090616040044dfa3bba1ab93a9f6d1` remains open for public C target
`task_070dbdb1d2e04a36b09a5eab25ddddb3` and callback boundaries
`task_d0997d4a11184689ac99a91b430340de`. Array policy task3717, full reconstruction
and release acceptance remain open. I do not admit reconstructed binary float
arithmetic in this change.

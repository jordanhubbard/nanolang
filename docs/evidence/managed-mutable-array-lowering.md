# My shape-checked mutable array lowering

I connect task41's reviewed runtime checkpoint (PR708, merge699f4011) to actual
native LLVM and import-free Wasm instructions. My production checkpoint is
ef75ddfb; 421574d3 includes the separate wrapper link repair and enum control.
My contract is `../NANOISA_MANAGED_MUTABLE_ARRAYS.md`. Aggregate488 and managed51da
remain open: nested/cyclic/nominal children, unsupported packed coercions,
deferred scalar transfers and host contracts are outside this bounded result.

I admit ARR_NEW/PUSH/SET/POP only after ordinary verification, closed managed
profile checks and successful whole-module element-shape analysis. I retain
runtime tag, bounds, allocation and ownership checks. My old read-only split
route is unchanged; mutable modules prepare boxed split storage before
publication. GET retains before frame consumption; POP transfers the child
before consuming its array; PUSH/SET retain the edge and consume their operands,
returning the same array owner only on success. Scalar out pointers preserve my
native/Wasm ABI. First failure survives frame cleanup and public status return.

## My checked targets

On 421574d3 I passed 51 managed target methods in 71.704 seconds, including six
new methods and all 45 existing methods. The same gate passed shared profile
checks (including new admitted/unresolved/nested API cases), runtime2,
package2, array/boxed/packed core3 and string core3. A later test-only expansion
passed seven focused methods in 12.371 seconds: it checks retained global
contents on entry after a failed call and proves prepared split SET succeeds
with native malloc forced to fail. No production changed for that expansion.

My new checks cover capacity8/16/32/64, finite packed coercions and endpoints,
subnormal/signed-zero values, boxed VOID/int/U8/float/bool/string/enum leaves,
local/global/call aliases, same-child SET, transferred last-child POP, optional
GET/POP, full-width bounds errors, exact-tag errors, initializer order,
repeated-entry/fresh-instance behavior, and prior-output refusal for unsupported
shapes/transfers. Existing private packed tests retain exact float payload-bit
coverage; ordinary value equality alone is not that evidence.

My value/alias matrix runs in normal NanoVM, instrumented native LLVM, Node
Wasm with zero imports, and Wasmtime; the paired helper also exercises
nvm2wasm's published artifact. Reentry, failure and finite-memory controls use
the explicit native/Node harnesses described above. Error controls compare corresponding VM/managed categories,
not numerical equality of their status enums. Native wrapped allocation failures
cover constructor and growth/child cleanup, followed by successful reentry.
Actual bounded Wasm memory exhaustion preserves a committed global array and
allows repeated entry and terminal reclamation. The private runtime/package
controls retain rollback, descriptor relocation and cross-target ABI evidence.

## My link closure

I added the shape analysis to all explicit verifier source consumers: main
Makefile, both examples commands, generated wrapper objects, nanoisa manifest
and ForthSEE manifest. The standalone injected-allocation test excludes the
new normal object and resets its private budget before comparing profile
results. This avoids duplicate symbols and keeps profile observations separate
from the deliberately failed analysis call.

Real ForthSEE native import and examples shared-library load passed (two methods,
7.771 seconds). The real generated wrapper then compiled and executed. I built
a fresh canonical native seed with a new module cache, ran its help, emitted
`examples/language/nl_hello.nano` with normal shadows, verified the bytecode and
executed it in NanoVM. Output was exactly `Hello from NanoLang!` plus newline.
This is a host-link/ordinary publication gate, not a new compiler fixed point.

## My retained corrections

I retain first failures separately. Production -O3 compilation exposed a
previously standalone analysis diagnostic-copy truncation warning; I bounded
its copy explicitly. Initial focused fixtures used a decimal subnormal token
rejected by the assembler and counted a static literal as a live allocated
string. Corrected fixtures derive the subnormal with typed division and use a
fresh concat child. Missing-tool/old-path and unsupported verify-subcommand
attempts preceded the corrected explicit commands; they are not compiler bugs.

The first full managed run passed 50/51 methods: an old blanket mutation-refusal
fixture expected newly supported instructions to fail. I retained unsupported
nested/packed-shape refusal and old-output assertions instead; admitted behavior
has independent positive tests. A real wrapper link exposed the existing
missing local_bindings.o dependency (task76d9). I recorded it before adding the
object and repeated actual wrapper execution. Neither failure is described as
an unexplained infrastructure incident.

Local logs are `/tmp/nanolang-mutable-lowering-*`; fresh seed artifacts/logs are
`/tmp/nanolang-mutable-lowering-seed/`. My native toolchain flag is
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
Darwin sanitizer7ba and historical evaluator791a remain separately open; I did
not replay either artifact or infer a cause.

## My final integration

I rebased onto main94cca515 (including exact F64 canonical transport710 and
component inventory711). On source a4e6bbeb, my reviewed lowering/verifier,
analysis and wrapper production files are byte-identical to 421574d3. Integrated
shape12 methods passed in 1.344 seconds, profile API checks in 0.493 seconds,
all seven new target methods in 12.151 seconds, and the real wrapper passed.
I rebuilt the canonical native seed with another fresh module cache, ran help,
then emitted, verified and executed the ordinary hello with normal shadows.
The output remains exactly `Hello from NanoLang!` plus newline. Integrated
artifacts/logs are `/tmp/nanolang-mutable-lowering-seed-integrated/`; gate log is
`/tmp/nanolang-mutable-lowering-integrated.log`. I did not repeat an unrelated
full compiler fixed point or claim Darwin acceptance.

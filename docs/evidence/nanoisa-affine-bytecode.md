# My bounded affine bytecode evidence

I implement `task_4d22d17c96864d9fbc6031cfba2dc927` after PR545. My separate
`nvm_affine_analyze_function` API consumes actual decoded function instructions,
checked ownership declarations and cloned affine local state. It does not
alter `nvm_verify` or the direct VM/native ownership refusals.

I distinguish scalar stack values from record observations tied to a checked
parameter root. A checked scalar field read removes the observation and
produces a scalar. I refuse observation duplication, permutation, stores,
returns and calls. Existing aggregate operations do not become implicit owner
moves. An owned parameter remains an obligation after observing its fields
and prevents an otherwise scalar return.

I propagate reachable branches with exact local liveness, scalar tags and
observation provenance. Ordinary initialized loops and equal diamonds pass;
different roots, scalar tags and unavailable-local states fail joins, including
back edges. Unsupported operations are refused even in dead code. Explicit
return checks reject extra values, escaped observations, wrong result tags and
unconsumed owners. I require Boolean branch conditions in this bounded slice.

I pass 300 checks over ordinary assembled functions and 419 checks with
allocation failures injected into the new analysis. These include valid shared
and exclusive parameter readers, ordinary nonresource record controls, scalar
loops, float scalar arithmetic, negative transfer/escape cases, and the exact
4096-instruction boundary. The 4097-instruction, 257-local and 257-stack-value
cases are explicit refusals. The 419-check variant also passes focused
ASan/UBSan with leak detection: the new analysis, affine state, ownership
contracts and place queries are instrumented; other linked objects are ordinary
builds. This is not a whole-VM sanitizer claim.

The existing 157 ordinary and 182 allocation-injection state checks also pass.
My declaration artifact test continues to execute ordinary code while refusing
resource contracts in VM and native publication. The two real Forth host
build/load methods, five regular/daemon wrapper links and seven publication
methods pass with the new source included in all five explicit build consumers.

My [reference contract](../NANOISA_REFERENCES.md) names the next move/store/
pack/unpack instructions and the `verify_function_impl` integration point.
Their wire codec, stack-owner transitions, caller-place alias substitution,
reference creation/access/end-region operations and actual VM/native semantics
remain required. This analyzer's entry-reference assumptions are not a proof
of caller alias compatibility. No reference producer or execution path is
enabled by analysis success. Float-record lowering remains task93574; my full
release and ownership parents remain open.

Local evidence is in `/tmp/nanolang-affine-bytecode-final.log`,
`/tmp/nanolang-affine-bytecode-sanitizers-final.log` and
`/tmp/nanolang-affine-bytecode-host.log`.

I also build the actual canonical native compiler seed through the updated
NanoISA host manifest. It runs help and emits an ordinary hello artifact;
that artifact verifies and executes in NanoVM. The logs are
`/tmp/nanolang-affine-bytecode-seed.log` and
`/tmp/nanolang-affine-bytecode-hello.log`. This dependency-closure check is not
a new compiler fixed-point or runtime-reference acceptance result.

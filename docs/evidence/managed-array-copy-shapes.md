# My literal and slice shape checkpoint

I extend the non-admitting analysis at production checkpoint `e51f2bc9`, based
on main `43ef9d3d` with private runtime PR718. I count every literal input and
preserve its declared storage; every input must satisfy the existing portable
packed matrix or boxed leaf set. My literal transfer explicitly sets both
counted pops and one result, matching the ordinary verifier.

I create each slice origin from its function, instruction PC and possible
source declaration. I preserve all alternatives at joins. Copies do not alias
source origins, but I weakly union possible source contents into the copy
summary. Repeated calls, recursive calls and copies of earlier instances at the
same site cannot reset old contents. Later source writes may overapproximate
copy contents; copy writes do not mutate the distinct source summary. Derived
origins share the existing 64-origin cap and return LIMIT on exhaustion.

My 18 focused methods pass in 2.367 seconds. Each analysis runs through ordinary
and Clang ASan/UBSan probes, compares reports and canonical module text, and
checks unchanged profile results. Accepted ordinary fixtures also run in NanoVM.
I cover literal counts 0/1/8/9/17, all finite packed pairs, boxed leaf tags,
fresh copies, mixed declaration joins, repeated/recursive copy sites, global
reentry, unknown origins, origin/stack caps, deterministic analysis allocation
failure and preservation of prior LLVM/Wasm output when admission refuses.

My first focused run found an incomplete stack-effect override in this new
implementation: I set literal pops but left the dynamic pushes marker. I kept
`/tmp/nanolang-copy-shapes-focused.log`, corrected the transfer to `count -> 1`,
and retained `/tmp/nanolang-copy-shapes-corrected.log`. This was a checked
analysis refusal, not an executed malformed module or a historical replay.

My result establishes only shape facts. No opcode/profile/emitter admission
changes here. Counted-root consumption, slice operand cleanup in generated
code, and paired actual native LLVM/Wasm instruction acceptance remain task
b702. Full aggregate488, managed51da and release acceptance remain open.

My frozen adjacent run passes `test-llvm-managed-strings` and
`test-verifier-profiles`: all 52 managed instruction methods pass in 79.288
seconds, along with required runtime/package/core and shared-profile targets.
I retain `/tmp/nanolang-copy-shapes-adjacent.log`. I did not change compiler
source or run a new compiler bootstrap, Darwin gate or full release acceptance.

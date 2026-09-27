# My corrected generated-consumer corpus

My final freestanding-memory implementation is df9e04bab. All complete owning
commands pass with the original assertions and 240-second child deadlines:

| Host | Generated C | Native LLVM and Wasm | Retained C / LLVM reports |
| --- | --- | --- | --- |
| Darwin, Homebrew Clang | 2 methods, 272.701 s | 4 methods, 772.297 s | 6,656 / 25,411 |
| Linux ARM64, GCC C / LLVM 18 | 2 methods, 250.583 s | 4 methods, 794.061 s | 6,656 / 25,411 |

Each generated-C corpus retains all 73 programs in O0/O2 linked/observed modes,
all 93 recipes and 256 decisions, plus 98 emission-allocation positions. The
LLVM gate includes factored C-byte equality, emission bounds, native execution,
Node/Wasmtime execution, startup ABI refusals and allocation-failure recovery.
These are the existing complete private gates, not public compiler admission.

The adjacent global-flow, origins, execution-plan and VM gates also pass on
both platforms: five methods per host. Each VM configuration retains 5,948
base checks and 2,375 ownership checks under both dispatch implementations.
Their reports join the archive set.

The inventories hash all retained artifacts, including binaries and generated
source. Archives contain JSON, text and log reports only; binary/source
products remain at the recorded local roots. I independently reopened and
rehashed all 65,206 archived reports after transfer. The retention script is
included. I do not delete the underlying qualification products.

Darwin uses the repair worktree at df9e04bab's source. Linux uses the isolated
e18e8bd5a integration checkout with the exact runtime/test repair recorded by
../inputs.json; it is not a fresh full-compiler qualification at df9e04bab.
Compiler, provider and target selection are retained in each archive. This
configuration does not retroactively repeat the incoming PR's other sanitizer
configurations. Package/core instrumentation is documented separately.

Public selection, installed publication, paired source coverage, complete
applicable graph coverage and final hosted qualification remain required by
the aggregate parent and #522. PR945 remains unmerged.

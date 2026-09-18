# My bounded Darwin arithmetic and array-graph qualification

I track `task_209f1e88044b4e29886228dc7357188d` against canonical source
`25a685adfb4f9bf32c68c9e6de04eb0298673c45` (PR732, including PR731).
This report contract changes no production source. My earlier e9 product
candidate and its evidence remain immutable; PR522 and publication stay held.

I qualify these newly integrated boundaries on the already authorized Darwin
peer in a fresh isolated detached checkout. I preserve existing worktrees,
tools, stashes and historical failure artifacts. I record the exact Git head
and tree, clean status, OS/architecture, compiler/linker/SDK/runtime versions,
selected commands, executable hashes, raw logs and each phase's outcome.

My ordered phases are:

1. Inspect prerequisites without changing the frozen source. Identify the
   installed C/LLVM toolchain, Node, Wasmtime/Python dependencies and supported
   linker. Missing prerequisites are an explicit incomplete outcome.
2. Build `nano_vm nvm2c nvm2wasm nanoisa_dump` from this exact source with the
   supported Darwin build tools. Retain actual compiler flags and tool hashes.
3. Run `python3 -m unittest -f -v tests.test_binary64_arithmetic`: two helper
   methods, including exact standalone generation, sanitizer/optimization/LTO
   modes and target guards.
4. Run `python3 -m unittest -f -v tests.test_binary64_arithmetic_backends`: four
   methods spanning scalar and managed paths, ordinary VM/native/LLVM/Wasm
   execution, exact arithmetic bits and unchanged operand transport.
5. Run `python3 -m unittest -f -v tests.test_llvm_managed_graphs`: eight methods
   covering actual generated nested owners, cycles, finite-memory pressure,
   persistent globals, first errors and cleanup across normal target routes.
6. Seal raw logs and a report with SHA256 hashes, elapsed times, selected tools,
   method counts, any skipped routes, first failure and unchanged source state.

I keep existing per-command deadlines, assertions, sanitizers and memory bounds.
Each phase has an outer 900-second bound and stops on its first failing test.
I do not treat a skipped target or unavailable dependency as qualified. I retain
the first failure and report it before changing source, tool selection or the
test harness; any correction receives its own recorded contract and qualification.
The helper's Clang LTO linker selection is part of the observed harness, not
permission to silently remove LTO on Darwin.

These tests use fresh ordinary modules and public verification/translation.
I do not replay historical crash/timeout artifacts. This is neither a full
quick suite nor a compiler bootstrap, new release candidate or release gate
completion. Source-wide arithmetic policy, full aggregate/runtime scope and
the unchanged affine-example blocker remain separate open obligations.

## My recorded prerequisite correction

The peer's first inventory finds neither the `wasmtime` executable on PATH nor
the Python `wasmtime` module. It stops before build or tests. I preserve that
terminal inventory report and hashes; it is not a passing qualification.

I authorize the existing Homebrew toolchain to install its Wasmtime formula,
after inspecting formula identity, and a dedicated temporary Python virtual
environment to install the Python Wasmtime dependency. I record package
versions, installation commands and logs. I do not modify shell startup files,
project source, existing virtual environments or the frozen product tools.
Per-command PATH selects the new environment and installed CLI explicitly.
If either installation fails, I retain that outcome and stop the dependent gate.

With both prerequisites verified, I start a new independently logged execution
of the original ordered phases at the same source pin. All assertions, LTO,
sanitizers, inner/outer deadlines and target routes remain required. The missing
dependency outcome remains separate from the corrected qualification.

## My explicit LLVM tool selection correction

My first corrected build stops after 0.821661 seconds with exit 2 because
`scripts/embed_managed_runtime.py` cannot execute `opt`. The peer confirms
that `/opt/homebrew/opt/llvm/bin/opt` exists outside the selected PATH. No test
method ran. I retain the separate build log and inventory; installing Wasmtime
did not qualify my build.

For a new logged qualification I prepend the existing Homebrew LLVM bin
directory to the dedicated venv and Homebrew PATH. I inventory the selected
clang, opt, llvm-link and linkers, including exact versions and executable
hashes, before starting. I use this explicit PATH for every phase; I do not
change shell startup files or install another LLVM. My source pin, test methods,
assertions, sanitizer/LTO requirements and deadlines remain unchanged. I stop
on the first new failure and preserve all previous terminal reports.

I retain both terminal prerequisite outcomes and installation logs in
[evidence](evidence/darwin-arithmetic-graphs-732/sealed-prerequisite-manifest.json).
The coordinator copied the sealed files over SSH and independently verified
every SHA256 against the peer report. No test method passed in either run;
the task remains open for the newly selected LLVM toolchain qualification.

## My Darwin LTO linker correction

My explicit LLVM build passes in 4.004839 seconds. The first helper method
then fails in 0.891003 seconds because its unconditional Clang `-fuse-ld=lld`
selects ld64.lld23, which rejects SDK27 arm64e.x1 TAPI targets. I preserve this
terminal outcome under task_39536e6a885e47b88bed4445bb289e5e before repair.

I retain the compiler's native default linker on Darwin and keep the existing
explicit LLD selection on other Clang hosts. I qualify Darwin with Apple clang
and Apple ld, retaining `-flto`, ASan/UBSan, all optimization modes and all 133
bit assertions for both direct and embedded source. I change no production
helper or expectation. A fresh detached qualification uses the corrected test
commit, with production source still identical to25a685ad; I record both pins.
All remaining backend/graph routes and deadlines remain required. This is a
linker selection correction, not qualification of LLD against this newer SDK.

At corrected harness b834e804, my unchanged Linux helper methods both pass
with GCC in 1.120 seconds and Clang in 1.313 seconds. Each run retains direct
and exact embedded-source O0/O2/O3-contraction/LTO sanitizer checks and target
refusals. These host results do not establish Darwin acceptance.

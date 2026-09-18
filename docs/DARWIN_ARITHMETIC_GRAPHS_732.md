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

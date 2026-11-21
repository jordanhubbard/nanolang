# My NanoLang passive closed-call analysis

I implement this #1142 prerequisite in `582e57fe6` on
`feat/5.1-nanolang-translator`, draft PR #1144. My pure NanoLang implementation
is `src_nano/nvm2c/passive_calls.nano`. My existing C analyzer and product
selection are unchanged.

I analyze a concrete callee under scalar actual arguments. I decode its whole
instruction span, follow reachable direct and tail calls, check stack joins,
and intersect initialized-local sets until convergence. I check local reads
after convergence, including loop predecessors. I refuse recursion, captures,
foreign/indirect calls and reachable operations outside the existing scalar
allowlist. A decoded but unreachable effect does not affect this summary;
ordinary module verification must still check that instruction.

My descriptor API preserves the current narrow function representation and
does not infer scalar values from signature tags. Its caller must establish
scalar actual arguments. Passive-node validation is not connected yet: guarded
inputs, node/block ranges, dependency/read accounting, topological order and
whole-module relationships remain required before passive admission.

## My paired checks

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Final callee corpus | 1498 cases match on five routes | 1498 cases match on five routes |
| Explicit callee expectations against actual C analysis | 415 pass | 415 pass |
| Both versioned scalar opcode allowlists | 512 decisions match per route | 512 decisions match per route |
| Actual C analyzer and ISA decoder, ASan/UBSan | All 1498 cases pass | All 1498 cases pass |
| Embedded-identifier gate | Five methods pass | Five methods pass |

My final differential run takes 119.954s on Linux and 84.818s on Darwin. These
observations are not a performance comparison. The first Linux run passes
1140 cases before I expand the controls; I retain that log separately. The
implementation does not change between those runs.

The five routes are C-seed native plus both C/NanoLang producers through NanoVM
and sanitizer-instrumented generated C. My driver is compiled from the actual
NanoLang analysis, not a foreign wrapper around the C oracle. Each route also
checks all 256 opcode values under both version-1 and version-2 scalar profiles.
All named functions have useful shadows, executed during fixture builds.

I retain explicit scalar/void result counts, argument consumption, direct/tail
calls, repeated checked callees, self/mutual recursion, depth checks before a
cached result, uninitialized reads, initialized/uninitialized joins and loops,
stack-height disagreement, missing targets, instruction-interior targets,
unreachable effects and unknown/truncated instructions. Local-set controls
exercise indices 0,31,32,63,64,127 and65534, plus the maximum 65535-parameter
descriptor. Random cases use a fixed seed. A matching summary is not a proof
of general compiler correctness or a termination theorem.

## My resource boundaries

I keep the existing limits, without test-only replacements:

| Limit | Passing control | Refused control |
| --- | --- | --- |
| 64 active call frames | Chains of 63 and 64 callees | Chain of 65; cached leaf reached at depth 64 |
| 65536 decoded instructions per callee | 65535 NOPs and RET | 65536 NOPs and RET |
| 2097152 local-state words per callee | 2048 instructions, 65535 locals, 1024 words per instruction | 2049 instructions with the same local shape |
| 1048576 worklist steps per callee | Generated cyclic graph converges in 4263 steps | Larger generated cyclic graph reaches 1048577 steps |

Both worklist graphs use only balanced scalar constants/stores, conditionals,
NOPs, jumps and a reachable void return. Distinct initialized-local sets enter
a cycle at different positions. The larger graph remains below instruction,
local-storage and call-depth limits and has no local reads. An independent
Python set-intersection traversal counts its propagation steps. Both actual
analyzers refuse it at the unchanged worklist bound. This tests the production
limit, rather than a reduced imitation of it.

I split each 64-bit local-state word into two nonnegative integer halves and
implement intersection in NanoLang. The word budget retains the original
64-bit counting rule; my representation uses two integer slots per word.
These checks do not simulate allocation failure or establish a whole-process
memory cap, complete leak freedom, concurrency safety or service isolation.

## My oracle, commands and retained inputs

`tests/nvm2c_passive_calls_reference.c` includes the existing `passive.c`, which
includes `passive_calls.inc`, and exposes its private `closed_function` and
`allowed` functions. I add no alternate C analysis implementation. The test
envelope carries explicit descriptors and code; it is not the NanoISA module
wire format. Its narrow-field checks run before either analyzer.

```sh
make -j8 obj/nvm2c_passive_calls_reference obj/nvm2c_module_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_passive_calls
cc -std=c99 -Wall -Wextra -Werror -O1 -g \
  -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  -Isrc/nanoisa tests/nvm2c_passive_calls_reference.c src/nanoisa/isa.c \
  -o .test_output/passive-calls-reference-asan
PYTHONPATH="$PWD" python3 .test_output/passive-calls-collect.py --corpus
make test-embedded-nano-identifiers
```

`make test-nanolang-translator-passive-calls` declares the complete build
dependencies for the ordinary three-route gate; setting the source compiler
environment variable adds the two NanoLang-producer routes. Darwin prepends
`/opt/homebrew/bin:/usr/local/bin` to PATH. Linux retains default leak detection;
I do not claim Darwin LeakSanitizer coverage.

Both hosts reuse the C-seed-built full NanoLang driver from 602dc495e. The C and
NanoLang compiler frontends are unchanged between that pin and 582e57fe6. The
new fixture imports and compiles the new analysis source on every route.
Generated-C routes instrument generated code and link the ordinary AOT runtime;
the independent C oracle sanitizer build instruments both analysis and decoding.
These development producers do not establish installed/no-C-seed execution,
raw Stage1/Stage2 equality or C emission in NanoLang.

My post-run manifests have identical source hashes at 582e57fe6. They record
tool hashes and explicit SDK observations, not a complete before/after SDK
closure. Temporary unittest binaries are not archived. My independent corpus
collector deterministically regenerates all 1498 inputs from the final test,
executes the actual sanitizer oracle and retains every input byte, explicit
expectation and C decision. Those complete corpus records match across hosts.
`retained-inputs.json` binds each compressed source/log/manifest/corpus to its
uncompressed length and SHA256 and its compressed SHA256.

I keep the full passive/ownership/service profile implementation, instruction
and host admission, representation analysis, NanoLang C emission, product and
bootstrap cutover and all remaining #976 release requirements open. No module
is newly authorized to execute by this checkpoint.

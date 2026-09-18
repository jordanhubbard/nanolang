# My existing floating array arithmetic checkpoint

I qualify existing VM, interpreter and paired legacy array leaves under
bd474b339063ea2fe3915b29f12f9c80, ordered scalar-left evaluation under
30c04e6d0ef594ef20bf7f9fec161969, paired typing/opcode selection under
90b473627cffdb98e69d5a5641937d30, legacy declared-call selection under
6ed0cd1295af4a42b04133ddc6d39d98 and the Darwin test-hook prerequisite
8674e6954ba44d3da54ea5809d0bba9c. My
[contract](../NANOISA_AGGREGATE_BINARY64_POLICY.md) precedes each repair.
Full array parent3717 and publication remain open.

I reuse the scalar binary64 helpers for each existing FLOAT leaf: canonical
quiet NaN, positive zero for either zero denominator, and binary64 rounding at
each operation. Integer behavior, mixed VM promotion, array shape/allocation,
aliases and cleanup retain their prior boundaries. Both legacy producers
snapshot scalar-left operands in source order before capturing array length.
C-seed's existing helper fallback retains its type selection; its sequencing
change is independently reviewed, without a new unknown-to-INT substitution.

My paired checker recognizes exact flat int/float arithmetic before scalar
promotion and diagnoses selected mismatches before publication. Canonical
emission uses generic ADD/SUB/MUL/DIV for those arrays. Legacy selfhost selection
also retains exact unbound declared numeric-array call results; any lexical
name, even one with missing type metadata, prevents declaration fallback.
Other legacy array profiles retain their checker path. I add no native-C,
LLVM or Wasm NanoISA aggregate admission from this checkpoint.

## My measured acceptance

| Pin and platform | Actual gate | Result |
|---|---|---|
| acb65661 Linux | Full Stage1/Stage2 bootstrap and installed compiler smoke | Pass, 245.961s |
| acb65661 Linux | Five source methods, all supported producers | Pass, 16.676s |
| 985321f7 Linux | Direct evaluator representations, callback neighbors and VM controls | Pass, 10.575s |
| 61a8dc7a Linux, integrated through PR758 | Rebuilt C/runtime tools and five source methods | Pass, 3.420s + 16.099s |
| 61a8dc7a Linux | Direct evaluator, VM and owned-string controls | Pass, 10.152s |
| d7e04722 Linux, integrated through PR759 | Rebuilt C/runtime tools and five source methods | Pass, 2.491s + 16.927s |
| acb65661 Darwin | Full Stage1/Stage2 bootstrap and installed compiler smoke | Pass, 290.866s |
| acb65661 Darwin | Same five source methods | Pass, 21.261s |
| acb65661 Darwin | Direct evaluator and callback neighbors | Pass; VM prerequisite compile stopped separately |
| 826c1b95 Darwin, acb plus test-hook repair13460195 | Corrected full VM gate | Pass, 6.718s |
| 13460195 Linux | Corrected formatting fault-hook gate | Pass |

The five methods preserve the original four arithmetic/order fixtures and add
12 diagnostic-specific mixed-type publication checks. Sixty-eight integer-bit
cases cover both signs and payloads of quiet/signaling NaNs, infinities, signed
zeros, zero denominators, halfway rounding, overflow and subnormal boundaries.
Each case checks array pairs and both broadcast directions plus unchanged input
bits. Interpreter, C-seed legacy and both selfhost legacy compilers execute;
C-seed and both selfhost canonical outputs verify and execute in the VM.
Ordered effects grow a shared array before length capture and retain aliases,
empty inputs and integer neighbors. No refused artifact executes.

My direct evaluator fixture supplies actual dynamic and legacy VAL_ARRAY values:
408 exact results and 24 empty results pass on both hosts. The existing optimized
callback fixture retains 24 results and eight input patterns. VM tests retain
274541 assertions, callback roots, heap and stack failures, recovery and original
array refcounts/contents. The arithmetic allocation control covers eight
existing generic/dedicated operations and three forms at its instrumented heap
allocation point; I do not claim an exhaustive malloc/realloc sweep. Integrated
owned-string controls retain 600 allocation checks and 150 proof/readiness
checks with nine admissions.

## My retained failures and exact identities

The first c547e2b7 full bootstrap passed, but frozen3087067f source tests refused
eight selfhost broadcast routes because the checker chose scalar FLOAT.
The af38155f corrected bootstrap passed; frozen985321f7 then passed exact bits,
canonical ordering and refusal controls, while two legacy ordering routes
refused generated C because identifier-only selection dropped a declared array
call result. I retain both logs and source/tool snapshots. The final reviewed
call classifier fixes that prerequisite without changing the positive fixtures.

Darwin's first adjacent make stops at strict compilation of the existing
substring fault-injection fixture: the SDK already defines snprintf. Both direct
evaluator fixtures pass in that same run. I retain the incomplete log and add
only a test-local undef before the existing hook. I keep production warnings,
fortification and assertions intact; the corrected VM gate passes in full.

Linux final, integrated and callback-integrated snapshots retain respectively
2032, 2045 and 2046 unchanged source/tool hashes. Integrated Stage1/Stage2 hashes
remain those of the actual acb65661 bootstrap. I do not claim a fresh bootstrap
at either later integration. Darwin source/tool snapshots remain unchanged in
each run; its corrected test fixture is a separate recorded commit. The runner
records explicit Apple Clang build selection and the installed tool inventory.
I copied and SHA-verified all 13 Darwin reports from the peer host.

My [report manifest](aggregate-binary64-policy/report-sha256.json) seals every
retained report and both Darwin runners. Evidence includes before/after hashes,
commands, times and full first failures. Canonical merge precedes task closure.
The broader aggregate backends/shapes, scalar matrix, product and full release
acceptance are separate requirements.

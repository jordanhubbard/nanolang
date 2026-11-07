# I reconstruct bounded direct owned-value calls

I extend my [bounded contract](../../NANOISA_OWNED_LOCAL_RECONSTRUCTION.md)
under #999 on `fix/5.1-owned-calls` from `36930e1d3`. I preserve original
`nvm_verify` admission; retained signature queries and affine consistency do
not grant executable authority. My region tree preserves nominal parameters,
owned results, live parameter slots, direct calls and ordered scalar snapshots.

## Source evidence and failed assumptions

`probe.nano`, `probe.log` and `probe.nasm` retain my first actual C producer
probe. It returns 24 in NanoVM and demonstrates owner construction, forwarding,
consumption, and a staged owner below scalar argument arithmetic. Reconstructed
Nano compilation also passes (`rebuilt-probe.log`).

`initial-calls.log` records my first fixture exceeding the source producer's
budget when its shadow entry is included. `calls-second.log` records a generated
entry wrapper exceeding that same budget. `calls-third.log` and
`calls-fourth.log` show that removing the wrapper alone does not account for
arithmetic helper declarations and their shadow entry. I preserve those failures.
I now emit the retained entry directly, count all generated functions, and refuse
Nano publication above seven declarations. My accepted fixture fits six retained
functions plus one arithmetic helper. The producer budget remains eight.

## Qualification

- Eight owned methods pass with the C seed and retained Stage1/Stage2 producers.
  `exact-final-owned.log` is the final run after rebuilding my facts helper.
  Each producer supplies actual source output for both the local and call tests.
- I compare original NanoVM, reconstructed C at O0/O2 with ASan/UBSan, and
  reconstructed NanoVM plus native output at O0/O2 with ASan/UBSan. The call
  fixture returns 74 after distinct Left/Right records, mixed arguments,
  constructor results, forwarding, BOOL results, nested and repeated calls.
- I compare regenerated layout identities and require every original positional
  parameter/result signature in the regenerated module. Generated arithmetic
  helpers may add their own scalar signatures. I count explicit materialized
  calls in each output and reject VM/goto/dispatcher source substitutions.
- Nine CRC-valid module mutations refuse both outputs atomically: entry and
  helper cycles, missing callee, wrong nominal argument/result, borrowed
  parameter mode, duplicate/lost owner and missing resource authority.
- Six scalar stack operations cannot consume, copy or rearrange a staged owner
  in the independent region analyzer. A valid helper branch is refused on both
  surfaces. A valid over-budget source is refused before replacing Nano output.
  The original four local-resource tests retain their authority/region controls.
- Eleven scalar reconstruction neighbors pass in 55.665 seconds
  (`scalar-neighbors.log`). Strict Clang C11 compilation of the facts helper
  passes with `-Wall -Wextra -Werror` (`final-build.log`).
- The existing `test-owned-reconstruction` target now runs both owned test
  modules, so the existing hosted producer gate includes this slice.

`providers.sha256` identifies copied tools and retained object inputs. I copied
C tools and objects from the aggregate before its later bootstrap changes, then
stopped reading aggregate products. Stage1/Stage2 and the scalar native C driver
come from `/Users/jkh/Src/nanolang-51-owned-reconstruction/bin`, whose prior
receipt records source `28e50b802` and bootstrap `run-m17ygl7e`. These are retained
producer checks, not freshly built generations at `36930e1d3`. I built only the
small changed facts front end and individual test outputs; no competing full
bootstrap ran.

This is bounded macOS evidence. Borrowed calls, recursion, ownership control
flow, nested owners, arrays/unions, generic and cross-module paths, fresh combined
generations and Linux/Darwin integration remain open under #999 and 5.1.

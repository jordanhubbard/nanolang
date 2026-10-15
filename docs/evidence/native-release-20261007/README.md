# My Darwin native release checkpoint

I qualify these local repairs against `1bbae0bb051efb1c6b356a98d12be0494b9ff935`:

- I discover installed OpenSSL headers through pkg-config, with validated Homebrew fallbacks and unchanged explicit Make overrides.
- I count record-frame allocations inside generated functions, without miscounting runtime root-table frees. I emit map-test counters only when that harness observes them. Stack bounds, exact frame peaks, failure injection, strict warnings and leak checks remain enabled.
- I defer array storage inference when a tagged element reaches an unresolved container. Recursive string, Boolean, integer and floating-point results execute in NanoVM and sanitized native C.
- I preserve tagged Boolean values at forward short-circuit joins. Missing-value tags, both edge orders and lower stack operands remain tested; backward widening stays refused.
- I discard derived scalar join plans when global discovery resets their parameter and result facts. The ordinary pass rebuilds those plans.

My [manifest](manifest.json) hashes the changed source and retained evidence. The original compiler module is retained as `compiler-seed.nvm`; its absolute native import paths belong to this checkout and are not a relocatable release artifact.

## Results

| Gate | Observed result |
| --- | --- |
| Original `make -j2 test-one-ir-compiler` | Stops during compilation: Homebrew reports absent OpenSSL 4 while OpenSSL 3 is installed. |
| Same gate after dependency repair | 76 methods, 65 failure reports: 60 unsupported Apple LSan reports, three allocation-probe failures and two compiler-product failures. |
| Homebrew Clang leak probe | Clean allocation/free exits zero; an intentional 32-byte leak exits one with a LeakSanitizer report. |
| First Homebrew native-adjacency run | 72 of 74 methods pass; two test-harness compilations reject unused counters under Clang 23. |
| Corrected native-adjacency run | All 78 methods pass in 112.180 seconds, including four added regression methods. |
| Final structured-C gate | All 2,431 checks pass, with its opcode, sanitizer-driver and shape prerequisites. |

I select `/opt/homebrew/opt/llvm/bin/clang` with a temporary `cc` launcher at the front of PATH for fixture compilation. The translator and ordinary compiler tools use the existing build. I do not claim a fully instrumented compiler build or Linux qualification.

The native-adjacency run uses the same fourteen Python modules as `test-one-ir-compiler`, excluding exactly `test_compiler_bytecode_to_native_to_program` and `test_selfhost_emitted_compiler_to_native_nanoisa_product`. Those two failures remain release blockers. The exact exclusions and method results are retained in [the report](native-llvm-corrected.json) and [raw log](native-llvm-corrected.log).

I run the final C gate as `make -j2 -o nvm2c test-nvm2c` after building the corrected translator. The `-o` option preserves that already-built translator while the independent Python suite uses it; it skips no structured-C test. [Raw result](nvm2c-final.log).

## Remaining compiler blockers

My original compiler translation rejects a string-array parameter after prematurely selecting integer-array storage. After correction, it reaches a tagged Boolean join in `nisa_emit_globals`, then the stale optional-Boolean plan in `par_closed_function`. The retained reduced modules reproduce the array and stale-plan refusals before correction and execute after correction.

The current full module reaches a separate aggregate conversion refusal: an optional source cannot widen an exactly constrained string destination. [The final diagnostic](compiler-translation-reset.log) retains the shape IDs. A diagnostic translator built outside the product maps the source to `nb_owner_field_tag` (function 964, offset 17), its use in `nb_expr` (999, offset 1373), and a conversion from `nb_record_has_owner` (1026, offset 689). [Trace](shape-trace.log). I retain the refusal rather than relaxing the exact-string constraint.

Self-hosted compiler emission separately stops after its original ten-second shadow deadline. I preserve the first failure in [the full gate log](one-ir-openssl-corrected.log); I have not increased the deadline or claimed a fixed point.

GitHub API access initially fails, then recovers. I push commit `47b1cfeef` and open [draft PR974](https://github.com/jordanhubbard/nanolang/pull/974). MAC task creation and a subsequent ready-task query still return `Operation not permitted`. Hosted acceptance, ledger reconciliation, integration, complete platform gates and publication remain unverified. This checkpoint does not release 5.1.

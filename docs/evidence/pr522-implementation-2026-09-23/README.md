# My resumed PR522 implementation

I retain the completed hosted run [35795377922](https://github.com/jordanhubbard/nanolang/actions/runs/35795377922) at `8c0b013604a642b6c69197d49d65e51b5d826469`. `hosted-failures.log.gz` contains the failed-job log export; the adjacent JSON seals its uncompressed bytes. `failed-cases.json` indexes reported test failures without replacing their terminals.

I resume the complete canonical-product work under the 2026-09-23 request. Selected ordinary payloads precede nested and owned union lowering. I keep every existing acceptance case and require new complete gates before claiming PR522 is resolved.

## First corrected slice

I preserve the concrete identity of complete ordinary selected patterns. My nine raw-producer cases exercise integer and string execution, empty variants, reordered fields, missing/duplicate/unknown fields, wrong variants and an unselected union. All eight emitter-driver methods pass. Fresh two-stage bootstrap, mandatory shadows and installed-compiler checks pass. Six unchanged selected-pattern methods pass across C-seed, Stage 1 and Stage 2. All ten canonical-publication methods pass with Stage 1 and Stage 2.

I replace two obsolete nested-integer-array refusals with a currently unsupported nested-record result and add VM/native execution for the now-supported integer case. The two native artifact consumers pass with an isolated Clang ASan/UBSan runtime and matching link flags (`detect_leaks=0`, matching hosted policy; strict UBSan stops on errors). I restore the ordinary runtime afterward. These checks do not qualify the remaining nested/owned generic-union or complete-release gates.

`focused-logs.json` seals each uncompressed terminal. The ordinary bootstrap uses the existing explicit 60-second shadow budget. The instrumented native consumer check does not claim a fully instrumented compiler bootstrap.

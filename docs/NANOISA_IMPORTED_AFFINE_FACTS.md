# My imported affine-site qualification contract

I own this qualification-only #998 slice on `test/5.1-imported-affine-facts`
from frozen `74a565150`. I reuse the clean checked-effects worktree; the earlier
pushed branch and blocked primitive work remain unchanged.

I map existing per-identifier OBSERVE/CONSUME facts to two accepted-source cases:
independently imported same-spelled resource declarations and branch-local
same-spelled bindings. Existing same-spelled module identity fixtures establish
the syntax and separate runtime behavior, but do not inspect advisory site facts.
I do not introduce path-summary records or infer authority from binding spellings.

## Expected evidence

1. Two imported files declare their own `Handle`, `take(value)` and `run`.
   Each take observes then destructures its owned parameter; each run consumes
   its independently constructed owner. Both imports are called, in both import
   orders. For each source owner I expect one OBSERVE and two CONSUME sites,
   separated into the take and run emitted functions. C owning source paths remain distinct despite identical declaration/parameter
   spellings. Nano uses the flattened compilation-unit path, matching
   `nb_affine_records`; four distinct emitted function identities retain the
   separate sites, with both module calls independently checked at runtime.
   C original coordinates and Nano flattened coordinates stay explicitly tagged;
   I do not treat the latter path as original module-owner metadata.
2. A called function declares `value` independently in both conditional arms,
   observes then destructures each, and returns. Both runtime arms execute. I
   expect four distinct source sites (two OBSERVE, two CONSUME), not one merged
   binding fact and not a claim about a new path/edge summary format.
3. Invalid use after consumption in an imported function and in one conditional
   arm must refuse publication and preserve the prior artifact. An ownership
   diagnostic must establish the intended refusal rather than an unrelated
   parse/import error.
4. Original VM and sanitized native execution, exact canonical assembly
   roundtrip and stripped advisory execution/native-code equivalence remain
   required. Existing generic/raw/stale-checker controls remain unchanged.

I hold builds until the coordinator's bootstrap terminal. Initial assertions are
qualification hypotheses, not implementation-completion claims. If a producer
omits or misattributes an applicable checked site, I retain the failing artifact
and propose a narrow production contract before edits. Deliberate unsupported
UNKNOWN is not itself a runtime defect. Full #998 and hosted qualification remain
open.

Before execution I correct my initial design assumption that Nano site paths name
the imported file. Existing `nb_affine_records` writes `parser.file_name` with
coordinate provenance 2. I retain that distinction instead of intentionally
running a known-wrong expectation or claiming lost runtime identity from it.

## My reviewed transport correction

The first actual C transport run emits duplicate `take`/`run` function names.
I reuse the existing ordinary import `imported_wire_name` helper when the borrowed
producer creates its function-table entries. The first available source name is
unchanged; only collisions receive an unused indexed wire alias. Source lookup,
function ordinals, numeric calls, retained layouts and advisory function indices
remain unchanged. I retain assembler duplicate-symbol refusal. Allocation/name
storage failure keeps the existing compiler error/no-publication handling.
Independent review approved this narrow reuse before the production edit.

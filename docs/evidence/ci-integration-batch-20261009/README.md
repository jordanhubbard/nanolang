# My CI integration repair batch

I profile the actual C shadow interpreter child before changing lookup. The
retained top-of-stack sample is dominated by `env_get_var_visible_at` and string
comparison. My earlier parent sample measured module compilation, so I do not
use it as an interpreter profile. A temporary child-PID print enabled sampling;
I remove it from the candidate.

I reuse the existing optional symbol-name index in source-position lookup.
Both passes retain newest-first order, live file/flow/scope checks, and the
priority of located locals, explicit imports, then unlocated bindings. Index
allocation failure still falls back to the complete linear scan. I add no
borrowed symbol pointer or cached visibility decision.

My unchanged emitter selection has 662 shadows in both retained logs, in the
same order, with zero failures. The baseline completes in 42.358958 seconds of
shadow execution and the candidate in 8.073218 seconds (about 5.25 times faster).
The baseline is the unprofiled shadow child; a separately sampled run takes
44.138246 seconds. Compilation wall time includes host compilation and is not
this shadow-only measurement. Both runs use the temporary checkout at
23885ce29; the candidate changes only indexed visibility lookup in production.

My focused gate passes indexed lookup/failure controls, 52 environment checks,
ten lexical-scope methods and the full evaluator suite. Added differential
checks exercise duplicate names across files, flow and scope boundaries,
explicit import precedence, popped/reused slots and all three index-allocation
failure points. Two thousand source-position lookups use 2,000 name comparisons
with over 4,096 symbols. My selected LLVM ASan/UBSan fixture includes the actual
environment implementation; linked common/runtime objects remain ordinary.

The coverage job at release 80a472c85 uses lcov 2.0 and fails while reading a
deleted `.nano-native.*/program.c`. I exclude only these ephemeral generated
sources during capture, before source reading, retaining compiler/runtime
coverage and the existing threshold. I checked the lcov v2.0 implementation's
source filtering; a real corrected Linux coverage run remains required.

The completed CI run reports 147 passing and 89 failing corpus cases on both
macOS arm64 and Linux x64, plus four user-guide snippets on each. Linux arm64
is cancelled. The retained diagnostics identify missing builtin/foreign
lowering, aggregate/type parity, module identity and source-fixture issues.
Those remain release work; this performance repair does not close them or #982.
I do not increase shadow deadlines or remove tests.

## My combined acquisition bootstrap and checkout consolidation

At ab82d2c41 the primary checkout completes all seventeen bootstrap steps in
run-2187jh8m with byte-identical raw modules. I verify every source/tool/product
hash before applying the lookup repair. Both rebuilt native drivers pass all
seven companion-acquisition/output-preservation cases each. This bootstrap
qualifies the combined C/Nano acquisition pin, not the subsequent lookup patch
or a final release candidate.

I retain the earlier interrupted run-l2kf2ron evidence. Its old session handle
is missing. A direct script resume omits Make's NANO_BUILD_CACHE environment,
so the strict host-closure comparison refuses the resulting alternative module
paths. I preserve that terminal, return through Make, and qualify the fresh run.
I do not weaken the host-identity check or describe the failed resume as a pass.

After applying the same lookup patch to the primary C/Nano acquisition tree,
my combined C seed build, indexed-symbol fixture, 52 environment assertions,
ten lexical-scope methods, full evaluator and C companion-driver tests pass.
I retain integrated-primary.log. I do not repeat the self-hosted bootstrap for
this C lookup-only change; final candidate qualification remains a later gate.

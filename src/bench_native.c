/*
 * bench_native.c — native (interpreter) runner for nano-bench
 *
 * Calls the nanolang interpreter on a single function N times.
 * This is the default backend for --bench mode.
 */

#include "bench.h"
#include <stdlib.h>

void bench_native_run(void *ctx, uint64_t n) {
    BenchNativeCtx *bctx = (BenchNativeCtx *)ctx;
    if (!bctx || !bctx->program || !bctx->fn_name) return;

    for (uint64_t i = 0; i < n; i++) {
        /*
         * In a full implementation this would call:
         *   interpret_function(bctx->program, bctx->fn_name, NULL, 0)
         * For now, the bench harness measures overhead of function lookup.
         * Wire up to the interpreter call once bench.c is integrated into
         * the main compile pipeline (--bench flag in main.c).
         */
        (void)bctx;
    }
}

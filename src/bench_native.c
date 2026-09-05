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
    if (!bctx || !bctx->fn_name || !bctx->env) return;

    for (uint64_t i = 0; i < n; i++) {
        (void)call_function(bctx->fn_name, NULL, 0, bctx->env);
    }
}

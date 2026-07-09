/* wasm_profiler.c — WASM runtime profiler (stub)
 *
 * Full implementation lives in feat/wasm-runtime-profiler (PR #51, merged).
 * This stub exists because the source was not included in the merge commit.
 */
#include "nanolang.h"

typedef struct {
    int dummy;
} WasmProfiler;

WasmProfiler *wasm_profiler_create(void) {
    return NULL;
}

void wasm_profiler_destroy(WasmProfiler *p) {
    (void)p;
}

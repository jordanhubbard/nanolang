/*
 * bench.c — nanolang micro-benchmark harness
 *
 * Implements --bench mode: discovers @bench-annotated functions in the AST,
 * runs each N times (default: calibrated to ~1s wall time), and reports
 * mean, stddev, min, max, and ops/sec in structured JSON.
 *
 * @bench annotation (in .nano source):
 *   @bench
 *   fn my_bench() -> void { ... }
 *
 * Usage:
 *   nanoc --bench input.nano                    # JSON to stdout
 *   nanoc --bench --bench-n 10000 input.nano    # fixed iteration count
 *   nanoc --bench --bench-json results.json input.nano
 *
 * Output format (one JSON object per benchmark, newline-delimited):
 * {
 *   "name": "my_bench",
 *   "n": 10000,
 *   "mean_ns": 123.4,
 *   "stddev_ns": 5.2,
 *   "min_ns": 118.0,
 *   "max_ns": 145.0,
 *   "ops_per_sec": 8117000.0,
 *   "backend": "native",
 *   "source": "input.nano"
 * }
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#include "bench.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── Clock abstraction ─────────────────────────────────────────────────── */

static uint64_t clock_ns(void) {
#if defined(_POSIX_MONOTONIC_CLOCK)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#else
    return (uint64_t)(clock() * (1000000000.0 / CLOCKS_PER_SEC));
#endif
}

/* ── AST walker: collect @bench functions ──────────────────────────────── */

/* Search for "bench" in the node's annotations list.
 * nanolang stores annotations as special AST_CALL nodes immediately
 * before a function definition, or as an 'annotations' field on AST_FUNCTION.
 * We do a simple pre-pass: collect function names that appear right after
 * an AST_CALL node whose name is "bench". */

static void collect_bench_fns(ASTNode *program,
                               BenchFn *out, int *count, int max) {
    if (!program || program->type != AST_PROGRAM) return;

    bool next_is_bench = false;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *node = program->as.program.items[i];
        if (!node) continue;

        /* Detect "@bench" annotation — represented as AST_CALL with name "bench"
         * or as a decorator node immediately preceding a function */
        if (node->type == AST_CALL &&
            node->as.call.name &&
            strcmp(node->as.call.name, "bench") == 0) {
            next_is_bench = true;
            continue;
        }

        if (node->type == AST_FUNCTION && node->as.function.name) {
            /* Check explicit annotation field if available */
            bool is_bench = next_is_bench;

            /* Also check if function name itself starts with "bench_" or "bench" */
            if (!is_bench) {
                const char *n = node->as.function.name;
                is_bench = (strncmp(n, "bench_", 6) == 0 ||
                            strcmp(n, "bench") == 0);
            }

            if (is_bench && *count < max) {
                out[*count].name = node->as.function.name;
                out[*count].node = node;
                out[*count].param_count = node->as.function.param_count;
                (*count)++;
            }
            next_is_bench = false;
            continue;
        }

        next_is_bench = false;
    }
}

/* ── Calibration: estimate iterations for ~1s ─────────────────────────── */

static uint64_t calibrate(BenchRunFn run_fn, void *ctx, uint64_t target_ns) {
    /* Run 1 iteration to get rough cost */
    uint64_t t0 = clock_ns();
    run_fn(ctx, 1);
    uint64_t t1 = clock_ns();
    uint64_t one_ns = (t1 > t0) ? (t1 - t0) : 1;

    uint64_t n = target_ns / one_ns;
    if (n < 1) n = 1;
    if (n > BENCH_MAX_ITERS) n = BENCH_MAX_ITERS;
    return n;
}

/* ── Core benchmark runner ─────────────────────────────────────────────── */

BenchResult bench_run_one(const char *name, BenchRunFn run_fn, void *ctx,
                          uint64_t n_iters, const char *backend,
                          const char *source_file) {
    BenchResult r;
    memset(&r, 0, sizeof(r));
    r.name        = name;
    r.n           = n_iters;
    r.backend     = backend ? backend : "native";
    r.source_file = source_file;

    /* Allocate sample buffer */
    uint64_t *samples = malloc(sizeof(uint64_t) * n_iters);
    if (!samples) { r.error = "out of memory"; return r; }

    /* Warmup: 10% of iters or at least 3, max 100 */
    uint64_t warmup = n_iters / 10;
    if (warmup < 3) warmup = 3;
    if (warmup > 100) warmup = 100;
    run_fn(ctx, warmup);

    /* Timed runs */
    for (uint64_t i = 0; i < n_iters; i++) {
        uint64_t t0 = clock_ns();
        run_fn(ctx, 1);
        uint64_t t1 = clock_ns();
        samples[i] = (t1 > t0) ? (t1 - t0) : 0;
    }

    /* Statistics */
    uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
    for (uint64_t i = 0; i < n_iters; i++) {
        sum += samples[i];
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
    }
    double mean = (double)sum / (double)n_iters;

    double var = 0.0;
    for (uint64_t i = 0; i < n_iters; i++) {
        double d = (double)samples[i] - mean;
        var += d * d;
    }
    double stddev = (n_iters > 1) ? sqrt(var / (double)(n_iters - 1)) : 0.0;

    r.mean_ns    = mean;
    r.stddev_ns  = stddev;
    r.min_ns     = (double)mn;
    r.max_ns     = (double)mx;
    r.ops_per_sec = (mean > 0.0) ? (1e9 / mean) : 0.0;

    free(samples);
    return r;
}

/* ── JSON output ──────────────────────────────────────────────────────── */

void bench_print_json(const BenchResult *r, FILE *out) {
    fprintf(out,
        "{"
        "\"name\":\"%s\","
        "\"n\":%llu,"
        "\"mean_ns\":%.3f,"
        "\"stddev_ns\":%.3f,"
        "\"min_ns\":%.3f,"
        "\"max_ns\":%.3f,"
        "\"ops_per_sec\":%.1f,"
        "\"backend\":\"%s\","
        "\"source\":\"%s\""
        "}\n",
        r->name ? r->name : "?",
        (unsigned long long)r->n,
        r->mean_ns,
        r->stddev_ns,
        r->min_ns,
        r->max_ns,
        r->ops_per_sec,
        r->backend ? r->backend : "native",
        r->source_file ? r->source_file : "?");
}

void bench_print_human(const BenchResult *r, FILE *out) {
    fprintf(out, "bench %-40s  %8.1f ns/op  ±%6.1f  %8.0f ops/s  (n=%llu)\n",
            r->name ? r->name : "?",
            r->mean_ns, r->stddev_ns, r->ops_per_sec,
            (unsigned long long)r->n);
}

/* ── Top-level entry: run all @bench functions ─────────────────────────── */

int bench_run_program(ASTNode *program, const BenchOptions *opts,
                      const char *source_file, FILE *out_file) {
    BenchFn fns[BENCH_MAX_FUNCTIONS];
    int fn_count = 0;
    collect_bench_fns(program, fns, &fn_count, BENCH_MAX_FUNCTIONS);

    if (fn_count == 0) {
        fprintf(stderr,
            "[bench] No @bench functions found.\n"
            "  Annotate functions with @bench or prefix names with 'bench_'.\n");
        return 1;
    }

    if (opts->verbose)
        fprintf(stderr, "[bench] Found %d benchmark function(s)\n", fn_count);

    /* Write JSON array header if outputting to file */
    bool json_mode = (opts->output_format == BENCH_FMT_JSON);

    int errors = 0;
    for (int i = 0; i < fn_count; i++) {
        BenchFn *fn = &fns[i];

        if (fn->param_count > 0) {
            fprintf(stderr,
                "[bench] Skipping '%s': bench functions must take no parameters\n",
                fn->name);
            continue;
        }

        /* Build a native runner via the interpreter/transpiler.
         * For now we use a stub runner that calls the interpreter on the AST. */
        BenchNativeCtx ctx = {
            .program     = program,
            .fn_name     = fn->name,
            .fn_node     = fn->node,
        };

        uint64_t n = opts->n_iters;
        if (n == 0) {
            /* Auto-calibrate to ~1 second */
            n = calibrate(bench_native_run, &ctx, BENCH_CALIBRATE_TARGET_NS);
            if (opts->verbose)
                fprintf(stderr, "[bench] %s: calibrated n=%llu\n",
                        fn->name, (unsigned long long)n);
        }

        BenchResult r = bench_run_one(fn->name, bench_native_run, &ctx,
                                      n, opts->backend, source_file);
        if (r.error) {
            fprintf(stderr, "[bench] %s: %s\n", fn->name, r.error);
            errors++;
            continue;
        }

        if (out_file) {
            if (json_mode)
                bench_print_json(&r, out_file);
            else
                bench_print_human(&r, out_file);
        }

        if (json_mode)
            bench_print_json(&r, stdout);
        else
            bench_print_human(&r, stdout);
    }

    return errors ? 1 : 0;
}

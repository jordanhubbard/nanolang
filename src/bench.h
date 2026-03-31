/*
 * bench.h — nanolang micro-benchmark harness API
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#pragma once
#ifndef BENCH_H
#define BENCH_H

#include "nanolang.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Limits ────────────────────────────────────────────────────────────── */

#define BENCH_MAX_FUNCTIONS        256
#define BENCH_MAX_ITERS            100000000ULL  /* 100M max iterations */
#define BENCH_CALIBRATE_TARGET_NS  1000000000ULL /* calibrate to ~1s */
#define BENCH_DEFAULT_WARMUP_FRAC  0.10          /* 10% warmup */

/* ── Output format ─────────────────────────────────────────────────────── */

typedef enum {
    BENCH_FMT_HUMAN = 0,  /* human-readable table */
    BENCH_FMT_JSON  = 1,  /* newline-delimited JSON (one object per bench) */
} BenchOutputFmt;

/* ── Options ───────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t       n_iters;       /* 0 = auto-calibrate */
    BenchOutputFmt output_format;
    const char    *json_out_path; /* NULL = stdout only */
    const char    *backend;       /* "native", "wasm", "ptx", "c", "llvm" */
    bool           verbose;
} BenchOptions;

/* ── Benchmark function descriptor ────────────────────────────────────── */

typedef struct {
    const char *name;
    ASTNode    *node;
    int         param_count;
} BenchFn;

/* ── Result ─────────────────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    uint64_t    n;
    double      mean_ns;
    double      stddev_ns;
    double      min_ns;
    double      max_ns;
    double      ops_per_sec;
    const char *backend;
    const char *source_file;
    const char *error;    /* NULL on success */
} BenchResult;

/* ── Runner callback type ──────────────────────────────────────────────── */

/* Called with ctx + number of iterations to perform */
typedef void (*BenchRunFn)(void *ctx, uint64_t n);

/* ── Native (interpreter) runner context ──────────────────────────────── */

typedef struct {
    ASTNode    *program;
    const char *fn_name;
    ASTNode    *fn_node;
} BenchNativeCtx;

/* Forward declaration — implemented in bench_native.c (auto-generated stub) */
void bench_native_run(void *ctx, uint64_t n);

/* ── Public API ─────────────────────────────────────────────────────────── */

/*
 * Run a single benchmark N times.
 * run_fn is called with (ctx, n_iters) per measurement sample.
 */
BenchResult bench_run_one(const char *name, BenchRunFn run_fn, void *ctx,
                          uint64_t n_iters, const char *backend,
                          const char *source_file);

/* Print result as JSON (one line) */
void bench_print_json(const BenchResult *r, FILE *out);

/* Print result in human-readable form */
void bench_print_human(const BenchResult *r, FILE *out);

/*
 * Discover @bench functions in the program AST and run them all.
 * Returns 0 on success, non-zero if any benchmark failed.
 */
int bench_run_program(ASTNode *program, const BenchOptions *opts,
                      const char *source_file, FILE *out_file);

#endif /* BENCH_H */

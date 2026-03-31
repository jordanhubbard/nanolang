/*
 * proptest.h — QuickCheck-style property-based test oracle for nanolang
 *
 * Discovers @property-annotated functions (detected by "prop_" prefix or an
 * AST_CALL node named "property" immediately preceding the function), generates
 * random typed inputs, runs each property 100 times, and shrinks counterexamples.
 *
 * Usage in .nano source:
 *   # @property
 *   fn prop_add_commutative(a: int, b: int) -> bool {
 *       return (== (+ a b) (+ b a))
 *   }
 *
 * CLI:
 *   nano --proptest input.nano
 *
 * Output:
 *   PASS [prop_add_commutative] (100 cases)
 *   FAIL [prop_my_broken]: counterexample: a=5 b=3
 *
 * PRNG: xorshift64, seed from NANO_PROPTEST_SEED env var or fixed default.
 * Shrinking: for ints try 0, sign flip, halving; for strings try shorter prefixes.
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#pragma once
#ifndef PROPTEST_H
#define PROPTEST_H

#include "nanolang.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Limits ────────────────────────────────────────────────────────────── */

#define PROPTEST_MAX_FUNCTIONS   256
#define PROPTEST_DEFAULT_RUNS    100
#define PROPTEST_INT_MIN         (-1000LL)
#define PROPTEST_INT_MAX         1000LL
#define PROPTEST_MAX_STRING_LEN  10
#define PROPTEST_SHRINK_STEPS    64   /* max shrink attempts per argument */

/* ── Options ───────────────────────────────────────────────────────────── */

typedef struct {
    int     n_runs;    /* number of random cases per property (default 100) */
    bool    verbose;   /* print each case */
    uint64_t seed;     /* PRNG seed (0 = read NANO_PROPTEST_SEED or use default) */
} PropTestOptions;

/* ── Property function descriptor ─────────────────────────────────────── */

typedef struct {
    const char *name;        /* function name */
    ASTNode    *node;        /* AST_FUNCTION node */
    int         param_count;
} PropFn;

/* ── Public API ─────────────────────────────────────────────────────────── */

/*
 * Discover property functions in the program AST and run them all.
 * Returns 0 if all properties passed, 1 if any failed or none found.
 */
int proptest_run_program(ASTNode *program, Environment *env,
                         const PropTestOptions *opts,
                         const char *source_file);

#endif /* PROPTEST_H */

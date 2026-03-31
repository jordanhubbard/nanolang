/*
 * proptest.c — QuickCheck-style property-based test oracle for nanolang
 *
 * See proptest.h for documentation.
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#include "proptest.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── xorshift64 PRNG ───────────────────────────────────────────────────── */

static uint64_t g_rng_state = 0;

static void rng_seed(uint64_t seed) {
    g_rng_state = seed ? seed : 0xdeadbeefcafe1234ULL;
}

static uint64_t rng_next(void) {
    uint64_t x = g_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng_state = x;
    return x;
}

/* Return a random int64 in [lo, hi] (inclusive) */
static long long rng_int_range(long long lo, long long hi) {
    if (lo >= hi) return lo;
    uint64_t range = (uint64_t)(hi - lo) + 1;
    return lo + (long long)(rng_next() % range);
}

/* Return a random double in [-1e6, 1e6] */
static double rng_float(void) {
    double t = (double)(rng_next() % 2000001ULL);
    return t - 1000000.0;
}

/* Return 0 or 1 */
static bool rng_bool(void) {
    return (rng_next() & 1) != 0;
}

/* Fill buf with a random ASCII printable string of length [0, max_len].
 * buf must be at least max_len+1 bytes. */
static void rng_string(char *buf, int max_len) {
    int len = (int)(rng_next() % ((unsigned)(max_len + 1)));
    for (int i = 0; i < len; i++) {
        /* ASCII printable: 0x20 (' ') to 0x7e ('~') */
        buf[i] = (char)(0x20 + (int)(rng_next() % 95));
    }
    buf[len] = '\0';
}

/* ── AST walker: collect @property / prop_ functions ──────────────────── */

static void collect_prop_fns(ASTNode *program,
                              PropFn *out, int *count, int max) {
    if (!program || program->type != AST_PROGRAM) return;

    bool next_is_prop = false;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *node = program->as.program.items[i];
        if (!node) continue;

        /* Detect "property" annotation — an AST_CALL with name "property"
         * immediately before a function definition. */
        if (node->type == AST_CALL &&
            node->as.call.name &&
            strcmp(node->as.call.name, "property") == 0) {
            next_is_prop = true;
            continue;
        }

        if (node->type == AST_FUNCTION && node->as.function.name) {
            bool is_prop = next_is_prop;

            /* Also detect by "prop_" prefix convention */
            if (!is_prop) {
                const char *n = node->as.function.name;
                is_prop = (strncmp(n, "prop_", 5) == 0);
            }

            if (is_prop && *count < max) {
                out[*count].name        = node->as.function.name;
                out[*count].node        = node;
                out[*count].param_count = node->as.function.param_count;
                (*count)++;
            }
            next_is_prop = false;
            continue;
        }

        next_is_prop = false;
    }
}

/* ── Random value generation ───────────────────────────────────────────── */

static Value make_int_val(long long v) {
    Value val;
    memset(&val, 0, sizeof(val));
    val.type = VAL_INT;
    val.as.int_val = v;
    return val;
}

static Value make_float_val(double v) {
    Value val;
    memset(&val, 0, sizeof(val));
    val.type = VAL_FLOAT;
    val.as.float_val = v;
    return val;
}

static Value make_bool_val(bool v) {
    Value val;
    memset(&val, 0, sizeof(val));
    val.type = VAL_BOOL;
    val.as.bool_val = v;
    return val;
}

/* String val owns a malloc'd copy; caller must free val.as.string_val */
static Value make_string_val(const char *s) {
    Value val;
    memset(&val, 0, sizeof(val));
    val.type = VAL_STRING;
    val.as.string_val = strdup(s);
    return val;
}

static Value random_value_for_type(Type t) {
    switch (t) {
        case TYPE_INT:
            return make_int_val(rng_int_range(PROPTEST_INT_MIN, PROPTEST_INT_MAX));
        case TYPE_FLOAT:
            return make_float_val(rng_float());
        case TYPE_BOOL:
            return make_bool_val(rng_bool());
        case TYPE_STRING: {
            char buf[PROPTEST_MAX_STRING_LEN + 1];
            rng_string(buf, PROPTEST_MAX_STRING_LEN);
            return make_string_val(buf);
        }
        default:
            /* Fallback to int for unsupported types */
            return make_int_val(rng_int_range(PROPTEST_INT_MIN, PROPTEST_INT_MAX));
    }
}

static void free_value_strings(Value *args, int count) {
    for (int i = 0; i < count; i++) {
        if (args[i].type == VAL_STRING && args[i].as.string_val) {
            free(args[i].as.string_val);
            args[i].as.string_val = NULL;
        }
    }
}

/* ── Argument printing ─────────────────────────────────────────────────── */

static void print_arg(FILE *f, const Parameter *p, const Value *v) {
    fprintf(f, "%s=", p->name ? p->name : "?");
    switch (v->type) {
        case VAL_INT:   fprintf(f, "%lld", (long long)v->as.int_val); break;
        case VAL_FLOAT: fprintf(f, "%g",   v->as.float_val);           break;
        case VAL_BOOL:  fprintf(f, "%s",   v->as.bool_val ? "true" : "false"); break;
        case VAL_STRING:
            fprintf(f, "\"%s\"", v->as.string_val ? v->as.string_val : ""); break;
        default:        fprintf(f, "?"); break;
    }
}

/* ── Value result: did the property hold? ─────────────────────────────── */

static bool result_is_pass(const Value *v) {
    switch (v->type) {
        case VAL_BOOL:   return v->as.bool_val;
        case VAL_INT:    return v->as.int_val != 0;
        case VAL_STRING:
            return v->as.string_val &&
                   (strcmp(v->as.string_val, "pass") == 0 ||
                    strcmp(v->as.string_val, "PASS") == 0 ||
                    strcmp(v->as.string_val, "ok")   == 0);
        default: return false;
    }
}

/* ── Shrinking ─────────────────────────────────────────────────────────── */

/* Try to shrink args[idx] to a simpler value of type t.
 * Returns true if there is a next candidate to try (written into *out).
 * step counts from 0 upward. */
static bool shrink_step(int step, Type t, const Value *cur, Value *out) {
    switch (t) {
        case TYPE_INT: {
            long long v = cur->as.int_val;
            if (step == 0) { *out = make_int_val(0);       return (v != 0); }
            if (step == 1) { *out = make_int_val(v < 0 ? -v : v); return (v != 0 && v != (v < 0 ? -v : v)); }
            if (step == 2) { *out = make_int_val(v / 2);   return (v != 0 && v / 2 != v); }
            if (step == 3) { *out = make_int_val(1);        return (v != 1 && v > 0); }
            if (step == 4) { *out = make_int_val(-1);       return (v != -1 && v < 0); }
            return false;
        }
        case TYPE_FLOAT: {
            double v = cur->as.float_val;
            if (step == 0) { *out = make_float_val(0.0);   return (v != 0.0); }
            if (step == 1) { *out = make_float_val(v < 0 ? -v : v); return true; }
            if (step == 2) { *out = make_float_val(v / 2); return (v / 2 != v); }
            return false;
        }
        case TYPE_BOOL: {
            if (step == 0) { *out = make_bool_val(false); return cur->as.bool_val; }
            return false;
        }
        case TYPE_STRING: {
            const char *s = cur->as.string_val ? cur->as.string_val : "";
            int len = (int)strlen(s);
            if (step == 0) { *out = make_string_val(""); return len > 0; }
            if (step < len) {
                char buf[PROPTEST_MAX_STRING_LEN + 2];
                int newlen = len - step;
                if (newlen < 0) newlen = 0;
                if (newlen > PROPTEST_MAX_STRING_LEN) newlen = PROPTEST_MAX_STRING_LEN;
                memcpy(buf, s, (size_t)newlen);
                buf[newlen] = '\0';
                *out = make_string_val(buf);
                return true;
            }
            return false;
        }
        default:
            return false;
    }
}

/* Given a failing set of args, find the smallest counterexample via shrinking.
 * Modifies args[] in-place to contain the minimal failing case. */
static void shrink_args(const PropFn *fn, Value *args, Environment *env) {
    ASTNode *fn_node = fn->node;
    Parameter *params = fn_node->as.function.params;
    int n = fn->param_count;

    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        for (int i = 0; i < n; i++) {
            Type t = params[i].type;
            for (int step = 0; step < PROPTEST_SHRINK_STEPS; step++) {
                Value candidate;
                memset(&candidate, 0, sizeof(candidate));
                bool has_next = shrink_step(step, t, &args[i], &candidate);
                if (!has_next) break;

                /* Try the candidate */
                Value saved = args[i];
                args[i] = candidate;
                Value result = call_function(fn->name, args, n, env);
                if (!result_is_pass(&result)) {
                    /* Smaller counterexample found — keep it */
                    if (saved.type == VAL_STRING && saved.as.string_val)
                        free(saved.as.string_val);
                    made_progress = true;
                    /* Keep searching this arg */
                } else {
                    /* Candidate passed — restore original and free candidate */
                    if (candidate.type == VAL_STRING && candidate.as.string_val)
                        free(candidate.as.string_val);
                    args[i] = saved;
                    break;
                }
            }
        }
    }
}

/* ── Single property runner ────────────────────────────────────────────── */

/* Returns true if property passed all runs */
static bool run_one_property(const PropFn *fn, Environment *env,
                              const PropTestOptions *opts) {
    ASTNode *fn_node = fn->node;
    if (!fn_node || fn_node->type != AST_FUNCTION) {
        fprintf(stderr, "[proptest] %s: not a function node\n", fn->name);
        return false;
    }

    int n = fn->param_count;
    Parameter *params = fn_node->as.function.params;
    Type ret_type = fn_node->as.function.return_type;

    /* Property functions must return bool or a pass/fail indicator */
    if (ret_type != TYPE_BOOL && ret_type != TYPE_INT &&
        ret_type != TYPE_STRING && ret_type != TYPE_UNKNOWN) {
        fprintf(stderr, "[proptest] %s: return type must be bool/int/string\n",
                fn->name);
        return false;
    }

    Value args[32];  /* max 32 params */
    if (n > 32) {
        fprintf(stderr, "[proptest] %s: too many parameters (%d)\n", fn->name, n);
        return false;
    }

    int runs = opts->n_runs > 0 ? opts->n_runs : PROPTEST_DEFAULT_RUNS;

    for (int trial = 0; trial < runs; trial++) {
        /* Generate random args */
        for (int i = 0; i < n; i++) {
            args[i] = random_value_for_type(params[i].type);
        }

        if (opts->verbose) {
            fprintf(stderr, "[proptest] %s trial %d:", fn->name, trial + 1);
            for (int i = 0; i < n; i++) {
                fprintf(stderr, " ");
                print_arg(stderr, &params[i], &args[i]);
            }
            fprintf(stderr, "\n");
        }

        Value result = call_function(fn->name, args, n, env);
        bool passed = result_is_pass(&result);
        if (result.type == VAL_STRING && result.as.string_val)
            free(result.as.string_val);

        if (!passed) {
            /* Shrink to find minimal counterexample */
            shrink_args(fn, args, env);

            printf("FAIL [%s]: counterexample:", fn->name);
            for (int i = 0; i < n; i++) {
                printf(" ");
                print_arg(stdout, &params[i], &args[i]);
            }
            printf("\n");

            free_value_strings(args, n);
            return false;
        }

        free_value_strings(args, n);
    }

    printf("PASS [%s] (%d cases)\n", fn->name, runs);
    return true;
}

/* ── Top-level entry ───────────────────────────────────────────────────── */

int proptest_run_program(ASTNode *program, Environment *env,
                         const PropTestOptions *opts,
                         const char *source_file) {
    (void)source_file;

    /* Seed PRNG */
    uint64_t seed = opts->seed;
    if (seed == 0) {
        const char *env_seed = getenv("NANO_PROPTEST_SEED");
        if (env_seed && *env_seed) {
            seed = (uint64_t)strtoull(env_seed, NULL, 10);
        }
        if (seed == 0) seed = 0xdeadbeefcafe1234ULL;
    }
    rng_seed(seed);

    /* Collect property functions */
    PropFn fns[PROPTEST_MAX_FUNCTIONS];
    int fn_count = 0;
    collect_prop_fns(program, fns, &fn_count, PROPTEST_MAX_FUNCTIONS);

    if (fn_count == 0) {
        fprintf(stderr,
            "[proptest] No property functions found.\n"
            "  Prefix function names with 'prop_' or annotate with 'property' before fn.\n");
        return 1;
    }

    if (opts->verbose)
        fprintf(stderr, "[proptest] Found %d property function(s) (seed=%llu)\n",
                fn_count, (unsigned long long)seed);

    int failures = 0;
    for (int i = 0; i < fn_count; i++) {
        if (!run_one_property(&fns[i], env, opts))
            failures++;
    }

    return failures ? 1 : 0;
}

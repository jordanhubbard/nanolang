/*
 * pgo_pass.h — Profile-Guided Optimization (PGO) pass for nanolang
 *
 * Reads a .nano.prof file (collapsed-stack format emitted by --profile-runtime),
 * identifies hot functions above a call-count threshold, and rewrites their
 * call sites in the AST to inline the callee body.
 *
 * The resulting AST can then be re-compiled with normal optimizations
 * (constant folding, DCE) to produce an optimized binary.
 *
 * Usage:
 *   nanoc --pgo <prog>.nano.prof input.nano -o optimised
 *
 *   or programmatically:
 *     PGOProfile *prof = pgo_load_profile("prog.nano.prof");
 *     if (prof) {
 *         pgo_apply(program, prof);
 *         pgo_profile_free(prof);
 *     }
 *
 * Pipeline position:  parse → typecheck → [PGO] → fold → DCE → transpile
 */

#ifndef NANOLANG_PGO_PASS_H
#define NANOLANG_PGO_PASS_H

#include "nanolang.h"
#include <stdint.h>
#include <stdbool.h>

/* ── Profile data ──────────────────────────────────────────────────────── */

#define PGO_MAX_ENTRIES  4096

typedef struct {
    char    *name;       /* function name */
    uint64_t calls;      /* call count from .nano.prof */
    bool     is_hot;     /* true if above hotness threshold */
} PGOEntry;

typedef struct {
    PGOEntry entries[PGO_MAX_ENTRIES];
    int      count;

    /* Hotness threshold: functions called more than hot_threshold times
     * are candidates for inlining at their call sites. */
    uint64_t hot_threshold;

    /* Statistics (filled by pgo_apply) */
    int      sites_inlined;   /* call sites rewritten */
    int      functions_hot;   /* distinct hot functions found */
} PGOProfile;

/* ── Inlining policy limits ────────────────────────────────────────────── */

/*
 * Maximum number of statements in a function body that we are willing to
 * inline.  Larger functions are skipped to avoid code size blow-up.
 */
#define PGO_MAX_INLINE_STMTS  32

/*
 * Maximum inline depth: guard against mutual recursion causing infinite
 * inlining at compile time.
 */
#define PGO_MAX_INLINE_DEPTH  4

/* ── Public API ─────────────────────────────────────────────────────────── */

/*
 * pgo_load_profile(path) — parse a .nano.prof file.
 * Returns a newly allocated PGOProfile, or NULL on error.
 * The hot_threshold is set automatically based on percentile.
 */
PGOProfile *pgo_load_profile(const char *path);

/*
 * pgo_load_profile_threshold(path, threshold) — as above, but override
 * the hot_threshold explicitly (useful for testing).
 */
PGOProfile *pgo_load_profile_threshold(const char *path, uint64_t threshold);

/*
 * pgo_profile_free(prof) — free a PGOProfile returned by pgo_load_profile.
 */
void pgo_profile_free(PGOProfile *prof);

/*
 * pgo_apply(program, prof) — rewrite call sites for hot functions in-place.
 *
 * For each AST_CALL node whose callee is a hot function, replace the call
 * with a let-bound copy of the callee's body with arguments substituted.
 *
 * Returns the number of call sites inlined (also stored in prof->sites_inlined).
 */
int pgo_apply(ASTNode *program, PGOProfile *prof);

/*
 * pgo_is_hot(prof, name) — query whether a function is hot.
 */
bool pgo_is_hot(const PGOProfile *prof, const char *name);

/*
 * pgo_print_report(prof) — print hotspot summary to stderr.
 */
void pgo_print_report(const PGOProfile *prof);

#endif /* NANOLANG_PGO_PASS_H */

/*
 * c_backend.h — my bounded hosted C99/C11 source target
 *
 * I emit a bounded hosted C99/C11 profile from the checked AST.
 * I preserve output on semantic refusal; I do not promise whole-language
 * or freestanding lowering.
 *
 * Type mappings
 * ─────────────
 *   nano int    → int64_t
 *   nano float  → double
 *   nano bool   → int (0 or 1)
 *   nano string → const char *  (literals are static; scalar conversions and
 *                               concatenation retain owned snapshots until exit)
 *   nano void   → void
 *   nano u8     → uint8_t
 *   nano enum   → int64_t carrier (existing enumerator declarations)
 *   local struct/union → exact declared C typedef in supported contexts
 *
 * Control flow
 * ────────────
 *   if/else    → C if/else
 *   while      → C while
 *   return     → C return
 *   let        → local variable declaration
 *
 * Profile boundaries
 * ──────────────────
 *   I support direct declared calls. I refuse arrays, first-class callable values,
 *   expression/local-bound callees, record spread, for loops, tuple values, effects, async/await
 *   and try propagation. Other list/map/binary-string/opaque/row/borrow/generic
 *   or unresolved value representations receive checked refusal. By-value record
 *   fields require prior complete local declarations. I require hosted library support. Exact supported
 *   scalar/local-union block values use scoped statements; unsupported insertion
 *   contexts receive checked refusal before output publication.
 *
 * Usage
 * ─────
 *   nanoc --target c input.nano -o output.c
 *   cc -std=c99 -I. output.c -o output
 *
 * My API no_main option omits the hosted wrapper; it refuses global startup.
 * I do not expose no-stdlib/no-main CLI flags or a freestanding runtime ABI.
 */
#pragma once
#ifndef C_BACKEND_H
#define C_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Options for C backend emission */
typedef struct {
    bool no_stdlib;      /* True is refused: my generated support is hosted. */
    bool no_main;        /* Omit hosted wrapper; globals require an entry and are refused. */
    bool static_strings; /* Compatibility alias: both values retain static literal storage. */
    bool verbose;
    /* Optional checked dependency closure. I borrow declarations and context
     * throughout planning/emission. Returned declarations must occur in root;
     * caller is NULL for root context, alias is NULL for unqualified calls. */
    ASTNode *(*resolve_function)(void *context, ASTNode *caller,
                                 const char *name, const char *alias);
    void *function_context;
} CBOptions;

/* Emit C source to a file path.
 * Returns 0 on success, non-zero on error.
 */
int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, const CBOptions *opts);

/* Emit C source to an already-open FILE*. */
int c_backend_emit_fp(ASTNode *root, FILE *out,
                      const char *source_file, const CBOptions *opts);

#endif /* C_BACKEND_H */

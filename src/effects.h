/*
 * effects.h — Algebraic effect system support for nanolang (row-poly branch).
 *
 * This file provides:
 *   - EffectRow  : set of effect names on arrow types (for type_infer.c)
 *   - EffectDecl : extended effect declaration with param_type fields
 *   - EffectRegistry : registry of declared effects
 *   - Stub functions used by type_infer.c
 *
 * nanolang.h already defines EffectOp and EffectDef; this file adds the
 * supplementary structures needed by the HM type inferencer.
 *
 * When the full effect system is merged from feat/dap-debugger, replace
 * this file with the complete implementation.
 */
#ifndef EFFECTS_H
#define EFFECTS_H

#include "nanolang.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* ── Effect row: a set of effect names on arrow types ───────────────────── */

#define EFFECT_ROW_MAX 16

typedef struct {
    char *effects[EFFECT_ROW_MAX]; /* effect names present in this row */
    int   count;
    bool  is_open;  /* true = row is polymorphic (can be extended) */
} EffectRow;

/* ── Effect declaration (for type_infer registry) ───────────────────────── */
/* Note: EffectOp is already defined in nanolang.h */

typedef struct {
    char      *name;      /* effect name, e.g. "IO" */
    EffectOp  *ops;       /* array of operations */
    int        op_count;
    Type       param_type;       /* simplified: single param type */
    char      *param_type_name;
    Type       return_type;
    char      *return_type_name;
} EffectDecl;

/* ── Effect registry ─────────────────────────────────────────────────────── */

#define EFFECT_REGISTRY_MAX 64

typedef struct {
    EffectDecl decls[EFFECT_REGISTRY_MAX];
    int        count;
} EffectRegistry;

/* ── Handler frame (for interpreter) ────────────────────────────────────── */

typedef struct EffectHandlerFrame EffectHandlerFrame;
struct EffectHandlerFrame {
    char                  *effect_name;
    ASTNode              **handler_bodies;
    char                 **handler_op_names;
    char                 **handler_param_names;
    int                    handler_count;
    EffectHandlerFrame    *outer;
    Environment           *env;
};

/* ── Inline helper functions ─────────────────────────────────────────────── */

static inline EffectRegistry *effect_registry_new(void) {
    EffectRegistry *r = (EffectRegistry *)calloc(1, sizeof(EffectRegistry));
    return r;
}

static inline void effect_registry_free(EffectRegistry *r) { free(r); }

static inline void effect_register_builtins(EffectRegistry *r) { (void)r; }

static inline void effect_row_add(EffectRow *row, const char *name) {
    if (!row || !name || row->count >= EFFECT_ROW_MAX) return;
    row->effects[row->count++] = (char *)name;
}

static inline char *effect_row_to_str(EffectRow *row) {
    if (!row || row->count == 0) return strdup("pure");
    size_t len = 1;
    for (int i = 0; i < row->count; i++) {
        if (row->effects[i]) len += strlen(row->effects[i]) + 2;
    }
    char *s = (char *)malloc(len);
    if (!s) return strdup("?");
    s[0] = '\0';
    for (int i = 0; i < row->count; i++) {
        if (!row->effects[i]) continue;
        if (i > 0) strcat(s, ",");
        strcat(s, row->effects[i]);
    }
    return s;
}

static inline EffectDecl *effect_lookup(EffectRegistry *r, const char *name) {
    if (!r || !name) return NULL;
    for (int i = 0; i < r->count; i++)
        if (r->decls[i].name && strcmp(r->decls[i].name, name) == 0)
            return &r->decls[i];
    return NULL;
}

static inline EffectOp *effect_op_lookup(EffectDecl *d, const char *name) {
    if (!d || !name) return NULL;
    for (int i = 0; i < d->op_count; i++)
        if (d->ops[i].name && strcmp(d->ops[i].name, name) == 0)
            return &d->ops[i];
    return NULL;
}

static inline void effect_register_from_ast(EffectRegistry *r, ASTNode *node) {
    (void)r; (void)node;
}

static inline bool effect_check_program(ASTNode *prog, EffectRegistry *r,
                                         void *ctx) {
    (void)prog; (void)r; (void)ctx;
    return true;
}

/* Handler stack stubs */
static inline void nl_effect_frame_push(EffectHandlerFrame *f) { (void)f; }
static inline void nl_effect_frame_pop(void) {}
static inline EffectHandlerFrame *nl_effect_find_handler(
    const char *e, const char *o, int *idx) {
    (void)e; (void)o; (void)idx; return NULL;
}
static inline void env_effect_register(Environment *env, ASTNode *node) {
    (void)env; (void)node;
}
static inline EffectDecl *env_effect_lookup(Environment *env, const char *name) {
    (void)env; (void)name; return NULL;
}

#endif /* EFFECTS_H */

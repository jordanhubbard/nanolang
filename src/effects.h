/*
 * effects.h — Algebraic Effect System for nanolang
 *
 * Lightweight row-based effect system layered on HM Algorithm W.
 * Effects tracked as sets of effect names ("effect rows") on arrow types.
 */
#ifndef EFFECTS_H
#define EFFECTS_H

#include "nanolang.h"
#include <stdbool.h>

/* ── Effect row: set of effect names on a function type ─────────────────── */

#define EFFECT_ROW_MAX 16

typedef struct {
    char *names[EFFECT_ROW_MAX];
    int   count;
    bool  is_open;   /* open row = may have more effects */
} EffectRow;

/* ── Effect declaration (registered in effect registry) ─────────────────── */

typedef struct {
    char      *name;       /* effect name, e.g. "IO" */
    EffectOp  *ops;        /* array of operations (EffectOp from nanolang.h) */
    int        op_count;
} EffectDecl;

/* ── Effect registry ────────────────────────────────────────────────────── */

#define EFFECT_REGISTRY_MAX 32

typedef struct {
    EffectDecl decls[EFFECT_REGISTRY_MAX];
    int        count;
} EffectRegistry;

/* ── Effect check context (for effect_check_program) ────────────────────── */

typedef struct {
    EffectRegistry *registry;
    EffectRow      *current_row;
    bool            has_errors;
} EffectCheckCtx;

/* ── API ────────────────────────────────────────────────────────────────── */

EffectRegistry *effect_registry_new(void);
void            effect_registry_free(EffectRegistry *reg);

bool effect_register(EffectRegistry *reg, const char *name,
                     EffectOp *ops, int op_count);
bool effect_register_from_ast(EffectRegistry *reg, ASTNode *node);
void effect_register_builtins(EffectRegistry *reg);

EffectDecl *effect_lookup(EffectRegistry *reg, const char *name);
EffectOp   *effect_op_lookup(EffectDecl *decl, const char *op_name);

void  effect_row_add(EffectRow *row, const char *name);
char *effect_row_to_str(EffectRow *row);

bool effect_check_program(ASTNode *program, EffectRegistry *reg,
                          Environment *env);

#endif /* EFFECTS_H */

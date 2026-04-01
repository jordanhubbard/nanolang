/*
 * effects.h — Algebraic Effect System for nanolang
 *
 * Implements:
 *   - Effect declarations: effect IO { print : String -> Unit }
 *   - Effect handlers:     handle { ... } with { print s -> ... }
 *   - Effect polymorphism: function types carry effect rows
 *   - Effect row integration with the HM type inferencer
 *
 * Design: a lightweight row-based effect system layered on top of HM
 * Algorithm W.  Effects are tracked as sets of effect names ("effect rows")
 * attached to arrow types.  Unification merges rows.  The typechecker
 * validates that every performed effect is handled.
 *
 * Supported built-in effects:
 *   IO    – I/O operations (print, read)
 *   Err   – Error/exception propagation (throw, catch_err)
 *   State – Mutable state (get, put)
 */
#ifndef EFFECTS_H
#define EFFECTS_H

#include "nanolang.h"
#include <stdbool.h>

/* ── Effect operation descriptor ────────────────────────────────────────── */
/* EffectOp is already defined in nanolang.h (Parameter*-based).
 * EffectSysOp is the simplified type used internally by the effect system. */
typedef struct {
    char *name;             /* operation name, e.g. "print" */
    Type  param_type;       /* parameter type */
    Type  return_type;      /* return type */
    char *param_type_name;  /* for struct/union param */
    char *return_type_name; /* for struct/union return */
} EffectSysOp;

/* ── Effect declaration (registered in effect registry) ─────────────────── */

typedef struct {
    char        *name;   /* effect name, e.g. "IO" */
    EffectSysOp *ops;    /* array of operations */
    int          op_count;
} EffectDecl;

/* ── Effect row: a set of effect names ──────────────────────────────────── */

#define EFFECT_ROW_MAX 16

typedef struct {
    char *effects[EFFECT_ROW_MAX]; /* effect names present in this row */
    int   count;
    bool  is_open;  /* true = row is polymorphic (can be extended) */
} EffectRow;

/* ── Effect registry ─────────────────────────────────────────────────────── */

#define EFFECT_REGISTRY_MAX 64

typedef struct {
    EffectDecl decls[EFFECT_REGISTRY_MAX];
    int        count;
} EffectRegistry;

/* ── Handler frame (runtime context for effect handlers) ─────────────────── */

typedef struct EffectHandlerFrame EffectHandlerFrame;
struct EffectHandlerFrame {
    char                  *effect_name;
    /* Handler bodies stored as AST; evaluation is left to the interpreter */
    ASTNode              **handler_bodies;
    char                 **handler_op_names;
    char                 **handler_param_names;
    int                    handler_count;
    EffectHandlerFrame    *outer;   /* enclosing handler (for nested handle) */
    Environment           *env;    /* environment captured at handle site */
};

/* ── Runtime handler stack (global, used by interpreter) ─────────────────── */

/* Push / pop a frame onto the interpreter handler stack */
void nl_effect_frame_push(EffectHandlerFrame *frame);
void nl_effect_frame_pop(void);

/* Find the nearest handler arm matching (effect_name, op_name).
 * Returns the frame that owns it, or NULL if no handler is active.
 * Sets *arm_idx_out to the index within frame->handler_op_names.  */
EffectHandlerFrame *nl_effect_find_handler(const char *effect_name,
                                           const char *op_name,
                                           int        *arm_idx_out);

/* Environment lookup helpers (wrappers that call into effects registry) */
void        env_effect_register(Environment *env, ASTNode *decl_node);
EffectDecl *env_effect_lookup(Environment *env, const char *name);

/* ── Effect inference context ────────────────────────────────────────────── */

typedef struct {
    EffectRegistry  *registry;
    /* Stack of active handler frames for static effect-checking */
    EffectHandlerFrame *handler_stack;
    bool has_error;
} EffectCtx;

/* ── Public API ──────────────────────────────────────────────────────────── */

/* Create / destroy */
EffectRegistry     *effect_registry_new(void);
void                effect_registry_free(EffectRegistry *reg);
EffectCtx          *effect_ctx_new(EffectRegistry *reg);
void                effect_ctx_free(EffectCtx *ctx);

/* Register a built-in or user-defined effect */
bool  effect_register(EffectRegistry *reg, const char *name,
                       EffectSysOp *ops, int op_count);

/* Register a user-declared effect from an AST_EFFECT_DECL node */
bool  effect_register_from_ast(EffectRegistry *reg, ASTNode *decl_node);

/* Look up a registered effect by name */
EffectDecl *effect_lookup(EffectRegistry *reg, const char *name);

/* Look up an operation within an effect */
EffectSysOp *effect_op_lookup(EffectDecl *decl, const char *op_name);

/* Effect row utilities */
EffectRow  *effect_row_new(void);
void        effect_row_free(EffectRow *row);
bool        effect_row_add(EffectRow *row, const char *effect_name);
bool        effect_row_contains(const EffectRow *row, const char *effect_name);
bool        effect_row_merge(EffectRow *dst, const EffectRow *src);  /* dst ∪= src */
bool        effect_row_subset(const EffectRow *sub, const EffectRow *sup);
char       *effect_row_to_str(const EffectRow *row); /* caller must free */

/* Type-check an effect declaration node */
bool effect_check_decl(EffectCtx *ctx, ASTNode *node);

/* Type-check a handle expression */
bool effect_check_handler(EffectCtx *ctx, ASTNode *node, Environment *env);

/* Type-check a perform expression */
bool effect_check_perform(EffectCtx *ctx, ASTNode *node);

/* Walk the entire program and check all effect nodes */
bool effect_check_program(ASTNode *program, EffectRegistry *reg, Environment *env);

/* Pretty-print a function type with effect row: (int) -[IO]-> string */
char *effect_arrow_type_str(const char *param_type, const char *ret_type,
                             const EffectRow *row);  /* caller must free */

/* Register built-in effects: IO, Err, State */
void effect_register_builtins(EffectRegistry *reg);

#endif /* EFFECTS_H */

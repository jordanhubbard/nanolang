/*
 * type_infer.h — Hindley-Milner Algorithm W type inference for nanolang
 *
 * Provides let-polymorphism via Robinson unification + occurs check.
 * Runs as an optional pass (--infer) before the existing type checker.
 *
 * Extended with algebraic effect rows: arrow types carry an EffectRow
 * recording which effects the function may perform.
 *
 * Extended with row-polymorphic records: HM_RECORD carries field labels,
 * field types, and an optional row-variable tail for open record types.
 * See docs/ROW_POLYMORPHIC_RECORDS_DESIGN.md for the full design.
 */
#ifndef TYPE_INFER_H
#define TYPE_INFER_H

#include "nanolang.h"
#include "effects.h"

/* ── HM type representation ───────────────────────────────────────────── */

typedef enum {
    HM_VAR,    /* type variable α */
    HM_CON,    /* concrete type: int, float, bool, string, void */
    HM_ARROW,  /* function type: τ1 -[ε]→ τ2 (ε = effect row) */
    HM_RECORD, /* record type: { f1:τ1, …, fn:τn | ρ }
                  row_tail == NULL → closed record
                  row_tail->kind == HM_VAR → open (row variable ρ) */
} HMKind;

/* Maximum fields per record type node (additional fields via row tail). */
#define HM_RECORD_MAX_FIELDS 64


typedef struct HMType HMType;
struct HMType {
    HMKind kind;
    union {
        int         var_id;        /* HM_VAR */
        const char *con_name;      /* HM_CON: "int", "float", "bool", "string", "void" */
        struct {
            HMType    *param;
            HMType    *ret;
            EffectRow *row;        /* HM_ARROW: effect row (NULL == pure) */
        } arrow;                   /* HM_ARROW */
        struct {
            char     **field_names;  /* sorted field label strings (owned by ctx) */
            HMType   **field_types;  /* parallel array of field types */
            int        field_count;
            HMType    *row_tail;     /* NULL = closed; HM_VAR node = open row var */
        } record;                  /* HM_RECORD */
    } as;
};

/* ── Type scheme: ∀ a1 … an . τ ────────────────────────────────────────── */

typedef struct {
    int    *bound_vars;    /* array of bound type-variable ids */
    int     bound_count;
    HMType *type;
} TypeScheme;

/* ── Substitution: id → HMType* ─────────────────────────────────────────── */

typedef struct SubstEntry {
    int               id;
    HMType           *type;
    struct SubstEntry *next;
} SubstEntry;

typedef struct {
    SubstEntry *head;
} Subst;

/* ── Type environment entry ──────────────────────────────────────────────── */

typedef struct HMEnvEntry {
    const char       *name;
    TypeScheme       *scheme;
    struct HMEnvEntry *next;
} HMEnvEntry;

typedef struct {
    HMEnvEntry *head;
} HMEnv;

/* ── Allocation list (arena-like cleanup) ────────────────────────────────── */

typedef struct AllocNode {
    void            *ptr;
    struct AllocNode *next;
} AllocNode;

/* ── Inference context ───────────────────────────────────────────────────── */

typedef struct {
    int             next_var_id;
    Subst           subst;
    bool            has_error;
    const char     *source_file;
    AllocNode      *allocs;      /* all malloc'd memory; freed by hm_ctx_free */
    EffectRow      *current_effects; /* effects accumulated in current function body */
    EffectRegistry *effect_registry; /* shared effect registry (may be NULL) */
} InferCtx;

/* ── Public API ──────────────────────────────────────────────────────────── */

/* Create / destroy inference context */
InferCtx *hm_ctx_new(const char *source_file);
void      hm_ctx_free(InferCtx *ctx);

/* Run Algorithm W over the whole program AST.
 * Returns true if no type errors were found. */
bool hm_infer_program(ASTNode *program, const char *source_file);

/* Lower-level utilities exposed for testing / LSP integration */
HMType *hm_tv_fresh(InferCtx *ctx);
HMType *hm_con_type(InferCtx *ctx, const char *name);
HMType *hm_arrow_type(InferCtx *ctx, HMType *param, HMType *ret);
HMType *hm_arrow_type_with_effects(InferCtx *ctx, HMType *param, HMType *ret,
                                    EffectRow *row);
HMType *hm_subst_apply(InferCtx *ctx, HMType *t);
bool    hm_unify(InferCtx *ctx, HMType *t1, HMType *t2, int line, int col);
char   *hm_type_to_str(InferCtx *ctx, HMType *t);  /* caller must NOT free — ctx owns it */

/* ── Row-polymorphic record API ──────────────────────────────────────────── */

/*
 * hm_record_type — Build a closed or open record type.
 *
 *   names       : array of field label strings (copied into ctx arena)
 *   types       : parallel array of field HMTypes
 *   count       : number of fields
 *   row_tail    : NULL for a closed record; an HM_VAR node for an open record
 *
 * Fields are sorted lexicographically before storage so that structural
 * equality checks and unification are canonical.
 */
HMType *hm_record_type(InferCtx *ctx,
                        const char **names, HMType **types, int count,
                        HMType *row_tail);

/*
 * hm_record_open — Shorthand: build a record type with a fresh row variable.
 * Equivalent to hm_record_type(ctx, names, types, count, hm_tv_fresh(ctx)).
 */
HMType *hm_record_open(InferCtx *ctx,
                        const char **names, HMType **types, int count);

/*
 * hm_record_closed — Shorthand: build a closed record type.
 * Equivalent to hm_record_type(ctx, names, types, count, NULL).
 */
HMType *hm_record_closed(InferCtx *ctx,
                          const char **names, HMType **types, int count);

/*
 * hm_record_spread — Model a spread expression: { ...base, extra_fields… }
 *
 *   base        : type of the record being spread
 *   extra_names : additional (or overriding) field labels
 *   extra_types : types for the extra fields
 *   extra_count : number of extra fields
 *
 * Returns a new record type that contains all fields from `base` that are
 * not overridden, plus the extra fields.  If `base` has a row variable,
 * the result is still open (carries the same or a fresh row variable).
 */
HMType *hm_record_spread(InferCtx *ctx, HMType *base,
                          const char **extra_names, HMType **extra_types,
                          int extra_count);

/*
 * hm_record_field_type — Look up the type of a field in a (possibly open)
 * record type after applying the current substitution.
 *
 * Returns NULL if the field is not present in the known part of the record
 * and the record is closed.  Returns a fresh type variable if the record is
 * open and the field is not in the known part (adding it to the row).
 */
HMType *hm_record_field_type(InferCtx *ctx, HMType *rec,
                              const char *field_name, int line, int col);

/* Effect-aware inference entry point */
bool hm_infer_program_with_effects(ASTNode *program, const char *source_file,
                                    EffectRegistry *reg);


#endif /* TYPE_INFER_H */

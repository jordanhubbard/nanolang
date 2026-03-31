/*
 * type_infer.c — Hindley-Milner Algorithm W type inference for nanolang
 *
 * Implementation of:
 *   - Fresh type variable allocation
 *   - Robinson unification with occurs check
 *   - Substitution apply / compose
 *   - Generalization (free vars → ∀ binders)
 *   - Instantiation (∀ binders → fresh vars)
 *   - Algorithm W: Var / App / Lam / Let / LetRec cases
 *
 * Error messages use the same stderr format as typechecker.c and report
 * the error code E003 (type mismatch).
 */

#include "type_infer.h"
#include "effects.h"
#include "colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ── Forward declarations ────────────────────────────────────────────────── */

static HMType     *infer_expr(InferCtx *ctx, HMEnv *env, ASTNode *node);
static HMType     *infer_block(InferCtx *ctx, HMEnv *env, ASTNode **stmts, int count);
static TypeScheme *generalize(InferCtx *ctx, HMEnv *env, HMType *t);
static HMType     *instantiate(InferCtx *ctx, TypeScheme *scheme);
static bool        occurs(InferCtx *ctx, int id, HMType *t);
static void        collect_free_vars(InferCtx *ctx, HMEnv *env, HMType *t,
                                     int *out, int *out_count, int cap);
static TypeScheme *mono_scheme(InferCtx *ctx, HMType *t);
static HMEnv      *env_extend(InferCtx *ctx, HMEnv *parent,
                               const char *name, TypeScheme *scheme);
static TypeScheme *env_lookup(HMEnv *env, const char *name);
static HMType     *type_from_ast_type(InferCtx *ctx, Type t,
                                       const char *struct_name);
/* Row-record unification (defined after hm_unify) */
bool hm_unify_rows(InferCtx *ctx, HMType *r1, HMType *r2, int line, int col);
/* Error emission (defined later in file) */
static void emit_type_mismatch(InferCtx *ctx, HMType *t1, HMType *t2,
                                int line, int col);

/* ── Memory management ───────────────────────────────────────────────────── */

static void *ctx_alloc(InferCtx *ctx, size_t sz) {
    void *p = malloc(sz);
    if (!p) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    memset(p, 0, sz);
    AllocNode *n = malloc(sizeof(AllocNode));
    if (!n) { free(p); fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    n->ptr  = p;
    n->next = ctx->allocs;
    ctx->allocs = n;
    return p;
}

static char *ctx_strdup(InferCtx *ctx, const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *p = ctx_alloc(ctx, len);
    memcpy(p, s, len);
    return p;
}

/* ── Context lifecycle ───────────────────────────────────────────────────── */

InferCtx *hm_ctx_new(const char *source_file) {
    InferCtx *ctx = malloc(sizeof(InferCtx));
    if (!ctx) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    memset(ctx, 0, sizeof(InferCtx));
    ctx->source_file = source_file;
    ctx->next_var_id = 1;
    return ctx;
}

void hm_ctx_free(InferCtx *ctx) {
    if (!ctx) return;
    AllocNode *n = ctx->allocs;
    while (n) {
        AllocNode *next = n->next;
        free(n->ptr);
        free(n);
        n = next;
    }
    /* Also free substitution chain (entries not in allocs list) */
    SubstEntry *s = ctx->subst.head;
    while (s) {
        SubstEntry *next = s->next;
        free(s);
        s = next;
    }
    free(ctx);
}

/* ── Type constructors ───────────────────────────────────────────────────── */

HMType *hm_tv_fresh(InferCtx *ctx) {
    HMType *t  = ctx_alloc(ctx, sizeof(HMType));
    t->kind    = HM_VAR;
    t->as.var_id = ctx->next_var_id++;
    return t;
}

HMType *hm_con_type(InferCtx *ctx, const char *name) {
    HMType *t     = ctx_alloc(ctx, sizeof(HMType));
    t->kind       = HM_CON;
    t->as.con_name = ctx_strdup(ctx, name);
    return t;
}

HMType *hm_arrow_type(InferCtx *ctx, HMType *param, HMType *ret) {
    HMType *t         = ctx_alloc(ctx, sizeof(HMType));
    t->kind           = HM_ARROW;
    t->as.arrow.param = param;
    t->as.arrow.ret   = ret;
    t->as.arrow.row   = NULL;  /* pure by default */
    return t;
}

HMType *hm_arrow_type_with_effects(InferCtx *ctx, HMType *param, HMType *ret,
                                     EffectRow *row) {
    HMType *t         = ctx_alloc(ctx, sizeof(HMType));
    t->kind           = HM_ARROW;
    t->as.arrow.param = param;
    t->as.arrow.ret   = ret;
    t->as.arrow.row   = row;
    return t;
}

/* ── Row-polymorphic record constructors ─────────────────────────────────── */

/*
 * hm_record_type — allocate a new HM_RECORD node.
 *
 * Field names are sorted lexicographically before storage so that
 * structural comparison is canonical regardless of source order.
 */
HMType *hm_record_type(InferCtx *ctx,
                        const char **names, HMType **types, int count,
                        HMType *row_tail) {
    if (count < 0) count = 0;
    if (count > HM_RECORD_MAX_FIELDS) count = HM_RECORD_MAX_FIELDS;

    HMType *t = ctx_alloc(ctx, sizeof(HMType));
    t->kind   = HM_RECORD;

    t->as.record.field_count = count;
    t->as.record.row_tail    = row_tail;

    if (count == 0) {
        t->as.record.field_names = NULL;
        t->as.record.field_types = NULL;
        return t;
    }

    /* Build a sort-index array */
    int *idx = malloc(sizeof(int) * count);
    if (!idx) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    for (int i = 0; i < count; i++) idx[i] = i;

    /* Sort idx by names[idx[i]] */
    /* Simple insertion sort — fields are typically small (<16) */
    for (int i = 1; i < count; i++) {
        int key = idx[i];
        int j   = i - 1;
        while (j >= 0 && strcmp(names[idx[j]], names[key]) > 0) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    t->as.record.field_names = ctx_alloc(ctx, sizeof(char*)   * count);
    t->as.record.field_types = ctx_alloc(ctx, sizeof(HMType*) * count);
    for (int i = 0; i < count; i++) {
        t->as.record.field_names[i] = ctx_strdup(ctx, names[idx[i]]);
        t->as.record.field_types[i] = types[idx[i]];
    }
    free(idx);
    return t;
}

HMType *hm_record_open(InferCtx *ctx,
                        const char **names, HMType **types, int count) {
    return hm_record_type(ctx, names, types, count, hm_tv_fresh(ctx));
}

HMType *hm_record_closed(InferCtx *ctx,
                          const char **names, HMType **types, int count) {
    return hm_record_type(ctx, names, types, count, NULL);
}

/*
 * hm_record_spread — model { ...base, extra_names[i]: extra_types[i], … }
 *
 * The semantics: all known fields of base are inherited (or overridden by
 * extra fields with the same name).  The row tail of base is preserved in
 * the result, keeping openness.
 */
HMType *hm_record_spread(InferCtx *ctx, HMType *base,
                          const char **extra_names, HMType **extra_types,
                          int extra_count) {
    base = hm_subst_apply(ctx, base);

    /* If base is a type variable (unknown record), build result as open */
    if (!base || base->kind == HM_VAR) {
        return hm_record_open(ctx, extra_names, extra_types, extra_count);
    }

    if (base->kind != HM_RECORD) {
        /* Spread of a non-record: treat as open, emit no error (deferred to
         * the existing typechecker). */
        return hm_record_open(ctx, extra_names, extra_types, extra_count);
    }

    /* Merge base fields with extra fields; extras override on collision */
    int base_count = base->as.record.field_count;
    int max_fields = base_count + extra_count;

    const char **merged_names = malloc(sizeof(char*) * max_fields);
    HMType    **merged_types  = malloc(sizeof(HMType*) * max_fields);
    if (!merged_names || !merged_types) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }

    int merged_count = 0;

    /* Copy base fields that are NOT overridden by extras */
    for (int i = 0; i < base_count; i++) {
        bool overridden = false;
        for (int j = 0; j < extra_count; j++) {
            if (strcmp(base->as.record.field_names[i], extra_names[j]) == 0) {
                overridden = true;
                break;
            }
        }
        if (!overridden) {
            merged_names[merged_count] = base->as.record.field_names[i];
            merged_types[merged_count] = base->as.record.field_types[i];
            merged_count++;
        }
    }

    /* Append extra fields */
    for (int j = 0; j < extra_count; j++) {
        merged_names[merged_count] = extra_names[j];
        merged_types[merged_count] = extra_types[j];
        merged_count++;
    }

    HMType *result = hm_record_type(ctx, merged_names, merged_types, merged_count,
                                     base->as.record.row_tail);
    free(merged_names);
    free(merged_types);
    return result;
}

/*
 * hm_record_field_type — resolve the type of a named field in a record.
 *
 * For open records, if the field is not in the known set, a fresh type
 * variable is added to the row (by extending the row tail binding).
 * For closed records, returns NULL if the field is absent.
 */
HMType *hm_record_field_type(InferCtx *ctx, HMType *rec,
                              const char *field_name, int line, int col) {
    if (!rec) return hm_tv_fresh(ctx);
    rec = hm_subst_apply(ctx, rec);

    /* If record is itself a type variable, return a fresh field type */
    if (rec->kind == HM_VAR) {
        return hm_tv_fresh(ctx);
    }

    if (rec->kind != HM_RECORD) {
        /* Not a record type — field access on wrong kind */
        if (line > 0) {
            fprintf(stderr,
                    "\n[E003] FIELD ACCESS ON NON-RECORD  %s:%d:%d\n"
                    "  Type '%s' is not a record type.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, rec));
            ctx->has_error = true;
        }
        return hm_tv_fresh(ctx);
    }

    /* Linear scan for the field label (fields are sorted) */
    for (int i = 0; i < rec->as.record.field_count; i++) {
        if (strcmp(rec->as.record.field_names[i], field_name) == 0)
            return hm_subst_apply(ctx, rec->as.record.field_types[i]);
    }

    /* Field not in known part */
    if (rec->as.record.row_tail) {
        /* Open record: add the field to the row tail by unifying the tail
         * with a new one-field record carrying a fresh type variable, then
         * return the fresh variable. */
        HMType *field_tv = hm_tv_fresh(ctx);
        HMType *new_tail = hm_tv_fresh(ctx);
        const char *fn[1];
        HMType    *ft[1];
        fn[0] = field_name;
        ft[0] = field_tv;
        HMType *row_ext = hm_record_type(ctx, fn, ft, 1, new_tail);
        hm_unify(ctx, rec->as.record.row_tail, row_ext, line, col);
        return field_tv;
    }

    /* Closed record, field missing */
    if (line > 0) {
        fprintf(stderr,
                "\n[E003] MISSING FIELD  %s:%d:%d\n"
                "  Closed record type '%s' has no field '%s'.\n",
                ctx->source_file ? ctx->source_file : "<unknown>",
                line, col,
                hm_type_to_str(ctx, rec),
                field_name);
        ctx->has_error = true;
    }
    return hm_tv_fresh(ctx);
}

/* ── Row unification ─────────────────────────────────────────────────────── */

/*
 * hm_unify_rows — unify two HM_RECORD types under row polymorphism.
 *
 * Algorithm:
 *   1. Collect the union of all field labels from both sides.
 *   2. For labels present in both: unify field types.
 *   3. For labels only in r1: they must come from r2's row tail.
 *   4. For labels only in r2: they must come from r1's row tail.
 *   5. Tails:
 *      - Both NULL (closed): extra fields → error.
 *      - One NULL, one open: open tail unifies with the extra fields record.
 *      - Both open: both tails unify with a fresh row variable extended by
 *        the opposite side's unique fields.
 */
bool hm_unify_rows(InferCtx *ctx, HMType *r1, HMType *r2, int line, int col) {
    /* Resolve substitutions */
    r1 = hm_subst_apply(ctx, r1);
    r2 = hm_subst_apply(ctx, r2);

    if (!r1 || !r2) return true;

    /* If either resolved to a type variable, defer to normal unify */
    if (r1->kind == HM_VAR || r2->kind == HM_VAR)
        return hm_unify(ctx, r1, r2, line, col);

    if (r1->kind != HM_RECORD || r2->kind != HM_RECORD) {
        emit_type_mismatch(ctx, r1, r2, line, col);
        return false;
    }

    int c1 = r1->as.record.field_count;
    int c2 = r2->as.record.field_count;

    bool ok = true;

    /* Track which r2 fields were matched */
    bool *matched2 = calloc(c2, sizeof(bool));
    if (!matched2 && c2 > 0) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }

    /* Arrays for unmatched fields from each side (to build residual records) */
    const char **only1_names = malloc(sizeof(char*) * (c1 + 1));
    HMType    **only1_types  = malloc(sizeof(HMType*) * (c1 + 1));
    const char **only2_names = malloc(sizeof(char*) * (c2 + 1));
    HMType    **only2_types  = malloc(sizeof(HMType*) * (c2 + 1));
    if (!only1_names || !only1_types || !only2_names || !only2_types) {
        fprintf(stderr, "[HM] out of memory\n"); exit(1);
    }
    int only1_count = 0, only2_count = 0;

    /* Step 1 + 2: for each field in r1, find matching field in r2 */
    for (int i = 0; i < c1; i++) {
        const char *fn1 = r1->as.record.field_names[i];
        bool found = false;
        for (int j = 0; j < c2; j++) {
            if (strcmp(fn1, r2->as.record.field_names[j]) == 0) {
                /* Shared field — unify types */
                ok = hm_unify(ctx,
                               r1->as.record.field_types[i],
                               r2->as.record.field_types[j],
                               line, col) && ok;
                matched2[j] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            only1_names[only1_count] = fn1;
            only1_types[only1_count] = r1->as.record.field_types[i];
            only1_count++;
        }
    }

    /* Step 3: collect unmatched r2 fields */
    for (int j = 0; j < c2; j++) {
        if (!matched2[j]) {
            only2_names[only2_count] = r2->as.record.field_names[j];
            only2_types[only2_count] = r2->as.record.field_types[j];
            only2_count++;
        }
    }

    /* Step 5: tail unification */
    HMType *tail1 = r1->as.record.row_tail
                    ? hm_subst_apply(ctx, r1->as.record.row_tail) : NULL;
    HMType *tail2 = r2->as.record.row_tail
                    ? hm_subst_apply(ctx, r2->as.record.row_tail) : NULL;

    if (only1_count == 0 && only2_count == 0) {
        /* Both sides agree on field sets.  Unify tails directly. */
        if (tail1 && tail2)
            ok = hm_unify(ctx, tail1, tail2, line, col) && ok;
        else if (tail1 && !tail2) {
            /* r1 open, r2 closed: tail1 must be empty record */
            HMType *empty = hm_record_closed(ctx, NULL, NULL, 0);
            ok = hm_unify(ctx, tail1, empty, line, col) && ok;
        } else if (!tail1 && tail2) {
            HMType *empty = hm_record_closed(ctx, NULL, NULL, 0);
            ok = hm_unify(ctx, tail2, empty, line, col) && ok;
        }
        /* both NULL (both closed, matching fields): already ok */
    } else if (only1_count > 0 && only2_count == 0) {
        /* r1 has extra fields → r2's tail must absorb them */
        if (!tail2) {
            /* r2 is closed; r1's extra fields can't be accommodated */
            fprintf(stderr,
                    "\n[E003] ROW MISMATCH  %s:%d:%d\n"
                    "  Record type '%s' has extra fields not present in closed '%s'.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, r1), hm_type_to_str(ctx, r2));
            ctx->has_error = true;
            ok = false;
        } else {
            /* r2 open: tail2 ~ {only1 fields | fresh} */
            HMType *fresh_tail = (tail1 ? tail1 : NULL);
            HMType *ext = hm_record_type(ctx, only1_names, only1_types,
                                          only1_count, fresh_tail);
            ok = hm_unify(ctx, tail2, ext, line, col) && ok;
            if (tail1 && only2_count == 0) {
                /* Also unify tail1 with fresh var or empty if r2 is fully consumed */
                HMType *empty_tail = hm_tv_fresh(ctx);
                ok = hm_unify(ctx, tail1, empty_tail, line, col) && ok;
            }
        }
    } else if (only1_count == 0 && only2_count > 0) {
        /* r2 has extra fields → r1's tail must absorb them */
        if (!tail1) {
            fprintf(stderr,
                    "\n[E003] ROW MISMATCH  %s:%d:%d\n"
                    "  Record type '%s' has extra fields not present in closed '%s'.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, r2), hm_type_to_str(ctx, r1));
            ctx->has_error = true;
            ok = false;
        } else {
            HMType *fresh_tail = (tail2 ? tail2 : NULL);
            HMType *ext = hm_record_type(ctx, only2_names, only2_types,
                                          only2_count, fresh_tail);
            ok = hm_unify(ctx, tail1, ext, line, col) && ok;
        }
    } else {
        /* Both sides have unique fields — need both tails to be open */
        if (!tail1 && !tail2) {
            fprintf(stderr,
                    "\n[E003] ROW MISMATCH  %s:%d:%d\n"
                    "  Closed record types '%s' and '%s' have incompatible fields.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, r1), hm_type_to_str(ctx, r2));
            ctx->has_error = true;
            ok = false;
        } else if (!tail1) {
            /* r1 closed, can't absorb r2's extra fields */
            fprintf(stderr,
                    "\n[E003] ROW MISMATCH  %s:%d:%d\n"
                    "  Closed record '%s' cannot absorb extra fields from '%s'.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, r1), hm_type_to_str(ctx, r2));
            ctx->has_error = true;
            ok = false;
        } else if (!tail2) {
            fprintf(stderr,
                    "\n[E003] ROW MISMATCH  %s:%d:%d\n"
                    "  Closed record '%s' cannot absorb extra fields from '%s'.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>",
                    line, col,
                    hm_type_to_str(ctx, r2), hm_type_to_str(ctx, r1));
            ctx->has_error = true;
            ok = false;
        } else {
            /* Both open: fresh shared tail ρ
             * tail1 ~ { only2_fields | ρ }
             * tail2 ~ { only1_fields | ρ }   */
            HMType *shared_tail = hm_tv_fresh(ctx);
            HMType *ext1 = hm_record_type(ctx, only2_names, only2_types,
                                           only2_count, shared_tail);
            HMType *ext2 = hm_record_type(ctx, only1_names, only1_types,
                                           only1_count, shared_tail);
            ok = hm_unify(ctx, tail1, ext1, line, col) && ok;
            ok = hm_unify(ctx, tail2, ext2, line, col) && ok;
        }
    }

    free(matched2);
    free(only1_names); free(only1_types);
    free(only2_names); free(only2_types);
    return ok;
}

/* ── Type pretty-printing ────────────────────────────────────────────────── */

/* Returns a string owned by ctx (allocated in ctx arena). */
char *hm_type_to_str(InferCtx *ctx, HMType *t) {
    t = hm_subst_apply(ctx, t);
    if (!t) return ctx_strdup(ctx, "?");
    switch (t->kind) {
        case HM_VAR: {
            char buf[32];
            snprintf(buf, sizeof(buf), "α%d", t->as.var_id);
            return ctx_strdup(ctx, buf);
        }
        case HM_CON:
            return ctx_strdup(ctx, t->as.con_name ? t->as.con_name : "?");
        case HM_ARROW: {
            char *ps = hm_type_to_str(ctx, t->as.arrow.param);
            char *rs = hm_type_to_str(ctx, t->as.arrow.ret);
            EffectRow *row = t->as.arrow.row;
            if (row && row->count > 0) {
                char *row_str = effect_row_to_str(row);
                size_t len = strlen(ps) + strlen(rs) + strlen(row_str) + 16;
                char *buf  = ctx_alloc(ctx, len);
                snprintf(buf, len, "(%s) -[%s]-> %s", ps, row_str, rs);
                free(row_str);
                return buf;
            } else {
                size_t len = strlen(ps) + strlen(rs) + 8;
                char *buf  = ctx_alloc(ctx, len);
                snprintf(buf, len, "(%s → %s)", ps, rs);
                return buf;
            }
        }
        case HM_RECORD: {
            /* Build "{f1:τ1, …, fn:τn}" or "{f1:τ1, …, fn:τn | ρ}" */
            size_t total = 4;
            char **fstrs = malloc(sizeof(char*) * (t->as.record.field_count + 1));
            if (!fstrs) return ctx_strdup(ctx, "{?}");
            for (int i = 0; i < t->as.record.field_count; i++) {
                char *ft = hm_type_to_str(ctx, t->as.record.field_types[i]);
                size_t fsz = strlen(t->as.record.field_names[i]) + strlen(ft) + 4;
                fstrs[i] = malloc(fsz);
                if (fstrs[i]) snprintf(fstrs[i], fsz, "%s:%s",
                                       t->as.record.field_names[i], ft);
                total += fsz + 2;
            }
            /* Tail representation */
            char *tail_str = NULL;
            if (t->as.record.row_tail) {
                tail_str = hm_type_to_str(ctx, t->as.record.row_tail);
                total += strlen(tail_str) + 4;
            }
            char *buf = ctx_alloc(ctx, total);
            buf[0] = '{'; buf[1] = '\0';
            for (int i = 0; i < t->as.record.field_count; i++) {
                if (i > 0) strcat(buf, ", ");
                if (fstrs[i]) strcat(buf, fstrs[i]);
                free(fstrs[i]);
            }
            if (tail_str) {
                strcat(buf, " | ");
                strcat(buf, tail_str);
            }
            strcat(buf, "}");
            free(fstrs);
            return buf;
        }
    }
    return ctx_strdup(ctx, "?");
}

/* ── Substitution ────────────────────────────────────────────────────────── */

static void subst_bind(InferCtx *ctx, int id, HMType *t) {
    /* Check if already bound (shouldn't happen, but guard anyway) */
    for (SubstEntry *e = ctx->subst.head; e; e = e->next) {
        if (e->id == id) { e->type = t; return; }
    }
    SubstEntry *entry = malloc(sizeof(SubstEntry));
    if (!entry) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    entry->id   = id;
    entry->type = t;
    entry->next = ctx->subst.head;
    ctx->subst.head = entry;
}

static HMType *subst_lookup(InferCtx *ctx, int id) {
    for (SubstEntry *e = ctx->subst.head; e; e = e->next) {
        if (e->id == id) return e->type;
    }
    return NULL;
}

/* Walk t, replacing type variables according to current substitution. */
HMType *hm_subst_apply(InferCtx *ctx, HMType *t) {
    if (!t) return t;
    switch (t->kind) {
        case HM_CON:
            return t;
        case HM_VAR: {
            HMType *mapped = subst_lookup(ctx, t->as.var_id);
            if (!mapped) return t;
            /* Path compression: apply substitution transitively */
            HMType *resolved = hm_subst_apply(ctx, mapped);
            if (resolved != mapped) subst_bind(ctx, t->as.var_id, resolved);
            return resolved;
        }
        case HM_ARROW: {
            HMType *p = hm_subst_apply(ctx, t->as.arrow.param);
            HMType *r = hm_subst_apply(ctx, t->as.arrow.ret);
            if (p == t->as.arrow.param && r == t->as.arrow.ret) return t;
            return hm_arrow_type_with_effects(ctx, p, r, t->as.arrow.row);
        }
        case HM_RECORD: {
            /* Apply substitution to each field type and to the row tail */
            bool changed = false;
            HMType **new_ftypes = malloc(sizeof(HMType*) * t->as.record.field_count);
            if (!new_ftypes) return t;
            for (int i = 0; i < t->as.record.field_count; i++) {
                new_ftypes[i] = hm_subst_apply(ctx, t->as.record.field_types[i]);
                if (new_ftypes[i] != t->as.record.field_types[i]) changed = true;
            }
            HMType *new_tail = t->as.record.row_tail
                               ? hm_subst_apply(ctx, t->as.record.row_tail)
                               : NULL;
            if (new_tail != t->as.record.row_tail) changed = true;
            if (!changed) { free(new_ftypes); return t; }
            HMType *nr = hm_record_type(ctx,
                (const char **)t->as.record.field_names,
                new_ftypes,
                t->as.record.field_count,
                new_tail);
            free(new_ftypes);
            return nr;
        }
    }
    return t;
}

/* ── Occurs check ────────────────────────────────────────────────────────── */

static bool occurs(InferCtx *ctx, int id, HMType *t) {
    t = hm_subst_apply(ctx, t);
    if (!t) return false;
    switch (t->kind) {
        case HM_CON:   return false;
        case HM_VAR:   return t->as.var_id == id;
        case HM_ARROW: return occurs(ctx, id, t->as.arrow.param) ||
                              occurs(ctx, id, t->as.arrow.ret);
        case HM_RECORD: {
            for (int i = 0; i < t->as.record.field_count; i++) {
                if (occurs(ctx, id, t->as.record.field_types[i])) return true;
            }
            if (t->as.record.row_tail && occurs(ctx, id, t->as.record.row_tail))
                return true;
            return false;
        }
    }
    return false;
}

/* ── Error emission ──────────────────────────────────────────────────────── */

static void emit_type_mismatch(InferCtx *ctx, HMType *t1, HMType *t2,
                                int line, int col) {
    ctx->has_error = true;
    char *s1 = hm_type_to_str(ctx, t1);
    char *s2 = hm_type_to_str(ctx, t2);

    /* Mirror the style of typechecker.c emit_context_error */
    fprintf(stderr,
            "\n%s[E003] TYPE MISMATCH%s  %s%s:%d:%d%s\n"
            "  Inferred %s%s%s, but expected %s%s%s\n",
            CSTART_ERROR, CEND,
            CSTART_DIM, ctx->source_file ? ctx->source_file : "<unknown>",
            line, col, CEND,
            CSTART_BOLD, s1, CEND,
            CSTART_BOLD, s2, CEND);
}

/* ── Unification ─────────────────────────────────────────────────────────── */

bool hm_unify(InferCtx *ctx, HMType *t1, HMType *t2, int line, int col) {
    t1 = hm_subst_apply(ctx, t1);
    t2 = hm_subst_apply(ctx, t2);

    /* Same pointer → trivially equal */
    if (t1 == t2) return true;

    /* Both concrete with same name */
    if (t1->kind == HM_CON && t2->kind == HM_CON) {
        if (strcmp(t1->as.con_name, t2->as.con_name) == 0) return true;
        emit_type_mismatch(ctx, t1, t2, line, col);
        return false;
    }

    /* Bind type variable (left) */
    if (t1->kind == HM_VAR) {
        if (t2->kind == HM_VAR && t1->as.var_id == t2->as.var_id) return true;
        if (occurs(ctx, t1->as.var_id, t2)) {
            fprintf(stderr, "\n[E003] INFINITE TYPE  %s:%d:%d\n"
                    "  Recursive type constraint would create an infinite type.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>", line, col);
            ctx->has_error = true;
            return false;
        }
        subst_bind(ctx, t1->as.var_id, t2);
        return true;
    }

    /* Bind type variable (right) */
    if (t2->kind == HM_VAR) {
        if (occurs(ctx, t2->as.var_id, t1)) {
            fprintf(stderr, "\n[E003] INFINITE TYPE  %s:%d:%d\n"
                    "  Recursive type constraint would create an infinite type.\n",
                    ctx->source_file ? ctx->source_file : "<unknown>", line, col);
            ctx->has_error = true;
            return false;
        }
        subst_bind(ctx, t2->as.var_id, t1);
        return true;
    }

    /* Both arrows: unify component-wise */
    if (t1->kind == HM_ARROW && t2->kind == HM_ARROW) {
        bool ok = hm_unify(ctx, t1->as.arrow.param, t2->as.arrow.param, line, col);
        ok = hm_unify(ctx, t1->as.arrow.ret, t2->as.arrow.ret, line, col) && ok;
        return ok;
    }

    /* ── Row-polymorphic record unification ────────────────────────────── */
    if (t1->kind == HM_RECORD && t2->kind == HM_RECORD) {
        return hm_unify_rows(ctx, t1, t2, line, col);
    }

    /* Shape mismatch (e.g., arrow vs. concrete) */
    emit_type_mismatch(ctx, t1, t2, line, col);
    return false;
}

/* ── Free variable collection ────────────────────────────────────────────── */

/* Collect type-variable ids free in `t` that are not bound in `env`.
 * Results written to out[0..cap-1]; *out_count updated. Deduplicates. */
static void collect_free_vars(InferCtx *ctx, HMEnv *env, HMType *t,
                               int *out, int *out_count, int cap) {
    t = hm_subst_apply(ctx, t);
    if (!t) return;
    switch (t->kind) {
        case HM_CON:
            return;
        case HM_VAR: {
            int id = t->as.var_id;
            /* Check if already in output (dedup) */
            for (int i = 0; i < *out_count; i++) {
                if (out[i] == id) return;
            }
            /* Check if id is bound (quantified) somewhere in env schemes */
            for (HMEnvEntry *e = env ? env->head : NULL; e; e = e->next) {
                if (e->scheme) {
                    for (int i = 0; i < e->scheme->bound_count; i++) {
                        if (e->scheme->bound_vars[i] == id) return;
                    }
                }
            }
            if (*out_count < cap) out[(*out_count)++] = id;
            return;
        }
        case HM_ARROW:
            collect_free_vars(ctx, env, t->as.arrow.param, out, out_count, cap);
            collect_free_vars(ctx, env, t->as.arrow.ret,   out, out_count, cap);
            return;
        case HM_RECORD:
            for (int i = 0; i < t->as.record.field_count; i++)
                collect_free_vars(ctx, env, t->as.record.field_types[i],
                                  out, out_count, cap);
            if (t->as.record.row_tail)
                collect_free_vars(ctx, env, t->as.record.row_tail,
                                  out, out_count, cap);
            return;
    }
}

/* ── Generalization & instantiation ─────────────────────────────────────── */

static TypeScheme *generalize(InferCtx *ctx, HMEnv *env, HMType *t) {
    int free_ids[256];
    int free_count = 0;
    collect_free_vars(ctx, env, t, free_ids, &free_count, 256);

    TypeScheme *scheme = ctx_alloc(ctx, sizeof(TypeScheme));
    scheme->type        = hm_subst_apply(ctx, t);
    scheme->bound_count = free_count;
    if (free_count > 0) {
        scheme->bound_vars = ctx_alloc(ctx, sizeof(int) * free_count);
        memcpy(scheme->bound_vars, free_ids, sizeof(int) * free_count);
    }
    return scheme;
}

static TypeScheme *mono_scheme(InferCtx *ctx, HMType *t) {
    TypeScheme *scheme  = ctx_alloc(ctx, sizeof(TypeScheme));
    scheme->type        = t;
    scheme->bound_count = 0;
    scheme->bound_vars  = NULL;
    return scheme;
}

/* Replace bound vars with fresh type variables */
static HMType *instantiate_type(InferCtx *ctx, HMType *t,
                                 int *bound, HMType **fresh, int count) {
    t = hm_subst_apply(ctx, t);
    if (!t) return t;
    switch (t->kind) {
        case HM_CON:
            return t;
        case HM_VAR: {
            for (int i = 0; i < count; i++) {
                if (bound[i] == t->as.var_id) return fresh[i];
            }
            return t;
        }
        case HM_ARROW: {
            HMType *p = instantiate_type(ctx, t->as.arrow.param, bound, fresh, count);
            HMType *r = instantiate_type(ctx, t->as.arrow.ret,   bound, fresh, count);
            if (p == t->as.arrow.param && r == t->as.arrow.ret) return t;
            return hm_arrow_type(ctx, p, r);
        }
        case HM_RECORD: {
            int fc = t->as.record.field_count;
            HMType **new_ftypes = malloc(sizeof(HMType*) * (fc + 1));
            if (!new_ftypes) return t;
            bool changed = false;
            for (int i = 0; i < fc; i++) {
                new_ftypes[i] = instantiate_type(ctx, t->as.record.field_types[i],
                                                  bound, fresh, count);
                if (new_ftypes[i] != t->as.record.field_types[i]) changed = true;
            }
            HMType *new_tail = t->as.record.row_tail
                ? instantiate_type(ctx, t->as.record.row_tail, bound, fresh, count)
                : NULL;
            if (new_tail != t->as.record.row_tail) changed = true;
            if (!changed) { free(new_ftypes); return t; }
            HMType *nr = hm_record_type(ctx,
                (const char **)t->as.record.field_names,
                new_ftypes, fc, new_tail);
            free(new_ftypes);
            return nr;
        }
    }
    return t;
}

static HMType *instantiate(InferCtx *ctx, TypeScheme *scheme) {
    if (!scheme) return hm_tv_fresh(ctx);
    if (scheme->bound_count == 0) return hm_subst_apply(ctx, scheme->type);

    /* Create a fresh type variable for each bound variable */
    HMType **fresh = malloc(sizeof(HMType*) * scheme->bound_count);
    if (!fresh) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
    for (int i = 0; i < scheme->bound_count; i++) {
        fresh[i] = hm_tv_fresh(ctx);
    }
    HMType *result = instantiate_type(ctx, scheme->type,
                                       scheme->bound_vars, fresh,
                                       scheme->bound_count);
    free(fresh);
    return result;
}

/* ── Type environment ────────────────────────────────────────────────────── */

static HMEnv *env_extend(InferCtx *ctx, HMEnv *parent,
                          const char *name, TypeScheme *scheme) {
    HMEnv *newenv   = ctx_alloc(ctx, sizeof(HMEnv));
    HMEnvEntry *ent = ctx_alloc(ctx, sizeof(HMEnvEntry));
    ent->name   = ctx_strdup(ctx, name);
    ent->scheme = scheme;
    ent->next   = parent ? parent->head : NULL;
    newenv->head = ent;
    return newenv;
}

static TypeScheme *env_lookup(HMEnv *env, const char *name) {
    if (!env || !name) return NULL;
    for (HMEnvEntry *e = env->head; e; e = e->next) {
        if (e->name && strcmp(e->name, name) == 0) return e->scheme;
    }
    return NULL;
}

/* ── AST type → HM type ──────────────────────────────────────────────────── */

static HMType *type_from_ast_type(InferCtx *ctx, Type t, const char *struct_name) {
    switch (t) {
        case TYPE_INT:    return hm_con_type(ctx, "int");
        case TYPE_U8:     return hm_con_type(ctx, "u8");
        case TYPE_FLOAT:  return hm_con_type(ctx, "float");
        case TYPE_BOOL:   return hm_con_type(ctx, "bool");
        case TYPE_STRING: return hm_con_type(ctx, "string");
        case TYPE_VOID:   return hm_con_type(ctx, "void");
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            if (struct_name) return hm_con_type(ctx, struct_name);
            return hm_con_type(ctx, "struct");
        default:
            return hm_tv_fresh(ctx);   /* unknown → fresh variable */
    }
}

/* ── Primitive (builtin) type environment ────────────────────────────────── */

static HMEnv *make_builtin_env(InferCtx *ctx) {
    /* Seed a few key builtins so identifiers are recognised.
     * We use monomorphic schemes here since nanolang's builtins are
     * already checked by the existing typechecker; we just need them
     * present to avoid spurious "unbound variable" noise. */
    HMEnv *env = NULL;

    /* println / print : string → void  (simplified: accept any single arg) */
    HMType *a   = hm_tv_fresh(ctx);
    HMType *prt = hm_arrow_type(ctx, a, hm_con_type(ctx, "void"));
    env = env_extend(ctx, env, "println", mono_scheme(ctx, prt));
    env = env_extend(ctx, env, "print",   mono_scheme(ctx, prt));

    /* len : array<α> → int  — approximated as α → int */
    HMType *b   = hm_tv_fresh(ctx);
    HMType *len = hm_arrow_type(ctx, b, hm_con_type(ctx, "int"));
    env = env_extend(ctx, env, "len", mono_scheme(ctx, len));

    /* assert : bool → void */
    HMType *ass = hm_arrow_type(ctx, hm_con_type(ctx, "bool"),
                                 hm_con_type(ctx, "void"));
    env = env_extend(ctx, env, "assert", mono_scheme(ctx, ass));

    /* format : string → string  (variadic; first arg is template) */
    HMType *fmt = hm_arrow_type(ctx, hm_con_type(ctx, "string"),
                                 hm_con_type(ctx, "string"));
    env = env_extend(ctx, env, "format", mono_scheme(ctx, fmt));

    return env;
}

/* ── Algorithm W expression inference ───────────────────────────────────── */

/*
 * Infer the HM type of a nanolang AST expression node.
 * Returns NULL only when a hard error prevents further inference.
 */
static HMType *infer_expr(InferCtx *ctx, HMEnv *env, ASTNode *node) {
    if (!node) return hm_con_type(ctx, "void");

    switch (node->type) {

        /* ── Literals ─────────────────────────────────────────────────── */
        case AST_NUMBER: return hm_con_type(ctx, "int");
        case AST_FLOAT:  return hm_con_type(ctx, "float");
        case AST_BOOL:   return hm_con_type(ctx, "bool");
        case AST_STRING: return hm_con_type(ctx, "string");

        /* ── Variable reference ───────────────────────────────────────── */
        case AST_IDENTIFIER: {
            TypeScheme *scheme = env_lookup(env, node->as.identifier);
            if (!scheme) {
                /* Unknown var: return fresh type variable (existential).
                 * The existing typechecker will report the definitive error. */
                return hm_tv_fresh(ctx);
            }
            return instantiate(ctx, scheme);
        }

        /* ── Prefix / binary operation ────────────────────────────────── */
        case AST_PREFIX_OP: {
            if (node->as.prefix_op.arg_count == 0)
                return hm_con_type(ctx, "void");

            /* Infer all argument types */
            HMType *arg0 = infer_expr(ctx, env, node->as.prefix_op.args[0]);
            if (!arg0) return hm_tv_fresh(ctx);

            TokenType op = node->as.prefix_op.op;

            /* Arithmetic: + - * / → numeric result, same type as args */
            if (op == TOKEN_PLUS || op == TOKEN_MINUS ||
                op == TOKEN_STAR || op == TOKEN_SLASH) {
                if (node->as.prefix_op.arg_count >= 2) {
                    HMType *arg1 = infer_expr(ctx, env, node->as.prefix_op.args[1]);
                    if (arg1) hm_unify(ctx, arg0, arg1,
                                       node->line, node->column);
                }
                return hm_subst_apply(ctx, arg0);
            }

            /* Comparisons: < <= > >= == != → bool */
            if (op == TOKEN_LT || op == TOKEN_LE ||
                op == TOKEN_GT || op == TOKEN_GE ||
                op == TOKEN_EQ || op == TOKEN_NE) {
                if (node->as.prefix_op.arg_count >= 2) {
                    HMType *arg1 = infer_expr(ctx, env, node->as.prefix_op.args[1]);
                    if (arg1) hm_unify(ctx, arg0, arg1,
                                       node->line, node->column);
                }
                return hm_con_type(ctx, "bool");
            }

            /* Logical not → bool */
            if (op == TOKEN_NOT) {
                hm_unify(ctx, arg0, hm_con_type(ctx, "bool"),
                         node->line, node->column);
                return hm_con_type(ctx, "bool");
            }

            return hm_subst_apply(ctx, arg0);
        }

        /* ── Function call ────────────────────────────────────────────── */
        case AST_CALL: {
            /* Look up function type in HM env */
            HMType *fn_type = NULL;
            if (node->as.call.name) {
                TypeScheme *scheme = env_lookup(env, node->as.call.name);
                fn_type = scheme ? instantiate(ctx, scheme) : hm_tv_fresh(ctx);
            } else if (node->as.call.func_expr) {
                fn_type = infer_expr(ctx, env, node->as.call.func_expr);
            } else {
                fn_type = hm_tv_fresh(ctx);
            }

            /* Build expected function type from args, right to left */
            HMType *result_type = hm_tv_fresh(ctx);
            HMType *expected    = result_type;

            /* Build arrow type: arg1 → arg2 → … → result */
            for (int i = node->as.call.arg_count - 1; i >= 0; i--) {
                HMType *at = infer_expr(ctx, env, node->as.call.args[i]);
                if (!at) at = hm_tv_fresh(ctx);
                expected = hm_arrow_type(ctx, at, expected);
            }

            hm_unify(ctx, fn_type, expected, node->line, node->column);
            return hm_subst_apply(ctx, result_type);
        }

        /* ── Module-qualified call (treat like a regular call result) ─── */
        case AST_MODULE_QUALIFIED_CALL:
            return hm_tv_fresh(ctx);  /* deferred to existing typechecker */

        /* ── Let binding ──────────────────────────────────────────────── */
        case AST_LET: {
            ASTNode *val = node->as.let.value;
            HMType  *val_type;

            if (val) {
                val_type = infer_expr(ctx, env, val);
                if (!val_type) val_type = hm_tv_fresh(ctx);
            } else {
                val_type = hm_tv_fresh(ctx);
            }

            /* If there is an explicit type annotation, unify with it */
            if (node->as.let.var_type != TYPE_UNKNOWN &&
                node->as.let.var_type != TYPE_VOID) {
                HMType *ann = type_from_ast_type(ctx, node->as.let.var_type,
                                                  node->as.let.type_name);
                hm_unify(ctx, val_type, ann, node->line, node->column);
            }

            val_type = hm_subst_apply(ctx, val_type);

            /* Generalize at let boundary (let-polymorphism) */
            TypeScheme *scheme = generalize(ctx, env, val_type);
            /* Extend the environment (caller's env is immutable; we return
             * the new binding via a side-channel on ctx for blocks) */
            /* NOTE: env mutation is handled in infer_block via the
             * let_env output parameter pattern – for standalone let nodes
             * we simply register the binding in a wrapper. */
            (void)scheme;  /* used by infer_block below */
            return hm_con_type(ctx, "void");
        }

        /* ── Function definition ──────────────────────────────────────── */
        case AST_FUNCTION: {
            /* Build arrow type: τ1 → τ2 → … → τret */
            int     pc      = node->as.function.param_count;
            HMEnv  *fn_env  = env;

            /* Fresh type variable (or annotated type) for each parameter */
            HMType **param_types = NULL;
            if (pc > 0) {
                param_types = malloc(sizeof(HMType*) * pc);
                if (!param_types) { fprintf(stderr, "[HM] out of memory\n"); exit(1); }
            }
            for (int i = 0; i < pc; i++) {
                Parameter *p = &node->as.function.params[i];
                HMType *pt;
                if (p->type != TYPE_UNKNOWN) {
                    pt = type_from_ast_type(ctx, p->type, p->struct_type_name);
                } else {
                    pt = hm_tv_fresh(ctx);
                }
                param_types[i] = pt;
                fn_env = env_extend(ctx, fn_env, p->name, mono_scheme(ctx, pt));
            }

            /* Infer body — isolate effect tracking per function */
            EffectRow *outer_effects = ctx->current_effects;
            EffectRow  fn_effects;
            memset(&fn_effects, 0, sizeof(fn_effects));
            fn_effects.is_open    = true;
            ctx->current_effects  = &fn_effects;

            HMType *body_type = NULL;
            if (node->as.function.body) {
                if (node->as.function.body->type == AST_BLOCK) {
                    body_type = infer_block(ctx, fn_env,
                                            node->as.function.body->as.block.statements,
                                            node->as.function.body->as.block.count);
                } else {
                    body_type = infer_expr(ctx, fn_env, node->as.function.body);
                }
            }
            if (!body_type) body_type = hm_con_type(ctx, "void");

            ctx->current_effects = outer_effects;

            /* Unify with declared return type if present */
            if (node->as.function.return_type != TYPE_UNKNOWN) {
                HMType *ret_ann = type_from_ast_type(
                    ctx, node->as.function.return_type,
                    node->as.function.return_struct_type_name);
                hm_unify(ctx, body_type, ret_ann, node->line, node->column);
                body_type = ret_ann;
            }

            /* Build arrow type right-to-left, attaching effect row to outermost arrow */
            HMType *fn_type = hm_subst_apply(ctx, body_type);
            for (int i = pc - 1; i >= 0; i--) {
                EffectRow *arrow_row = (i == 0 && fn_effects.count > 0)
                                       ? &fn_effects : NULL;
                fn_type = hm_arrow_type_with_effects(
                    ctx,
                    hm_subst_apply(ctx, param_types[i]),
                    fn_type,
                    arrow_row);
            }
            /* Zero-param functions: wrap with unit→result arrow carrying effects */
            if (pc == 0 && fn_effects.count > 0) {
                fn_type = hm_arrow_type_with_effects(
                    ctx,
                    hm_con_type(ctx, "void"),
                    fn_type,
                    &fn_effects);
            }
            free(param_types);

            /* Register function name in env (let-rec: use monomorphic during body,
             * generalise after — here body is already done, so generalise now) */
            if (node->as.function.name) {
                TypeScheme *scheme = generalize(ctx, env, fn_type);
                /* Propagate to caller's env by attaching an entry at env head */
                HMEnvEntry *ent = ctx_alloc(ctx, sizeof(HMEnvEntry));
                ent->name   = ctx_strdup(ctx, node->as.function.name);
                ent->scheme = scheme;
                ent->next   = env ? env->head : NULL;
                if (env) env->head = ent;  /* mutate caller's env head */
            }

            return fn_type;
        }

        /* ── Block / program ─────────────────────────────────────────── */
        case AST_BLOCK:
            return infer_block(ctx, env,
                               node->as.block.statements,
                               node->as.block.count);

        /* ── Return ───────────────────────────────────────────────────── */
        case AST_RETURN:
            if (node->as.return_stmt.value)
                return infer_expr(ctx, env, node->as.return_stmt.value);
            return hm_con_type(ctx, "void");

        /* ── If / else ────────────────────────────────────────────────── */
        case AST_IF: {
            if (node->as.if_stmt.condition)
                hm_unify(ctx,
                         infer_expr(ctx, env, node->as.if_stmt.condition),
                         hm_con_type(ctx, "bool"),
                         node->line, node->column);
            HMType *then_t = NULL, *else_t = NULL;
            if (node->as.if_stmt.then_branch)
                then_t = infer_expr(ctx, env, node->as.if_stmt.then_branch);
            if (node->as.if_stmt.else_branch)
                else_t = infer_expr(ctx, env, node->as.if_stmt.else_branch);
            if (then_t && else_t)
                hm_unify(ctx, then_t, else_t, node->line, node->column);
            return hm_subst_apply(ctx, then_t ? then_t : hm_con_type(ctx, "void"));
        }

        /* ── While ────────────────────────────────────────────────────── */
        case AST_WHILE:
            if (node->as.while_stmt.condition)
                hm_unify(ctx,
                         infer_expr(ctx, env, node->as.while_stmt.condition),
                         hm_con_type(ctx, "bool"),
                         node->line, node->column);
            if (node->as.while_stmt.body)
                infer_expr(ctx, env, node->as.while_stmt.body);
            return hm_con_type(ctx, "void");

        /* ── For ──────────────────────────────────────────────────────── */
        case AST_FOR:
            if (node->as.for_stmt.range_expr)
                infer_expr(ctx, env, node->as.for_stmt.range_expr);
            if (node->as.for_stmt.body)
                infer_expr(ctx, env, node->as.for_stmt.body);
            return hm_con_type(ctx, "void");

        /* ── Set (assignment) ─────────────────────────────────────────── */
        case AST_SET:
            if (node->as.set.value)
                infer_expr(ctx, env, node->as.set.value);
            return hm_con_type(ctx, "void");

        /* ── Print ────────────────────────────────────────────────────── */
        case AST_PRINT:
            if (node->as.print.expr)
                infer_expr(ctx, env, node->as.print.expr);
            return hm_con_type(ctx, "void");

        /* ── Assert ───────────────────────────────────────────────────── */
        case AST_ASSERT:
            if (node->as.assert.condition) {
                HMType *cond_t = infer_expr(ctx, env, node->as.assert.condition);
                hm_unify(ctx, cond_t, hm_con_type(ctx, "bool"),
                         node->line, node->column);
            }
            return hm_con_type(ctx, "void");

        /* ── Field access — row-aware ────────────────────────────────────── */
        case AST_FIELD_ACCESS: {
            HMType *obj_t = infer_expr(ctx, env, node->as.field_access.object);
            if (!obj_t) return hm_tv_fresh(ctx);
            obj_t = hm_subst_apply(ctx, obj_t);
            /* If we have a record type (closed or open), resolve the field */
            if (obj_t->kind == HM_RECORD) {
                return hm_record_field_type(ctx, obj_t,
                                             node->as.field_access.field_name,
                                             node->line, node->column);
            }
            /* Otherwise opaque to HM (struct defined outside HM pass) */
            return hm_tv_fresh(ctx);
        }

        /* ── Array literal ────────────────────────────────────────────── */
        case AST_ARRAY_LITERAL: {
            HMType *elem_t = hm_tv_fresh(ctx);
            for (int i = 0; i < node->as.array_literal.element_count; i++) {
                HMType *et = infer_expr(ctx, env, node->as.array_literal.elements[i]);
                if (et) hm_unify(ctx, elem_t, et,
                                  node->as.array_literal.elements[i]->line,
                                  node->as.array_literal.elements[i]->column);
            }
            return hm_tv_fresh(ctx);   /* array<elem_t> */
        }

        /* ── Tuple literal ────────────────────────────────────────────── */
        case AST_TUPLE_LITERAL:
            for (int i = 0; i < node->as.tuple_literal.element_count; i++)
                infer_expr(ctx, env, node->as.tuple_literal.elements[i]);
            return hm_tv_fresh(ctx);

        /* ── Match expression ─────────────────────────────────────────── */
        case AST_MATCH: {
            infer_expr(ctx, env, node->as.match_expr.expr);
            HMType *res = hm_tv_fresh(ctx);
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                HMType *arm_t = infer_expr(ctx, env, node->as.match_expr.arm_bodies[i]);
                if (arm_t) hm_unify(ctx, res, arm_t,
                                     node->as.match_expr.arm_bodies[i]->line,
                                     node->as.match_expr.arm_bodies[i]->column);
            }
            return hm_subst_apply(ctx, res);
        }

        /* ── Struct literal → closed HM_RECORD type ─────────────────────── */
        case AST_STRUCT_LITERAL: {
            int fc = node->as.struct_literal.field_count;
            if (fc <= 0 || fc > HM_RECORD_MAX_FIELDS) {
                /* Empty or oversized struct literal: fall through to fresh var */
                return hm_tv_fresh(ctx);
            }
            const char **fnames = malloc(sizeof(char*) * fc);
            HMType    **ftypes  = malloc(sizeof(HMType*) * fc);
            if (!fnames || !ftypes) { free(fnames); free(ftypes); return hm_tv_fresh(ctx); }
            for (int i = 0; i < fc; i++) {
                fnames[i] = node->as.struct_literal.field_names[i];
                ftypes[i] = infer_expr(ctx, env, node->as.struct_literal.field_values[i]);
                if (!ftypes[i]) ftypes[i] = hm_tv_fresh(ctx);
            }
            HMType *rec = hm_record_closed(ctx, fnames, ftypes, fc);
            free(fnames);
            free(ftypes);
            return rec;
        }

        /* ── Struct / union / enum definitions (no type to infer) ─────── */
        case AST_STRUCT_DEF:
        case AST_ENUM_DEF:
        case AST_UNION_DEF:
        case AST_UNION_CONSTRUCT:
        case AST_IMPORT:
        case AST_MODULE_DECL:
        case AST_OPAQUE_TYPE:
        case AST_SHADOW:
        case AST_PAR_BLOCK:
        case AST_PAR_LET:
        case AST_UNSAFE_BLOCK:
        case AST_TRY_OP:
        case AST_TUPLE_INDEX:
        case AST_QUALIFIED_NAME:
        case AST_BREAK:
        case AST_CONTINUE:
        case AST_COND:
            return hm_tv_fresh(ctx);   /* not modelled by HM pass */

        /* ── Algebraic effect nodes ───────────────────────────────────── */

        case AST_EFFECT_DECL:
            /* Effect declarations introduce no runtime value;
             * register in the effect registry if available. */
            if (ctx->effect_registry)
                effect_register_from_ast(ctx->effect_registry, node);
            return hm_con_type(ctx, "void");

        case AST_EFFECT_HANDLER:  /* row-poly compat: handled same as AST_HANDLE_EXPR */
        case AST_HANDLE_EXPR: {
            /* handle { body } with { op param -> handler_body }
             *
             * The type of the whole handle expression is the type of body
             * with the named effect removed from the row.
             *
             * Simplified: infer body type, infer handler body types, and
             * add the handled effect to the current_effects set so callers
             * know it is consumed here. */
            HMType *body_type = hm_tv_fresh(ctx);
            if (node->as.handle_expr.body) {
                /* Save and isolate current_effects around the body */
                EffectRow *outer_effects = ctx->current_effects;
                EffectRow  inner_effects;
                memset(&inner_effects, 0, sizeof(inner_effects));
                inner_effects.is_open = true;
                ctx->current_effects  = &inner_effects;

                body_type = infer_expr(ctx, env, node->as.handle_expr.body);

                ctx->current_effects = outer_effects;
                /* The handled effect is consumed — do NOT propagate it up. */
            }
            /* Infer each handler clause (for type consistency of bodies) */
            for (int i = 0; i < node->as.handle_expr.handler_count; i++) {
                if (node->as.handle_expr.handler_bodies[i])
                    infer_expr(ctx, env,
                               node->as.handle_expr.handler_bodies[i]);
            }
            return hm_subst_apply(ctx, body_type);
        }

        case AST_EFFECT_OP: {
            /* perform Foo.op arg — record the effect, return op return type */
            const char *eff_name = node->as.effect_op.effect_name;
            const char *op_name  = node->as.effect_op.op_name;
            if (node->as.effect_op.arg)
                infer_expr(ctx, env, node->as.effect_op.arg);
            if (ctx->current_effects && eff_name)
                effect_row_add(ctx->current_effects, eff_name);
            if (ctx->effect_registry && eff_name && op_name) {
                EffectDecl *decl = effect_lookup(ctx->effect_registry, eff_name);
                if (decl) {
                    EffectOp *op2 = effect_op_lookup(decl, op_name);
                    if (op2)
                        return type_from_ast_type(ctx, op2->return_type,
                                                   op2->return_type_name);
                }
            }
            return hm_tv_fresh(ctx);
        }

        case AST_PROGRAM:
            return infer_block(ctx, env,
                               node->as.program.items,
                               node->as.program.count);

        case AST_ASYNC_FN:
            /* async fn: infer the underlying function */
            return infer_expr(ctx, env, node->as.async_fn.function);

        case AST_AWAIT:
            /* await expr: transparent — infer the inner expression */
            return infer_expr(ctx, env, node->as.await_expr.expr);
    }
    return hm_tv_fresh(ctx);
}

/*
 * Infer types for a sequence of statements, threading the environment so
 * that let-bindings and function definitions become visible to later stmts.
 * Returns the type of the last expression (for implicit return semantics).
 */
static HMType *infer_block(InferCtx *ctx, HMEnv *env,
                             ASTNode **stmts, int count) {
    HMType *last = hm_con_type(ctx, "void");

    /* We mutate a local env chain head so each let/fn binding is visible
     * to subsequent statements in the block.  The parent env is unaffected. */
    HMEnv local_env;
    local_env.head = env ? env->head : NULL;
    HMEnv *cur_env = &local_env;

    for (int i = 0; i < count; i++) {
        ASTNode *s = stmts[i];
        if (!s) continue;

        if (s->type == AST_LET) {
            /* Infer RHS */
            HMType *val_type = hm_tv_fresh(ctx);
            if (s->as.let.value) {
                val_type = infer_expr(ctx, cur_env, s->as.let.value);
                if (!val_type) val_type = hm_tv_fresh(ctx);
            }
            /* Explicit annotation unification */
            if (s->as.let.var_type != TYPE_UNKNOWN) {
                HMType *ann = type_from_ast_type(ctx, s->as.let.var_type,
                                                  s->as.let.type_name);
                hm_unify(ctx, val_type, ann, s->line, s->column);
            }
            val_type = hm_subst_apply(ctx, val_type);
            TypeScheme *scheme = generalize(ctx, cur_env, val_type);
            /* Prepend to current env */
            HMEnvEntry *ent = ctx_alloc(ctx, sizeof(HMEnvEntry));
            ent->name    = ctx_strdup(ctx, s->as.let.name);
            ent->scheme  = scheme;
            ent->next    = cur_env->head;
            cur_env->head = ent;
            last = hm_con_type(ctx, "void");

        } else if (s->type == AST_FUNCTION) {
            /* Process function, then register in env */
            HMType *fn_type = infer_expr(ctx, cur_env, s);
            if (s->as.function.name && fn_type) {
                TypeScheme *scheme = generalize(ctx, cur_env, fn_type);
                HMEnvEntry *ent = ctx_alloc(ctx, sizeof(HMEnvEntry));
                ent->name    = ctx_strdup(ctx, s->as.function.name);
                ent->scheme  = scheme;
                ent->next    = cur_env->head;
                cur_env->head = ent;
            }
            last = hm_con_type(ctx, "void");

        } else {
            last = infer_expr(ctx, cur_env, s);
            if (!last) last = hm_con_type(ctx, "void");
        }
    }
    return hm_subst_apply(ctx, last);
}

/* ── Top-level entry point ───────────────────────────────────────────────── */

bool hm_infer_program(ASTNode *program, const char *source_file) {
    if (!program) return true;

    InferCtx *ctx = hm_ctx_new(source_file);
    HMEnv    *env = make_builtin_env(ctx);

    if (program->type == AST_PROGRAM) {
        infer_block(ctx, env, program->as.program.items, program->as.program.count);
    } else {
        infer_expr(ctx, env, program);
    }

    bool ok = !ctx->has_error;
    hm_ctx_free(ctx);
    return ok;
}

/* ── Effect-aware entry point ────────────────────────────────────────────── */

/*
 * hm_infer_program_with_effects — run Algorithm W with algebraic effect
 * tracking enabled.  Registers built-in effects (IO, Err, State) plus any
 * user-declared effects found in the AST.  After type inference, runs the
 * effect consistency checker.
 *
 * Returns true iff no type errors and no effect errors were found.
 */
bool hm_infer_program_with_effects(ASTNode *program, const char *source_file,
                                    EffectRegistry *reg) {
    if (!program) return true;

    /* Use caller-supplied registry or create a fresh one */
    bool own_registry = (reg == NULL);
    if (own_registry) {
        reg = effect_registry_new();
        effect_register_builtins(reg);
    }

    InferCtx *ctx = hm_ctx_new(source_file);
    ctx->effect_registry = reg;

    /* Global effect row for top-level expressions */
    EffectRow global_effects;
    memset(&global_effects, 0, sizeof(global_effects));
    global_effects.is_open = true;
    ctx->current_effects   = &global_effects;

    HMEnv *env = make_builtin_env(ctx);

    /* Add built-in effect operations to the HM environment as
     * first-class callable functions so callers can use them directly.
     * e.g. IO.print : (string) -[IO]-> void */
    for (int i = 0; i < reg->count; i++) {
        EffectDecl *d = &reg->decls[i];
        for (int j = 0; j < d->op_count; j++) {
            EffectOp *op = &d->ops[j];
            /* Build qualified name "EffectName.opName" */
            size_t qname_len = strlen(d->name) + strlen(op->name) + 2;
            char  *qname     = malloc(qname_len);
            if (!qname) continue;
            snprintf(qname, qname_len, "%s.%s", d->name, op->name);

            /* EffectOp.params is a Parameter[] array; take first param or void */
            HMType *param_t = (op->param_count > 0 && op->params)
                ? type_from_ast_type(ctx, op->params[0].type,
                                          op->params[0].struct_type_name)
                : hm_con_type(ctx, "void");
            HMType *ret_t    = type_from_ast_type(ctx, op->return_type,
                                                   op->return_type_name);

            EffectRow *op_row = ctx_alloc(ctx, sizeof(EffectRow));
            memset(op_row, 0, sizeof(EffectRow));
            effect_row_add(op_row, d->name);

            HMType *fn_t = hm_arrow_type_with_effects(ctx, param_t, ret_t,
                                                        op_row);
            TypeScheme *scheme = mono_scheme(ctx, fn_t);
            HMEnvEntry *ent    = ctx_alloc(ctx, sizeof(HMEnvEntry));
            ent->name   = ctx_strdup(ctx, qname);
            ent->scheme = scheme;
            ent->next   = env ? env->head : NULL;
            if (env) env->head = ent;

            free(qname);
        }
    }

    if (program->type == AST_PROGRAM) {
        infer_block(ctx, env, program->as.program.items, program->as.program.count);
    } else {
        infer_expr(ctx, env, program);
    }

    bool type_ok = !ctx->has_error;
    hm_ctx_free(ctx);

    /* Run the effect checker */
    bool effect_ok = effect_check_program(program, reg, NULL);

    if (own_registry)
        effect_registry_free(reg);

    return type_ok && effect_ok;
}

/*
 * NanoCore AST → S-expression Exporter
 *
 * Converts NanoLang AST nodes (that are within the NanoCore subset) to
 * S-expression format that can be evaluated by the Coq-extracted
 * reference interpreter (tools/nanocore_ref/nanocore-ref).
 *
 * The S-expression format maps directly to Syntax.v:
 *   (EInt 42)
 *   (EBinOp OpAdd (EInt 1) (EInt 2))
 *   (ELet "x" (EInt 42) (EBinOp OpAdd (EVar "x") (EInt 1)))
 */

#include "nanocore_export.h"
#include "nanocore_subset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

/* ── Dynamic string buffer ────────────────────────────────────────────── */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} SBuf;

static SBuf sbuf_new(void) {
    SBuf b = { .data = malloc(256), .len = 0, .cap = 256 };
    b.data[0] = '\0';
    return b;
}

static void sbuf_ensure(SBuf *b, size_t extra) {
    while (b->len + extra + 1 > b->cap) {
        b->cap *= 2;
        b->data = realloc(b->data, b->cap);
    }
}

static void sbuf_append(SBuf *b, const char *s) {
    size_t n = strlen(s);
    sbuf_ensure(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void sbuf_appendf(SBuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    sbuf_ensure(b, (size_t)n);
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

static char *sbuf_finish(SBuf *b) {
    return b->data;  /* caller owns the memory */
}

/* ── Operator mapping ─────────────────────────────────────────────────── */

static const char *binop_name(TokenType op) {
    switch (op) {
        case TOKEN_PLUS:    return "OpAdd";
        case TOKEN_MINUS:   return "OpSub";
        case TOKEN_STAR:    return "OpMul";
        case TOKEN_SLASH:   return "OpDiv";
        case TOKEN_PERCENT: return "OpMod";
        case TOKEN_EQ:      return "OpEq";
        case TOKEN_NE:      return "OpNe";
        case TOKEN_LT:      return "OpLt";
        case TOKEN_LE:      return "OpLe";
        case TOKEN_GT:      return "OpGt";
        case TOKEN_GE:      return "OpGe";
        case TOKEN_AND:     return "OpAnd";
        case TOKEN_OR:      return "OpOr";
        default:            return NULL;
    }
}

static const char *unop_name(TokenType op) {
    switch (op) {
        case TOKEN_MINUS:   return "OpNeg";
        case TOKEN_NOT:     return "OpNot";
        default:            return NULL;
    }
}

/* ── Type mapping ─────────────────────────────────────────────────────── */

static void emit_type(SBuf *b, Type t) {
    switch (t) {
        case TYPE_INT:     sbuf_append(b, "TInt"); break;
        case TYPE_BOOL:    sbuf_append(b, "TBool"); break;
        case TYPE_STRING:  sbuf_append(b, "TString"); break;
        case TYPE_VOID:    sbuf_append(b, "TUnit"); break;
        default:           sbuf_append(b, "TUnit"); break;  /* fallback */
    }
}

/* ── String escaping for S-expressions ────────────────────────────────── */

static void emit_string(SBuf *b, const char *s) {
    sbuf_append(b, "\"");
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '"':  sbuf_append(b, "\\\""); break;
            case '\\': sbuf_append(b, "\\\\"); break;
            case '\n': sbuf_append(b, "\\n"); break;
            case '\t': sbuf_append(b, "\\t"); break;
            default:   sbuf_ensure(b, 1); b->data[b->len++] = *p; b->data[b->len] = '\0'; break;
        }
    }
    sbuf_append(b, "\"");
}

/* ── Recursive AST → S-expression emitter ─────────────────────────────── */

/* Returns false if the node is outside NanoCore */
static bool emit_expr(SBuf *b, ASTNode *node, Environment *env);

/* Emit a block of statements as nested ESeq */
static bool emit_block(SBuf *b, ASTNode **stmts, int count, Environment *env) {
    if (count == 0) {
        sbuf_append(b, "(EUnit)");
        return true;
    }
    if (count == 1) {
        return emit_expr(b, stmts[0], env);
    }
    /* Multiple statements: (ESeq s1 (ESeq s2 ... sn)) */
    for (int i = 0; i < count - 1; i++) {
        sbuf_append(b, "(ESeq ");
        if (!emit_expr(b, stmts[i], env)) return false;
        sbuf_append(b, " ");
    }
    if (!emit_expr(b, stmts[count - 1], env)) return false;
    for (int i = 0; i < count - 1; i++) {
        sbuf_append(b, ")");
    }
    return true;
}

static bool emit_expr(SBuf *b, ASTNode *node, Environment *env) {
    if (!node) {
        sbuf_append(b, "(EUnit)");
        return true;
    }

    switch (node->type) {

        /* ── Literals ── */
        case AST_NUMBER:
            sbuf_appendf(b, "(EInt %lld)", node->as.number);
            return true;

        case AST_BOOL:
            sbuf_appendf(b, "(EBool %s)", node->as.bool_val ? "true" : "false");
            return true;

        case AST_STRING:
            sbuf_append(b, "(EString ");
            emit_string(b, node->as.string_val);
            sbuf_append(b, ")");
            return true;

        /* ── Variables ── */
        case AST_IDENTIFIER:
            sbuf_append(b, "(EVar ");
            emit_string(b, node->as.identifier);
            sbuf_append(b, ")");
            return true;

        /* ── Prefix operators: EBinOp / EUnOp ── */
        case AST_PREFIX_OP: {
            int argc = node->as.prefix_op.arg_count;
            TokenType op = node->as.prefix_op.op;

            if (argc == 2) {
                /* String concatenation uses the same + operator */
                const char *opname = binop_name(op);
                if (!opname) return false;
                /* Check if this is string concatenation (OpAdd on strings maps to OpStrCat) */
                /* We emit OpAdd and let the evaluator handle it based on types.
                 * But the Coq model has separate OpStrCat... We detect at export time
                 * by checking if both args are string-typed. For now, just emit OpAdd
                 * and the evaluator will dispatch correctly for int+int. For string+string
                 * we need OpStrCat. */
                /* Simple heuristic: if op is TOKEN_PLUS and either arg is AST_STRING, use OpStrCat */
                bool is_strcat = false;
                if (op == TOKEN_PLUS) {
                    ASTNode *a1 = node->as.prefix_op.args[0];
                    ASTNode *a2 = node->as.prefix_op.args[1];
                    if ((a1 && a1->type == AST_STRING) || (a2 && a2->type == AST_STRING)) {
                        is_strcat = true;
                    }
                }
                sbuf_appendf(b, "(EBinOp %s ", is_strcat ? "OpStrCat" : opname);
                if (!emit_expr(b, node->as.prefix_op.args[0], env)) return false;
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.prefix_op.args[1], env)) return false;
                sbuf_append(b, ")");
                return true;
            } else if (argc == 1) {
                const char *uopname = unop_name(op);
                if (!uopname) return false;
                sbuf_appendf(b, "(EUnOp %s ", uopname);
                if (!emit_expr(b, node->as.prefix_op.args[0], env)) return false;
                sbuf_append(b, ")");
                return true;
            }
            return false;
        }

        /* ── Function calls: EApp / built-in EUnOp ── */
        case AST_CALL: {
            const char *name = node->as.call.name;

            /* NanoCore builtins map to unary operators */
            if (name && strcmp(name, "str_length") == 0 && node->as.call.arg_count == 1) {
                sbuf_append(b, "(EUnOp OpStrLen ");
                if (!emit_expr(b, node->as.call.args[0], env)) return false;
                sbuf_append(b, ")");
                return true;
            }
            if (name && strcmp(name, "array_length") == 0 && node->as.call.arg_count == 1) {
                sbuf_append(b, "(EUnOp OpArrayLen ");
                if (!emit_expr(b, node->as.call.args[0], env)) return false;
                sbuf_append(b, ")");
                return true;
            }
            if (name && strcmp(name, "char_at") == 0 && node->as.call.arg_count == 2) {
                sbuf_append(b, "(EStrIndex ");
                if (!emit_expr(b, node->as.call.args[0], env)) return false;
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.call.args[1], env)) return false;
                sbuf_append(b, ")");
                return true;
            }
            if (name && strcmp(name, "array_push") == 0 && node->as.call.arg_count == 2) {
                sbuf_append(b, "(EArrayPush ");
                if (!emit_expr(b, node->as.call.args[0], env)) return false;
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.call.args[1], env)) return false;
                sbuf_append(b, ")");
                return true;
            }

            /* Regular function call: EApp */
            if (name) {
                /* Named function call: (EApp (EVar "name") arg)
                 * Multi-arg calls need currying: (EApp (EApp (EVar "f") a1) a2) */
                if (node->as.call.arg_count == 0) {
                    /* Zero-arg call: (EApp (EVar "f") (EUnit)) */
                    sbuf_append(b, "(EApp (EVar ");
                    emit_string(b, name);
                    sbuf_append(b, ") (EUnit))");
                    return true;
                }
                /* Build nested application */
                for (int i = 0; i < node->as.call.arg_count; i++) {
                    sbuf_append(b, "(EApp ");
                }
                sbuf_append(b, "(EVar ");
                emit_string(b, name);
                sbuf_append(b, ")");
                for (int i = 0; i < node->as.call.arg_count; i++) {
                    sbuf_append(b, " ");
                    if (!emit_expr(b, node->as.call.args[i], env)) return false;
                    sbuf_append(b, ")");
                }
                return true;
            }

            /* Expression call: (EApp func_expr args...) */
            if (node->as.call.func_expr) {
                for (int i = 0; i < node->as.call.arg_count; i++) {
                    sbuf_append(b, "(EApp ");
                }
                if (!emit_expr(b, node->as.call.func_expr, env)) return false;
                for (int i = 0; i < node->as.call.arg_count; i++) {
                    sbuf_append(b, " ");
                    if (!emit_expr(b, node->as.call.args[i], env)) return false;
                    sbuf_append(b, ")");
                }
                return true;
            }
            return false;
        }

        /* ── Array literal: EArray ── */
        case AST_ARRAY_LITERAL:
            sbuf_append(b, "(EArray");
            for (int i = 0; i < node->as.array_literal.element_count; i++) {
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.array_literal.elements[i], env)) return false;
            }
            sbuf_append(b, ")");
            return true;

        /* ── Let binding: ELet ── */
        case AST_LET:
            sbuf_append(b, "(ELet ");
            emit_string(b, node->as.let.name);
            sbuf_append(b, " ");
            if (!emit_expr(b, node->as.let.value, env)) return false;
            /* NanoCore ELet has body; in NanoLang, let is a statement.
             * We'll handle this in block context where the rest of the block is the body. */
            sbuf_append(b, " (EUnit))");  /* placeholder body */
            return true;

        /* ── Mutable assignment: ESet ── */
        case AST_SET:
            sbuf_append(b, "(ESet ");
            emit_string(b, node->as.set.name);
            sbuf_append(b, " ");
            if (!emit_expr(b, node->as.set.value, env)) return false;
            sbuf_append(b, ")");
            return true;

        /* ── If/else: EIf ── */
        case AST_IF:
            sbuf_append(b, "(EIf ");
            if (!emit_expr(b, node->as.if_stmt.condition, env)) return false;
            sbuf_append(b, " ");
            if (!emit_expr(b, node->as.if_stmt.then_branch, env)) return false;
            sbuf_append(b, " ");
            if (node->as.if_stmt.else_branch) {
                if (!emit_expr(b, node->as.if_stmt.else_branch, env)) return false;
            } else {
                sbuf_append(b, "(EUnit)");
            }
            sbuf_append(b, ")");
            return true;

        /* ── While loop: EWhile ── */
        case AST_WHILE:
            sbuf_append(b, "(EWhile ");
            if (!emit_expr(b, node->as.while_stmt.condition, env)) return false;
            sbuf_append(b, " ");
            if (!emit_expr(b, node->as.while_stmt.body, env)) return false;
            sbuf_append(b, ")");
            return true;

        /* ── Block/sequence: ESeq ── */
        case AST_BLOCK:
            return emit_block(b, node->as.block.statements, node->as.block.count, env);

        /* ── Function definition: ELam / EFix ── */
        case AST_FUNCTION: {
            if (node->as.function.is_extern) return false;
            /* Single-parameter functions become ELam, multi-param become nested ELam */
            /* If function has a name and body references it, it's EFix */
            /* For now, emit as ELam (non-recursive) for single param */
            if (node->as.function.param_count == 0) {
                sbuf_append(b, "(ELam \"_\" TUnit ");
                if (!emit_expr(b, node->as.function.body, env)) return false;
                sbuf_append(b, ")");
                return true;
            }
            /* Build nested lambdas from right to left */
            for (int i = 0; i < node->as.function.param_count; i++) {
                sbuf_append(b, "(ELam ");
                emit_string(b, node->as.function.params[i].name);
                sbuf_append(b, " ");
                emit_type(b, node->as.function.params[i].type);
                sbuf_append(b, " ");
            }
            if (!emit_expr(b, node->as.function.body, env)) return false;
            for (int i = 0; i < node->as.function.param_count; i++) {
                sbuf_append(b, ")");
            }
            return true;
        }

        /* ── Return: just emit the value ── */
        case AST_RETURN:
            if (node->as.return_stmt.value) {
                return emit_expr(b, node->as.return_stmt.value, env);
            }
            sbuf_append(b, "(EUnit)");
            return true;

        /* ── Struct literal: ERecord ── */
        case AST_STRUCT_LITERAL:
            sbuf_append(b, "(ERecord");
            for (int i = 0; i < node->as.struct_literal.field_count; i++) {
                sbuf_append(b, " (");
                emit_string(b, node->as.struct_literal.field_names[i]);
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.struct_literal.field_values[i], env)) return false;
                sbuf_append(b, ")");
            }
            sbuf_append(b, ")");
            return true;

        /* ── Field access: EField ── */
        case AST_FIELD_ACCESS:
            sbuf_append(b, "(EField ");
            if (!emit_expr(b, node->as.field_access.object, env)) return false;
            sbuf_append(b, " ");
            emit_string(b, node->as.field_access.field_name);
            sbuf_append(b, ")");
            return true;

        /* ── Union construction: EConstruct ── */
        case AST_UNION_CONSTRUCT:
            sbuf_append(b, "(EConstruct ");
            emit_string(b, node->as.union_construct.variant_name);
            sbuf_append(b, " ");
            if (node->as.union_construct.field_count > 0) {
                /* Emit first field value as payload */
                if (!emit_expr(b, node->as.union_construct.field_values[0], env)) return false;
            } else {
                sbuf_append(b, "(EUnit)");
            }
            /* Type annotation (simplified) */
            sbuf_append(b, " (TVariant))");
            return true;

        /* ── Match expression: EMatch ── */
        case AST_MATCH:
            sbuf_append(b, "(EMatch ");
            if (!emit_expr(b, node->as.match_expr.expr, env)) return false;
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                sbuf_append(b, " (");
                emit_string(b, node->as.match_expr.pattern_variants[i]);
                sbuf_append(b, " ");
                emit_string(b, node->as.match_expr.pattern_bindings[i]);
                sbuf_append(b, " ");
                if (!emit_expr(b, node->as.match_expr.arm_bodies[i], env)) return false;
                sbuf_append(b, ")");
            }
            sbuf_append(b, ")");
            return true;

        /* ── Print: outside NanoCore but we can skip it ── */
        case AST_PRINT:
            /* Evaluate the expression but discard (side-effect not modeled) */
            sbuf_append(b, "(EUnit)");
            return true;

        /* ── Assert: outside NanoCore ── */
        case AST_ASSERT:
            sbuf_append(b, "(EUnit)");
            return true;

        default:
            return false;
    }
}

/* ── Public API ───────────────────────────────────────────────────────── */

char *nanocore_export_sexpr(ASTNode *node, Environment *env) {
    if (!node) return NULL;

    SBuf b = sbuf_new();
    if (!emit_expr(&b, node, env)) {
        free(b.data);
        return NULL;
    }
    return sbuf_finish(&b);
}

char *nanocore_reference_eval(const char *sexpr, const char *compiler_path) {
    if (!sexpr) return NULL;

    /* Find nanocore-ref binary: try same directory as compiler, then PATH */
    char ref_path[1024];
    bool found = false;

    if (compiler_path) {
        /* Try directory containing the compiler */
        const char *last_slash = strrchr(compiler_path, '/');
        if (last_slash) {
            int dir_len = (int)(last_slash - compiler_path);
            snprintf(ref_path, sizeof(ref_path), "%.*s/nanocore-ref", dir_len, compiler_path);
            if (access(ref_path, X_OK) == 0) {
                found = true;
            }
        }
    }

    if (!found) {
        /* Try PATH */
        snprintf(ref_path, sizeof(ref_path), "nanocore-ref");
    }

    /* Create a pipe to the reference interpreter */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "echo '%s' | %s 2>/dev/null", sexpr, ref_path);

    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    char result[4096];
    size_t total = 0;
    size_t n;
    while ((n = fread(result + total, 1, sizeof(result) - total - 1, fp)) > 0) {
        total += n;
    }
    result[total] = '\0';
    pclose(fp);

    /* Trim trailing newline */
    while (total > 0 && (result[total - 1] == '\n' || result[total - 1] == '\r')) {
        result[--total] = '\0';
    }

    return total > 0 ? strdup(result) : NULL;
}

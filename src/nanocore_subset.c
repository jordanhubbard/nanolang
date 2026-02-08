/*
 * NanoCore Subset Checker + Trust Levels
 *
 * Determines which functions in a NanoLang program fall within the formally
 * verified NanoCore subset. The NanoCore subset corresponds exactly to the
 * expressions defined in formal/Syntax.v, for which type soundness,
 * determinism, and big-step/small-step equivalence are proven in Coq.
 *
 * Trust levels:
 *   1 (verified)    - NanoCore only. Proven sound in Coq.
 *   2 (typechecked) - Full NanoLang types. Checked but not proven.
 *   3 (ffi)         - Calls FFI/extern C code. No verification.
 *   4 (unsafe)      - Inside unsafe blocks.
 */

#include "nanocore_subset.h"
#include <stdio.h>
#include <string.h>

/* ── NanoCore binary operator check ──────────────────────────────────────── */

/* Returns true if the token operator is in the NanoCore verified subset.
 * Maps to Coq Syntax.v binop: OpAdd..OpStrCat */
static bool is_nanocore_binop(TokenType op) {
    switch (op) {
        case TOKEN_PLUS:    /* OpAdd / OpStrCat */
        case TOKEN_MINUS:   /* OpSub */
        case TOKEN_STAR:    /* OpMul */
        case TOKEN_SLASH:   /* OpDiv */
        case TOKEN_PERCENT: /* OpMod */
        case TOKEN_EQ:      /* OpEq */
        case TOKEN_NE:      /* OpNe */
        case TOKEN_LT:      /* OpLt */
        case TOKEN_LE:      /* OpLe */
        case TOKEN_GT:      /* OpGt */
        case TOKEN_GE:      /* OpGe */
        case TOKEN_AND:     /* OpAnd */
        case TOKEN_OR:      /* OpOr */
            return true;
        default:
            return false;
    }
}

/* ── NanoCore unary operator check ───────────────────────────────────────── */

/* Returns true if the prefix op is in the NanoCore verified subset.
 * Maps to Coq Syntax.v unop: OpNeg, OpNot, OpStrLen, OpArrayLen */
static bool is_nanocore_unop(TokenType op, int arg_count) {
    if (arg_count == 1) {
        switch (op) {
            case TOKEN_MINUS:  /* OpNeg (unary minus) */
            case TOKEN_NOT:    /* OpNot */
                return true;
            default:
                break;
        }
    }
    return false;
}

/* Returns true if a call to a built-in function is in the NanoCore subset.
 * str_length → OpStrLen, array_length → OpArrayLen */
static bool is_nanocore_builtin(const char *name) {
    return (strcmp(name, "str_length") == 0 ||
            strcmp(name, "array_length") == 0 ||
            strcmp(name, "char_at") == 0 ||
            strcmp(name, "array_push") == 0);
}

/* ── Subset reason tracking ──────────────────────────────────────────────── */

typedef enum {
    SUBSET_OK,
    SUBSET_OUTSIDE_AST,    /* AST node type not in NanoCore */
    SUBSET_OUTSIDE_TYPE,   /* Type not in NanoCore */
    SUBSET_FFI_CALL,       /* Calls an extern/FFI function */
    SUBSET_MODULE_CALL,    /* Calls a module-qualified function */
    SUBSET_UNSAFE_BLOCK,   /* Contains unsafe block */
    SUBSET_NON_CORE_BINOP, /* Uses binary operator not in NanoCore */
} SubsetViolation;

typedef struct {
    SubsetViolation violation;
    TrustLevel level;
    const char *detail;
} SubsetResult;

static SubsetResult ok_result(void) {
    return (SubsetResult){ .violation = SUBSET_OK, .level = TRUST_VERIFIED, .detail = NULL };
}

static SubsetResult outside_ast(const char *detail) {
    return (SubsetResult){ .violation = SUBSET_OUTSIDE_AST, .level = TRUST_TYPECHECKED, .detail = detail };
}

static SubsetResult ffi_call(const char *detail) {
    return (SubsetResult){ .violation = SUBSET_FFI_CALL, .level = TRUST_FFI, .detail = detail };
}

static SubsetResult module_call(const char *detail) {
    return (SubsetResult){ .violation = SUBSET_MODULE_CALL, .level = TRUST_FFI, .detail = detail };
}

static SubsetResult unsafe_block(void) {
    return (SubsetResult){ .violation = SUBSET_UNSAFE_BLOCK, .level = TRUST_UNSAFE, .detail = "unsafe block" };
}

/* Merge two results: the worse (higher trust level) wins */
static SubsetResult merge_results(SubsetResult a, SubsetResult b) {
    return (a.level >= b.level) ? a : b;
}

/* ── Recursive AST walker ────────────────────────────────────────────────── */

/* Forward declaration */
static SubsetResult check_subset(ASTNode *node, Environment *env);

/* Check a list of AST nodes, return worst result */
static SubsetResult check_subset_list(ASTNode **nodes, int count, Environment *env) {
    SubsetResult result = ok_result();
    for (int i = 0; i < count; i++) {
        if (nodes[i]) {
            result = merge_results(result, check_subset(nodes[i], env));
        }
    }
    return result;
}

/*
 * Recursively check whether an AST node falls within the NanoCore subset.
 *
 * NanoCore expression forms (from formal/Syntax.v):
 *   EInt, EBool, EString, EUnit, EVar,
 *   EBinOp, EUnOp, EIf, ELet, ESet, ESeq, EWhile,
 *   ELam, EApp, EFix,
 *   EArray, EIndex, EArraySet, EArrayPush,
 *   ERecord, EField, ESetField,
 *   EConstruct, EMatch, EStrIndex
 */
static SubsetResult check_subset(ASTNode *node, Environment *env) {
    if (!node) return ok_result();

    switch (node->type) {

        /* ── Literals: EInt, EBool, EString ── */
        case AST_NUMBER:     /* EInt */
        case AST_BOOL:       /* EBool */
        case AST_STRING:     /* EString */
            return ok_result();

        /* ── Variables: EVar ── */
        case AST_IDENTIFIER:
            return ok_result();

        /* ── Prefix operators: EUnOp (OpNeg, OpNot) ── */
        case AST_PREFIX_OP: {
            /* Check if it's a NanoCore binop or unop */
            int argc = node->as.prefix_op.arg_count;
            TokenType op = node->as.prefix_op.op;

            if (argc == 2) {
                /* Binary operator */
                if (!is_nanocore_binop(op)) {
                    return (SubsetResult){
                        .violation = SUBSET_NON_CORE_BINOP,
                        .level = TRUST_TYPECHECKED,
                        .detail = "non-NanoCore binary operator"
                    };
                }
                return check_subset_list(node->as.prefix_op.args, argc, env);
            } else if (argc == 1) {
                /* Unary operator */
                if (!is_nanocore_unop(op, argc)) {
                    return outside_ast("non-NanoCore unary operator");
                }
                return check_subset(node->as.prefix_op.args[0], env);
            }
            return outside_ast("unexpected prefix operator arity");
        }

        /* ── Function calls: EApp ── */
        case AST_CALL: {
            const char *name = node->as.call.name;

            /* Check for built-in NanoCore functions */
            if (name && is_nanocore_builtin(name)) {
                return check_subset_list(node->as.call.args, node->as.call.arg_count, env);
            }

            /* Check if calling an extern (FFI) function */
            if (name) {
                Function *func = env_get_function(env, name);
                if (func && func->is_extern) {
                    return ffi_call(name);
                }
            }

            /* Function expression call (first-class functions) */
            SubsetResult result = ok_result();
            if (node->as.call.func_expr) {
                result = merge_results(result, check_subset(node->as.call.func_expr, env));
            }

            /* Check all arguments */
            result = merge_results(result,
                check_subset_list(node->as.call.args, node->as.call.arg_count, env));
            return result;
        }

        /* ── Module-qualified calls: outside NanoCore ── */
        case AST_MODULE_QUALIFIED_CALL:
            return module_call(node->as.module_qualified_call.function_name);

        /* ── Array literal: EArray ── */
        case AST_ARRAY_LITERAL:
            return check_subset_list(node->as.array_literal.elements,
                                     node->as.array_literal.element_count, env);

        /* ── Let binding: ELet ── */
        case AST_LET: {
            /* Check for non-NanoCore types */
            switch (node->as.let.var_type) {
                case TYPE_INT:
                case TYPE_BOOL:
                case TYPE_STRING:
                case TYPE_VOID:
                case TYPE_ARRAY:
                case TYPE_STRUCT:
                case TYPE_UNION:
                case TYPE_FUNCTION:
                case TYPE_UNKNOWN:  /* Type inference - checked later */
                    break;
                default:
                    return outside_ast("non-NanoCore type in let binding");
            }
            return check_subset(node->as.let.value, env);
        }

        /* ── Mutable assignment: ESet ── */
        case AST_SET:
            return check_subset(node->as.set.value, env);

        /* ── If/else: EIf ── */
        case AST_IF: {
            SubsetResult result = check_subset(node->as.if_stmt.condition, env);
            result = merge_results(result, check_subset(node->as.if_stmt.then_branch, env));
            result = merge_results(result, check_subset(node->as.if_stmt.else_branch, env));
            return result;
        }

        /* ── While loop: EWhile ── */
        case AST_WHILE: {
            SubsetResult result = check_subset(node->as.while_stmt.condition, env);
            result = merge_results(result, check_subset(node->as.while_stmt.body, env));
            return result;
        }

        /* ── For loop: sugar for while+let — equivalent to verified ── */
        case AST_FOR: {
            /* AST_FOR is syntactic sugar for a let+while pattern.
             * We flag it as typechecked since the desugaring isn't
             * explicitly in NanoCore, but note the equivalence. */
            SubsetResult result = check_subset(node->as.for_stmt.range_expr, env);
            result = merge_results(result, check_subset(node->as.for_stmt.body, env));
            /* Bump to typechecked since for isn't directly in the Coq model */
            if (result.level < TRUST_TYPECHECKED) {
                result.level = TRUST_TYPECHECKED;
                result.detail = "for-loop (equivalent to verified while+let)";
                result.violation = SUBSET_OUTSIDE_AST;
            }
            return result;
        }

        /* ── Block/sequence: ESeq ── */
        case AST_BLOCK:
            return check_subset_list(node->as.block.statements,
                                     node->as.block.count, env);

        /* ── Function definition: ELam / EFix ── */
        case AST_FUNCTION: {
            /* Check for extern (FFI) functions */
            if (node->as.function.is_extern) {
                return ffi_call(node->as.function.name);
            }

            /* Check parameter types */
            for (int i = 0; i < node->as.function.param_count; i++) {
                switch (node->as.function.params[i].type) {
                    case TYPE_INT:
                    case TYPE_BOOL:
                    case TYPE_STRING:
                    case TYPE_VOID:
                    case TYPE_ARRAY:
                    case TYPE_STRUCT:
                    case TYPE_UNION:
                    case TYPE_FUNCTION:
                        break;
                    default:
                        return outside_ast("non-NanoCore parameter type");
                }
            }

            /* Check return type */
            switch (node->as.function.return_type) {
                case TYPE_INT:
                case TYPE_BOOL:
                case TYPE_STRING:
                case TYPE_VOID:
                case TYPE_ARRAY:
                case TYPE_STRUCT:
                case TYPE_UNION:
                case TYPE_FUNCTION:
                    break;
                default:
                    return outside_ast("non-NanoCore return type");
            }

            return check_subset(node->as.function.body, env);
        }

        /* ── Struct literal: ERecord ── */
        case AST_STRUCT_LITERAL: {
            SubsetResult result = ok_result();
            for (int i = 0; i < node->as.struct_literal.field_count; i++) {
                result = merge_results(result,
                    check_subset(node->as.struct_literal.field_values[i], env));
            }
            return result;
        }

        /* ── Field access: EField ── */
        case AST_FIELD_ACCESS:
            return check_subset(node->as.field_access.object, env);

        /* ── Union construction: EConstruct ── */
        case AST_UNION_CONSTRUCT: {
            SubsetResult result = ok_result();
            for (int i = 0; i < node->as.union_construct.field_count; i++) {
                result = merge_results(result,
                    check_subset(node->as.union_construct.field_values[i], env));
            }
            return result;
        }

        /* ── Pattern matching: EMatch ── */
        case AST_MATCH: {
            SubsetResult result = check_subset(node->as.match_expr.expr, env);
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                result = merge_results(result,
                    check_subset(node->as.match_expr.arm_bodies[i], env));
            }
            return result;
        }

        /* ── Return: treated as transparent for NanoCore purposes.
         *    In Coq, function bodies are expressions (last expr = return value).
         *    NanoLang requires explicit 'return', which is semantically equivalent. ── */
        case AST_RETURN: {
            if (node->as.return_stmt.value) {
                return check_subset(node->as.return_stmt.value, env);
            }
            return ok_result();  /* return with no value = unit */
        }

        /* ── Unsafe block: trust level 4 ── */
        case AST_UNSAFE_BLOCK:
            return unsafe_block();

        /* ── Shadow tests: transparent (check inner body) ── */
        case AST_SHADOW:
            return check_subset(node->as.shadow.body, env);

        /* ── Program root: check all items ── */
        case AST_PROGRAM:
            return check_subset_list(node->as.program.items,
                                     node->as.program.count, env);

        /* ── Everything else: outside NanoCore ── */
        case AST_FLOAT:
            return outside_ast("float type (not in NanoCore)");
        case AST_COND:
            return outside_ast("cond expression (not in NanoCore)");
        case AST_PRINT:
            return outside_ast("print (not in NanoCore)");
        case AST_ASSERT:
            return outside_ast("assert (not in NanoCore)");
        case AST_IMPORT:
            return outside_ast("import (not in NanoCore)");
        case AST_MODULE_DECL:
            return outside_ast("module declaration (not in NanoCore)");
        case AST_ENUM_DEF:
            return outside_ast("enum definition (not in NanoCore)");
        case AST_OPAQUE_TYPE:
            return outside_ast("opaque type (not in NanoCore)");
        case AST_TUPLE_LITERAL:
            return outside_ast("tuple literal (not in NanoCore)");
        case AST_TUPLE_INDEX:
            return outside_ast("tuple index (not in NanoCore)");
        case AST_QUALIFIED_NAME:
            return outside_ast("qualified name (not in NanoCore)");
        case AST_BREAK:
            return outside_ast("break (not in NanoCore)");
        case AST_CONTINUE:
            return outside_ast("continue (not in NanoCore)");

        /* Struct/union definitions are type declarations, not expressions.
         * They don't affect the trust level of function bodies. */
        case AST_STRUCT_DEF:
        case AST_UNION_DEF:
            return ok_result();

        default:
            return outside_ast("unknown AST node type");
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

bool nanocore_is_subset(ASTNode *node, Environment *env) {
    SubsetResult result = check_subset(node, env);
    return result.level == TRUST_VERIFIED;
}

TrustLevel nanocore_function_trust(ASTNode *func, Environment *env) {
    SubsetResult result = check_subset(func, env);
    return result.level;
}

TrustReport *nanocore_trust_report(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) return NULL;

    TrustReport *report = malloc(sizeof(TrustReport));
    report->entries = NULL;
    report->count = 0;
    report->capacity = 0;

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (!item || item->type != AST_FUNCTION) continue;
        if (item->as.function.name == NULL) continue;

        /* Skip shadow test functions */
        if (strncmp(item->as.function.name, "shadow_", 7) == 0) continue;

        SubsetResult result = check_subset(item, env);

        /* Grow array if needed */
        if (report->count >= report->capacity) {
            report->capacity = report->capacity == 0 ? 16 : report->capacity * 2;
            report->entries = realloc(report->entries,
                sizeof(FunctionTrust) * report->capacity);
        }

        const char *reason;
        switch (result.level) {
            case TRUST_VERIFIED:
                reason = "NanoCore subset, proven sound";
                break;
            case TRUST_TYPECHECKED:
                reason = result.detail ? result.detail : "Uses features not in NanoCore";
                break;
            case TRUST_FFI:
                if (result.detail) {
                    /* Build a more informative reason */
                    static char ffi_reason[256];
                    snprintf(ffi_reason, sizeof(ffi_reason), "Calls %s (C FFI)", result.detail);
                    reason = ffi_reason;
                } else {
                    reason = "Calls FFI/extern C code";
                }
                break;
            case TRUST_UNSAFE:
                reason = "Contains unsafe block";
                break;
            default:
                reason = "Unknown trust level";
                break;
        }

        report->entries[report->count++] = (FunctionTrust){
            .function_name = item->as.function.name,
            .level = result.level,
            .reason = reason,
            .return_type = item->as.function.return_type,
            .param_count = item->as.function.param_count,
        };
    }

    return report;
}

/* ── Trust level labels ──────────────────────────────────────────────────── */

static const char *trust_label(TrustLevel level) {
    switch (level) {
        case TRUST_VERIFIED:    return "verified";
        case TRUST_TYPECHECKED: return "typechecked";
        case TRUST_FFI:         return "ffi";
        case TRUST_UNSAFE:      return "unsafe";
        default:                return "unknown";
    }
}

/* Format a type signature string for a function */
static void format_signature(char *buf, size_t buflen, const char *name,
                              FunctionTrust *entry) {
    const char *ret = type_to_string(entry->return_type);
    if (entry->param_count == 0) {
        snprintf(buf, buflen, "%s() -> %s", name, ret);
    } else {
        /* We don't have full param type info in FunctionTrust, so just show count */
        snprintf(buf, buflen, "%s(...) -> %s", name, ret);
    }
}

void nanocore_print_trust_report(TrustReport *report, const char *filename) {
    if (!report) return;

    printf("Trust Report for %s:\n", filename);

    /* Count by level */
    int counts[5] = {0};
    for (int i = 0; i < report->count; i++) {
        if (report->entries[i].level >= 1 && report->entries[i].level <= 4) {
            counts[report->entries[i].level]++;
        }
    }

    for (int i = 0; i < report->count; i++) {
        FunctionTrust *entry = &report->entries[i];
        char sig[256];
        format_signature(sig, sizeof(sig), entry->function_name, entry);

        printf("  %-40s [%-11s]  %s\n", sig, trust_label(entry->level), entry->reason);
    }

    printf("\nSummary: %d verified, %d typechecked, %d ffi, %d unsafe\n",
           counts[TRUST_VERIFIED], counts[TRUST_TYPECHECKED],
           counts[TRUST_FFI], counts[TRUST_UNSAFE]);
}

void nanocore_free_trust_report(TrustReport *report) {
    if (!report) return;
    free(report->entries);
    free(report);
}

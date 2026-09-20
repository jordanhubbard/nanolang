/* I expose static cleanup units only inside this fixture's actual interpreter TU. */
#include <assert.h>
#include "../src/eval.c"

static void fixture_free_union_shell(Value v) {
    assert(v.type == VAL_UNION && v.as.union_val);
    UnionValue *u = v.as.union_val;
    for (int i = 0; i < u->field_count; ++i) free(u->field_names[i]);
    free(u->field_names); free(u->field_values);
    free(u->union_name); free(u->variant_name); free(u);
}

void file_match_cleanup_units(void) {
    ASTNode literal = {0}; literal.type = AST_STRUCT_LITERAL;
    Value first = create_union("Choice", 0, "None", NULL, NULL, 0);
    assert(eval_match_owns_empty_literal(&literal, first));
    literal.type = AST_UNION_CONSTRUCT;
    assert(eval_match_owns_empty_literal(&literal, first));
    literal.as.union_construct.field_count = 1;
    assert(!eval_match_owns_empty_literal(&literal, first));
    literal.as.union_construct.field_count = 0;
    const ASTNodeType excluded[] = {AST_IDENTIFIER, AST_CALL, AST_FIELD_ACCESS, AST_IF, AST_MATCH};
    for (size_t i = 0; i < sizeof(excluded)/sizeof(*excluded); ++i) {
        literal.type = excluded[i]; assert(!eval_match_owns_empty_literal(&literal, first));
    }
    assert(!eval_match_owns_empty_literal(NULL, first));
    literal.type = AST_STRUCT_LITERAL;
    assert(!eval_match_owns_empty_literal(&literal, create_int(1)));
    /* I directly exercise the defensive exact-pointer escape exclusion. */
    eval_match_release_empty_literal(first, true, first);
    assert(!strcmp(first.as.union_val->variant_name, "None"));
    Value second = create_union("Choice", 0, "None", NULL, NULL, 0);
    assert(first.as.union_val != second.as.union_val);
    eval_match_release_empty_literal(first, true, second);
    assert(!strcmp(second.as.union_val->union_name, "Choice"));
    eval_match_release_empty_literal(second, false, create_void());
    assert(second.as.union_val->field_count == 0);
    fixture_free_union_shell(second);

    /* I own these roots explicitly; the actual match evaluates an alias. */
    for (int nested = 0; nested < 3; ++nested) {
        Value child = create_union("Choice", 0, "None", NULL, NULL, 0);
        char *names[] = {"value"};
        Value payload = nested == 2 ? child : create_int(41);
        Value root = nested ? create_union("Choice", 1, "Some", names, &payload, 1) : child;
        assert(eval_match_owns_empty_literal(&literal, root) == (nested == 0));
        Environment *env = create_environment(); assert(env);
        env_define_var(env, "borrowed", TYPE_UNION, false, root);
        ASTNode id = {0}, number = {0}, matched = {0};
        id.type = AST_IDENTIFIER; id.as.identifier = "borrowed";
        number.type = AST_NUMBER; number.as.number = 7;
        char *variants[] = {nested ? "Some" : "None"};
        char *bindings[] = {"_"}; ASTNode *bodies[] = {&number};
        matched.type = AST_MATCH; matched.as.match_expr.expr = &id;
        matched.as.match_expr.arm_count = 1; matched.as.match_expr.pattern_variants = variants;
        matched.as.match_expr.pattern_bindings = bindings; matched.as.match_expr.arm_bodies = bodies;
        for (int i = 0; i < 16; ++i) {
            Value result = eval_expression(&matched, env);
            assert(result.type == VAL_INT && result.as.int_val == 7 && env->symbol_count == 1);
            assert(env->symbols[0].value.as.union_val == root.as.union_val);
            assert(!strcmp(root.as.union_val->union_name, "Choice"));
            if (nested == 2) assert(root.as.union_val->field_values[0].as.union_val == child.as.union_val);
        }
        free_environment(env);
        fixture_free_union_shell(root);
        if (nested) fixture_free_union_shell(child);
    }
    puts("match-cleanup-units:exact-escape:alias:payload:nested:retained");
}

void file_match_cleanup_guard_return(ASTNode *matched, Environment *env) {
    assert(matched->type == AST_MATCH && matched->as.match_expr.arm_count == 2);
    ASTNode number = {0}, returning = {0};
    number.type = AST_NUMBER; number.as.number = 29;
    returning.type = AST_RETURN; returning.as.return_stmt.value = &number;
    ASTNode *guards[] = {&returning, NULL};
    ASTNode **saved = matched->as.match_expr.guard_exprs;
    matched->as.match_expr.guard_exprs = guards;
    int before = env->symbol_count;
    Value result = eval_expression(matched, env);
    assert(result.type == VAL_INT && result.as.int_val == 29 && result.is_return);
    assert(env->symbol_count == before);
    matched->as.match_expr.guard_exprs = saved;
}

void file_match_cleanup_free_returned(Value value) {
    assert(value.type == VAL_UNION && value.as.union_val->field_count == 0);
    fixture_free_union_shell(value);
}

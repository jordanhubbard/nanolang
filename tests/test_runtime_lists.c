/**
 * test_runtime_lists.c — unit tests for all runtime/list_AST*.c files
 *
 * Exercises the generic list API for all 33 compiled AST node list types:
 *   nl_list_X_new, nl_list_X_with_capacity, nl_list_X_push, nl_list_X_pop,
 *   nl_list_X_insert, nl_list_X_remove, nl_list_X_set, nl_list_X_get,
 *   nl_list_X_clear, nl_list_X_length, nl_list_X_capacity, nl_list_X_is_empty,
 *   nl_list_X_free
 *
 * Note: list_ASTMatchClause.c is NOT in RUNTIME_SOURCES and is excluded here.
 */

#include "../src/nanolang.h"

/* Include non-AST list headers */
#include "../src/runtime/list_int.h"
#include "../src/runtime/list_string.h"
#include "../src/runtime/list_LexerToken.h"
#include "../src/runtime/list_token.h"
#include "../src/runtime/list_CompilerDiagnostic.h"
#include "../src/runtime/list_CompilerSourceLocation.h"

/* Include all list headers for the 33 compiled AST types */
#include "../src/runtime/list_ASTArrayLiteral.h"
#include "../src/runtime/list_ASTAssert.h"
#include "../src/runtime/list_ASTBinaryOp.h"
#include "../src/runtime/list_ASTBlock.h"
#include "../src/runtime/list_ASTBool.h"
#include "../src/runtime/list_ASTCall.h"
#include "../src/runtime/list_ASTEnum.h"
#include "../src/runtime/list_ASTFieldAccess.h"
#include "../src/runtime/list_ASTFloat.h"
#include "../src/runtime/list_ASTFor.h"
#include "../src/runtime/list_ASTFunction.h"
#include "../src/runtime/list_ASTIdentifier.h"
#include "../src/runtime/list_ASTIf.h"
#include "../src/runtime/list_ASTImport.h"
#include "../src/runtime/list_ASTLet.h"
#include "../src/runtime/list_ASTMatch.h"
#include "../src/runtime/list_ASTModuleQualifiedCall.h"
#include "../src/runtime/list_ASTNumber.h"
#include "../src/runtime/list_ASTOpaqueType.h"
#include "../src/runtime/list_ASTPrint.h"
#include "../src/runtime/list_ASTReturn.h"
#include "../src/runtime/list_ASTSet.h"
#include "../src/runtime/list_ASTShadow.h"
#include "../src/runtime/list_ASTStmtRef.h"
#include "../src/runtime/list_ASTString.h"
#include "../src/runtime/list_ASTStruct.h"
#include "../src/runtime/list_ASTStructLiteral.h"
#include "../src/runtime/list_ASTTupleIndex.h"
#include "../src/runtime/list_ASTTupleLiteral.h"
#include "../src/runtime/list_ASTUnion.h"
#include "../src/runtime/list_ASTUnionConstruct.h"
#include "../src/runtime/list_ASTUnsafeBlock.h"
#include "../src/runtime/list_ASTWhile.h"

#include <stdio.h>
#include <string.h>

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#define ASSERT(cond) \
    if (!(cond)) { \
        printf("\n  FAILED: %s at line %d\n", #cond, __LINE__); \
        exit(1); \
    }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        printf("\n  FAILED: %s == %s at line %d (got %lld, expected %lld)\n", \
            #a, #b, __LINE__, (long long)(a), (long long)(b)); \
        exit(1); \
    }

/*
 * Macro: TEST_LIST(TypeName)
 *
 * Exercises the full list API for a given AST node list type.
 * Uses a zero-initialised struct value (safe because list ops don't
 * dereference inner pointers — they just copy the struct by value).
 */
#define TEST_LIST(TypeName) \
    do { \
        struct nl_##TypeName zero_val; \
        memset(&zero_val, 0, sizeof(zero_val)); \
        \
        /* new + is_empty + length + capacity */ \
        List_##TypeName *lst = nl_list_##TypeName##_new(); \
        ASSERT(lst != NULL); \
        ASSERT(nl_list_##TypeName##_is_empty(lst)); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 0); \
        ASSERT(nl_list_##TypeName##_capacity(lst) > 0); \
        \
        /* push + get + length + is_empty */ \
        nl_list_##TypeName##_push(lst, zero_val); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 1); \
        ASSERT(!nl_list_##TypeName##_is_empty(lst)); \
        nl_list_##TypeName##_get(lst, 0); \
        \
        /* push second element */ \
        nl_list_##TypeName##_push(lst, zero_val); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 2); \
        \
        /* set */ \
        nl_list_##TypeName##_set(lst, 0, zero_val); \
        \
        /* insert at index 0 */ \
        nl_list_##TypeName##_insert(lst, 0, zero_val); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 3); \
        \
        /* remove at index 0 */ \
        nl_list_##TypeName##_remove(lst, 0); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 2); \
        \
        /* pop */ \
        nl_list_##TypeName##_pop(lst); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 1); \
        \
        /* clear */ \
        nl_list_##TypeName##_clear(lst); \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 0); \
        \
        /* with_capacity explicit */ \
        List_##TypeName *lst2 = nl_list_##TypeName##_with_capacity(4); \
        ASSERT(lst2 != NULL); \
        nl_list_##TypeName##_free(lst2); \
        \
        /* push 10 items to exercise ensure_capacity growth path */ \
        for (int _i = 0; _i < 10; _i++) { \
            nl_list_##TypeName##_push(lst, zero_val); \
        } \
        ASSERT_EQ(nl_list_##TypeName##_length(lst), 10); \
        \
        nl_list_##TypeName##_free(lst); \
        printf("  ✓ List_%s\n", #TypeName); \
    } while (0)

/* Test the non-AST list types (int, string, LexerToken, Token, CompilerDiagnostic, CompilerSourceLocation) */
static void test_non_ast_lists(void) {
    printf("\nTesting non-AST list types:\n");

    /* list_int */
    {
        List_int *li = list_int_new();
        ASSERT(li != NULL);
        ASSERT(list_int_is_empty(li));
        ASSERT_EQ(list_int_length(li), 0);
        ASSERT(list_int_capacity(li) > 0);
        list_int_push(li, 42);
        list_int_push(li, 99);
        ASSERT_EQ(list_int_length(li), 2);
        ASSERT(!list_int_is_empty(li));
        ASSERT_EQ(list_int_get(li, 0), 42);
        list_int_set(li, 0, 7);
        ASSERT_EQ(list_int_get(li, 0), 7);
        list_int_insert(li, 0, 1);
        ASSERT_EQ(list_int_length(li), 3);
        list_int_remove(li, 0);
        ASSERT_EQ(list_int_length(li), 2);
        list_int_pop(li);
        ASSERT_EQ(list_int_length(li), 1);
        list_int_clear(li);
        ASSERT_EQ(list_int_length(li), 0);
        List_int *li2 = list_int_with_capacity(4);
        ASSERT(li2 != NULL);
        list_int_free(li2);
        for (int i = 0; i < 10; i++) list_int_push(li, (int64_t)i);
        ASSERT_EQ(list_int_length(li), 10);
        list_int_free(li);
        printf("  ✓ List_int\n");
    }

    /* list_string */
    {
        List_string *ls = list_string_new();
        ASSERT(ls != NULL);
        ASSERT(list_string_is_empty(ls));
        ASSERT_EQ(list_string_length(ls), 0);
        ASSERT(list_string_capacity(ls) > 0);
        list_string_push(ls, "hello");
        list_string_push(ls, "world");
        ASSERT_EQ(list_string_length(ls), 2);
        ASSERT(!list_string_is_empty(ls));
        list_string_get(ls, 0);
        list_string_set(ls, 0, "foo");
        list_string_insert(ls, 0, "bar");
        ASSERT_EQ(list_string_length(ls), 3);
        list_string_remove(ls, 0);
        ASSERT_EQ(list_string_length(ls), 2);
        list_string_pop(ls);
        ASSERT_EQ(list_string_length(ls), 1);
        list_string_clear(ls);
        ASSERT_EQ(list_string_length(ls), 0);
        List_string *ls2 = list_string_with_capacity(4);
        ASSERT(ls2 != NULL);
        list_string_free(ls2);
        for (int i = 0; i < 10; i++) list_string_push(ls, "x");
        ASSERT_EQ(list_string_length(ls), 10);
        list_string_free(ls);
        printf("  ✓ List_string\n");
    }

    /* nl_list_LexerToken */
    {
        struct nl_LexerToken zero_lt;
        memset(&zero_lt, 0, sizeof(zero_lt));
        List_LexerToken *ll = nl_list_LexerToken_new();
        ASSERT(ll != NULL);
        ASSERT(nl_list_LexerToken_is_empty(ll));
        nl_list_LexerToken_push(ll, zero_lt);
        nl_list_LexerToken_push(ll, zero_lt);
        ASSERT_EQ(nl_list_LexerToken_length(ll), 2);
        nl_list_LexerToken_get(ll, 0);
        nl_list_LexerToken_set(ll, 0, zero_lt);
        nl_list_LexerToken_insert(ll, 0, zero_lt);
        nl_list_LexerToken_remove(ll, 0);
        nl_list_LexerToken_pop(ll);
        ASSERT_EQ(nl_list_LexerToken_length(ll), 1);
        nl_list_LexerToken_clear(ll);
        ASSERT_EQ(nl_list_LexerToken_length(ll), 0);
        List_LexerToken *ll2 = nl_list_LexerToken_with_capacity(4);
        ASSERT(ll2 != NULL);
        nl_list_LexerToken_free(ll2);
        ASSERT(nl_list_LexerToken_capacity(ll) > 0);
        for (int i = 0; i < 10; i++) nl_list_LexerToken_push(ll, zero_lt);
        nl_list_LexerToken_free(ll);
        printf("  ✓ List_LexerToken\n");
    }

    /* nl_list_Token (uses struct nl_LexerToken as element type) */
    {
        struct nl_LexerToken zero_lt;
        memset(&zero_lt, 0, sizeof(zero_lt));
        List_Token *lt = nl_list_Token_new();
        ASSERT(lt != NULL);
        ASSERT(nl_list_Token_is_empty(lt));
        nl_list_Token_push(lt, zero_lt);
        nl_list_Token_push(lt, zero_lt);
        ASSERT_EQ(nl_list_Token_length(lt), 2);
        nl_list_Token_get(lt, 0);
        nl_list_Token_set(lt, 0, zero_lt);
        nl_list_Token_insert(lt, 0, zero_lt);
        nl_list_Token_remove(lt, 0);
        nl_list_Token_pop(lt);
        ASSERT_EQ(nl_list_Token_length(lt), 1);
        nl_list_Token_clear(lt);
        ASSERT_EQ(nl_list_Token_length(lt), 0);
        List_Token *lt2 = nl_list_Token_with_capacity(4);
        ASSERT(lt2 != NULL);
        nl_list_Token_free(lt2);
        ASSERT(nl_list_Token_capacity(lt) > 0);
        for (int i = 0; i < 10; i++) nl_list_Token_push(lt, zero_lt);
        nl_list_Token_free(lt);
        printf("  ✓ List_Token\n");
    }

    /* nl_list_CompilerDiagnostic */
    {
        struct nl_CompilerDiagnostic zero_cd;
        memset(&zero_cd, 0, sizeof(zero_cd));
        List_CompilerDiagnostic *lcd = nl_list_CompilerDiagnostic_new();
        ASSERT(lcd != NULL);
        ASSERT(nl_list_CompilerDiagnostic_is_empty(lcd));
        nl_list_CompilerDiagnostic_push(lcd, zero_cd);
        nl_list_CompilerDiagnostic_push(lcd, zero_cd);
        ASSERT_EQ(nl_list_CompilerDiagnostic_length(lcd), 2);
        nl_list_CompilerDiagnostic_get(lcd, 0);
        nl_list_CompilerDiagnostic_set(lcd, 0, zero_cd);
        nl_list_CompilerDiagnostic_insert(lcd, 0, zero_cd);
        nl_list_CompilerDiagnostic_remove(lcd, 0);
        nl_list_CompilerDiagnostic_pop(lcd);
        ASSERT_EQ(nl_list_CompilerDiagnostic_length(lcd), 1);
        nl_list_CompilerDiagnostic_clear(lcd);
        ASSERT_EQ(nl_list_CompilerDiagnostic_length(lcd), 0);
        List_CompilerDiagnostic *lcd2 = nl_list_CompilerDiagnostic_with_capacity(4);
        ASSERT(lcd2 != NULL);
        nl_list_CompilerDiagnostic_free(lcd2);
        ASSERT(nl_list_CompilerDiagnostic_capacity(lcd) > 0);
        for (int i = 0; i < 10; i++) nl_list_CompilerDiagnostic_push(lcd, zero_cd);
        nl_list_CompilerDiagnostic_free(lcd);
        printf("  ✓ List_CompilerDiagnostic\n");
    }

    /* nl_list_CompilerSourceLocation */
    {
        struct nl_CompilerSourceLocation zero_csl;
        memset(&zero_csl, 0, sizeof(zero_csl));
        List_CompilerSourceLocation *lcsl = nl_list_CompilerSourceLocation_new();
        ASSERT(lcsl != NULL);
        ASSERT(nl_list_CompilerSourceLocation_is_empty(lcsl));
        nl_list_CompilerSourceLocation_push(lcsl, zero_csl);
        nl_list_CompilerSourceLocation_push(lcsl, zero_csl);
        ASSERT_EQ(nl_list_CompilerSourceLocation_length(lcsl), 2);
        nl_list_CompilerSourceLocation_get(lcsl, 0);
        nl_list_CompilerSourceLocation_set(lcsl, 0, zero_csl);
        nl_list_CompilerSourceLocation_insert(lcsl, 0, zero_csl);
        nl_list_CompilerSourceLocation_remove(lcsl, 0);
        nl_list_CompilerSourceLocation_pop(lcsl);
        ASSERT_EQ(nl_list_CompilerSourceLocation_length(lcsl), 1);
        nl_list_CompilerSourceLocation_clear(lcsl);
        ASSERT_EQ(nl_list_CompilerSourceLocation_length(lcsl), 0);
        List_CompilerSourceLocation *lcsl2 = nl_list_CompilerSourceLocation_with_capacity(4);
        ASSERT(lcsl2 != NULL);
        nl_list_CompilerSourceLocation_free(lcsl2);
        ASSERT(nl_list_CompilerSourceLocation_capacity(lcsl) > 0);
        for (int i = 0; i < 10; i++) nl_list_CompilerSourceLocation_push(lcsl, zero_csl);
        nl_list_CompilerSourceLocation_free(lcsl);
        printf("  ✓ List_CompilerSourceLocation\n");
    }
}

int main(void) {
    printf("=== Runtime List Tests ===\n\n");
    printf("Testing list operations for all 33 compiled AST node list types:\n");

    TEST_LIST(ASTArrayLiteral);
    TEST_LIST(ASTAssert);
    TEST_LIST(ASTBinaryOp);
    TEST_LIST(ASTBlock);
    TEST_LIST(ASTBool);
    TEST_LIST(ASTCall);
    TEST_LIST(ASTEnum);
    TEST_LIST(ASTFieldAccess);
    TEST_LIST(ASTFloat);
    TEST_LIST(ASTFor);
    TEST_LIST(ASTFunction);
    TEST_LIST(ASTIdentifier);
    TEST_LIST(ASTIf);
    TEST_LIST(ASTImport);
    TEST_LIST(ASTLet);
    TEST_LIST(ASTMatch);
    TEST_LIST(ASTModuleQualifiedCall);
    TEST_LIST(ASTNumber);
    TEST_LIST(ASTOpaqueType);
    TEST_LIST(ASTPrint);
    TEST_LIST(ASTReturn);
    TEST_LIST(ASTSet);
    TEST_LIST(ASTShadow);
    TEST_LIST(ASTStmtRef);
    TEST_LIST(ASTString);
    TEST_LIST(ASTStruct);
    TEST_LIST(ASTStructLiteral);
    TEST_LIST(ASTTupleIndex);
    TEST_LIST(ASTTupleLiteral);
    TEST_LIST(ASTUnion);
    TEST_LIST(ASTUnionConstruct);
    TEST_LIST(ASTUnsafeBlock);
    TEST_LIST(ASTWhile);

    test_non_ast_lists();

    printf("\n✓ All runtime list tests passed! (33 AST + 6 non-AST list types)\n");
    return 0;
}

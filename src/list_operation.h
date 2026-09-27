#ifndef NANO_LIST_OPERATION_H
#define NANO_LIST_OPERATION_H
#include <string.h>

/* I name the list types backed by the dedicated compiler-schema runtime. */
static inline int nl_list_has_schema_runtime(const char *name) {
    static const char *types[] = {
        "ASTArrayLiteral", "ASTAssert", "ASTBinaryOp", "ASTBlock", "ASTBool",
        "ASTCall", "ASTEnum", "ASTFieldAccess", "ASTFloat", "ASTFor",
        "ASTFunction", "ASTIdentifier", "ASTIf", "ASTImport", "ASTLet",
        "ASTMatch", "ASTModuleQualifiedCall", "ASTNumber", "ASTOpaqueType",
        "ASTPrint", "ASTReturn", "ASTServiceDecl", "ASTSet", "ASTShadow",
        "ASTStmtRef", "ASTString", "ASTStruct", "ASTStructLiteral",
        "ASTTupleIndex", "ASTTupleLiteral", "ASTUnion", "ASTUnionConstruct",
        "ASTUnsafeBlock", "ASTWhile", "CompilerDiagnostic",
        "CompilerSourceLocation", "LexerToken",
    };
    if (!name) return 0;
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i)
        if (!strcmp(name, types[i])) return 1;
    return 0;
}

/* I separate the element name from complete compound operation suffixes. */
static inline const char *nl_list_operation_separator(const char *name) {
    if (!name) return NULL;
    size_t length = strlen(name);
    const char *compound[] = {"_with_capacity", "_is_empty"};
    for (size_t i = 0; i < sizeof(compound) / sizeof(compound[0]); i++) {
        size_t suffix_length = strlen(compound[i]);
        if (length > suffix_length && !strcmp(name + length - suffix_length, compound[i]))
            return name + length - suffix_length;
    }
    return strrchr(name, '_');
}
#endif

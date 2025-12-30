/* Schema type list implementations for self-hosted compiler */
#ifndef SCHEMA_LISTS_H
#define SCHEMA_LISTS_H

#include <stdbool.h>

/* Forward declarations for schema types (defined in transpiled compiler_ast.nano) */
typedef struct nl_ASTTupleIndex {
    int node_type;
    int line;
    int column;
    int tuple;
    int tuple_type;
    int index;
} nl_ASTTupleIndex;

typedef struct nl_CompilerSourceLocation {
    char *file;
    int line;
    int column;
} nl_CompilerSourceLocation;

typedef struct nl_CompilerDiagnostic {
    int phase;
    int severity;
    char *code;
    char *message;
    nl_CompilerSourceLocation location;
} nl_CompilerDiagnostic;

/* List type definitions */
typedef struct List_ASTTupleIndex {
    nl_ASTTupleIndex *data;
    int length;
    int capacity;
} List_ASTTupleIndex;

typedef struct List_CompilerDiagnostic {
    nl_CompilerDiagnostic *data;
    int length;
    int capacity;
} List_CompilerDiagnostic;

/* List<ASTTupleIndex> functions */
List_ASTTupleIndex* nl_list_ASTTupleIndex_new(void);
void nl_list_ASTTupleIndex_push(List_ASTTupleIndex *list, nl_ASTTupleIndex value);
nl_ASTTupleIndex nl_list_ASTTupleIndex_get(List_ASTTupleIndex *list, int index);
int nl_list_ASTTupleIndex_length(List_ASTTupleIndex *list);
void nl_list_ASTTupleIndex_free(List_ASTTupleIndex *list);

/* List<CompilerDiagnostic> functions */
List_CompilerDiagnostic* nl_list_CompilerDiagnostic_new(void);
void nl_list_CompilerDiagnostic_push(List_CompilerDiagnostic *list, nl_CompilerDiagnostic value);
nl_CompilerDiagnostic nl_list_CompilerDiagnostic_get(List_CompilerDiagnostic *list, int index);
int nl_list_CompilerDiagnostic_length(List_CompilerDiagnostic *list);
void nl_list_CompilerDiagnostic_free(List_CompilerDiagnostic *list);

#endif /* SCHEMA_LISTS_H */


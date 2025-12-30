#ifndef LIST_COMPILERDIAGNOSTIC_H
#define LIST_COMPILERDIAGNOSTIC_H

#include <stdint.h>
#include <stdbool.h>
#include "../generated/compiler_schema.h"

/* Dynamic list of CompilerDiagnostic */
typedef struct List_CompilerDiagnostic {
    struct nl_CompilerDiagnostic *data;
    int length;
    int capacity;
} List_CompilerDiagnostic;

List_CompilerDiagnostic* nl_list_CompilerDiagnostic_new(void);
List_CompilerDiagnostic* nl_list_CompilerDiagnostic_with_capacity(int capacity);
void nl_list_CompilerDiagnostic_push(List_CompilerDiagnostic *list, struct nl_CompilerDiagnostic value);
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_pop(List_CompilerDiagnostic *list);
void nl_list_CompilerDiagnostic_insert(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value);
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_remove(List_CompilerDiagnostic *list, int index);
void nl_list_CompilerDiagnostic_set(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value);
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_get(List_CompilerDiagnostic *list, int index);
void nl_list_CompilerDiagnostic_clear(List_CompilerDiagnostic *list);
int nl_list_CompilerDiagnostic_length(List_CompilerDiagnostic *list);
int nl_list_CompilerDiagnostic_capacity(List_CompilerDiagnostic *list);
bool nl_list_CompilerDiagnostic_is_empty(List_CompilerDiagnostic *list);
void nl_list_CompilerDiagnostic_free(List_CompilerDiagnostic *list);

#endif /* LIST_COMPILERDIAGNOSTIC_H */

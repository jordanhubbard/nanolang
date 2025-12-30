#ifndef LIST_ASTTUPLEINDEX_H
#define LIST_ASTTUPLEINDEX_H

#include <stdint.h>
#include <stdbool.h>
#include "../generated/compiler_schema.h"

/* Dynamic list of ASTTupleIndex */
typedef struct List_ASTTupleIndex {
    struct nl_ASTTupleIndex *data;
    int length;
    int capacity;
} List_ASTTupleIndex;

List_ASTTupleIndex* nl_list_ASTTupleIndex_new(void);
List_ASTTupleIndex* nl_list_ASTTupleIndex_with_capacity(int capacity);
void nl_list_ASTTupleIndex_push(List_ASTTupleIndex *list, struct nl_ASTTupleIndex value);
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_pop(List_ASTTupleIndex *list);
void nl_list_ASTTupleIndex_insert(List_ASTTupleIndex *list, int index, struct nl_ASTTupleIndex value);
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_remove(List_ASTTupleIndex *list, int index);
void nl_list_ASTTupleIndex_set(List_ASTTupleIndex *list, int index, struct nl_ASTTupleIndex value);
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_get(List_ASTTupleIndex *list, int index);
void nl_list_ASTTupleIndex_clear(List_ASTTupleIndex *list);
int nl_list_ASTTupleIndex_length(List_ASTTupleIndex *list);
int nl_list_ASTTupleIndex_capacity(List_ASTTupleIndex *list);
bool nl_list_ASTTupleIndex_is_empty(List_ASTTupleIndex *list);
void nl_list_ASTTupleIndex_free(List_ASTTupleIndex *list);

#endif /* LIST_ASTTUPLEINDEX_H */

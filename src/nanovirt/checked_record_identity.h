#ifndef NANOVIRT_CHECKED_RECORD_IDENTITY_H
#define NANOVIRT_CHECKED_RECORD_IDENTITY_H
#include "nanolang.h"

/* I use the registered checked declaration, never a same-spelled lookup. */
static inline bool nanovirt_checked_record_identity(const ASTNode *declaration,
        const char *name, char **fields, int count, bool ambiguous) {
    return declaration && !ambiguous && declaration->type == AST_STRUCT_DEF &&
        declaration->as.struct_def.name == name &&
        declaration->as.struct_def.field_names == fields &&
        declaration->as.struct_def.field_count == count;
}
#endif

/* I share exact pure-query definitions with standalone C-profile fixtures.
 * Each consumer selects its definitions and keeps their external linkage. */
#include "nanolang.h"

#ifdef NL_DEFINE_ROW_FIELD_INFO
const TypeInfo *row_field_info(const TypeInfo *info, int index, TypeInfo *fallback) {
    if (!info || index < 0 || index >= info->row_field_count) return NULL;
    if (info->row_field_type_info && info->row_field_type_info[index]) return info->row_field_type_info[index];
    if (!fallback || !info->row_field_types) return NULL;
    *fallback = (TypeInfo){.base_type = info->row_field_types[index],
        .generic_name = info->row_field_type_names ? info->row_field_type_names[index] : NULL};
    return fallback;
}
#endif

#ifdef NL_DEFINE_AST_SERVICE_QUERY
bool ast_has_service_declaration(const ASTNode *program) {
    if (!program) return false;
    if (program->type == AST_SERVICE_DECL) return true;
    if (program->type != AST_PROGRAM) return false;
    for (int i = 0; i < program->as.program.count; ++i)
        if (ast_has_service_declaration(program->as.program.items[i])) return true;
    return false;
}
#endif

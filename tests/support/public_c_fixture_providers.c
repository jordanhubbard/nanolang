/* I compile the exact query definitions used by the compiler. */
#define NL_DEFINE_ROW_FIELD_INFO
#define NL_DEFINE_AST_SERVICE_QUERY
#include "../../src/type_metadata_queries.inc"
#undef NL_DEFINE_AST_SERVICE_QUERY
#undef NL_DEFINE_ROW_FIELD_INFO

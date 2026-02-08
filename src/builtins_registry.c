/*
 * builtins_registry.c - Unified Builtin Function Registry
 *
 * Single authoritative table of all nanolang builtin functions, reconciled
 * from the former four independent tables in env.c, codegen.c, transpiler.c,
 * and vm_builtins.h.
 */
#include "builtins_registry.h"
#include <string.h>

/* Import NanoISA opcodes — 0 means "not an inline opcode" */
#define OP_NOP           0x00
#define OP_STR_LEN       0x40
#define OP_STR_CONCAT    0x41
#define OP_STR_SUBSTR    0x42
#define OP_STR_CONTAINS  0x43
#define OP_STR_EQ        0x44
#define OP_STR_CHAR_AT   0x45
#define OP_STR_FROM_INT  0x46
#define OP_STR_FROM_FLOAT 0x47
#define OP_CAST_INT      0x88
#define OP_CAST_FLOAT    0x89
#define OP_CAST_BOOL     0x8A
#define OP_CAST_STRING   0x8B

/*
 * The unified registry. Every builtin function known to the language appears
 * exactly once here. The columns are:
 *
 *   name         — nanolang source-level name
 *   c_name       — symbol emitted by the C transpiler (or NULL = same as name)
 *   arity        — number of parameters
 *   param_types  — up to 4 parameter types (rest TYPE_UNKNOWN)
 *   return_type  — return type
 *   nano_opcode  — NanoISA opcode if inlined in the VM (0 = FFI/extern)
 *   flags        — BUILTIN_PURE | BUILTIN_INLINE_VM | BUILTIN_FFI | BUILTIN_IO
 */

#define U TYPE_UNKNOWN
#define I TYPE_INT
#define F TYPE_FLOAT
#define B TYPE_BOOL
#define S TYPE_STRING
#define V TYPE_VOID
#define A TYPE_ARRAY
#define O TYPE_OPAQUE
#define L BUILTIN_LANG  /* Known to the typechecker */

const BuiltinEntry builtin_registry[] = {
    /* ── Core ──────────────────────────────────────────────────────── */
    {"range",           "range",             2, {I,I,U,U}, I, OP_NOP, L|BUILTIN_PURE},

    /* ── Print / IO ───────────────────────────────────────────────── */
    {"print",           "print",             1, {U,U,U,U}, V, OP_NOP, L|BUILTIN_IO},
    {"println",         "println",           1, {U,U,U,U}, V, OP_NOP, L|BUILTIN_IO},

    /* ── Math ─────────────────────────────────────────────────────── */
    {"abs",             "nl_abs",            1, {I,U,U,U}, I, OP_NOP, L|BUILTIN_PURE},
    {"min",             "nl_min",            2, {I,I,U,U}, I, OP_NOP, L|BUILTIN_PURE},
    {"max",             "nl_max",            2, {I,I,U,U}, I, OP_NOP, L|BUILTIN_PURE},
    {"sqrt",            "sqrt",              1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"pow",             "pow",               2, {F,F,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"floor",           "floor",             1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"ceil",            "ceil",              1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"round",           "round",             1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"sin",             "sin",               1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"cos",             "cos",               1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"tan",             "tan",               1, {F,U,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"atan2",           "atan2",             2, {F,F,U,U}, F, OP_NOP, L|BUILTIN_PURE|BUILTIN_FFI},
    {"asin",            "asin",              1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"acos",            "acos",              1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"atan",            "atan",              1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"log",             "log",               1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"log2",            "log2",              1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"log10",           "log10",             1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"exp",             "exp",               1, {F,U,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},
    {"fmod",            "fmod",              2, {F,F,U,U}, F, OP_NOP, BUILTIN_PURE|BUILTIN_FFI},

    /* ── Type casting ─────────────────────────────────────────────── */
    {"cast_int",        "nl_cast_int",       1, {U,U,U,U}, I, OP_CAST_INT,    L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"cast_float",      "nl_cast_float",     1, {U,U,U,U}, F, OP_CAST_FLOAT,  L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"cast_bool",       "nl_cast_bool",      1, {U,U,U,U}, B, OP_CAST_BOOL,   L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"cast_string",     "cast_string",       1, {U,U,U,U}, S, OP_CAST_STRING, L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"to_string",       "to_string",         1, {U,U,U,U}, S, OP_CAST_STRING, L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"int_to_string",   "int_to_string",     1, {I,U,U,U}, S, OP_STR_FROM_INT, BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"float_to_string", "float_to_string",   1, {F,U,U,U}, S, OP_STR_FROM_FLOAT, BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"bool_to_string",  "bool_to_string",    1, {B,U,U,U}, S, OP_CAST_STRING, BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"null_opaque",     "nl_null_opaque",    0, {U,U,U,U}, O, OP_NOP, L|BUILTIN_PURE},

    /* ── String operations ────────────────────────────────────────── */
    {"str_length",      "strlen",            1, {S,U,U,U}, I, OP_STR_LEN,      L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"str_concat",      "nl_str_concat",     2, {S,S,U,U}, S, OP_STR_CONCAT,   L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"str_substring",   "nl_str_substring",  3, {S,I,I,U}, S, OP_STR_SUBSTR,   L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"str_contains",    "nl_str_contains",   2, {S,S,U,U}, B, OP_STR_CONTAINS, L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"str_equals",      "nl_str_equals",     2, {S,S,U,U}, B, OP_STR_EQ,       L|BUILTIN_PURE|BUILTIN_INLINE_VM},
    /* str_index_of: handled by module FFI, not a builtin (VM has vm_str_index_of) */
    {"char_at",         "char_at",           2, {S,I,U,U}, I, OP_STR_CHAR_AT,  BUILTIN_PURE|BUILTIN_INLINE_VM},
    {"string_from_char","string_from_char",  1, {I,U,U,U}, S, OP_NOP, BUILTIN_PURE},
    {"string_to_int",   "string_to_int",     1, {S,U,U,U}, I, OP_NOP, BUILTIN_PURE},
    {"string_to_float", "string_to_float",   1, {S,U,U,U}, F, OP_NOP, BUILTIN_PURE},

    /* ── Character classification ─────────────────────────────────── */
    {"is_digit",        "is_digit",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_alpha",        "is_alpha",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_alnum",        "is_alnum",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_space",        "is_space",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_whitespace",   "is_whitespace",     1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_upper",        "is_upper",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"is_lower",        "is_lower",          1, {I,U,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"digit_value",     "digit_value",       1, {I,U,U,U}, I, OP_NOP, BUILTIN_PURE},
    {"char_to_lower",   "char_to_lower",     1, {I,U,U,U}, I, OP_NOP, BUILTIN_PURE},
    {"char_to_upper",   "char_to_upper",     1, {I,U,U,U}, I, OP_NOP, BUILTIN_PURE},

    /* ── Array operations ─────────────────────────────────────────── */
    {"array_length",    "dyn_array_length",  1, {A,U,U,U}, I, OP_NOP, L|BUILTIN_PURE},
    {"array_new",       "array_new",         2, {I,U,U,U}, A, OP_NOP, L|BUILTIN_PURE},
    {"array_set",       "dyn_array_put",     3, {A,I,U,U}, V, OP_NOP, L},
    {"at",              "dyn_array_get",     2, {A,I,U,U}, U, OP_NOP, L|BUILTIN_PURE},
    {"array_get",       "array_get",         2, {A,I,U,U}, U, OP_NOP, BUILTIN_PURE},
    {"array_push",      "array_push",        2, {A,U,U,U}, V, OP_NOP, 0},
    {"array_pop",       "array_pop",         1, {A,U,U,U}, U, OP_NOP, 0},
    {"array_remove_at", "dyn_array_remove_at", 2, {A,I,U,U}, V, OP_NOP, 0},
    {"array_slice",     "nl_array_slice",    3, {A,I,I,U}, A, OP_NOP, BUILTIN_PURE},
    {"array_concat",    "array_concat",      2, {A,A,U,U}, A, OP_NOP, BUILTIN_PURE},

    /* ── Hashmap operations ───────────────────────────────────────── */
    {"hashmap_new",     "hashmap_new",       0, {U,U,U,U}, U, OP_NOP, 0},
    {"hashmap_get",     "hashmap_get",       2, {U,S,U,U}, U, OP_NOP, BUILTIN_PURE},
    {"hashmap_set",     "hashmap_set",       3, {U,S,U,U}, V, OP_NOP, 0},
    {"hashmap_has",     "hashmap_has",       2, {U,S,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"hashmap_delete",  "hashmap_delete",    2, {U,S,U,U}, V, OP_NOP, 0},
    {"hashmap_keys",    "hashmap_keys",      1, {U,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"hashmap_values",  "hashmap_values",    1, {U,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"hashmap_length",  "hashmap_length",    1, {U,U,U,U}, I, OP_NOP, BUILTIN_PURE},
    /* map_* aliases (transpiler emits the same C names) */
    {"map_new",         "hashmap_new",       0, {U,U,U,U}, U, OP_NOP, 0},
    {"map_get",         "hashmap_get",       2, {U,S,U,U}, U, OP_NOP, BUILTIN_PURE},
    {"map_set",         "hashmap_set",       3, {U,S,U,U}, V, OP_NOP, 0},
    {"map_has",         "hashmap_has",       2, {U,S,U,U}, B, OP_NOP, BUILTIN_PURE},
    {"map_delete",      "hashmap_delete",    2, {U,S,U,U}, V, OP_NOP, 0},
    {"map_keys",        "hashmap_keys",      1, {U,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"map_values",      "hashmap_values",    1, {U,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"map_length",      "hashmap_length",    1, {U,U,U,U}, I, OP_NOP, BUILTIN_PURE},

    /* ── Higher-order ─────────────────────────────────────────────── */
    {"filter",          "filter",            2, {A,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"map",             "map",               2, {A,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"reduce",          "reduce",            3, {A,U,U,U}, U, OP_NOP, BUILTIN_PURE},

    /* ── Result<T, E> helpers ─────────────────────────────────────── */
    {"result_is_ok",      "result_is_ok",      1, {U,U,U,U}, B, OP_NOP, L|BUILTIN_PURE},
    {"result_is_err",     "result_is_err",     1, {U,U,U,U}, B, OP_NOP, L|BUILTIN_PURE},
    {"result_unwrap",     "result_unwrap",     1, {U,U,U,U}, U, OP_NOP, L|BUILTIN_PURE},
    {"result_unwrap_err", "result_unwrap_err", 1, {U,U,U,U}, U, OP_NOP, L|BUILTIN_PURE},
    {"result_unwrap_or",  "result_unwrap_or",  2, {U,U,U,U}, U, OP_NOP, L|BUILTIN_PURE},
    {"result_map",        "result_map",        2, {U,U,U,U}, U, OP_NOP, L|BUILTIN_PURE},
    {"result_and_then",   "result_and_then",   2, {U,U,U,U}, U, OP_NOP, L|BUILTIN_PURE},

    /* ── OS / File System (FFI) ───────────────────────────────────── */
    {"file_read",       "nl_os_file_read",   1, {S,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_read_bytes", "nl_os_file_read_bytes", 1, {S,U,U,U}, A, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_write",      "nl_os_file_write",  2, {S,S,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_append",     "nl_os_file_append", 2, {S,S,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_remove",     "nl_os_file_remove", 1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_rename",     "nl_os_file_rename", 2, {S,S,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_exists",     "nl_os_file_exists", 1, {S,U,U,U}, B, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"file_size",       "nl_os_file_size",   1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"tmp_dir",         "nl_os_tmp_dir",     0, {U,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"mktemp",          "nl_os_mktemp",      1, {S,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"mktemp_dir",      "nl_os_mktemp_dir",  1, {S,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"dir_create",      "nl_os_dir_create",  1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"dir_remove",      "nl_os_dir_remove",  1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"dir_list",        "nl_os_dir_list",    1, {S,U,U,U}, A, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"dir_exists",      "nl_os_dir_exists",  1, {S,U,U,U}, B, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"getcwd",          "nl_os_getcwd",      0, {U,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"chdir",           "nl_os_chdir",       1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"fs_walkdir",      "nl_os_walkdir",     1, {S,U,U,U}, A, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"path_isfile",     "nl_os_path_isfile", 1, {S,U,U,U}, B, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"path_isdir",      "nl_os_path_isdir",  1, {S,U,U,U}, B, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"path_join",       "nl_os_path_join",   2, {S,S,U,U}, S, OP_NOP, BUILTIN_PURE | BUILTIN_FFI},
    {"path_basename",   "nl_os_path_basename", 1, {S,U,U,U}, S, OP_NOP, BUILTIN_PURE | BUILTIN_FFI},
    {"path_dirname",    "nl_os_path_dirname",  1, {S,U,U,U}, S, OP_NOP, BUILTIN_PURE | BUILTIN_FFI},
    {"path_normalize",  "nl_os_path_normalize",1, {S,U,U,U}, S, OP_NOP, BUILTIN_PURE | BUILTIN_FFI},
    {"system",          "nl_os_system",      1, {S,U,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"exit",            "nl_os_exit",        1, {I,U,U,U}, V, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"getenv",          "nl_os_getenv",      1, {S,U,U,U}, S, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"setenv",          "nl_os_setenv",      2, {S,S,U,U}, I, OP_NOP, BUILTIN_IO | BUILTIN_FFI},
    {"process_run",     "nl_os_process_run", 1, {S,U,U,U}, A, OP_NOP, BUILTIN_IO | BUILTIN_FFI},

    /* ── Binary string ────────────────────────────────────────────── */
    {"bytes_from_string",  "nl_bytes_from_string",  1, {S,U,U,U}, A, OP_NOP, BUILTIN_PURE},
    {"string_from_bytes",  "nl_string_from_bytes",  1, {A,U,U,U}, S, OP_NOP, BUILTIN_PURE},
    {"bstr_utf8_length",   "bstr_utf8_length",      1, {S,U,U,U}, I, OP_NOP, BUILTIN_PURE},
    {"bstr_utf8_char_at",  "bstr_utf8_char_at",     2, {S,I,U,U}, I, OP_NOP, BUILTIN_PURE},
    {"bstr_validate_utf8", "bstr_validate_utf8",    1, {S,U,U,U}, B, OP_NOP, BUILTIN_PURE},
};

const int builtin_registry_count = sizeof(builtin_registry) / sizeof(builtin_registry[0]);

#undef U
#undef I
#undef F
#undef B
#undef S
#undef V
#undef A
#undef O
#undef L

/* ── Lookup functions ─────────────────────────────────────────────── */

const BuiltinEntry *builtin_find(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < builtin_registry_count; i++) {
        if (strcmp(builtin_registry[i].name, name) == 0) {
            return &builtin_registry[i];
        }
    }
    return NULL;
}

const char *builtin_c_name(const char *name) {
    const BuiltinEntry *e = builtin_find(name);
    return e ? e->c_name : NULL;
}

bool builtin_is_known(const char *name) {
    return builtin_find(name) != NULL;
}

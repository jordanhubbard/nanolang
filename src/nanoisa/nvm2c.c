/*
 * Structured C11 from a closed NanoISA subset.
 *
 * Temps are C arrays so backward goto is valid C. I64_ADD becomes
 * `t[i] = a + b`. Strings live in a parallel `s[]` of C string pointers.
 * The operand stack exists only while translating.
 */

#include "nvm2c.h"
#include "../binary64_bits.h"
#include "../binary64_format.h"
#include "../binary64_arithmetic_source.h"
#include "binary64_parse_source.h"
#include "isa.h"
#include "utf8.h"
#include "nvm2c_shape.h"
#include "ownership_contracts.h"
#include "affine_state.h"
#include "affine_bytecode.h"
#include "verifier.h"
#include "../nanovm/vm_decode.h"

#include <stdarg.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NVM2C_MAX_LOCALS 1024
#define NVM2C_CALL_SIZE (64 + NVM2C_MAX_LOCALS * 32)

/* I inspect my generated executable text, not literal data or comments.
 * This is an internal architecture assertion, not a C security validator. */
static bool contains_vm_wrapper_code(const char *source) {
    const char *p = source;
    while (*p) {
        if (*p == '"' || *p == '\'') {
            char quote = *p++;
            while (*p && *p != quote) {
                if (*p == '\\' && p[1]) p++;
                p++;
            }
            if (*p) p++;
        } else if (p[0] == '/' && p[1] == '*') {
            p += 2;
            while (*p && !(p[0] == '*' && p[1] == '/')) p++;
            if (*p) p += 2;
        } else if (p[0] == '/' && p[1] == '/') {
            while (*p && *p != '\n') p++;
        } else {
            if (!strncmp(p, "nano_vm", 7) || !strncmp(p, "nvm_blob", 8)) return true;
            p++;
        }
    }
    return false;
}

#define NVM2C_VK_INT 0
#define NVM2C_VK_STR 1
#define NVM2C_VK_UNK 2
#define NVM2C_VK_ARR 3
#define NVM2C_VK_REC 4
#define NVM2C_VK_SARR 5
#define NVM2C_VK_RARR 6
#define NVM2C_VK_MAP 7
#define NVM2C_VK_VALUE 8
#define NVM2C_VK_BOOL 9
#define NVM2C_VK_BARR 10
#define NVM2C_VK_FLOAT 11
#define NVM2C_VK_FARR 12

/* Classifier-only field facts: low bits retain observed finite payload members.
 * These codes never appear in generated runtime record tags. */
#define NVM2C_FIELD_VARIANT_BASE 0x40
#define NVM2C_FIELD_VARIANT_OPTIONAL 0x10
#define NVM2C_FIELD_VARIANT_INT_ARRAY 0x20
#define NVM2C_SCALAR_VARIANT UINT16_C(0x4000)
static int variant_field(uint8_t kind) {
    return kind > NVM2C_FIELD_VARIANT_BASE && kind < NVM2C_FIELD_VARIANT_BASE + 64;
}
static uint8_t variant_field_for(uint8_t kind) {
    return NVM2C_FIELD_VARIANT_BASE | (kind == NVM2C_VK_INT ? 1 :
        kind == NVM2C_VK_BOOL ? 2 : kind == NVM2C_VK_FLOAT ? 4 : kind == NVM2C_VK_ARR ? NVM2C_FIELD_VARIANT_INT_ARRAY : 8);
}
static uint16_t variant_field_tags(uint8_t kind) {
    if (kind & NVM2C_FIELD_VARIANT_OPTIONAL) return 0;
    return NVM2C_SCALAR_VARIANT | ((kind & 1) ? 1u << TAG_INT : 0) |
        ((kind & 2) ? 1u << TAG_BOOL : 0) | ((kind & 4) ? 1u << TAG_FLOAT : 0) |
        ((kind & 8) ? 1u << TAG_STRING : 0) |
        ((kind & NVM2C_FIELD_VARIANT_INT_ARRAY) ? 1u << TAG_ARRAY : 0);
}
static int variant_scalar_tags(uint16_t tags) {
    const unsigned members = (1u << TAG_INT) | (1u << TAG_BOOL) |
                             (1u << TAG_FLOAT) | (1u << TAG_STRING);
    return (tags & NVM2C_SCALAR_VARIANT) && (tags & members) &&
           !(tags & ~(NVM2C_SCALAR_VARIANT | members | (1u << TAG_VOID)));
}

static int variant_payload_tags(uint16_t tags) {
    if (variant_scalar_tags(tags)) return 1;
    const unsigned members = (1u << TAG_INT) | (1u << TAG_BOOL) |
        (1u << TAG_FLOAT) | (1u << TAG_STRING) | (1u << TAG_ARRAY);
    return (tags & NVM2C_SCALAR_VARIANT) && (tags & members) &&
        !(tags & ~(NVM2C_SCALAR_VARIANT | members | (1u << TAG_VOID)));
}

static int variant_scalar_kind(uint8_t kind) {
    return kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL ||
           kind == NVM2C_VK_FLOAT || kind == NVM2C_VK_STR;
}

/* I share physical 64-bit slots, not element type identity. Float slots
 * contain memcpy-preserved double bits and never undergo integer arithmetic. */
static int word_array_storage(uint8_t kind) {
    return kind == NVM2C_VK_ARR || kind == NVM2C_VK_BARR || kind == NVM2C_VK_FARR;
}

static int boolean_result(uint8_t opcode) {
    switch (opcode) {
    case OP_CAST_BOOL: case OP_AND: case OP_OR: case OP_NOT:
    case OP_PUSH_BOOL: case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT:
    case OP_EQ: case OP_NE: case OP_I64_EQ: case OP_I64_NE:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH: case OP_STR_CONTAINS:
    case OP_HM_HAS: case OP_TYPE_CHECK: return 1;
    default: return 0;
    }
}

typedef struct Nvm2cFieldBlock {
    struct Nvm2cFieldBlock *next;
    uint8_t data[];
} Nvm2cFieldBlock;

typedef struct Nvm2cJoinShape {
    struct Nvm2cJoinShape *next;
    size_t target;
    int count;
    NvmShapeId shapes[];
} Nvm2cJoinShape;

typedef struct Nvm2cScalarJoin {
    struct Nvm2cScalarJoin *next;
    uint32_t function;
    size_t target;
    int count;
    uint16_t tags[];
} Nvm2cScalarJoin;

#define NVM2C_SCALAR_UNKNOWN UINT16_C(0x8000)

/* A boxed carrier is not proof of a scalar payload. These masks come only
 * from exact scalar producers, their copies, and conservative local stores. */
static uint16_t scalar_kind_tags(uint8_t kind) {
    return kind == NVM2C_VK_INT ? 1u << TAG_INT :
           kind == NVM2C_VK_BOOL ? 1u << TAG_BOOL :
           kind == NVM2C_VK_FLOAT ? 1u << TAG_FLOAT :
           kind == NVM2C_VK_STR ? 1u << TAG_STRING : NVM2C_SCALAR_UNKNOWN;
}

static int boxed_carrier_tags(uint16_t tags) {
    if (variant_payload_tags(tags)) return 1;
    const unsigned numeric = (1u << TAG_INT) | (1u << TAG_FLOAT);
    return (tags & numeric) == numeric && !(tags & ~(numeric | (1u << TAG_VOID)));
}

static int optional_scalar_tags(uint16_t tags) {
    unsigned payload = tags & ~(1u << TAG_VOID);
    return (tags & (1u << TAG_VOID)) && payload && !(payload & (payload - 1)) &&
           !(tags & ~((1u << (TAG_BOOL + 1)) - 1));
}

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    char *err;
    size_t err_len;
    int failed;
    size_t sim_stack_capacity;
    size_t record_width;
    int has_maps;
    int has_float_arithmetic;
    int has_string_arrays;
    int has_integer_arrays;
    int has_record_array_allocations;
    int has_record_array_getter;
    int has_owned_strings;
    int has_owned_aggregates;
    size_t global_count;
    size_t local_width;
    uint8_t *array_results;
    uint8_t *emitted_functions;
    uint8_t *required_functions;
    uint8_t *tagged_locals;
    uint16_t *local_scalar_tags;
    Nvm2cScalarJoin *scalar_joins;
    uint16_t array_shape_kinds;
    Nvm2cFieldBlock *field_blocks;
    uint8_t *default_fields;
    NvmShapeGraph shapes;
    NvmShapeId *shape_locals, *shape_results, *shape_globals, **shape_outputs;
    NvmShapeId *shape_current;
    Nvm2cJoinShape **join_shapes;
    int shape_generic_array;
    int track_shapes;
    uint8_t shape_opcode;
    uint32_t classify_function_index;
    size_t classify_offset;
} Nvm2cBuf;

static void nvm2c_fail(Nvm2cBuf *b, const char *fmt, ...) {
    if (b->failed) return;
    b->failed = 1;
    if (!b->err || b->err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b->err, b->err_len, fmt, ap);
    va_end(ap);
}

static int nvm2c_grow(Nvm2cBuf *b, size_t need) {
    if (b->failed) return 0;
    if (b->len == SIZE_MAX || need > SIZE_MAX - b->len - 1) {
        nvm2c_fail(b, "I cannot represent this output size");
        return 0;
    }
    if (b->len + need + 1 <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + need + 1) {
        if (cap > (size_t)-1 / 2) {
            nvm2c_fail(b, "output too large");
            return 0;
        }
        cap *= 2;
    }
    char *n = realloc(b->data, cap);
    if (!n) {
        nvm2c_fail(b, "out of memory");
        return 0;
    }
    b->data = n;
    b->cap = cap;
    return 1;
}

static void nvm2c_puts(Nvm2cBuf *b, const char *s) {
    size_t n = strlen(s);
    if (!nvm2c_grow(b, n)) return;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void nvm2c_printf(Nvm2cBuf *b, const char *fmt, ...) {
    if (b->failed) return;
    va_list ap;
    va_start(ap, fmt);
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(aq);
        nvm2c_fail(b, "format error");
        return;
    }
    if (!nvm2c_grow(b, (size_t)n)) {
        va_end(aq);
        return;
    }
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, aq);
    va_end(aq);
    b->len += (size_t)n;
}

static int ident_ok(const char *s) {
    if (!s || !*s) return 0;
    unsigned char c = (unsigned char)*s;
    if (!(nl_ascii_isalpha(c) || c == '_')) return 0;
    for (s++; *s; s++) {
        c = (unsigned char)*s;
        if (!(nl_ascii_isalnum(c) || c == '_')) return 0;
    }
    return 1;
}

static void fn_c_name(const NvmModule *mod, uint32_t idx, char *out, size_t n) {
    const char *name = NULL;
    if (idx < mod->function_count) {
        uint32_t ni = mod->functions[idx].name_idx;
        if (ni < mod->string_count) name = mod->strings[ni];
    }
    if (ident_ok(name)) {
        snprintf(out, n, "nl_%s", name);
    } else {
        snprintf(out, n, "nl_fn_%u", idx);
    }
}

static int aggregate_value_tag(uint8_t tag) {
    return tag == TAG_STRUCT || tag == TAG_UNION || tag == TAG_TUPLE;
}

static uint8_t aggregate_kind_for_tag(uint8_t tag) {
    return tag == TAG_STRUCT ? AGG_RECORD :
           tag == TAG_UNION ? AGG_VARIANT : AGG_TUPLE;
}

static const char *c_result_type(const Nvm2cBuf *b, const NvmFunctionEntry *fn, uint32_t idx) {
    if (fn->result_count == 0 || fn->result_tag == TAG_VOID) return "void";
    if (fn->result_count != 1) return NULL;
    if (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL) return "int64_t";
    if (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM) return "nmap_value";
    if (fn->result_tag == TAG_FLOAT) return "double";
    if (fn->result_tag == TAG_STRING) return "const char *";
    if (fn->result_tag == TAG_ARRAY) {
        if (word_array_storage(b->array_results[idx])) return "narr_t";
        if (b->array_results[idx] == NVM2C_VK_SARR) return "nsarr_t";
        if (b->array_results[idx] == NVM2C_VK_RARR) return "nrarr_t";
        return NULL;
    }
    if (fn->result_tag == TAG_HASHMAP) return "nmap_t";
    if (aggregate_value_tag(fn->result_tag)) return "nrec_t";
    return NULL;
}

static int result_is_i64(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 &&
           (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL);
}

static const char *c_local_type(uint8_t kind) {
    if (kind == NVM2C_VK_FLOAT) return "double";
    if (kind == NVM2C_VK_STR) return "const char *";
    if (word_array_storage(kind)) return "narr_t";
    if (kind == NVM2C_VK_SARR) return "nsarr_t";
    if (kind == NVM2C_VK_REC) return "nrec_t";
    if (kind == NVM2C_VK_RARR) return "nrarr_t";
    if (kind == NVM2C_VK_MAP) return "nmap_t";
    if (kind == NVM2C_VK_VALUE) return "nmap_value";
    return "int64_t";
}

static uint8_t fn_local_kind(const Nvm2cBuf *b, const uint8_t *kinds, uint32_t fn, uint16_t slot) {
    return kinds[(size_t)fn * b->local_width + slot];
}

typedef struct {
    const char *name;
    const char *c_name;
    uint8_t argc, parameter, result;
} Nvm2cHost;

/* I recognize my builtin namespace, not arbitrary libraries exporting a name. */
static const Nvm2cHost host_adapters[] = {
    {"strlen", "nhost_strlen", 1, TAG_STRING, TAG_INT},
    {"atan", "atan", 1, TAG_FLOAT, TAG_FLOAT},
    {"vm_getcwd", "nhost_getcwd", 0, TAG_VOID, TAG_STRING},
    {"vm_getenv", "nhost_getenv", 1, TAG_STRING, TAG_STRING},
    {"nl_os_getenv", "nhost_getenv", 1, TAG_STRING, TAG_STRING},
    {"vm_tmp_dir", "nhost_tmp_dir", 0, TAG_VOID, TAG_STRING},
    {"get_argc", "nhost_argc", 0, TAG_VOID, TAG_INT},
    {"get_argv", "nhost_argv", 1, TAG_INT, TAG_STRING},
    {"file_read", "nhost_file_read", 1, TAG_STRING, TAG_STRING},
    {"vm_file_read", "nhost_file_read", 1, TAG_STRING, TAG_STRING},
    {"nl_os_file_read", "nhost_file_read", 1, TAG_STRING, TAG_STRING},
    {"file_write", "nhost_file_write", 2, TAG_STRING, TAG_INT},
    {"vm_file_write", "nhost_file_write", 2, TAG_STRING, TAG_INT},
    {"nl_os_file_write", "nhost_file_write", 2, TAG_STRING, TAG_INT},
    {"file_exists", "nhost_file_exists", 1, TAG_STRING, TAG_BOOL},
    {"vm_file_exists", "nhost_file_exists", 1, TAG_STRING, TAG_BOOL},
    {"nl_os_file_exists", "nhost_file_exists", 1, TAG_STRING, TAG_BOOL},
    {"dir_exists", "nhost_dir_exists", 1, TAG_STRING, TAG_BOOL},
    {"vm_dir_exists", "nhost_dir_exists", 1, TAG_STRING, TAG_BOOL},
    {"nl_os_dir_exists", "nhost_dir_exists", 1, TAG_STRING, TAG_BOOL},
    {"file_delete", "nhost_remove", 1, TAG_STRING, TAG_INT},
    {"file_remove", "nhost_remove", 1, TAG_STRING, TAG_INT},
    {"nl_os_file_delete", "nhost_remove", 1, TAG_STRING, TAG_INT},
    {"nl_os_file_remove", "nhost_remove", 1, TAG_STRING, TAG_INT},
    {"file_rename", "nhost_rename", 2, TAG_STRING, TAG_INT},
    {"nl_os_file_rename", "nhost_rename", 2, TAG_STRING, TAG_INT},
    {"file_compare_identity", "nhost_identity", 2, TAG_STRING, TAG_INT},
    {"file_compare_destinations", "nhost_destinations", 2, TAG_STRING, TAG_INT},
    {"path_normalize", "nhost_normalize", 1, TAG_STRING, TAG_STRING},
    {"nl_os_path_normalize", "nhost_normalize", 1, TAG_STRING, TAG_STRING},
    {"nl_exec_shell", "nhost_shell", 1, TAG_STRING, TAG_INT},
    {"nl_exec_capture", "nhost_capture", 1, TAG_STRING, TAG_STRING},
    {"vm_is_digit", "nhost_is_digit", 1, TAG_INT, TAG_BOOL},
    {"vm_is_alpha", "nhost_is_alpha", 1, TAG_INT, TAG_BOOL},
    {"vm_is_alnum", "nhost_is_alnum", 1, TAG_INT, TAG_BOOL},
    {"vm_is_space", "nhost_is_space", 1, TAG_INT, TAG_BOOL},
    {"vm_is_upper", "nhost_is_upper", 1, TAG_INT, TAG_BOOL},
    {"vm_is_lower", "nhost_is_lower", 1, TAG_INT, TAG_BOOL},
    {"vm_is_whitespace", "nhost_is_whitespace", 1, TAG_INT, TAG_BOOL},
    {"vm_digit_value", "nhost_digit_value", 1, TAG_INT, TAG_INT},
    {"vm_string_from_char", "nhost_from_char", 1, TAG_INT, TAG_STRING},
    {"string_from_char", "nhost_from_char", 1, TAG_INT, TAG_STRING},
    {"vm_mktemp_dir", "nhost_mktemp_dir", 1, TAG_STRING, TAG_STRING},
};

/* These native contracts have homogeneous string parameters. I do not infer
 * an arbitrary artifact's ABI from its coarse NanoISA return tag. String
 * results must be independent of argument storage, or copied by nhost_snapshot;
 * I do not admit general interior-pointer/borrowed-input result contracts. The
 * name/signature check trusts the artifact to implement this lifetime contract. */
static const Nvm2cHost artifact_adapters[] = {
    /* I snapshot the facade's transient borrowed strings before its next call. */
    {"nlc_module_artifact", "nhost_snapshot", 1, TAG_STRING, TAG_STRING},
    {"nl_nanoisa_load_print", "nhost_snapshot", 1, TAG_STRING, TAG_STRING},
    {"nl_nanoisa_load_pretty", "nhost_snapshot", 1, TAG_STRING, TAG_STRING},
    {"nl_nanoisa_last_error", "nhost_snapshot", 0, TAG_VOID, TAG_STRING},
    {"nl_nanoisa_assemble_save", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"nl_nanoisa_assemble_text_save", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"fs_walkdir", "nhost_walk", 1, TAG_STRING, TAG_ARRAY},
    {"nl_fs_list_files", "nhost_walk", 2, TAG_STRING, TAG_ARRAY},
    {"nl_fs_list_files_ci", "nhost_walk", 2, TAG_STRING, TAG_ARRAY},
    {"nl_fs_list_dirs", "nhost_walk", 1, TAG_STRING, TAG_ARRAY},
    {"nl_fs_parent_dir", "nhost_snapshot", 1, TAG_STRING, TAG_STRING},
    {"nl_fs_join_path", "nhost_snapshot", 2, TAG_STRING, TAG_STRING},
    {"nl_fs_is_directory", "nhost_artifact", 1, TAG_STRING, TAG_INT},
    {"nl_fs_file_exists", "nhost_artifact", 1, TAG_STRING, TAG_INT},
    {"nl_fs_file_size", "nhost_artifact", 1, TAG_STRING, TAG_INT},
    {"path_normalize", "nhost_artifact", 1, TAG_STRING, TAG_STRING},
    {"path_canonical", "nhost_artifact", 1, TAG_STRING, TAG_STRING},
    {"path_join", "nhost_artifact", 2, TAG_STRING, TAG_STRING},
    {"path_basename", "nhost_artifact", 1, TAG_STRING, TAG_STRING},
    {"path_dirname", "nhost_artifact", 1, TAG_STRING, TAG_STRING},
    {"path_relpath", "nhost_artifact", 2, TAG_STRING, TAG_STRING},
    {"file_read", "nhost_artifact", 1, TAG_STRING, TAG_STRING},
    {"file_write", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"file_append", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"file_exists", "nhost_artifact", 1, TAG_STRING, TAG_BOOL},
    {"file_delete", "nhost_artifact", 1, TAG_STRING, TAG_INT},
    {"fs_mkdir_p", "nhost_artifact", 1, TAG_STRING, TAG_INT},
    {"file_copy", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"dir_copy", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"file_compare_identity", "nhost_artifact", 2, TAG_STRING, TAG_INT},
    {"file_compare_destinations", "nhost_artifact", 2, TAG_STRING, TAG_INT},
};

static bool scalar_artifact_adapter(const Nvm2cHost *host) {
    return host && (!strcmp(host->c_name, "nhost_artifact") ||
                    !strcmp(host->c_name, "nhost_snapshot"));
}

static const Nvm2cHost *import_host(const NvmModule *mod, uint32_t index) {
    if (index >= mod->import_count || !mod->imports) return NULL;
    const NvmImportEntry *imp = &mod->imports[index];
    const char *module = nvm_get_string(mod, imp->module_name_idx);
    const char *name = nvm_get_string(mod, imp->function_name_idx);
    if (module && name && mod->string_lengths &&
        imp->kind == NVM_IMPORT_ARTIFACT && module[0] == '/' &&
        mod->string_lengths[imp->module_name_idx] == strlen(module) &&
        mod->string_lengths[imp->function_name_idx] == strlen(name)) {
        for (size_t i = 0; i < sizeof artifact_adapters / sizeof artifact_adapters[0]; ++i) {
            const Nvm2cHost *host = &artifact_adapters[i];
            if (strcmp(name, host->name) || imp->param_count != host->argc ||
                imp->return_type != host->result || (host->argc &&
                (!mod->import_param_types || !mod->import_param_types[index]))) continue;
            for (uint8_t p = 0; p < host->argc; ++p)
                if (mod->import_param_types[index][p] != host->parameter) return NULL;
            return host;
        }
    }
    if (!module || module[0] || !name || imp->kind != NVM_IMPORT_FFI ||
        !mod->string_lengths || mod->string_lengths[imp->module_name_idx] != 0 ||
        mod->string_lengths[imp->function_name_idx] != strlen(name)) return NULL;
    for (size_t i = 0; i < sizeof host_adapters / sizeof host_adapters[0]; ++i) {
        const Nvm2cHost *host = &host_adapters[i];
        if (strcmp(name, host->name) != 0 || imp->param_count != host->argc ||
            imp->return_type != host->result) continue;
        if (host->argc && (!mod->import_param_types || !mod->import_param_types[index])) return NULL;
        for (uint8_t p = 0; p < host->argc; ++p)
            if (mod->import_param_types[index][p] != host->parameter) return NULL;
        return host;
    }
    return NULL;
}

static void emit_c_string_lit(Nvm2cBuf *b, const char *s, uint32_t len) {
    uint32_t i;
    for (i = 0; i < len; i++) {
        if (s[i] == '\0') {
            nvm2c_fail(b, "PUSH_STR: embedded NUL is not in the nvm2c subset");
            return;
        }
    }
    nvm2c_puts(b, "\"");
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\' || c == '"' || c == '?') {
            nvm2c_printf(b, "\\%c", (char)c);
        } else if (c == '\n') {
            nvm2c_puts(b, "\\n");
        } else if (c == '\t') {
            nvm2c_puts(b, "\\t");
        } else if (c == '\r') {
            nvm2c_puts(b, "\\r");
        } else if (c >= 32 && c < 127) {
            char tmp[2] = {(char)c, 0};
            nvm2c_puts(b, tmp);
        } else {
            /* UTF-8 payload bytes and remaining controls. Three-digit octal
             * so a following hex/octal digit cannot extend the escape. */
            nvm2c_printf(b, "\\%03o", (unsigned)c);
        }
    }
    nvm2c_puts(b, "\"");
}

typedef struct {
    uint8_t kind;
    int origin;
    uint8_t *rec_k;
    NvmShapeId shape;
    uint16_t scalar_tags;
} Nvm2cSimSlot;

typedef struct {
    uint8_t *parameters;
    uint8_t *fields;
    uint8_t *results;
    uint8_t *global_kinds;
    uint8_t *global_fields;
    uint8_t *global_stored;
    int changed;
    int discover_globals;
    int final;
} Nvm2cFacts;

enum {
    NVM2C_GLOBAL_UNSTORED,
    NVM2C_GLOBAL_EXACT_CANDIDATE,
    NVM2C_GLOBAL_DYNAMIC
};

/* Classifier field vectors are immutable after publication. Branch merges
 * clone before changing them; one function-scoped arena owns every vector. */
static uint8_t *sim_fields(Nvm2cBuf *b, const uint8_t *source, uint8_t fill) {
    Nvm2cFieldBlock *block = malloc(sizeof *block + b->record_width);
    if (!block) {
        nvm2c_fail(b, "I cannot allocate aggregate field facts");
        return NULL;
    }
    block->next = b->field_blocks;
    b->field_blocks = block;
    if (source) memcpy(block->data, source, b->record_width);
    else memset(block->data, fill, b->record_width);
    return block->data;
}

static int merge_fact(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, uint8_t kind) {
    if (kind == NVM2C_VK_UNK || *dest == kind) return 1;
    if (*dest != NVM2C_VK_UNK) {
        char target[96];
        if (dest < facts->fields) {
            size_t offset = (size_t)(dest - facts->parameters);
            snprintf(target, sizeof target, "parameter %zu of function %zu",
                     offset % b->local_width, offset / b->local_width);
        } else if (dest < facts->results) {
            size_t offset = (size_t)(dest - facts->fields);
            size_t parameter = offset / b->record_width;
            snprintf(target, sizeof target, "field %zu of parameter %zu of function %zu",
                     offset % b->record_width, parameter % b->local_width,
                     parameter / b->local_width);
        } else if (dest >= b->array_results) {
            snprintf(target, sizeof target, "array result of function %zu", (size_t)(dest - b->array_results));
        } else {
            size_t offset = (size_t)(dest - facts->results);
            snprintf(target, sizeof target, "result field %zu of function %zu",
                     offset % b->record_width, offset / b->record_width);
        }
        nvm2c_fail(b, "I cannot assign conflicting kinds to a function parameter or aggregate field "
                      "(function %u, offset %zu: %s versus %s at %s)",
                   b->classify_function_index, b->classify_offset,
                   *dest == NVM2C_VK_BOOL ? "bool" : c_local_type(*dest),
                   kind == NVM2C_VK_BOOL ? "bool" : c_local_type(kind), target);
        return 0;
    }
    *dest = kind;
    facts->changed = 1;
    return 1;
}

static int merge_fields(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, const uint8_t *fields) {
    for (size_t i = 0; i < b->record_width; i++) {
        if (!merge_fact(b, facts, &dest[i], fields[i])) return 0;
    }
    return 1;
}

/* Callers opt into present-scalar to optional storage widening. Ordinary
 * aggregate field merging stays exact; the graph checks optional payloads. */
static int merge_parameter(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, uint8_t kind) {
    if (*dest == NVM2C_VK_VALUE && (kind == NVM2C_VK_STR || kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL || kind == NVM2C_VK_FLOAT)) return 1;
    if ((*dest == NVM2C_VK_STR || *dest == NVM2C_VK_INT || *dest == NVM2C_VK_BOOL || *dest == NVM2C_VK_FLOAT) && kind == NVM2C_VK_VALUE) {
        *dest = NVM2C_VK_VALUE;
        facts->changed = 1;
        return 1;
    }
    return merge_fact(b, facts, dest, kind);
}

/* I box primitive array and map handles at call boundaries. Record-field
 * storage keeps its separate, exact representation contract. */
static int merge_call_parameter(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, uint8_t kind) {
    if (*dest == NVM2C_VK_VALUE && (word_array_storage(kind) || kind == NVM2C_VK_SARR || kind == NVM2C_VK_MAP)) return 1;
    if ((word_array_storage(*dest) || *dest == NVM2C_VK_SARR || *dest == NVM2C_VK_MAP) && kind == NVM2C_VK_VALUE) {
        *dest = NVM2C_VK_VALUE;
        facts->changed = 1;
        return 1;
    }
    return merge_parameter(b, facts, dest, kind);
}

static int merge_record_results(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, const uint8_t *fields) {
    for (size_t i = 0; i < b->record_width; ++i) {
        if (variant_field(dest[i]) && variant_field(fields[i])) {
            uint8_t joined = dest[i] | fields[i];
            if (joined != dest[i]) { dest[i] = joined; facts->changed = 1; }
        } else if ((variant_field(dest[i]) && fields[i] == NVM2C_VK_VALUE) ||
                   (dest[i] == NVM2C_VK_VALUE && variant_field(fields[i]))) {
            /* Keep the explicit scalar-set destination, but discard projected
             * tag provenance until every optional producer is checked. */
            uint8_t joined = (variant_field(dest[i]) ? dest[i] : fields[i]) |
                             NVM2C_FIELD_VARIANT_OPTIONAL;
            if (dest[i] != joined) { dest[i] = joined; facts->changed = 1; }
        } else if (!merge_parameter(b, facts, &dest[i], fields[i])) return 0;
    }
    return 1;
}

static int merge_global_kind(Nvm2cBuf *b, Nvm2cFacts *facts, uint32_t slot, uint8_t kind) {
    if (kind == NVM2C_VK_UNK || kind == NVM2C_VK_VALUE) return 1;
    uint8_t *dest = &facts->global_kinds[slot];
    if (*dest == kind) return 1;
    if (*dest == NVM2C_VK_UNK) {
        *dest = kind;
        facts->changed = 1;
        return 1;
    }
    if (*dest == NVM2C_VK_RARR || kind == NVM2C_VK_RARR) {
        nvm2c_fail(b, "I require record-array global %u to retain one exact representation "
                      "(function %u, offset %zu)",
                   slot, b->classify_function_index, b->classify_offset);
        return 0;
    }
    if (*dest != NVM2C_VK_VALUE) {
        *dest = NVM2C_VK_VALUE;
        facts->changed = 1;
    }
    return 1;
}

static int merge_global_fields(Nvm2cBuf *b, Nvm2cFacts *facts, uint32_t slot,
                               const uint8_t *fields) {
    uint8_t *dest = facts->global_fields + (size_t)slot * b->record_width;
    for (size_t i = 0; i < b->record_width; ++i) {
        if (fields[i] == NVM2C_VK_UNK || dest[i] == fields[i]) continue;
        if (dest[i] != NVM2C_VK_UNK) {
            nvm2c_fail(b, "I found conflicting field %zu representations in record-array global %u "
                          "(function %u, offset %zu)",
                       i, slot, b->classify_function_index, b->classify_offset);
            return 0;
        }
        dest[i] = fields[i];
        facts->changed = 1;
    }
    return 1;
}

static int shape_ok(Nvm2cBuf *b) {
    if (b->shapes.error) {
        const InstructionInfo *info = isa_get_info(b->shape_opcode);
        nvm2c_fail(b, "I found invalid aggregate shape constraints during %s in function %u at offset %zu: %s",
                   info ? info->name : "classification", b->classify_function_index,
                   b->classify_offset, b->shapes.error);
    }
    return !b->failed;
}

static NvmShapeId shape_variable(Nvm2cBuf *b, NvmShapeId *slot) {
    if (!b->track_shapes) return 0;
    if (!*slot) *slot = nvm_shape_new(&b->shapes, NVM_SHAPE_UNKNOWN);
    shape_ok(b);
    return *slot;
}

static int shape_type(Nvm2cBuf *b, NvmShapeId id, NvmShapeKind kind) {
    if (!b->track_shapes) return 1;
    if (kind == NVM_SHAPE_UNKNOWN) return shape_ok(b);
    if (nvm_shape_kind(&b->shapes, id) != kind) {
        NvmShapeId typed = nvm_shape_new(&b->shapes, kind);
        nvm_shape_unify(&b->shapes, id, typed);
    }
    return shape_ok(b);
}

static int shape_kind(Nvm2cBuf *b, NvmShapeId id, uint8_t kind) {
    if (!b->track_shapes) return 1;
    if (kind == NVM2C_VK_UNK) return shape_ok(b);
    if (kind == NVM2C_VK_INT) return shape_type(b, id, NVM_SHAPE_INT);
    if (kind == NVM2C_VK_BOOL) return shape_type(b, id, NVM_SHAPE_BOOL);
    if (kind == NVM2C_VK_FLOAT) return shape_type(b, id, NVM_SHAPE_FLOAT);
    if (kind == NVM2C_VK_STR) return shape_type(b, id, NVM_SHAPE_STRING);
    if (kind == NVM2C_VK_REC) return shape_type(b, id, NVM_SHAPE_RECORD);
    if (kind == NVM2C_VK_MAP) return shape_type(b, id, NVM_SHAPE_MAP);
    if (variant_field(kind)) {
        return shape_type(b, id, NVM_SHAPE_OPTIONAL) &&
            shape_type(b, nvm_shape_child(&b->shapes, id, 0),
                kind & NVM2C_FIELD_VARIANT_INT_ARRAY ? NVM_SHAPE_VARIANT_INT_ARRAY : NVM_SHAPE_VARIANT_SCALAR);
    }
    if (kind == NVM2C_VK_VALUE) return shape_type(b, id, NVM_SHAPE_OPTIONAL);
    if (!shape_type(b, id, NVM_SHAPE_ARRAY)) return 0;
    if (b->shape_generic_array) return 1;
    NvmShapeId element = nvm_shape_child(&b->shapes, id, 0);
    return shape_type(b, element, kind == NVM2C_VK_RARR ? NVM_SHAPE_RECORD :
                                kind == NVM2C_VK_SARR ? NVM_SHAPE_STRING :
                                kind == NVM2C_VK_FARR ? NVM_SHAPE_FLOAT :
                                kind == NVM2C_VK_BARR ? NVM_SHAPE_BOOL : NVM_SHAPE_INT);
}

static int shape_equal(Nvm2cBuf *b, NvmShapeId a, NvmShapeId c) {
    if (!b->track_shapes) return 1;
    nvm_shape_unify(&b->shapes, a, c);
    return shape_ok(b);
}

static NvmShapeId shape_child(Nvm2cBuf *b, NvmShapeId parent, uint32_t index) {
    if (!b->track_shapes) return 0;
    NvmShapeId child = nvm_shape_child(&b->shapes, parent, index);
    shape_ok(b);
    return child;
}

/* I give proved numeric or constructor-derived finite storage its own payload set. */
static int shape_carrier_box(Nvm2cBuf *b, NvmShapeId id, uint16_t tags) {
    if (!b->track_shapes) return 1;
    return shape_type(b, id, NVM_SHAPE_OPTIONAL) &&
           shape_type(b, shape_child(b, id, 0), variant_payload_tags(tags) ?
               ((tags & (1u << TAG_ARRAY)) ? NVM_SHAPE_VARIANT_INT_ARRAY : NVM_SHAPE_VARIANT_SCALAR) : NVM_SHAPE_NUMERIC);
}

/* A flat scalar fact can omit a nested caller's optional representation.
 * I seed inferred scalar storage with a directed flow, not an exact type.
 * Constructors and native array/map payload constraints remain exact. */
static int shape_field_kind(Nvm2cBuf *b, NvmShapeId id, uint8_t kind) {
    if (!b->track_shapes) return 1;
    if (kind != NVM2C_VK_STR && kind != NVM2C_VK_INT && kind != NVM2C_VK_BOOL && kind != NVM2C_VK_FLOAT)
        return shape_kind(b, id, kind);
    NvmShapeId source = nvm_shape_new(&b->shapes, kind == NVM2C_VK_STR ? NVM_SHAPE_STRING :
                                     kind == NVM2C_VK_BOOL ? NVM_SHAPE_BOOL :
                                     kind == NVM2C_VK_FLOAT ? NVM_SHAPE_FLOAT : NVM_SHAPE_INT);
    return source && nvm_shape_convert(&b->shapes, source, id) && shape_ok(b);
}

/* Returning a present scalar into optional record storage is a conversion,
 * not equality between the source scalar and an optional shape. */
static int shape_record_return(Nvm2cBuf *b, NvmShapeId source, NvmShapeId result,
                               const uint8_t *source_fields, const uint8_t *result_fields) {
    if (!b->track_shapes) return 1;
    if (!shape_type(b, source, NVM_SHAPE_RECORD) || !shape_type(b, result, NVM_SHAPE_RECORD)) return 0;
    for (size_t i = 0; i < b->record_width; ++i) {
        if (source_fields[i] == NVM2C_VK_UNK && result_fields[i] == NVM2C_VK_UNK) continue;
        NvmShapeId from = shape_child(b, source, (uint32_t)i);
        NvmShapeId to = shape_child(b, result, (uint32_t)i);
        if (!shape_field_kind(b, from, source_fields[i]) || !shape_field_kind(b, to, result_fields[i])) return 0;
        /* A present boxed field is not absent variant padding. Materialize
         * its payload obligation even when the producer has no known tag. */
        if (source_fields[i] == NVM2C_VK_VALUE && variant_field(result_fields[i]) &&
            !shape_child(b, from, 0)) return 0;
    }
    return nvm_shape_convert(&b->shapes, source, result) && shape_ok(b);
}

static int sim_push_slot(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                         Nvm2cSimSlot slot) {
    if ((size_t)*sp >= b->sim_stack_capacity) {
        nvm2c_fail(b, "function %u: operand stack overflow", idx);
        return 0;
    }
    if (!slot.shape) slot.shape = shape_variable(b, b->shape_current);
    if (b->shape_opcode == OP_AGG_GET ||
        (b->shape_opcode == OP_LOAD_LOCAL &&
         (slot.kind == NVM2C_VK_STR || slot.kind == NVM2C_VK_INT || slot.kind == NVM2C_VK_BOOL || slot.kind == NVM2C_VK_FLOAT))) {
        if (!shape_field_kind(b, slot.shape, slot.kind)) return 0;
    } else if (!shape_kind(b, slot.shape, slot.kind)) return 0;
    if (!slot.scalar_tags) slot.scalar_tags = scalar_kind_tags(slot.kind);
    if (slot.kind == NVM2C_VK_VALUE && boxed_carrier_tags(slot.scalar_tags) &&
        !shape_carrier_box(b, slot.shape, slot.scalar_tags)) return 0;
    stk[*sp] = slot;
    if (!stk[*sp].rec_k) stk[*sp].rec_k = b->default_fields;
    (*sp)++;
    return 1;
}

/* Read only resolved representation facts. An array with no established
 * element kind is not evidence for an integer-array representation. */
static uint8_t resolved_shape_kind(Nvm2cBuf *b, NvmShapeId id) {
    if (!id) return NVM2C_VK_UNK;
    switch (nvm_shape_kind(&b->shapes, id)) {
    case NVM_SHAPE_INT: return NVM2C_VK_INT;
    case NVM_SHAPE_BOOL: return NVM2C_VK_BOOL;
    case NVM_SHAPE_FLOAT: return NVM2C_VK_FLOAT;
    case NVM_SHAPE_STRING: return NVM2C_VK_STR;
    case NVM_SHAPE_RECORD: return NVM2C_VK_REC;
    case NVM_SHAPE_MAP: return NVM2C_VK_MAP;
    case NVM_SHAPE_OPTIONAL: return NVM2C_VK_VALUE;
    case NVM_SHAPE_ARRAY: {
        NvmShapeId element = nvm_shape_lookup(&b->shapes, id, 0);
        if (!element) return NVM2C_VK_UNK;
        switch (nvm_shape_kind(&b->shapes, element)) {
        case NVM_SHAPE_INT: return NVM2C_VK_ARR;
        case NVM_SHAPE_BOOL: return NVM2C_VK_BARR;
        case NVM_SHAPE_FLOAT: return NVM2C_VK_FARR;
        case NVM_SHAPE_STRING: return NVM2C_VK_SARR;
        case NVM_SHAPE_RECORD: return NVM2C_VK_RARR;
        default: return NVM2C_VK_UNK;
        }
    }
    default: return NVM2C_VK_UNK;
    }
}

static int sim_push(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                    uint8_t kind, int origin) {
    if (kind == NVM2C_VK_INT && boolean_result(b->shape_opcode)) kind = NVM2C_VK_BOOL;
    Nvm2cSimSlot slot;
    memset(&slot, 0, sizeof slot);
    slot.kind = kind;
    slot.origin = origin;
    if (b->shape_opcode == OP_PUSH_VOID) slot.scalar_tags = 1u << TAG_VOID;
    if (b->shape_opcode == OP_PUSH_U8) slot.scalar_tags = 1u << TAG_U8;
    if (b->shape_opcode == OP_ENUM_VAL) slot.scalar_tags = 1u << TAG_ENUM;
    return sim_push_slot(b, idx, stk, sp, slot);
}

static int sim_pop(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                   Nvm2cSimSlot *out) {
    if (*sp <= 0) {
        nvm2c_fail(b, "function %u: operand stack underflow", idx);
        return 0;
    }
    *out = stk[--(*sp)];
    return 1;
}

static void mark_origin(uint8_t *local_kind, uint16_t nloc, int origin, uint8_t kind) {
    if (origin >= 0 && (uint16_t)origin < nloc) {
        if (local_kind[origin] != NVM2C_VK_VALUE) local_kind[origin] = kind;
    }
}

static int mark_string_operand(Nvm2cBuf *b, uint8_t *local_kind, uint16_t nloc,
                               Nvm2cSimSlot value) {
    mark_origin(local_kind, nloc, value.origin, NVM2C_VK_STR);
    /* I propagate a consumer's inferred storage back to an unresolved
     * projection. Observed tagged values retain their runtime unboxing. */
    return value.kind != NVM2C_VK_UNK || shape_field_kind(b, value.shape, NVM2C_VK_STR);
}

static int require_typed_integer_operand(Nvm2cBuf *b, uint8_t *local_kind, uint16_t nloc,
                                         Nvm2cSimSlot value) {
    mark_origin(local_kind, nloc, value.origin, NVM2C_VK_INT);
    if (value.kind == NVM2C_VK_UNK)
        return shape_field_kind(b, value.shape, NVM2C_VK_INT);
    if (value.kind == NVM2C_VK_INT || value.kind == NVM2C_VK_VALUE) return 1;
    nvm2c_fail(b, "I require integer operands for typed I64 operations");
    return 0;
}

static const uint8_t *fn_rec_k_const(const Nvm2cBuf *b, const uint8_t *tab, uint32_t fn, uint16_t slot) {
    return tab + ((size_t)fn * b->local_width + slot) * b->record_width;
}

typedef struct {
    Nvm2cSimSlot *slots;
    int sp;
    int set;
} Nvm2cSimJoin;

static int sim_join(Nvm2cBuf *b, uint32_t idx, size_t target, Nvm2cSimJoin *join,
                    const Nvm2cSimSlot *stack, int sp, Nvm2cFacts *facts) {
    Nvm2cScalarJoin *plan = b->scalar_joins;
    while (plan && (plan->function != idx || plan->target != target)) plan = plan->next;
    if (plan && plan->count != sp) {
        nvm2c_fail(b, "I found inconsistent scalar join height in function %u", idx);
        return 0;
    }
    if (!join->set) {
        if (sp) {
            join->slots = malloc((size_t)sp * sizeof *stack);
            if (!join->slots) {
                nvm2c_fail(b, "I cannot allocate classifier branch stack");
                return 0;
            }
            memcpy(join->slots, stack, (size_t)sp * sizeof *stack);
            for (int i = 0; i < sp; ++i) {
                if (plan && plan->tags[i]) {
                    uint16_t tags = plan->tags[i] | stack[i].scalar_tags;
                    if (!optional_scalar_tags(tags) && !boxed_carrier_tags(tags)) {
                        nvm2c_fail(b, "I require proved finite payload provenance at a boxed join");
                        return 0;
                    }
                    join->slots[i].kind = NVM2C_VK_VALUE;
                    join->slots[i].scalar_tags = tags;
                }
                if (!b->track_shapes) continue;
                uint8_t kind = join->slots[i].kind;
                if (kind == NVM2C_VK_REC) {
                    /* A join owns destination storage. Later finite variant
                     * members must not unify their exact producer payloads. */
                    NvmShapeId storage = nvm_shape_new(&b->shapes, NVM_SHAPE_RECORD);
                    if (!storage || !nvm_shape_convert(&b->shapes, stack[i].shape, storage) ||
                        !shape_ok(b)) return 0;
                    join->slots[i].shape = storage;
                    continue;
                }
                if (kind != NVM2C_VK_STR && kind != NVM2C_VK_INT &&
                    kind != NVM2C_VK_BOOL && kind != NVM2C_VK_VALUE) continue;
                /* Destination storage never rewrites the exact producer. */
                NvmShapeId storage = 0;
                storage = shape_variable(b, &storage);
                if (!shape_field_kind(b, storage, kind) ||
                    (kind == NVM2C_VK_VALUE && boxed_carrier_tags(join->slots[i].scalar_tags) &&
                     !shape_carrier_box(b, storage, join->slots[i].scalar_tags)) ||
                    !nvm_shape_convert(&b->shapes, stack[i].shape, storage) || !shape_ok(b)) return 0;
                join->slots[i].shape = storage;
            }
        }
        join->sp = sp;
        join->set = 1;
        return 1;
    }
    if (join->sp != sp) {
        nvm2c_fail(b, "I found incompatible stack heights at a join in function %u", idx);
        return 0;
    }
    for (int i = 0; i < sp; i++) {
        uint16_t tags = join->slots[i].scalar_tags | stack[i].scalar_tags;
        if ((boxed_carrier_tags(join->slots[i].scalar_tags) || boxed_carrier_tags(stack[i].scalar_tags)) &&
            !boxed_carrier_tags(tags)) {
            nvm2c_fail(b, "I cannot join proved finite storage with unproved payload provenance");
            return 0;
        }
        int scalar_join = optional_scalar_tags(tags) || boxed_carrier_tags(tags);
        if (plan && plan->tags[i] && !scalar_join) {
            nvm2c_fail(b, "I require proved finite payload provenance at a boxed join");
            return 0;
        }
        if (scalar_join) {
            if (!plan) {
                if ((size_t)sp > (SIZE_MAX - sizeof *plan) / sizeof *plan->tags) {
                    nvm2c_fail(b, "I cannot represent scalar join facts"); return 0;
                }
                plan = calloc(1, sizeof *plan + (size_t)sp * sizeof *plan->tags);
                if (!plan) { nvm2c_fail(b, "I cannot allocate scalar join facts"); return 0; }
                plan->function = idx; plan->target = target; plan->count = sp;
                plan->next = b->scalar_joins; b->scalar_joins = plan;
            }
            if (plan->tags[i] != tags) {
                if (facts->final) {
                    nvm2c_fail(b, "I require stable scalar join facts before shape emission"); return 0;
                }
                plan->tags[i] = tags; facts->changed = 1;
            }
            join->slots[i].kind = NVM2C_VK_VALUE;
            b->has_maps = 1;
        }
        int string_join = (join->slots[i].kind == NVM2C_VK_STR || join->slots[i].kind == NVM2C_VK_VALUE) &&
                          (stack[i].kind == NVM2C_VK_STR || stack[i].kind == NVM2C_VK_VALUE);
        if (!scalar_join && string_join && join->slots[i].kind != stack[i].kind) {
            if (target <= b->classify_offset) {
                nvm2c_fail(b, "I cannot yet widen tagged string storage on a backward stack edge in function %u", idx);
                return 0;
            }
            join->slots[i].kind = NVM2C_VK_VALUE;
            b->has_maps = 1;
        }
        int record_join = join->slots[i].kind == NVM2C_VK_REC && stack[i].kind == NVM2C_VK_REC;
        if (record_join) {
            uint8_t *fields = sim_fields(b, join->slots[i].rec_k, 0);
            if (!fields) return 0;
            for (size_t f = 0; f < b->record_width; ++f) {
                uint8_t incoming = stack[i].rec_k[f];
                if (fields[f] == NVM2C_VK_UNK) fields[f] = incoming;
                else if (variant_field(fields[f]) && variant_field(incoming)) fields[f] |= incoming;
                else if (incoming != NVM2C_VK_UNK && incoming != fields[f]) {
                    nvm2c_fail(b, "I found incompatible aggregate fields at a join in function %u", idx);
                    return 0;
                }
            }
            if (!shape_record_return(b, stack[i].shape, join->slots[i].shape,
                                     stack[i].rec_k, fields)) return 0;
            join->slots[i].rec_k = fields;
        }
        if (b->track_shapes && !record_join && !((string_join || scalar_join)
                ? nvm_shape_convert(&b->shapes, stack[i].shape, join->slots[i].shape)
                : nvm_shape_unify(&b->shapes, join->slots[i].shape, stack[i].shape))) {
            nvm2c_fail(b, "I found incompatible shapes at a join in function %u: %s", idx, b->shapes.error);
            return 0;
        }
        join->slots[i].scalar_tags = tags;
        int origin = join->slots[i].origin == stack[i].origin ? stack[i].origin : -1;
        if (join->slots[i].kind == NVM2C_VK_UNK) {
            join->slots[i] = stack[i];
            join->slots[i].origin = origin;
            continue;
        }
        join->slots[i].origin = origin;
        if (stack[i].kind == NVM2C_VK_UNK) continue;
        if (join->slots[i].kind != stack[i].kind && !string_join && !scalar_join) {
            nvm2c_fail(b, "I found incompatible stack kinds at a join in function %u after offset %zu (slot %d: %u versus %u)",
                       idx, b->classify_offset, i, join->slots[i].kind, stack[i].kind);
            return 0;
        }
        if (stack[i].kind == NVM2C_VK_REC || stack[i].kind == NVM2C_VK_RARR || stack[i].kind == NVM2C_VK_MAP) {
            join->slots[i].rec_k = sim_fields(b, join->slots[i].rec_k, 0);
            if (!join->slots[i].rec_k) return 0;
            for (size_t field = 0; field < b->record_width; field++) {
                uint8_t incoming = stack[i].rec_k[field];
                uint8_t *current = &join->slots[i].rec_k[field];
                if (*current == NVM2C_VK_UNK) *current = incoming;
                else if (variant_field(*current) && variant_field(incoming)) *current |= incoming;
                else if (incoming != NVM2C_VK_UNK && incoming != *current) {
                    nvm2c_fail(b, "I found incompatible aggregate fields at a join in function %u", idx);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int jump_target(Nvm2cBuf *b, uint32_t idx, size_t start, int32_t rel,
                       size_t remaining, size_t *out);

/* I use tagged storage when a reachable read can precede the first store. */
static int mark_uninitialized_locals(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    size_t WIDTH = ((size_t)fn->local_count + 7) / 8;
    if (!WIDTH) WIDTH = 1;
    size_t count = (size_t)fn->code_length + 1;
    if (count > SIZE_MAX / WIDTH || count > SIZE_MAX / sizeof(size_t)) {
        nvm2c_fail(b, "I cannot allocate local initialization facts"); return 0;
    }
    uint8_t *states = calloc(count, WIDTH), *flags = calloc(count, 1);
    size_t *queue = malloc(count * sizeof *queue);
    if (!states || !flags || !queue) {
        nvm2c_fail(b, "I cannot allocate local initialization facts"); goto done;
    }
    for (uint16_t i = 0; i < fn->arity; ++i) states[i / 8] |= (uint8_t)(1u << (i % 8));
    size_t head = 0, tail = 1 % count, pending = 1;
    queue[0] = 0; flags[0] = 3; /* seen, queued */
    while (pending && !b->failed) {
        size_t pc = queue[head]; head = (head + 1) % count; --pending;
        flags[pc] &= (uint8_t)~2u;
        if (pc == fn->code_length) continue;
        DecodedInstruction ins;
        uint32_t n = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
        if (!n) { nvm2c_fail(b, "I cannot decode local initialization flow"); break; }
        uint8_t out[NVM2C_MAX_LOCALS / 8];
        memcpy(out, states + pc * WIDTH, WIDTH);
        if (ins.opcode == OP_STORE_LOCAL || ins.opcode == OP_LOAD_LOCAL) {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count || slot >= NVM2C_MAX_LOCALS) {
                nvm2c_fail(b, "I found an invalid local during initialization analysis"); break;
            }
            if (ins.opcode == OP_STORE_LOCAL) out[slot / 8] |= (uint8_t)(1u << (slot % 8));
        }
        if (ins.opcode == OP_RET || ins.opcode == OP_HALT || ins.opcode == OP_TAIL_CALL) continue;
        size_t next[2] = {pc + n, 0}, successors = 1;
        if (ins.opcode == OP_JMP || ins.opcode == OP_JMP_FALSE || ins.opcode == OP_JMP_TRUE) {
            size_t target;
            if (!jump_target(b, idx, pc, ins.operands[0].i32, fn->code_length, &target)) break;
            if (ins.opcode == OP_JMP) next[0] = target;
            else next[successors++] = target;
        }
        for (size_t s = 0; s < successors; ++s) {
            size_t at = next[s];
            uint8_t *dest = states + at * WIDTH;
            int changed = !(flags[at] & 1);
            if (changed) memcpy(dest, out, WIDTH);
            else for (size_t j = 0; j < WIDTH; ++j) {
                uint8_t intersection = dest[j] & out[j];
                if (intersection != dest[j]) changed = 1;
                dest[j] = intersection;
            }
            flags[at] |= 1;
            if (changed && !(flags[at] & 2)) {
                queue[tail] = at; tail = (tail + 1) % count; ++pending;
                flags[at] |= 2;
            }
        }
    }
    for (size_t pc = 0; pc < fn->code_length && !b->failed;) {
        DecodedInstruction ins;
        uint32_t n = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
        if (!n) { nvm2c_fail(b, "I cannot inspect local initialization flow"); break; }
        if ((flags[pc] & 1) && ins.opcode == OP_LOAD_LOCAL) {
            uint16_t slot = ins.operands[0].u16;
            if (!(states[pc * WIDTH + slot / 8] & (1u << (slot % 8))))
                b->tagged_locals[(size_t)idx * b->local_width + slot] = 1;
        }
        pc += n;
    }
done:
    free(states); free(flags); free(queue);
    return !b->failed;
}

static int classify_function_body(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                                  uint8_t *local_kind, uint8_t *rec_fields,
                                  Nvm2cSimJoin *joins, const uint8_t *targets,
                                  Nvm2cFacts *facts, Nvm2cSimSlot *stk) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    uint16_t nloc = fn->local_count;
    b->track_shapes = facts->final;
    b->classify_function_index = idx;
    b->classify_offset = 0;
    uint16_t i;
    memset(local_kind, NVM2C_VK_UNK, nloc);
    memset(rec_fields, NVM2C_VK_UNK, (size_t)nloc * b->record_width);
    memcpy(local_kind, facts->parameters + (size_t)idx * b->local_width, fn->arity);
    for (i = 0; i < nloc; ++i) {
        if (b->tagged_locals[(size_t)idx * b->local_width + i]) {
            local_kind[i] = NVM2C_VK_VALUE;
            b->has_maps = 1;
        }
    }
    memcpy(rec_fields, facts->fields + (size_t)idx * b->local_width * b->record_width,
           (size_t)nloc * b->record_width);

    if (fn->code_offset > mod->code_size ||
        fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "function %u: code range is outside the module", idx);
        return 0;
    }

    const uint8_t *code = mod->code + fn->code_offset;
    size_t remaining = fn->code_length;
    int sp = 0;
    size_t pc = 0;
    int terminated = 0;
    int previous_false_push = 0;

    while (pc < remaining) {
        size_t start = pc;
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            return 0;
        }
        pc += n;
        if (terminated && !joins[start].set) {
            previous_false_push = 0;
            continue;
        }
        if (targets[start]) {
            previous_false_push = 0;
            if (!terminated && !sim_join(b, idx, start, &joins[start], stk, sp, facts)) return 0;
            sp = joins[start].sp;
            if (sp) memcpy(stk, joins[start].slots, (size_t)sp * sizeof *stk);
            terminated = 0;
        }

        b->shape_current = facts->final ? &b->shape_outputs[idx][start] : NULL;
        b->shape_opcode = ins.opcode;
        b->classify_offset = start;
        b->shape_generic_array = !(ins.opcode == OP_ARR_LITERAL || ins.opcode == OP_CALL_EXTERN ||
            (ins.opcode == OP_ARR_NEW && ins.operands[0].u8 != TAG_INT));
        switch (ins.opcode) {
        case OP_NOP:
            break;
        case OP_PUSH_I64:
        case OP_PUSH_BOOL:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        case OP_PUSH_U8:
        case OP_ENUM_VAL:
        case OP_PUSH_VOID:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
            break;
        case OP_PUSH_F64:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_FLOAT, -1)) return 0;
            break;
        case OP_PUSH_STR:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        case OP_DUP: {
            if (sp <= 0) {
                nvm2c_fail(b, "function %u: DUP on empty stack", idx);
                return 0;
            }
            if (!sim_push_slot(b, idx, stk, &sp, stk[sp - 1])) return 0;
            break;
        }
        case OP_POP: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            break;
        }
        case OP_PRINT:
        case OP_PRINTLN: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            break;
        }
        case OP_ASSERT: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            if (previous_false_push) terminated = 1;
            break;
        }
        case OP_SWAP: {
            Nvm2cSimSlot x, y;
            if (!sim_pop(b, idx, stk, &sp, &x)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &y)) return 0;
            if (!sim_push_slot(b, idx, stk, &sp, x)) return 0;
            if (!sim_push_slot(b, idx, stk, &sp, y)) return 0;
            break;
        }
        case OP_ROT3: {
            Nvm2cSimSlot top, middle, bottom;
            if (!sim_pop(b, idx, stk, &sp, &top) ||
                !sim_pop(b, idx, stk, &sp, &middle) ||
                !sim_pop(b, idx, stk, &sp, &bottom)) return 0;
            if ((top.kind != NVM2C_VK_INT && top.kind != NVM2C_VK_BOOL) ||
                (middle.kind != NVM2C_VK_INT && middle.kind != NVM2C_VK_BOOL) ||
                (bottom.kind != NVM2C_VK_INT && bottom.kind != NVM2C_VK_BOOL)) {
                nvm2c_fail(b, "I require exact int/bool operands for native ROT3");
                return 0;
            }
            if (!sim_push_slot(b, idx, stk, &sp, top) ||
                !sim_push_slot(b, idx, stk, &sp, bottom) ||
                !sim_push_slot(b, idx, stk, &sp, middle)) return 0;
            break;
        }
        case OP_LOAD_GLOBAL: {
            uint32_t slot = ins.operands[0].u32;
            if (facts->global_kinds[slot] == NVM2C_VK_RARR) {
                Nvm2cSimSlot loaded = {0};
                loaded.kind = NVM2C_VK_RARR;
                loaded.origin = -1;
                loaded.shape = shape_variable(b, &b->shape_globals[slot]);
                loaded.rec_k = sim_fields(b,
                    facts->global_fields + (size_t)slot * b->record_width, NVM2C_VK_UNK);
                if (!loaded.rec_k || !sim_push_slot(b, idx, stk, &sp, loaded)) return 0;
            } else if (facts->final &&
                       facts->global_stored[slot] == NVM2C_GLOBAL_EXACT_CANDIDATE &&
                       facts->global_kinds[slot] == NVM2C_VK_UNK) {
                /* A nested projection can reveal an exact record-array only
                 * after graph construction. Do not publish the tagged global
                 * fallback into its consumers before that shape closes. */
                Nvm2cSimSlot loaded = {0};
                loaded.kind = NVM2C_VK_UNK;
                loaded.origin = -1;
                loaded.shape = shape_variable(b, &b->shape_globals[slot]);
                loaded.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!loaded.rec_k || !sim_push_slot(b, idx, stk, &sp, loaded)) return 0;
            } else if (!sim_push(b, idx, stk, &sp,
                                 facts->discover_globals ? NVM2C_VK_UNK : NVM2C_VK_VALUE, -1)) return 0;
            break;
        }
        case OP_STORE_GLOBAL: {
            Nvm2cSimSlot value;
            uint32_t slot = ins.operands[0].u32;
            if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
            uint8_t stored = value.kind == NVM2C_VK_VALUE
                ? NVM2C_GLOBAL_DYNAMIC : NVM2C_GLOBAL_EXACT_CANDIDATE;
            if (facts->global_stored[slot] < stored) {
                facts->global_stored[slot] = stored;
                facts->changed = 1;
            }
            if (!merge_global_kind(b, facts, slot, value.kind)) return 0;
            if (value.kind == NVM2C_VK_RARR) {
                if (!merge_global_fields(b, facts, slot, value.rec_k)) return 0;
                NvmShapeId global = shape_variable(b, &b->shape_globals[slot]);
                if (!shape_type(b, global, NVM_SHAPE_ARRAY) ||
                    !shape_equal(b, shape_child(b, value.shape, 0),
                                 shape_child(b, global, 0))) return 0;
            } else if (facts->final && value.kind == NVM2C_VK_UNK &&
                       facts->global_stored[slot] == NVM2C_GLOBAL_EXACT_CANDIDATE &&
                       facts->global_kinds[slot] == NVM2C_VK_UNK) {
                if (!shape_equal(b, value.shape,
                                 shape_variable(b, &b->shape_globals[slot]))) return 0;
            } else if (facts->global_kinds[slot] == NVM2C_VK_RARR &&
                       value.kind != NVM2C_VK_UNK) {
                nvm2c_fail(b, "I require record-array global %u to retain its exact representation "
                              "(function %u, offset %zu)", slot, idx, start);
                return 0;
            }
            /* I collect final-pass graph facts before resolving nested fields.
             * Emission still requires supported concrete or tagged storage. */
            if (value.kind != NVM2C_VK_INT && value.kind != NVM2C_VK_BOOL &&
                value.kind != NVM2C_VK_STR && value.kind != NVM2C_VK_FLOAT && value.kind != NVM2C_VK_VALUE &&
                !word_array_storage(value.kind) && value.kind != NVM2C_VK_SARR &&
                value.kind != NVM2C_VK_RARR && value.kind != NVM2C_VK_MAP &&
                value.kind != NVM2C_VK_UNK) {
                nvm2c_fail(b, "I cannot yet store an aggregate or unresolved global in function %u at offset %zu", idx, start);
                return 0;
            }
            break;
        }
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            Nvm2cSimSlot loaded;
            if (slot >= nloc) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                return 0;
            }
            memset(&loaded, 0, sizeof loaded);
            loaded.kind = local_kind[slot];
            if (loaded.kind == NVM2C_VK_VALUE)
                loaded.scalar_tags = b->local_scalar_tags[(size_t)idx * b->local_width + slot];
            loaded.origin = (int)slot;
            loaded.shape = shape_variable(b, &b->shape_locals[(size_t)idx * b->local_width + slot]);
            if (loaded.kind == NVM2C_VK_REC || loaded.kind == NVM2C_VK_RARR || loaded.kind == NVM2C_VK_MAP) {
                loaded.rec_k = sim_fields(b, rec_fields + (size_t)slot * b->record_width, 0);
                if (!loaded.rec_k) return 0;
            }
            if (!sim_push_slot(b, idx, stk, &sp, loaded)) return 0;
            break;
        }
        case OP_STORE_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            Nvm2cSimSlot v;
            if (slot >= nloc) {
                nvm2c_fail(b, "function %u: STORE_LOCAL %u out of range", idx, slot);
                return 0;
            }
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            size_t local_at = (size_t)idx * b->local_width + slot;
            uint16_t tags = b->local_scalar_tags[local_at] |
                (v.kind == NVM2C_VK_UNK ? 0 : v.scalar_tags);
            if (v.kind != NVM2C_VK_UNK &&
                (boxed_carrier_tags(b->local_scalar_tags[local_at]) || boxed_carrier_tags(v.scalar_tags)) &&
                !boxed_carrier_tags(tags)) {
                nvm2c_fail(b, "I cannot mix numeric local storage with unproved scalar or heap provenance");
                return 0;
            }
            if (tags != b->local_scalar_tags[local_at]) {
                b->local_scalar_tags[local_at] = tags; facts->changed = 1;
            }
            if ((v.kind == NVM2C_VK_VALUE || boxed_carrier_tags(tags)) && !b->tagged_locals[local_at]) {
                /* I retain tagged storage across every assignment and path,
                 * including earlier scalar writes revisited during inference. */
                b->tagged_locals[local_at] = 1;
                facts->changed = 1;
                if (slot < fn->arity) facts->parameters[local_at] = NVM2C_VK_VALUE;
            }
            NvmShapeId destination = shape_variable(b, &b->shape_locals[local_at]);
            if (b->tagged_locals[(size_t)idx * b->local_width + slot]) {
                if (!shape_type(b, destination, NVM_SHAPE_OPTIONAL)) return 0;
                if (boxed_carrier_tags(tags) && !shape_carrier_box(b, destination, tags)) return 0;
                if (b->track_shapes && !nvm_shape_convert(&b->shapes, v.shape, destination)) return 0;
                local_kind[slot] = NVM2C_VK_VALUE;
                break;
            }
            if (v.kind == NVM2C_VK_REC) {
                /* A local joins incoming record storage; it does not redefine
                 * the producer's exact field representation. */
                if (!shape_type(b, destination, NVM_SHAPE_RECORD)) return 0;
                if (b->track_shapes && !nvm_shape_convert(&b->shapes, v.shape, destination)) return 0;
            } else if (v.kind == NVM2C_VK_INT || v.kind == NVM2C_VK_BOOL || v.kind == NVM2C_VK_STR || v.kind == NVM2C_VK_FLOAT) {
                /* Local storage can later receive an optional projection.
                 * It must not equate that projection to an earlier literal. */
                if (b->track_shapes && !nvm_shape_convert(&b->shapes, v.shape, destination)) return 0;
            } else if (!shape_equal(b, v.shape, destination)) return 0;
            if (v.kind == NVM2C_VK_BOOL) {
                local_kind[slot] = NVM2C_VK_BOOL;
            } else if (v.kind == NVM2C_VK_VALUE) {
                local_kind[slot] = NVM2C_VK_VALUE;
            } else if (v.kind == NVM2C_VK_STR) {
                local_kind[slot] = NVM2C_VK_STR;
            } else if (v.kind == NVM2C_VK_FLOAT) {
                local_kind[slot] = NVM2C_VK_FLOAT;
            } else if (word_array_storage(v.kind)) {
                local_kind[slot] = v.kind;
            } else if (v.kind == NVM2C_VK_SARR) {
                local_kind[slot] = NVM2C_VK_SARR;
            } else if (v.kind == NVM2C_VK_REC) {
                local_kind[slot] = NVM2C_VK_REC;
                /* I retain every incoming record field across branches and
                 * inference passes, not just the last textual assignment. */
                uint8_t *fields = facts->fields +
                    ((size_t)idx * b->local_width + slot) * b->record_width;
                if (!merge_record_results(b, facts, fields, v.rec_k)) return 0;
                memcpy(rec_fields + (size_t)slot * b->record_width,
                       fields, b->record_width);
            } else if (v.kind == NVM2C_VK_RARR || v.kind == NVM2C_VK_MAP) {
                local_kind[slot] = v.kind;
                memcpy(rec_fields + (size_t)slot * b->record_width,
                       v.rec_k, b->record_width);
            } else if (v.kind == NVM2C_VK_INT && local_kind[slot] != NVM2C_VK_STR
                       && !word_array_storage(local_kind[slot])
                        && local_kind[slot] != NVM2C_VK_SARR
                        && local_kind[slot] != NVM2C_VK_REC
                        && local_kind[slot] != NVM2C_VK_RARR
                        && local_kind[slot] != NVM2C_VK_MAP
                        && local_kind[slot] != NVM2C_VK_VALUE
                        && local_kind[slot] != NVM2C_VK_BOOL) {
                local_kind[slot] = NVM2C_VK_INT;
            }
            break;
        }
        case OP_F64_FROM_BITS: case OP_F64_TO_BITS: {
            Nvm2cSimSlot value;
            if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
            uint8_t expected = ins.opcode == OP_F64_FROM_BITS ? NVM2C_VK_INT : NVM2C_VK_FLOAT;
            if (value.kind == NVM2C_VK_UNK) {
                mark_origin(local_kind, nloc, value.origin, expected);
                if (!shape_field_kind(b, value.shape, expected)) return 0;
            } else if (value.kind != expected && value.kind != NVM2C_VK_VALUE) {
                nvm2c_fail(b, "I require the exact input kind for binary64 bit transport"); return 0;
            }
            if (!sim_push(b, idx, stk, &sp,
                          expected == NVM2C_VK_INT ? NVM2C_VK_FLOAT : NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
        case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE:
        case OP_F64_GT: case OP_F64_GE: case OP_F64_NEG: {
            unsigned count = ins.opcode == OP_F64_NEG ? 1 : 2;
            for (unsigned operand = 0; operand < count; ++operand) {
                Nvm2cSimSlot value;
                if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
                if (value.kind == NVM2C_VK_UNK) {
                    mark_origin(local_kind, nloc, value.origin, NVM2C_VK_FLOAT);
                    if (!shape_field_kind(b, value.shape, NVM2C_VK_FLOAT)) return 0;
                } else if (value.kind != NVM2C_VK_FLOAT && value.kind != NVM2C_VK_VALUE) {
                    nvm2c_fail(b, "I require float operands for typed F64 operations"); return 0;
                }
            }
            if (!sim_push(b, idx, stk, &sp,
                          boolean_result(ins.opcode) ? NVM2C_VK_BOOL : NVM2C_VK_FLOAT, -1)) return 0;
            break;
        }
        case OP_I64_ADD_CARRY: case OP_I64_SUB_BORROW:
        case OP_I64_MUL_WIDE_S: case OP_I64_MUL_WIDE_U: {
            unsigned count = ins.opcode == OP_I64_ADD_CARRY || ins.opcode == OP_I64_SUB_BORROW ? 3 : 2;
            for (unsigned i = 0; i < count; ++i) {
                Nvm2cSimSlot value;
                if (!sim_pop(b, idx, stk, &sp, &value) ||
                    !require_typed_integer_operand(b, local_kind, nloc, value)) return 0;
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1) ||
                !sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ADD:
        case OP_I64_ADD:
        case OP_SUB:
        case OP_I64_SUB:
        case OP_MUL:
        case OP_I64_MUL:
        case OP_DIV:
        case OP_I64_DIV_S:
        case OP_MOD:
        case OP_I64_REM_S:
        case OP_I64_EQ:
        case OP_I64_NE:
        case OP_I64_LT_S:
        case OP_I64_LE_S:
        case OP_I64_GT_S:
        case OP_I64_GE_S:
        case OP_BOOL_AND:
        case OP_BOOL_OR: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (ins.opcode == OP_I64_ADD || ins.opcode == OP_I64_SUB ||
                ins.opcode == OP_I64_MUL || ins.opcode == OP_I64_DIV_S ||
                ins.opcode == OP_I64_REM_S || ins.opcode == OP_I64_EQ ||
                ins.opcode == OP_I64_NE || ins.opcode == OP_I64_LT_S ||
                ins.opcode == OP_I64_LE_S || ins.opcode == OP_I64_GT_S ||
                ins.opcode == OP_I64_GE_S) {
                if (!require_typed_integer_operand(b, local_kind, nloc, rhs) ||
                    !require_typed_integer_operand(b, local_kind, nloc, lhs)) return 0;
            } else if (ins.opcode == OP_BOOL_AND || ins.opcode == OP_BOOL_OR) {
                mark_origin(local_kind, nloc, rhs.origin, NVM2C_VK_BOOL);
                mark_origin(local_kind, nloc, lhs.origin, NVM2C_VK_BOOL);
            }
            uint8_t result = NVM2C_VK_INT;
            if ((ins.opcode == OP_ADD || ins.opcode == OP_SUB ||
                 ins.opcode == OP_MUL || ins.opcode == OP_DIV) &&
                (rhs.kind == NVM2C_VK_FLOAT || lhs.kind == NVM2C_VK_FLOAT))
                result = NVM2C_VK_FLOAT;
            if ((ins.opcode == OP_ADD || ins.opcode == OP_SUB ||
                 ins.opcode == OP_MUL || ins.opcode == OP_DIV) &&
                (rhs.kind == NVM2C_VK_VALUE || lhs.kind == NVM2C_VK_VALUE))
                result = NVM2C_VK_VALUE;
            if (!sim_push(b, idx, stk, &sp, result, -1)) return 0;
            if (result == NVM2C_VK_VALUE) {
                stk[sp - 1].scalar_tags = (1u << TAG_INT) | (1u << TAG_FLOAT);
                if (!shape_carrier_box(b, stk[sp - 1].shape, stk[sp - 1].scalar_tags)) return 0;
            }
            break;
        }
        case OP_NEG:
        case OP_I64_NEG:
        case OP_BOOL_NOT: {
            Nvm2cSimSlot x;
            if (!sim_pop(b, idx, stk, &sp, &x)) return 0;
            if (ins.opcode == OP_I64_NEG) {
                if (!require_typed_integer_operand(b, local_kind, nloc, x)) return 0;
            } else if (ins.opcode == OP_BOOL_NOT) {
                mark_origin(local_kind, nloc, x.origin, NVM2C_VK_BOOL);
            }
            uint8_t result = ins.opcode == OP_NEG && x.kind == NVM2C_VK_FLOAT ?
                NVM2C_VK_FLOAT : NVM2C_VK_INT;
            if (ins.opcode == OP_NEG && x.kind == NVM2C_VK_VALUE)
                result = NVM2C_VK_VALUE;
            if (!sim_push(b, idx, stk, &sp, result, -1)) return 0;
            if (result == NVM2C_VK_VALUE) {
                stk[sp - 1].scalar_tags = (1u << TAG_INT) | (1u << TAG_FLOAT);
                if (!shape_carrier_box(b, stk[sp - 1].shape, stk[sp - 1].scalar_tags)) return 0;
            }
            break;
        }
        case OP_STR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (!mark_string_operand(b, local_kind, nloc, v)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CONCAT: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (!mark_string_operand(b, local_kind, nloc, rhs) ||
                !mark_string_operand(b, local_kind, nloc, lhs)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_TRIM: {
            Nvm2cSimSlot value;
            if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
            if (!mark_string_operand(b, local_kind, nloc, value)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_SUBSTR: {
            Nvm2cSimSlot len, start, s;
            if (!sim_pop(b, idx, stk, &sp, &len)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &start)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)len;
            (void)start;
            if (!mark_string_operand(b, local_kind, nloc, s)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_STARTS_WITH:
        case OP_STR_ENDS_WITH:
        case OP_STR_CONTAINS: {
            Nvm2cSimSlot needle, hay;
            if (!sim_pop(b, idx, stk, &sp, &needle)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &hay)) return 0;
            if (!mark_string_operand(b, local_kind, nloc, needle) ||
                !mark_string_operand(b, local_kind, nloc, hay)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CHAR_AT: {
            Nvm2cSimSlot ix, s;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)ix;
            if (!mark_string_operand(b, local_kind, nloc, s)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_CAST_BOOL: case OP_AND: case OP_OR: case OP_NOT: {
            unsigned count = ins.opcode == OP_AND || ins.opcode == OP_OR ? 2 : 1;
            for (unsigned operand = 0; operand < count; ++operand) {
                Nvm2cSimSlot value;
                if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
                if (value.kind != NVM2C_VK_UNK && value.kind != NVM2C_VK_INT &&
                    value.kind != NVM2C_VK_BOOL && value.kind != NVM2C_VK_FLOAT &&
                    value.kind != NVM2C_VK_VALUE) {
                    nvm2c_fail(b, "I support scalar truthiness only for void, int, u8, bool, float and enum");
                    return 0;
                }
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_BOOL, -1)) return 0;
            break;
        }
        case OP_CAST_FLOAT: {
            Nvm2cSimSlot value;
            if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_FLOAT, -1)) return 0;
            break;
        }
        case OP_CAST_INT: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_TYPE_CHECK: {
            Nvm2cSimSlot value;
            if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
            if (value.kind != NVM2C_VK_VALUE && value.kind != NVM2C_VK_UNK &&
                value.kind != NVM2C_VK_BOOL && value.kind != NVM2C_VK_INT &&
                value.kind != NVM2C_VK_STR && value.kind != NVM2C_VK_FLOAT && facts->final) {
                nvm2c_fail(b, "I require preserved runtime tags for TYPE_CHECK"); return 0;
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_CAST_STRING: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            (void)v;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_EQ:
        case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        case OP_NE: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            /* Generic comparison observes tags, not equal operand kinds. */
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_HM_NEW: {
            uint8_t key = ins.operands[0].u8, value = ins.operands[1].u8;
            if (key != TAG_STRING || (value != TAG_INT && value != TAG_STRING)) {
                nvm2c_fail(b, "I support string-keyed maps with integer or string values");
                return 0;
            }
            Nvm2cSimSlot map = {0};
            map.kind = NVM2C_VK_MAP; map.origin = -1;
            map.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
            if (!map.rec_k) return 0;
            map.rec_k[0] = NVM2C_VK_STR;
            map.rec_k[1] = value == TAG_INT ? NVM2C_VK_INT : NVM2C_VK_STR;
            if (!sim_push_slot(b, idx, stk, &sp, map)) return 0;
            if (!shape_type(b, shape_child(b, stk[sp - 1].shape, 0), NVM_SHAPE_STRING) ||
                !shape_kind(b, shape_child(b, stk[sp - 1].shape, 1), map.rec_k[1])) return 0;
            break;
        }
        case OP_HM_SET:
        case OP_HM_GET:
        case OP_HM_HAS:
        case OP_HM_DELETE:
        case OP_HM_LEN: {
            Nvm2cSimSlot map, key = {0}, value = {0};
            if (ins.opcode == OP_HM_SET && !sim_pop(b, idx, stk, &sp, &value)) return 0;
            if (ins.opcode != OP_HM_LEN && !sim_pop(b, idx, stk, &sp, &key)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &map)) return 0;
            if (map.kind != NVM2C_VK_MAP && map.kind != NVM2C_VK_VALUE && map.kind != NVM2C_VK_UNK) {
                nvm2c_fail(b, "I require a map for this hashmap operation"); return 0;
            }
            /* A tagged global keeps its optional storage. The checked runtime
             * operation validates its map tag and value kind independently. */
            if (map.kind == NVM2C_VK_VALUE) {
                NvmShapeId checked = 0;
                map.shape = shape_variable(b, &checked);
            } else mark_origin(local_kind, nloc, map.origin, NVM2C_VK_MAP);
            if (!shape_type(b, map.shape, NVM_SHAPE_MAP)) return 0;
            if (ins.opcode != OP_HM_LEN) {
                if (key.kind != NVM2C_VK_STR && key.kind != NVM2C_VK_UNK && key.kind != NVM2C_VK_VALUE) {
                    nvm2c_fail(b, "I require a string hashmap key"); return 0;
                }
                if (!mark_string_operand(b, local_kind, nloc, key)) return 0;
                NvmShapeId key_shape = key.shape;
                if (key.kind == NVM2C_VK_VALUE) {
                    /* Checked extraction constrains this use, not the optional
                     * producer. Emission retains nvalue_require_string. */
                    NvmShapeId checked = 0;
                    key_shape = shape_variable(b, &checked);
                    if (!shape_type(b, key_shape, NVM_SHAPE_STRING)) return 0;
                    if (b->track_shapes &&
                        (!nvm_shape_convert(&b->shapes, shape_child(b, key.shape, 0), key_shape) ||
                         !shape_ok(b))) return 0;
                }
                if (!shape_type(b, key_shape, NVM_SHAPE_STRING) ||
                    !shape_equal(b, shape_child(b, map.shape, 0), key_shape)) return 0;
            }
            if (ins.opcode == OP_HM_SET) {
                if (value.kind != NVM2C_VK_INT && value.kind != NVM2C_VK_STR && value.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "I require an integer or string hashmap value"); return 0;
                }
                if (map.kind == NVM2C_VK_MAP && map.rec_k[1] != NVM2C_VK_UNK &&
                    value.kind != NVM2C_VK_UNK && map.rec_k[1] != value.kind) {
                    nvm2c_fail(b, "I found conflicting hashmap value representations"); return 0;
                }
                if (!shape_equal(b, shape_child(b, map.shape, 1), value.shape)) return 0;
            }
            if (ins.opcode == OP_HM_GET) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1) ||
                    !shape_equal(b, shape_child(b, stk[sp - 1].shape, 0),
                                 shape_child(b, map.shape, 1))) return 0;
            } else if (ins.opcode == OP_HM_HAS || ins.opcode == OP_HM_LEN) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            } else {
                map.kind = NVM2C_VK_MAP; map.origin = -1;
                if (!sim_push_slot(b, idx, stk, &sp, map)) return 0;
            }
            break;
        }
        case OP_ARR_LITERAL: {
            uint8_t tag = ins.operands[0].u8;
            uint16_t count = ins.operands[1].u16;
            uint16_t ai;
            if (tag == TAG_STRUCT) {
                Nvm2cSimSlot array = {0};
                array.kind = NVM2C_VK_RARR;
                array.origin = -1;
                array.shape = shape_variable(b, b->shape_current);
                array.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!array.rec_k || !shape_type(b, array.shape, NVM_SHAPE_ARRAY)) return 0;
                NvmShapeId element = shape_child(b, array.shape, 0);
                if (!shape_type(b, element, NVM_SHAPE_RECORD)) return 0;
                for (ai = 0; ai < count; ++ai) {
                    Nvm2cSimSlot value;
                    if (!sim_pop(b, idx, stk, &sp, &value)) return 0;
                    if (value.kind != NVM2C_VK_REC && value.kind != NVM2C_VK_UNK) {
                        nvm2c_fail(b, "I require record elements in a record-array literal");
                        return 0;
                    }
                    if (value.kind == NVM2C_VK_REC) for (size_t f = 0; f < b->record_width; ++f) {
                        if (array.rec_k[f] == NVM2C_VK_UNK) array.rec_k[f] = value.rec_k[f];
                        else if (value.rec_k[f] != NVM2C_VK_UNK && array.rec_k[f] != value.rec_k[f]) {
                            nvm2c_fail(b, "I found conflicting record-array literal field representations");
                            return 0;
                        }
                    }
                    if (!shape_equal(b, element, value.shape)) return 0;
                }
                if (!sim_push_slot(b, idx, stk, &sp, array)) return 0;
                break;
            }
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                (void)v;
            }
            if (tag == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else {
                if (!sim_push(b, idx, stk, &sp, tag == TAG_FLOAT ? NVM2C_VK_FARR : tag == TAG_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR, -1)) return 0;
            }
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            if (tag == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else if (tag == TAG_INT || tag == TAG_BOOL || tag == TAG_FLOAT) {
                if (!sim_push(b, idx, stk, &sp, tag == TAG_FLOAT ? NVM2C_VK_FARR : tag == TAG_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR, -1)) return 0;
            } else if (tag == TAG_STRUCT) {
                Nvm2cSimSlot array;
                memset(&array, 0, sizeof array);
                array.kind = NVM2C_VK_RARR;
                array.origin = -1;
                array.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!array.rec_k) return 0;
                if (!sim_push_slot(b, idx, stk, &sp, array)) return 0;
            } else {
                nvm2c_fail(b, "function %u: I support int, bool, float, string or struct elements in ARR_NEW", idx);
                return 0;
            }
            break;
        }
        case OP_ARR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            /* Length determines the container, not its element storage. */
            if (v.kind == NVM2C_VK_UNK && !shape_type(b, v.shape, NVM_SHAPE_ARRAY)) return 0;
            if (v.kind == NVM2C_VK_RARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_RARR);
            } else if (v.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_SARR);
            } else if (v.kind != NVM2C_VK_UNK && v.kind != NVM2C_VK_VALUE) {
                mark_origin(local_kind, nloc, v.origin, v.kind == NVM2C_VK_FARR ? NVM2C_VK_FARR : v.kind == NVM2C_VK_BARR ? NVM2C_VK_BARR : NVM2C_VK_ARR);
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ARR_GET: {
            Nvm2cSimSlot ix, arr;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            (void)ix;
            if (arr.kind == NVM2C_VK_VALUE) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
                break;
            }
            if (!shape_type(b, arr.shape, NVM_SHAPE_ARRAY)) return 0;
            NvmShapeId element_shape = shape_child(b, arr.shape, 0);
            bool optional_scalar = word_array_storage(arr.kind) || arr.kind == NVM2C_VK_SARR;
            NvmShapeId result_shape = shape_variable(b, b->shape_current);
            if (optional_scalar) {
                if (!shape_type(b, result_shape, NVM_SHAPE_OPTIONAL) ||
                    !shape_equal(b, shape_child(b, result_shape, 0), element_shape)) return 0;
            } else if (!shape_equal(b, result_shape, element_shape)) return 0;
            if (arr.kind == NVM2C_VK_RARR) {
                Nvm2cSimSlot rec;
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_RARR);
                memset(&rec, 0, sizeof rec);
                rec.kind = NVM2C_VK_REC;
                rec.origin = -1;
                rec.rec_k = arr.rec_k;
                if (!sim_push_slot(b, idx, stk, &sp, rec)) return 0;
            } else if (arr.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_SARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
            } else if (arr.kind == NVM2C_VK_UNK) {
                Nvm2cSimSlot value = {0};
                value.kind = NVM2C_VK_UNK;
                value.origin = -1;
                value.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!value.rec_k || !sim_push_slot(b, idx, stk, &sp, value)) return 0;
            } else {
                mark_origin(local_kind, nloc, arr.origin, arr.kind == NVM2C_VK_FARR ? NVM2C_VK_FARR : arr.kind == NVM2C_VK_BARR ? NVM2C_VK_BARR : NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
                stk[sp - 1].scalar_tags = (1u << TAG_VOID) |
                    (1u << (arr.kind == NVM2C_VK_FARR ? TAG_FLOAT : arr.kind == NVM2C_VK_BARR ? TAG_BOOL : TAG_INT));
            }
            break;
        }
        case OP_ARR_SET:
        case OP_ARR_PUSH: {
            Nvm2cSimSlot val, arr;
            if (!sim_pop(b, idx, stk, &sp, &val)) return 0;
            if (ins.opcode == OP_ARR_SET) {
                Nvm2cSimSlot index;
                if (!sim_pop(b, idx, stk, &sp, &index)) return 0;
                /* I validate dynamic tags with nvalue_require_int at emission. */
                if (index.kind != NVM2C_VK_INT && index.kind != NVM2C_VK_UNK &&
                    index.kind != NVM2C_VK_VALUE) {
                    nvm2c_fail(b, "ARR_SET index must be an integer");
                    return 0;
                }
            }
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            if (arr.kind == NVM2C_VK_VALUE) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
                break;
            }
            if (ins.opcode == OP_ARR_PUSH && arr.kind == NVM2C_VK_RARR && val.kind == NVM2C_VK_REC) {
                for (size_t f = 0; f < b->record_width; ++f) {
                    if (arr.rec_k[f] != NVM2C_VK_UNK && val.rec_k[f] != NVM2C_VK_UNK &&
                        arr.rec_k[f] != val.rec_k[f]) {
                        nvm2c_fail(b, "ARR_PUSH record field representation mismatch "
                                      "(function %u, offset %zu, field %zu: %s versus %s)",
                                   idx, start, f, c_local_type(arr.rec_k[f]), c_local_type(val.rec_k[f]));
                        return 0;
                    }
                }
            }
            if (ins.opcode == OP_ARR_SET && arr.kind != NVM2C_VK_UNK &&
                val.kind != NVM2C_VK_UNK) {
                uint8_t expected = val.kind == NVM2C_VK_REC ? NVM2C_VK_RARR :
                    val.kind == NVM2C_VK_VALUE && word_array_storage(arr.kind) ? arr.kind :
                    val.kind == NVM2C_VK_VALUE && arr.kind == NVM2C_VK_SARR ? NVM2C_VK_SARR :
                    val.kind == NVM2C_VK_STR ? NVM2C_VK_SARR :
                    val.kind == NVM2C_VK_FLOAT ? NVM2C_VK_FARR : val.kind == NVM2C_VK_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR;
                if (arr.kind != expected) {
                    nvm2c_fail(b, "ARR_SET element representation mismatch");
                    return 0;
                }
                if (expected == NVM2C_VK_RARR) {
                    for (size_t f = 0; f < b->record_width; ++f) {
                        /* Unknown flat fields can acquire recursive facts
                         * later. The element graph is still unified below. */
                        if (arr.rec_k[f] != NVM2C_VK_UNK &&
                            val.rec_k[f] != NVM2C_VK_UNK &&
                            arr.rec_k[f] != val.rec_k[f]) {
                            nvm2c_fail(b, "ARR_SET record field representation mismatch (function %u, offset %zu, field %zu, kinds %u/%u, final=%d)",
                                       idx, start, f, arr.rec_k[f], val.rec_k[f], facts->final);
                            return 0;
                        }
                    }
                }
            }
            if (val.kind == NVM2C_VK_UNK) {
                /* Missing element facts cannot erase a constructor's known
                 * array kind or invent an integer-array representation. */
                Nvm2cSimSlot pushed = arr;
                pushed.origin = -1;
                if (!sim_push_slot(b, idx, stk, &sp, pushed)) return 0;
            } else if (val.kind == NVM2C_VK_REC) {
                Nvm2cSimSlot pushed = arr;
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_RARR);
                if (arr.origin >= 0 && (uint16_t)arr.origin < nloc) {
                    memcpy(rec_fields + (size_t)arr.origin * b->record_width,
                           val.rec_k, b->record_width);
                }
                pushed.kind = NVM2C_VK_RARR;
                pushed.origin = -1;
                pushed.rec_k = val.rec_k;
                if (!sim_push_slot(b, idx, stk, &sp, pushed)) return 0;
            } else if (val.kind == NVM2C_VK_STR || arr.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_SARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else {
                uint8_t kind = val.kind == NVM2C_VK_FLOAT || arr.kind == NVM2C_VK_FARR ? NVM2C_VK_FARR :
                    val.kind == NVM2C_VK_BOOL || arr.kind == NVM2C_VK_BARR ? NVM2C_VK_BARR : NVM2C_VK_ARR;
                mark_origin(local_kind, nloc, arr.origin, kind);
                if (!sim_push(b, idx, stk, &sp, kind, -1)) return 0;
            }
            /* A scalar write checks and unboxes its source at emission.
             * I keep the array payload exact without constraining a projected
             * source which can resolve to optional storage later. */
            uint8_t written_array = stk[sp - 1].kind;
            if (written_array == NVM2C_VK_SARR || word_array_storage(written_array)) {
                uint8_t expected = written_array == NVM2C_VK_SARR ? NVM2C_VK_STR :
                    written_array == NVM2C_VK_FARR ? NVM2C_VK_FLOAT : written_array == NVM2C_VK_BARR ? NVM2C_VK_BOOL : NVM2C_VK_INT;
                if (val.kind != NVM2C_VK_UNK && val.kind != NVM2C_VK_VALUE && val.kind != expected) {
                    nvm2c_fail(b, "I require the matching scalar shape at this array write");
                    return 0;
                }
                NvmShapeKind payload = written_array == NVM2C_VK_SARR ? NVM_SHAPE_STRING :
                    written_array == NVM2C_VK_FARR ? NVM_SHAPE_FLOAT : written_array == NVM2C_VK_BARR ? NVM_SHAPE_BOOL : NVM_SHAPE_INT;
                if (!shape_type(b, shape_child(b, arr.shape, 0), payload)) return 0;
                /* An already tagged value retains its known payload contract. */
                if (val.kind == NVM2C_VK_VALUE &&
                    !shape_equal(b, shape_child(b, arr.shape, 0), shape_child(b, val.shape, 0))) return 0;
            } else if (!shape_equal(b, shape_child(b, arr.shape, 0), val.shape)) return 0;
            if (!shape_equal(b, stk[sp - 1].shape, arr.shape)) return 0;
            break;
        }
        case OP_AGG_PACK: {
            uint8_t aggregate_kind = ins.operands[0].u8;
            uint16_t count = ins.operands[3].u16;
            Nvm2cSimSlot packed;
            uint16_t ai;
            if (aggregate_kind != AGG_RECORD && aggregate_kind != AGG_VARIANT &&
                aggregate_kind != AGG_TUPLE) {
                nvm2c_fail(b, "function %u: I support record, variant or tuple packing, not aggregate kind %u",
                           idx, aggregate_kind);
                return 0;
            }
            memset(&packed, 0, sizeof packed);
            packed.kind = NVM2C_VK_REC;
            packed.origin = -1;
            packed.shape = shape_variable(b, b->shape_current);
            if (!shape_type(b, packed.shape, NVM_SHAPE_RECORD)) return 0;
            /* I have no value or scalar-kind evidence beyond this variant's
             * declared payload width. Padding must not constrain its callers. */
            packed.rec_k = sim_fields(b, NULL, ins.operands[0].u8 == AGG_VARIANT
                                     ? NVM2C_VK_UNK : NVM2C_VK_INT);
            if (!packed.rec_k) return 0;
            if (count > b->record_width) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                return 0;
            }
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (v.kind != NVM2C_VK_INT && v.kind != NVM2C_VK_STR &&
                    !word_array_storage(v.kind) && v.kind != NVM2C_VK_SARR &&
                    v.kind != NVM2C_VK_RARR && v.kind != NVM2C_VK_REC && v.kind != NVM2C_VK_VALUE &&
                    v.kind != NVM2C_VK_BOOL && v.kind != NVM2C_VK_FLOAT && v.kind != NVM2C_VK_MAP &&
                    v.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "function %u at offset %zu: AGG_PACK field %u kind %u requires supported aggregate shape facts (final=%d)",
                               idx, start, (unsigned)(count - 1 - ai), v.kind, facts->final);
                    return 0;
                }
                NvmShapeId field = shape_child(b, packed.shape, count - 1 - ai);
                if (aggregate_kind == AGG_VARIANT && (variant_scalar_kind(v.kind) || v.kind == NVM2C_VK_ARR)) {
                    packed.rec_k[count - 1 - ai] = variant_field_for(v.kind);
                    /* This constructor records an integer-array member. The
                     * legacy empty-array placeholder must agree, not acquire
                     * an unrelated element kind from a later consumer. */
                    if (v.kind == NVM2C_VK_ARR &&
                        !shape_type(b, shape_child(b, v.shape, 0), NVM_SHAPE_INT)) return 0;
                    if (!shape_type(b, field, NVM_SHAPE_OPTIONAL) ||
                        !shape_type(b, shape_child(b, field, 0), v.kind == NVM2C_VK_ARR ?
                            NVM_SHAPE_VARIANT_INT_ARRAY : NVM_SHAPE_VARIANT_SCALAR)) return 0;
                    if (b->track_shapes && !nvm_shape_convert(&b->shapes, v.shape, field)) return 0;
                } else {
                    packed.rec_k[count - 1 - ai] = v.kind;
                    if (!shape_equal(b, field, v.shape)) return 0;
                }
            }
            if (!sim_push_slot(b, idx, stk, &sp, packed)) return 0;
            break;
        }
        case OP_AGG_TAG: {
            Nvm2cSimSlot aggregate;
            if (!sim_pop(b, idx, stk, &sp, &aggregate)) return 0;
            mark_origin(local_kind, nloc, aggregate.origin, NVM2C_VK_REC);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_AGG_GET: {
            Nvm2cSimSlot rec;
            uint16_t fi = ins.operands[0].u16;
            uint8_t fk;
            if (!sim_pop(b, idx, stk, &sp, &rec)) return 0;
            /* A nested projection can have no local origin. I constrain the
             * consumed shape itself, not only the flat local-kind vector. */
            if (!shape_type(b, rec.shape, NVM_SHAPE_RECORD)) return 0;
            mark_origin(local_kind, nloc, rec.origin, NVM2C_VK_REC);
            if (fi >= b->record_width) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                return 0;
            }
            fk = rec.rec_k[fi];
            if (!shape_equal(b, shape_variable(b, b->shape_current),
                             shape_child(b, rec.shape, fi))) return 0;
            Nvm2cSimSlot field = {0};
            field.kind = variant_field(fk) ? NVM2C_VK_VALUE : fk;
            if (variant_field(fk)) field.scalar_tags = variant_field_tags(fk);
            field.origin = -1;
            if (fk == NVM2C_VK_RARR || fk == NVM2C_VK_REC || fk == NVM2C_VK_MAP || fk == NVM2C_VK_UNK) {
                /* A flat parent vector describes this field's representation,
                 * not the representations inside its nested elements. */
                field.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!field.rec_k) return 0;
            }
            if (!sim_push_slot(b, idx, stk, &sp, field)) return 0;
            break;
        }
        case OP_CALL:
        case OP_TAIL_CALL: {
            uint32_t callee = ins.operands[0].u32;
            if (callee >= mod->function_count) {
                nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
                return 0;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (cf->arity > NVM2C_MAX_LOCALS || cf->arity > cf->local_count) {
                nvm2c_fail(b, "I cannot classify a call with invalid parameter counts");
                return 0;
            }
            for (i = cf->arity; i > 0; i--) {
                Nvm2cSimSlot arg;
                if (!sim_pop(b, idx, stk, &sp, &arg)) return 0;
                size_t at = (size_t)callee * b->local_width + i - 1;
                uint16_t tags = b->local_scalar_tags[at] |
                    (arg.kind == NVM2C_VK_UNK ? 0 : arg.scalar_tags);
                if (arg.kind != NVM2C_VK_UNK &&
                    (boxed_carrier_tags(b->local_scalar_tags[at]) || boxed_carrier_tags(arg.scalar_tags)) &&
                    !boxed_carrier_tags(tags)) {
                    nvm2c_fail(b, "I cannot mix finite parameter storage with unproved payload provenance");
                    return 0;
                }
                if (tags != b->local_scalar_tags[at]) {
                    b->local_scalar_tags[at] = tags; facts->changed = 1;
                }
                if (boxed_carrier_tags(tags)) {
                    if (facts->parameters[at] != NVM2C_VK_VALUE) {
                        facts->parameters[at] = NVM2C_VK_VALUE; facts->changed = 1;
                    }
                    b->tagged_locals[at] = 1;
                } else if (!merge_call_parameter(b, facts, &facts->parameters[at], arg.kind)) return 0;
                NvmShapeId parameter = shape_variable(b, &b->shape_locals[at]);
                if (boxed_carrier_tags(tags)) {
                    if (!shape_carrier_box(b, parameter, tags)) return 0;
                    if (b->track_shapes && !nvm_shape_convert(&b->shapes, arg.shape, parameter)) return 0;
                } else if (arg.kind == NVM2C_VK_REC) {
                    uint8_t *fields = facts->fields + at * b->record_width;
                    if (!merge_record_results(b, facts, fields, arg.rec_k) ||
                        !shape_record_return(b, arg.shape, parameter, arg.rec_k, fields)) return 0;
                } else if (arg.kind == NVM2C_VK_RARR) {
                    uint8_t *fields = facts->fields + at * b->record_width;
                    if (!merge_record_results(b, facts, fields, arg.rec_k) ||
                        !shape_type(b, parameter, NVM_SHAPE_ARRAY) ||
                        !shape_record_return(b, shape_child(b, arg.shape, 0),
                                             shape_child(b, parameter, 0), arg.rec_k, fields)) return 0;
                } else if ((facts->parameters[at] == NVM2C_VK_STR || facts->parameters[at] == NVM2C_VK_INT ||
                            facts->parameters[at] == NVM2C_VK_BOOL || facts->parameters[at] == NVM2C_VK_FLOAT) &&
                           (arg.kind == facts->parameters[at] || arg.kind == NVM2C_VK_UNK)) {
                    /* A projected scalar can resolve to tagged storage later.
                     * I convert into parameter storage without equating it to
                     * a producer's exact constructor or array element shape. */
                    if (!shape_field_kind(b, parameter, facts->parameters[at])) return 0;
                    if (b->track_shapes && !nvm_shape_convert(&b->shapes, arg.shape, parameter)) return 0;
                } else if (facts->parameters[at] == NVM2C_VK_VALUE &&
                           (arg.kind == NVM2C_VK_STR || arg.kind == NVM2C_VK_INT ||
                            arg.kind == NVM2C_VK_BOOL || arg.kind == NVM2C_VK_FLOAT || word_array_storage(arg.kind) ||
                            arg.kind == NVM2C_VK_SARR || arg.kind == NVM2C_VK_MAP || arg.kind == NVM2C_VK_UNK)) {
                    /* A projected field can resolve after flat classification.
                     * Its storage conversion must wait for those graph facts. */
                    if (!shape_type(b, parameter, NVM_SHAPE_OPTIONAL)) return 0;
                    if (b->track_shapes && !nvm_shape_convert(&b->shapes, arg.shape, parameter)) return 0;
                } else if (!shape_equal(b, arg.shape, parameter)) return 0;
                if (arg.kind == NVM2C_VK_UNK) {
                    mark_origin(local_kind, nloc, arg.origin, facts->parameters[at]);
                }
                if (arg.kind == NVM2C_VK_MAP) {
                    if (!merge_fields(b, facts, facts->fields + at * b->record_width, arg.rec_k)) return 0;
                }
            }
            if (ins.opcode == OP_CALL) {
                if (cf->result_count == 1 && (cf->result_tag == TAG_U8 || cf->result_tag == TAG_ENUM)) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
                    stk[sp - 1].scalar_tags = 1u << cf->result_tag;
                } else if (cf->result_count == 1 && cf->result_tag == TAG_FLOAT) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_FLOAT, -1)) return 0;
                } else if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
                } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                    Nvm2cSimSlot result = {0};
                    result.kind = b->array_results[callee];
                    result.origin = -1;
                    result.rec_k = sim_fields(b, facts->results + (size_t)callee * b->record_width, 0);
                    if (!result.rec_k || !sim_push_slot(b, idx, stk, &sp, result)) return 0;
                } else if (result_is_i64(cf)) {
                    if (!sim_push(b, idx, stk, &sp, cf->result_tag == TAG_BOOL ? NVM2C_VK_BOOL : NVM2C_VK_INT, -1)) return 0;
                } else if (cf->result_count == 1 &&
                           (aggregate_value_tag(cf->result_tag) || cf->result_tag == TAG_HASHMAP)) {
                    Nvm2cSimSlot result;
                    memset(&result, 0, sizeof result);
                    result.kind = cf->result_tag == TAG_HASHMAP ? NVM2C_VK_MAP : NVM2C_VK_REC;
                    result.origin = -1;
                    result.rec_k = sim_fields(b, facts->results + (size_t)callee * b->record_width, 0);
                    if (!result.rec_k) return 0;
                    if (!sim_push_slot(b, idx, stk, &sp, result)) return 0;
                }
            } else if (aggregate_value_tag(cf->result_tag)) {
                if (!merge_record_results(b, facts, facts->results + (size_t)idx * b->record_width,
                                          facts->results + (size_t)callee * b->record_width)) return 0;
            } else if (cf->result_tag == TAG_ARRAY || cf->result_tag == TAG_HASHMAP) {
                if (cf->result_tag == TAG_ARRAY &&
                    !merge_fact(b, facts, &b->array_results[idx], b->array_results[callee])) return 0;
                uint8_t *dest = facts->results + (size_t)idx * b->record_width;
                const uint8_t *source = facts->results + (size_t)callee * b->record_width;
                if (b->array_results[callee] == NVM2C_VK_RARR && cf->result_tag == TAG_ARRAY) {
                    if (!merge_record_results(b, facts, dest, source)) return 0;
                } else if (!merge_fields(b, facts, dest, source)) return 0;
            }
            if (ins.opcode == OP_CALL && cf->result_count == 1 && sp > 0) {
                if (!shape_equal(b, stk[sp - 1].shape, shape_variable(b, &b->shape_results[callee]))) return 0;
            } else if (ins.opcode == OP_TAIL_CALL && cf->result_count == 1) {
                if (aggregate_value_tag(cf->result_tag)) {
                    if (!shape_record_return(b, shape_variable(b, &b->shape_results[callee]),
                                             shape_variable(b, &b->shape_results[idx]),
                                             facts->results + (size_t)callee * b->record_width,
                                             facts->results + (size_t)idx * b->record_width)) return 0;
                } else if (cf->result_tag == TAG_ARRAY && b->array_results[callee] == NVM2C_VK_RARR) {
                    NvmShapeId from = shape_variable(b, &b->shape_results[callee]);
                    NvmShapeId to = shape_variable(b, &b->shape_results[idx]);
                    if (!shape_type(b, to, NVM_SHAPE_ARRAY) ||
                        !shape_record_return(b, shape_child(b, from, 0), shape_child(b, to, 0),
                                             facts->results + (size_t)callee * b->record_width,
                                             facts->results + (size_t)idx * b->record_width)) return 0;
                } else if (!shape_equal(b, shape_variable(b, &b->shape_results[idx]),
                                        shape_variable(b, &b->shape_results[callee]))) return 0;
            }
            break;
        }
        case OP_CALL_EXTERN: {
            const Nvm2cHost *host = import_host(mod, ins.operands[0].u32);
            if (!host) {
                nvm2c_fail(b, "function %u: CALL_EXTERN has no exact builtin host ABI", idx);
                return 0;
            }
            for (uint8_t p = 0; p < host->argc; ++p) {
                Nvm2cSimSlot arg;
                if (!sim_pop(b, idx, stk, &sp, &arg)) return 0;
                uint8_t expected = host->parameter == TAG_STRING ? NVM2C_VK_STR :
                                   host->parameter == TAG_FLOAT ? NVM2C_VK_FLOAT : NVM2C_VK_INT;
                /* I check tagged arguments when the host consumes them; that
                 * use does not change their caller-owned representation. */
                if (arg.kind == NVM2C_VK_VALUE) continue;
                if (arg.kind != expected && arg.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "function %u at offset %zu: CALL_EXTERN %s argument kind mismatch at %u: %u requires %u",
                               idx, start, host->c_name, (unsigned)(host->argc - p - 1), arg.kind, expected);
                    return 0;
                }
                /* I wait for caller facts before constraining an unknown
                 * parameter: its eventual storage may retain runtime tags. */
                if (arg.kind != NVM2C_VK_UNK || facts->final)
                    mark_origin(local_kind, nloc, arg.origin, expected);
            }
            if (!sim_push(b, idx, stk, &sp,
                          host->result == TAG_ARRAY ? NVM2C_VK_SARR :
                          host->result == TAG_STRING ? NVM2C_VK_STR :
                          host->result == TAG_BOOL ? NVM2C_VK_BOOL :
                          host->result == TAG_FLOAT ? NVM2C_VK_FLOAT : NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_JMP_TRUE:
        case OP_JMP_FALSE: {
            Nvm2cSimSlot cond;
            if (!sim_pop(b, idx, stk, &sp, &cond)) return 0;
            (void)cond;
            break;
        }
        case OP_RET:
        case OP_HALT: {
            if (fn->result_count == 1 &&
                (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL || (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM) || fn->result_tag == TAG_FLOAT ||
                 fn->result_tag == TAG_STRING || fn->result_tag == TAG_ARRAY ||
                 aggregate_value_tag(fn->result_tag) || fn->result_tag == TAG_HASHMAP) &&
                sp > 0) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (facts->final && variant_payload_tags(v.scalar_tags) &&
                    fn->result_tag < 16 && !(v.scalar_tags & (1u << fn->result_tag))) {
                    nvm2c_fail(b, "I reject RET with a known incompatible variant scalar payload");
                    return 0;
                }
                if (fn->result_tag == TAG_BOOL) {
                    mark_origin(local_kind, nloc, v.origin, NVM2C_VK_BOOL);
                } else if (fn->result_tag == TAG_FLOAT && v.kind == NVM2C_VK_UNK) {
                    mark_origin(local_kind, nloc, v.origin, NVM2C_VK_FLOAT);
                } else if (fn->result_tag == TAG_STRING) {
                    if (!mark_string_operand(b, local_kind, nloc, v)) return 0;
                } else if (fn->result_tag == TAG_ARRAY) {
                    if (v.kind == NVM2C_VK_VALUE && variant_payload_tags(v.scalar_tags) &&
                        (v.scalar_tags & (1u << TAG_ARRAY))) {
                        if (!merge_fact(b, facts, &b->array_results[idx], NVM2C_VK_ARR) ||
                            !shape_kind(b, shape_variable(b, &b->shape_results[idx]), NVM2C_VK_ARR)) return 0;
                        break; /* Emission checks tag, integer storage and presence. */
                    }
                    if (v.kind != NVM2C_VK_UNK && !word_array_storage(v.kind) &&
                        v.kind != NVM2C_VK_SARR && v.kind != NVM2C_VK_RARR) {
                        nvm2c_fail(b, "I require an array representation at return in function %u", idx);
                        return 0;
                    }
                    if (!merge_fact(b, facts, &b->array_results[idx], v.kind)) return 0;
                    mark_origin(local_kind, nloc, v.origin, b->array_results[idx]);
                    if (v.kind == NVM2C_VK_RARR &&
                        !merge_record_results(b, facts, facts->results + (size_t)idx * b->record_width,
                                      v.rec_k)) return 0;
                } else if (aggregate_value_tag(fn->result_tag) || fn->result_tag == TAG_HASHMAP) {
                    mark_origin(local_kind, nloc, v.origin, fn->result_tag == TAG_HASHMAP ? NVM2C_VK_MAP : NVM2C_VK_REC);
                    if (v.kind == NVM2C_VK_REC &&
                        !merge_record_results(b, facts, facts->results + (size_t)idx * b->record_width,
                                              v.rec_k)) return 0;
                    if (v.kind == NVM2C_VK_MAP &&
                        !merge_fields(b, facts, facts->results + (size_t)idx * b->record_width,
                                      v.rec_k)) return 0;
                }
                NvmShapeKind declared = (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM) ? NVM_SHAPE_OPTIONAL :
                    fn->result_tag == TAG_STRING ? NVM_SHAPE_STRING :
                    fn->result_tag == TAG_BOOL ? NVM_SHAPE_BOOL :
                    fn->result_tag == TAG_FLOAT ? NVM_SHAPE_FLOAT :
                    fn->result_tag == TAG_HASHMAP ? NVM_SHAPE_MAP :
                    fn->result_tag == TAG_ARRAY ? NVM_SHAPE_ARRAY :
                    aggregate_value_tag(fn->result_tag) ? NVM_SHAPE_RECORD : NVM_SHAPE_INT;
                /* RET checks and unboxes a scalar or tagged map at emission.
                 * An inferred scalar projection may become optional later;
                 * its declared result never rewrites source storage. */
                if (((v.kind == NVM2C_VK_VALUE || v.kind == NVM2C_VK_UNK ||
                      v.kind == NVM2C_VK_INT || v.kind == NVM2C_VK_BOOL ||
                      v.kind == NVM2C_VK_FLOAT || v.kind == NVM2C_VK_STR) &&
                     (declared == NVM_SHAPE_INT || declared == NVM_SHAPE_BOOL ||
                      declared == NVM_SHAPE_FLOAT || declared == NVM_SHAPE_STRING)) ||
                    (v.kind == NVM2C_VK_VALUE && declared == NVM_SHAPE_MAP)) {
                    if (!shape_type(b, shape_variable(b, &b->shape_results[idx]), declared)) return 0;
                    break;
                }
                if (!shape_type(b, v.shape, declared)) return 0;
                if (declared == NVM_SHAPE_ARRAY && v.kind == NVM2C_VK_RARR) {
                    NvmShapeId result = shape_variable(b, &b->shape_results[idx]);
                    if (!shape_type(b, result, NVM_SHAPE_ARRAY) ||
                        !shape_record_return(b, shape_child(b, v.shape, 0), shape_child(b, result, 0),
                                             v.rec_k, facts->results + (size_t)idx * b->record_width)) return 0;
                } else if (declared == NVM_SHAPE_RECORD) {
                    if (!shape_record_return(b, v.shape, shape_variable(b, &b->shape_results[idx]),
                                             v.rec_k, facts->results + (size_t)idx * b->record_width)) return 0;
                } else if (!shape_equal(b, v.shape, shape_variable(b, &b->shape_results[idx]))) return 0;
            }
            break;
        }
        case OP_JMP:
            break;
        default: {
            const InstructionInfo *info = isa_get_info(ins.opcode);
            nvm2c_fail(b, "I cannot classify unsupported opcode %s (0x%02X) "
                          "in function %u at offset %zu",
                       info ? info->name : "UNKNOWN", ins.opcode, idx, start);
            return 0;
        }
        }
        if (ins.opcode == OP_JMP || ins.opcode == OP_JMP_FALSE || ins.opcode == OP_JMP_TRUE) {
            size_t target;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &target) ||
                !sim_join(b, idx, target, &joins[target], stk, sp, facts)) return 0;
        }
        if (ins.opcode == OP_JMP || ins.opcode == OP_RET ||
            ins.opcode == OP_HALT || ins.opcode == OP_TAIL_CALL) terminated = 1;
        previous_false_push = ins.opcode == OP_PUSH_BOOL && ins.operands[0].u8 == 0;
        if (b->failed) return 0;
    }

    for (i = 0; i < nloc; i++) {
        if (i < fn->arity &&
            !merge_parameter(b, facts, &facts->parameters[(size_t)idx * b->local_width + i], local_kind[i])) return 0;
    }
    return 1;
}

static int classify_function(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                             uint8_t *local_kind, uint8_t *rec_fields, Nvm2cFacts *facts) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    if (fn->local_count > NVM2C_MAX_LOCALS) {
        nvm2c_fail(b, "function %u: too many locals", idx);
        return 0;
    }
    if (fn->arity > fn->local_count) {
        nvm2c_fail(b, "function %u: arity exceeds local_count", idx);
        return 0;
    }
    if (fn->code_offset > mod->code_size || fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "I cannot classify function %u outside the module code", idx);
        return 0;
    }
    size_t length = fn->code_length;
    /* Every supported instruction adds at most one stack value. A reachable
     * positive-growth cycle is rejected by join-height checking. */
    if (length >= INT_MAX || length > SIZE_MAX / sizeof(Nvm2cSimSlot) - 1 ||
        length > SIZE_MAX / sizeof(Nvm2cSimJoin) - 1) {
        nvm2c_fail(b, "I cannot represent this function's classifier stack");
        return 0;
    }
    b->sim_stack_capacity = length + 1;
    Nvm2cSimSlot *stack = malloc((length + 1) * sizeof *stack);
    Nvm2cSimJoin *joins = calloc(length + 1, sizeof *joins);
    uint8_t *targets = calloc(length + 1, 1);
    uint8_t *starts = calloc(length + 1, 1);
    int ok = 0;
    if (!stack || !joins || !targets || !starts) {
        nvm2c_fail(b, "I cannot allocate classifier control-flow state");
        goto done;
    }
    for (size_t pc = 0; pc < length;) {
        DecodedInstruction ins;
        uint32_t n = isa_decode(mod->code + fn->code_offset + pc, length - pc, &ins);
        if (!n) {
            nvm2c_fail(b, "I cannot decode function %u at offset %zu", idx, pc);
            goto done;
        }
        starts[pc] = 1;
        if (ins.opcode == OP_JMP || ins.opcode == OP_JMP_FALSE || ins.opcode == OP_JMP_TRUE) {
            size_t target;
            if (!jump_target(b, idx, pc, ins.operands[0].i32, length, &target)) goto done;
            targets[target] = 1;
        }
        pc += n;
    }
    starts[length] = 1;
    for (size_t pc = 0; pc <= length; pc++) {
        if (targets[pc] && !starts[pc]) {
            nvm2c_fail(b, "I found a jump to a non-instruction boundary in function %u", idx);
            goto done;
        }
    }
    if (facts->final && !b->shape_outputs[idx]) {
        b->shape_outputs[idx] = calloc(length ? length : 1, sizeof(NvmShapeId));
        if (!b->shape_outputs[idx]) {
            nvm2c_fail(b, "I cannot allocate function instruction shapes");
            goto done;
        }
    }
    b->default_fields = sim_fields(b, NULL, NVM2C_VK_UNK);
    if (!b->default_fields) goto done;
    ok = classify_function_body(b, mod, idx, local_kind, rec_fields, joins, targets, facts, stack);
    if (ok && facts->final) {
        for (size_t pc = 0; pc <= length; ++pc) {
            if (!joins[pc].set || !joins[pc].sp) continue;
            size_t count = (size_t)joins[pc].sp;
            if (count > (SIZE_MAX - sizeof(Nvm2cJoinShape)) / sizeof(NvmShapeId)) {
                nvm2c_fail(b, "I cannot represent these join shapes"); ok = 0; break;
            }
            Nvm2cJoinShape *saved = malloc(sizeof *saved + count * sizeof(NvmShapeId));
            if (!saved) { nvm2c_fail(b, "I cannot allocate join shapes"); ok = 0; break; }
            saved->target = pc; saved->count = joins[pc].sp;
            for (size_t i = 0; i < count; ++i) saved->shapes[i] = joins[pc].slots[i].shape;
            saved->next = b->join_shapes[idx]; b->join_shapes[idx] = saved;
        }
    }
done:
    while (b->field_blocks) {
        Nvm2cFieldBlock *next = b->field_blocks->next;
        free(b->field_blocks);
        b->field_blocks = next;
    }
    b->default_fields = NULL;
    if (joins) for (size_t pc = 0; pc <= length; ++pc) free(joins[pc].slots);
    free(stack);
    free(joins);
    free(targets);
    free(starts);
    return ok;
}

static void emit_prototype(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                           const uint8_t *kinds) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(b, fn, idx);
    if (!rt) {
        nvm2c_fail(b, "function %u: only void or a single supported value result is supported",
                   idx);
        return;
    }
    if (fn->upvalue_count != 0) {
        nvm2c_fail(b, "function %u: upvalues are not in the nvm2c subset", idx);
        return;
    }
    if (fn->local_count > NVM2C_MAX_LOCALS) {
        nvm2c_fail(b, "function %u: too many locals", idx);
        return;
    }
    if (fn->arity > fn->local_count) {
        nvm2c_fail(b, "function %u: arity exceeds local_count", idx);
        return;
    }
    char name[64];
    fn_c_name(mod, idx, name, sizeof name);
    nvm2c_printf(b, "static %s %s(", rt, name);
    if (fn->arity == 0) {
        nvm2c_puts(b, "void");
    } else {
        uint16_t i;
        for (i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(b, kinds, idx, i)), (unsigned)i);
        }
    }
    nvm2c_puts(b, ");\n");
}

typedef struct {
    int *slots;
    uint8_t *kinds;
    size_t capacity;
    uint8_t **rec_k;
    uint8_t **rarr_k;
    size_t rec_capacity;
    int sp;
    int next_temp;
    int next_str;
    int next_arr;
    int next_sarr;
    int next_rec;
    int next_rarr;
    int next_map;
    int next_value;
    int next_float;
} Nvm2cStack;

static int stack_push_temp(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if ((size_t)st->next_temp >= st->capacity) {
        nvm2c_fail(b, "too many temporaries");
        return -1;
    }
    int t = st->next_temp++;
    nvm2c_printf(b, "    t[%d] = %s;\n", t, rhs);
    st->slots[st->sp] = t;
    st->kinds[st->sp] = NVM2C_VK_INT;
    st->sp++;
    return t;
}

static int stack_push_bool(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    int slot = stack_push_temp(b, st, rhs);
    if (slot >= 0) st->kinds[st->sp - 1] = NVM2C_VK_BOOL;
    return slot;
}

static int stack_push_str(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if ((size_t)st->next_str >= st->capacity) {
        nvm2c_fail(b, "too many string temporaries");
        return -1;
    }
    int s = st->next_str++;
    nvm2c_printf(b, "    s[%d] = %s;\n", s, rhs);
    st->slots[st->sp] = s;
    st->kinds[st->sp] = NVM2C_VK_STR;
    st->sp++;
    return s;
}

static int stack_push_float(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity || (size_t)st->next_float >= st->capacity) {
        nvm2c_fail(b, "too many float temporaries");
        return -1;
    }
    int f = st->next_float++;
    nvm2c_printf(b, "    f[%d] = %s;\n", f, rhs);
    st->slots[st->sp] = f;
    st->kinds[st->sp++] = NVM2C_VK_FLOAT;
    return f;
}

static int stack_push_arr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if ((size_t)st->next_arr >= st->capacity) {
        nvm2c_fail(b, "too many array temporaries");
        return -1;
    }
    int a = st->next_arr++;
    nvm2c_printf(b, "    a[%d] = %s;\n", a, rhs);
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_ARR;
    st->sp++;
    return a;
}

static int stack_push_iarray(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs, uint8_t kind) {
    int result = stack_push_arr(b, st, rhs);
    if (result >= 0) st->kinds[st->sp - 1] = kind;
    return result;
}

static int stack_push_sarr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if ((size_t)st->next_sarr >= st->capacity) {
        nvm2c_fail(b, "too many string-array temporaries");
        return -1;
    }
    int a = st->next_sarr++;
    nvm2c_printf(b, "    sa[%d] = %s;\n", a, rhs);
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_SARR;
    st->sp++;
    return a;
}

static int stack_push_rec(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if ((size_t)st->next_rec >= st->rec_capacity) {
        nvm2c_fail(b, "too many record temporaries");
        return -1;
    }
    int r = st->next_rec++;
    nvm2c_printf(b, "    r[%d] = %s;\n", r, rhs);
    st->slots[st->sp] = r;
    st->kinds[st->sp] = NVM2C_VK_REC;
    st->sp++;
    return r;
}

static int stack_push_rarr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity || (size_t)st->next_rarr >= st->rec_capacity) {
        nvm2c_fail(b, "too many record-array temporaries");
        return -1;
    }
    int a = st->next_rarr++;
    nvm2c_printf(b, "    ra[%d] = %s;\n", a, rhs);
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_RARR;
    st->sp++;
    return a;
}

static int stack_push_map(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity || (size_t)st->next_map >= st->capacity) {
        nvm2c_fail(b, "I cannot allocate another map temporary"); return -1;
    }
    int map = st->next_map++;
    nvm2c_printf(b, "    m[%d] = %s;\n", map, rhs);
    st->slots[st->sp] = map; st->kinds[st->sp++] = NVM2C_VK_MAP;
    return map;
}

static int stack_push_value(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if ((size_t)st->sp >= st->capacity || (size_t)st->next_value >= st->capacity) {
        nvm2c_fail(b, "I cannot allocate another tagged value temporary"); return -1;
    }
    int value = st->next_value++;
    nvm2c_printf(b, "    v[%d] = %s;\n", value, rhs);
    st->slots[st->sp] = value; st->kinds[st->sp++] = NVM2C_VK_VALUE;
    return value;
}

static int stack_pop_kind(Nvm2cBuf *b, Nvm2cStack *st, uint8_t *kind_out) {
    if (st->sp <= 0) {
        nvm2c_fail(b, "operand stack underflow");
        return -1;
    }
    st->sp--;
    if (kind_out) *kind_out = st->kinds[st->sp];
    return st->slots[st->sp];
}

static int stack_pop(Nvm2cBuf *b, Nvm2cStack *st) {
    return stack_pop_kind(b, st, NULL);
}

static const char *stack_array_name(uint8_t kind);

static int stack_pop_expect(Nvm2cBuf *b, Nvm2cStack *st, uint8_t kind, const char *what) {
    uint8_t got = NVM2C_VK_INT;
    int slot = stack_pop_kind(b, st, &got);
    if (b->failed) return -1;
    if (got == NVM2C_VK_FLOAT && kind == NVM2C_VK_VALUE) {
        char expression[64];
        snprintf(expression, sizeof expression, "nvalue_from_float(f[%d])", slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_VALUE && kind == NVM2C_VK_FLOAT) {
        char expression[64];
        snprintf(expression, sizeof expression, "nvalue_require_float(v[%d])", slot);
        stack_push_float(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_MAP && kind == NVM2C_VK_VALUE) {
        char expression[80];
        snprintf(expression, sizeof expression, "(nmap_value){13, 0, (char *)m[%d]}", slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_VALUE && kind == NVM2C_VK_MAP) {
        char expression[64];
        snprintf(expression, sizeof expression, "nvalue_require_map(v[%d])", slot);
        stack_push_map(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_STR && kind == NVM2C_VK_VALUE) {
        char expression[80];
        snprintf(expression, sizeof expression, "(nmap_value){5, 0, (char *)s[%d]}", slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if ((got == NVM2C_VK_INT || got == NVM2C_VK_BOOL) && kind == NVM2C_VK_VALUE) {
        char expression[80];
        snprintf(expression, sizeof expression, "(nmap_value){%u, t[%d], NULL}",
                 got == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT, slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if ((word_array_storage(got) || got == NVM2C_VK_SARR) && kind == NVM2C_VK_VALUE) {
        char expression[80];
        snprintf(expression, sizeof expression, "(nmap_value){7, %u, (char *)%s[%d]}",
                 got, stack_array_name(got), slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_VALUE && (kind == NVM2C_VK_INT || kind == NVM2C_VK_STR || kind == NVM2C_VK_BOOL)) {
        char expression[64];
        snprintf(expression, sizeof expression, "nvalue_require_%s(v[%d])",
                 kind == NVM2C_VK_INT ? "int" : kind == NVM2C_VK_BOOL ? "bool" : "string", slot);
        if (kind == NVM2C_VK_INT) stack_push_temp(b, st, expression);
        else if (kind == NVM2C_VK_BOOL) stack_push_bool(b, st, expression);
        else stack_push_str(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_VALUE && kind == NVM2C_VK_ARR) {
        char expression[80];
        snprintf(expression, sizeof expression, "nvalue_require_int_array(v[%d])", slot);
        stack_push_arr(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got != kind) {
        const char *want = "int";
        if (kind == NVM2C_VK_BOOL) want = "bool";
        if (kind == NVM2C_VK_STR) want = "string";
        else if (kind == NVM2C_VK_FLOAT) want = "float";
        else if (word_array_storage(kind)) want = "array";
        else if (kind == NVM2C_VK_SARR) want = "string array";
        else if (kind == NVM2C_VK_REC) want = "record";
        else if (kind == NVM2C_VK_RARR) want = "record array";
        else if (kind == NVM2C_VK_MAP) want = "hashmap";
        nvm2c_fail(b, "%s: expected %s value", what, want);
        return -1;
    }
    return slot;
}

static int stack_pop_condition(Nvm2cBuf *b, Nvm2cStack *st, const char *what) {
    if (st->sp && st->kinds[st->sp - 1] == NVM2C_VK_VALUE) {
        int value = stack_pop(b, st);
        char expression[384];
        snprintf(expression, sizeof expression,
                 "((v[%d].kind == 3 && nvalue_require_float(v[%d]) != 0.0) || v[%d].kind == 5 || v[%d].kind == 7 || v[%d].kind == 13 || ((v[%d].kind == 1 || v[%d].kind == 2 || v[%d].kind == 4 || v[%d].kind == 9) && v[%d].integer != 0))",
                 value, value, value, value, value, value, value, value, value, value);
        stack_push_temp(b, st, expression);
    }
    if (st->sp && st->kinds[st->sp - 1] == NVM2C_VK_FLOAT) {
        int value = stack_pop(b, st);
        char expression[48];
        snprintf(expression, sizeof expression, "(f[%d] != 0.0)", value);
        stack_push_bool(b, st, expression);
    }
    if (st->sp) {
        uint8_t kind = st->kinds[st->sp - 1];
        if (kind == NVM2C_VK_STR || word_array_storage(kind) ||
            kind == NVM2C_VK_SARR || kind == NVM2C_VK_RARR ||
            kind == NVM2C_VK_MAP || kind == NVM2C_VK_REC) {
            int value = stack_pop(b, st);
            char expression[48];
            /* My by-value record represents a constructed VM heap object. */
            if (kind == NVM2C_VK_REC) snprintf(expression, sizeof expression, "1");
            else snprintf(expression, sizeof expression, "(%s[%d] != NULL)",
                          stack_array_name(kind), value);
            stack_push_bool(b, st, expression);
        }
    }
    if (st->sp && st->kinds[st->sp - 1] == NVM2C_VK_BOOL)
        return stack_pop_expect(b, st, NVM2C_VK_BOOL, what);
    return stack_pop_expect(b, st, NVM2C_VK_INT, what);
}

/* I keep this new opcode profile scalar; existing branch truthiness is unchanged. */
static int stack_pop_scalar_condition(Nvm2cBuf *b, Nvm2cStack *st) {
    if (st->sp) {
        uint8_t kind = st->kinds[st->sp - 1];
        if (kind == NVM2C_VK_VALUE) {
            int slot = st->slots[st->sp - 1];
            nvm2c_printf(b, "    if (v[%d].kind != 0 && v[%d].kind != 1 && v[%d].kind != 2 && v[%d].kind != 3 && v[%d].kind != 4 && v[%d].kind != 9) NVM2C_ABORT();\n",
                        slot, slot, slot, slot, slot, slot);
        } else if (kind != NVM2C_VK_INT && kind != NVM2C_VK_BOOL && kind != NVM2C_VK_FLOAT) {
            nvm2c_fail(b, "I support scalar truthiness only for void, int, u8, bool, float and enum");
            return -1;
        }
    }
    return stack_pop_condition(b, st, "scalar truthiness");
}

static void scalar_value_expression(Nvm2cBuf *b, char *out, size_t size, uint8_t kind, int slot) {
    if (kind == NVM2C_VK_VALUE) snprintf(out, size, "v[%d]", slot);
    else if (kind == NVM2C_VK_FLOAT) snprintf(out, size, "nvalue_from_float(f[%d])", slot);
    else if (kind == NVM2C_VK_STR) snprintf(out, size, "(nmap_value){5, 0, (char *)s[%d]}", slot);
    else if (kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL)
        snprintf(out, size, "(nmap_value){%u, t[%d], NULL}", kind == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT, slot);
    else nvm2c_fail(b, "I require a tagged or scalar array element");
}

/* I give record locals stable invocation-owned addresses. Other locals retain
 * their scalar/handle storage; every local use shares this spelling. */
static void local_operand(char out[32], const Nvm2cBuf *b, const uint8_t *kinds,
                          uint32_t idx, uint16_t slot) {
    if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_REC) {
        unsigned record_slot = 0;
        for (uint16_t i = 0; i < slot; ++i)
            if (fn_local_kind(b, kinds, idx, i) == NVM2C_VK_REC) ++record_slot;
        snprintf(out, 32, "rl[%u]", record_slot);
    } else {
        snprintf(out, 32, "l%u", (unsigned)slot);
    }
}

/* I publish only live operand slots, not stale high-water temporaries. Caller
 * snapshots stay registered while callees run; mutable aggregates are traced
 * from their current contents at collection time. */
static void emit_map_roots(Nvm2cBuf *b, const Nvm2cStack *st,
                           const NvmFunctionEntry *fn, const uint8_t *kinds,
                           uint32_t idx) {
    if (!b->has_maps) return;
    nvm2c_puts(b, "    nroot_reset(&nroots.live);\n");
    for (uint16_t i = 0; i < fn->local_count; ++i) {
        uint8_t k = fn_local_kind(b, kinds, idx, i);
        if (k == NVM2C_VK_INT || k == NVM2C_VK_BOOL || k == NVM2C_VK_FLOAT) continue;
        char local[32];
        local_operand(local, b, kinds, idx, i);
        nvm2c_printf(b, "    nroot_add(&nroots.live, %u, %s%s);\n", k,
                     k == NVM2C_VK_REC || k == NVM2C_VK_VALUE ? "&" : "", local);
    }
    for (int i = 0; i < st->sp; ++i) {
        uint8_t k = st->kinds[i];
        if (k == NVM2C_VK_INT || k == NVM2C_VK_BOOL || k == NVM2C_VK_FLOAT) continue;
        nvm2c_printf(b, "    nroot_add(&nroots.live, %u, %s%s[%d]);\n", k,
                     k == NVM2C_VK_REC || k == NVM2C_VK_VALUE ? "&" : "",
                     stack_array_name(k), st->slots[i]);
    }
}

/* I coerce enum ordinals only at the VM's typed binary integer boundary.
 * Producer slots and exact integer consumers retain their original tags. */
static int prepare_typed_integer_pair(Nvm2cBuf *b, Nvm2cStack *st, uint8_t opcode) {
    switch (opcode) {
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL:
    case OP_I64_DIV_S: case OP_I64_REM_S:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S:
    case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
        break;
    default: return 1;
    }
    if (st->sp < 2) { nvm2c_fail(b, "typed integer operand stack underflow"); return 0; }
    for (int at = st->sp - 2; at < st->sp; ++at) {
        if (st->kinds[at] != NVM2C_VK_VALUE) continue;
        if ((size_t)st->next_temp >= st->capacity) {
            nvm2c_fail(b, "too many typed integer temporaries"); return 0;
        }
        int value = st->slots[at], temp = st->next_temp++;
        nvm2c_printf(b, "    if (v[%d].kind != 1 && v[%d].kind != 9) NVM2C_ABORT();\n"
                         "    t[%d] = v[%d].integer;\n", value, value, temp, value);
        st->slots[at] = temp;
        st->kinds[at] = NVM2C_VK_INT;
    }
    return !b->failed;
}

static void emit_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    uint8_t input_kind = strcmp(op, "&&") == 0 || strcmp(op, "||") == 0 ? NVM2C_VK_BOOL : NVM2C_VK_INT;
    int rhs = stack_pop_expect(b, st, input_kind, "binary op rhs");
    int lhs = stack_pop_expect(b, st, input_kind, "binary op lhs");
    if (b->failed) return;
    char expr[128];
    if (!strcmp(op, "+") || !strcmp(op, "-") || !strcmp(op, "*"))
        snprintf(expr, sizeof expr, "ni64_from_bits((uint64_t)t[%d] %s (uint64_t)t[%d])", lhs, op, rhs);
    else snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    stack_push_temp(b, st, expr);
}

static void emit_unop(Nvm2cBuf *b, Nvm2cStack *st, const char *prefix) {
    int x = stack_pop_expect(b, st, strcmp(prefix, "!") == 0 ? NVM2C_VK_BOOL : NVM2C_VK_INT, "unary op");
    if (b->failed) return;
    char expr[64];
    if (!strcmp(prefix, "-"))
        snprintf(expr, sizeof expr, "ni64_from_bits(UINT64_C(0) - (uint64_t)t[%d])", x);
    else snprintf(expr, sizeof expr, "(%s t[%d])", prefix, x);
    stack_push_temp(b, st, expr);
}

static void stack_keep_high_water(Nvm2cStack *st, const Nvm2cStack *other) {
    if (other->next_temp > st->next_temp) st->next_temp = other->next_temp;
    if (other->next_str > st->next_str) st->next_str = other->next_str;
    if (other->next_arr > st->next_arr) st->next_arr = other->next_arr;
    if (other->next_sarr > st->next_sarr) st->next_sarr = other->next_sarr;
    if (other->next_rec > st->next_rec) st->next_rec = other->next_rec;
    if (other->next_rarr > st->next_rarr) st->next_rarr = other->next_rarr;
    if (other->next_map > st->next_map) st->next_map = other->next_map;
    if (other->next_value > st->next_value) st->next_value = other->next_value;
    if (other->next_float > st->next_float) st->next_float = other->next_float;
}

static const char *stack_array_name(uint8_t kind) {
    switch (kind) {
    case NVM2C_VK_STR: return "s";
    case NVM2C_VK_ARR: return "a";
    case NVM2C_VK_BARR: case NVM2C_VK_FARR: return "a";
    case NVM2C_VK_SARR: return "sa";
    case NVM2C_VK_REC: return "r";
    case NVM2C_VK_RARR: return "ra";
    case NVM2C_VK_MAP: return "m";
    case NVM2C_VK_VALUE: return "v";
    case NVM2C_VK_FLOAT: return "f";
    default: return "t";
    }
}

static uint8_t **field_table_new(Nvm2cBuf *b, size_t rows) {
    if (!rows) return NULL;
    if (rows > SIZE_MAX / sizeof(uint8_t *) || rows > SIZE_MAX / b->record_width) {
        nvm2c_fail(b, "I cannot represent aggregate field storage");
        return NULL;
    }
    uint8_t **table = malloc(rows * sizeof *table);
    uint8_t *data = calloc(rows, b->record_width);
    if (!table || !data) {
        free(table);
        free(data);
        nvm2c_fail(b, "I cannot allocate aggregate field storage");
        return NULL;
    }
    for (size_t row = 0; row < rows; ++row) table[row] = data + row * b->record_width;
    return table;
}

static void field_table_free(uint8_t **table) {
    if (table) free(table[0]);
    free(table);
}

static int record_join_storage(Nvm2cBuf *b, uint32_t idx, Nvm2cStack *joins, uint8_t *set,
                       size_t tgt, const Nvm2cStack *st) {
    int i;
    if (!set[tgt]) {
        int *slots = st->sp ? malloc((size_t)st->sp * sizeof *slots) : NULL;
        uint8_t *kinds = st->sp ? malloc((size_t)st->sp * sizeof *kinds) : NULL;
        size_t records = (size_t)(st->next_rec > st->next_rarr ? st->next_rec : st->next_rarr);
        uint8_t **fields = field_table_new(b, records);
        uint8_t **array_fields = field_table_new(b, records);
        if ((st->sp && (!slots || !kinds)) || (records && (!fields || !array_fields))) {
            free(slots);
            free(kinds);
            field_table_free(fields);
            field_table_free(array_fields);
            nvm2c_fail(b, "I cannot allocate emitter branch stack");
            return 0;
        }
        if (st->sp) {
            memcpy(slots, st->slots, (size_t)st->sp * sizeof *slots);
            memcpy(kinds, st->kinds, (size_t)st->sp * sizeof *kinds);
        }
        joins[tgt] = *st;
        joins[tgt].slots = slots;
        joins[tgt].kinds = kinds;
        joins[tgt].capacity = (size_t)st->sp;
        joins[tgt].rec_k = fields;
        joins[tgt].rarr_k = array_fields;
        joins[tgt].rec_capacity = records;
        if (records) memcpy(fields[0], st->rec_k[0], records * b->record_width);
        if (records) memcpy(array_fields[0], st->rarr_k[0], records * b->record_width);
        set[tgt] = 1;
        return 1;
    }
    if (joins[tgt].sp != st->sp) {
        nvm2c_fail(b, "function %u: join at %zu has stack height %d, incoming %d",
                   idx, tgt, joins[tgt].sp, st->sp);
        return 0;
    }
    nvm2c_puts(b, "    {\n");
    for (i = 0; i < st->sp; i++) {
        if (joins[tgt].kinds[i] != st->kinds[i]) {
            nvm2c_fail(b, "function %u: join at %zu has a value-kind mismatch", idx, tgt);
            return 0;
        }
        if (joins[tgt].slots[i] == st->slots[i]) continue;
        nvm2c_printf(b, "    %s j%d = %s[%d];\n", c_local_type(st->kinds[i]), i,
                     stack_array_name(st->kinds[i]), st->slots[i]);
    }
    for (i = 0; i < st->sp; i++) {
        if (joins[tgt].slots[i] == st->slots[i]) continue;
        nvm2c_printf(b, "    %s[%d] = j%d;\n", stack_array_name(st->kinds[i]),
                     joins[tgt].slots[i], i);
        if (st->kinds[i] == NVM2C_VK_REC) {
            memcpy(joins[tgt].rec_k[joins[tgt].slots[i]],
                    st->rec_k[st->slots[i]], b->record_width);
        } else if (st->kinds[i] == NVM2C_VK_RARR) {
            memcpy(joins[tgt].rarr_k[joins[tgt].slots[i]],
                   st->rarr_k[st->slots[i]], b->record_width);
        }
    }
    nvm2c_puts(b, "    }\n");
    stack_keep_high_water(&joins[tgt], st);
    return 1;
}

/* I normalize only the taken edge, preserving a conditional's fallthrough
 * stack. Parallel join assignments then retain their existing cycle safety. */
static int record_join(Nvm2cBuf *b, uint32_t idx, Nvm2cStack *joins, uint8_t *set,
                       size_t target, Nvm2cStack *stack) {
    Nvm2cJoinShape *shape = b->join_shapes[idx];
    while (shape && shape->target != target) shape = shape->next;
    if (!shape) return record_join_storage(b, idx, joins, set, target, stack);
    if (shape->count != stack->sp) {
        nvm2c_fail(b, "I found inconsistent classified join height in function %u", idx);
        return 0;
    }
    int needed = 0;
    for (int i = 0; i < stack->sp; ++i)
        if (scalar_kind_tags(stack->kinds[i]) != NVM2C_SCALAR_UNKNOWN &&
            nvm_shape_kind(&b->shapes, shape->shapes[i]) == NVM_SHAPE_OPTIONAL) needed = 1;
    if (!needed) return record_join_storage(b, idx, joins, set, target, stack);
    Nvm2cStack edge = *stack;
    edge.slots = malloc((size_t)edge.sp * sizeof *edge.slots);
    edge.kinds = malloc((size_t)edge.sp * sizeof *edge.kinds);
    if (!edge.slots || !edge.kinds) {
        free(edge.slots); free(edge.kinds);
        nvm2c_fail(b, "I cannot allocate a tagged join edge"); return 0;
    }
    memcpy(edge.slots, stack->slots, (size_t)edge.sp * sizeof *edge.slots);
    memcpy(edge.kinds, stack->kinds, (size_t)edge.sp * sizeof *edge.kinds);
    for (int i = 0; i < edge.sp; ++i) {
        if (scalar_kind_tags(edge.kinds[i]) == NVM2C_SCALAR_UNKNOWN ||
            nvm_shape_kind(&b->shapes, shape->shapes[i]) != NVM_SHAPE_OPTIONAL) continue;
        if ((size_t)edge.next_value >= edge.capacity) {
            nvm2c_fail(b, "I cannot represent another tagged join temporary"); break;
        }
        int value = edge.next_value++;
        if (edge.kinds[i] == NVM2C_VK_STR)
            nvm2c_printf(b, "    v[%d] = (nmap_value){5, 0, (char *)s[%d]};\n", value, edge.slots[i]);
        else if (edge.kinds[i] == NVM2C_VK_FLOAT)
            nvm2c_printf(b, "    v[%d] = nvalue_from_float(f[%d]);\n", value, edge.slots[i]);
        else
            nvm2c_printf(b, "    v[%d] = (nmap_value){%u, t[%d], NULL};\n", value,
                         edge.kinds[i] == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT, edge.slots[i]);
        edge.slots[i] = value; edge.kinds[i] = NVM2C_VK_VALUE;
    }
    int ok = !b->failed && record_join_storage(b, idx, joins, set, target, &edge);
    stack_keep_high_water(stack, &edge);
    free(edge.slots); free(edge.kinds);
    return ok;
}

static void stack_restore_join(Nvm2cBuf *b, Nvm2cStack *st, const Nvm2cStack *join) {
    Nvm2cStack cur = *st;
    *st = *join;
    st->slots = cur.slots;
    st->kinds = cur.kinds;
    st->capacity = cur.capacity;
    st->rec_k = cur.rec_k;
    st->rarr_k = cur.rarr_k;
    st->rec_capacity = cur.rec_capacity;
    memset(st->rec_k[0], 0, st->rec_capacity * b->record_width);
    if (join->rec_capacity)
        memcpy(st->rec_k[0], join->rec_k[0], join->rec_capacity * b->record_width);
    memset(st->rarr_k[0], 0, st->rec_capacity * b->record_width);
    if (join->rec_capacity)
        memcpy(st->rarr_k[0], join->rarr_k[0], join->rec_capacity * b->record_width);
    if (st->sp) {
        memcpy(st->slots, join->slots, (size_t)st->sp * sizeof *st->slots);
        memcpy(st->kinds, join->kinds, (size_t)st->sp * sizeof *st->kinds);
    }
    stack_keep_high_water(st, &cur);
}

static int jump_target(Nvm2cBuf *b, uint32_t idx, size_t start, int32_t rel,
                       size_t remaining, size_t *out) {
    int64_t tgt = (int64_t)start + (int64_t)rel;
    if (tgt < 0 || (uint64_t)tgt > (uint64_t)remaining) {
        nvm2c_fail(b, "function %u: jump at offset %zu is out of range", idx, start);
        return 0;
    }
    *out = (size_t)tgt;
    return 1;
}

static int emit_self_tail_restart(Nvm2cBuf *b, Nvm2cStack *st, uint32_t idx,
                                  const NvmFunctionEntry *fn, const uint8_t *kinds) {
    int args[NVM2C_MAX_LOCALS];
    for (int i = (int)fn->arity - 1; i >= 0; --i) {
        args[i] = stack_pop_expect(b, st, fn_local_kind(b, kinds, idx, (uint16_t)i),
                                  "self TAIL_CALL argument");
        if (b->failed) return 0;
    }
    if (st->sp != 0) {
        nvm2c_fail(b, "self TAIL_CALL leaves extra stack values");
        return 0;
    }
    nvm2c_puts(b, "    {\n");
    for (uint16_t i = 0; i < fn->arity; ++i) {
        uint8_t kind = fn_local_kind(b, kinds, idx, i);
        nvm2c_printf(b, "        %s tc%u = %s[%d];\n", c_local_type(kind),
                     (unsigned)i, stack_array_name(kind), args[i]);
    }
    for (uint16_t i = 0; i < fn->arity; ++i) {
        char local[32];
        local_operand(local, b, kinds, idx, i);
        nvm2c_printf(b, "        %s = tc%u;\n", local, (unsigned)i);
    }
    for (uint16_t i = fn->arity; i < fn->local_count; ++i) {
        uint8_t kind = fn_local_kind(b, kinds, idx, i);
        char local[32];
        local_operand(local, b, kinds, idx, i);
        if (kind == NVM2C_VK_STR)
            nvm2c_printf(b, "        %s = \"\";\n", local);
        else
            nvm2c_printf(b, "        %s = (%s){0};\n", local, c_local_type(kind));
    }
    nvm2c_puts(b, "        goto L_tco;\n    }\n");
    return 1;
}

static int build_direct_call(Nvm2cBuf *b, Nvm2cStack *st, const NvmModule *mod,
                             uint32_t idx, uint32_t callee, const uint8_t *kinds,
                             char *call, size_t call_sz) {
    if (callee >= mod->function_count) {
        nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
        return 0;
    }
    const NvmFunctionEntry *cf = &mod->functions[callee];
    if (c_result_type(b, cf, callee) == NULL) {
        nvm2c_fail(b, "function %u: CALL target %u has an unsupported result", idx, callee);
        return 0;
    }
    int args[NVM2C_MAX_LOCALS];
    uint8_t argk[NVM2C_MAX_LOCALS];
    int i;
    for (i = (int)cf->arity - 1; i >= 0; i--) {
        uint8_t pk = fn_local_kind(b, kinds, callee, (uint16_t)i);
        argk[i] = pk;
        args[i] = stack_pop_expect(b, st, pk, "CALL argument");
        if (b->failed) return 0;
    }
    char cname[64];
    fn_c_name(mod, callee, cname, sizeof cname);
    size_t pos = 0;
    int written = snprintf(call, call_sz, "%s(", cname);
    if (written < 0 || (size_t)written >= call_sz) {
        nvm2c_fail(b, "I cannot fit the native call name"); return 0;
    }
    pos = (size_t)written;
    for (uint16_t a = 0; a < cf->arity; a++) {
        written = snprintf(call + pos, call_sz - pos, "%s%s[%d]",
                           a ? ", " : "", stack_array_name(argk[a]), args[a]);
        if (written < 0 || (size_t)written >= call_sz - pos) {
            nvm2c_fail(b, "function %u: CALL argument list overflow", idx);
            return 0;
        }
        pos += (size_t)written;
    }
    if (call_sz - pos < 2) {
        nvm2c_fail(b, "I cannot close the native call argument list"); return 0;
    }
    call[pos++] = ')'; call[pos] = '\0';
    return 1;
}

static int scalar_return_profile(const NvmFunctionEntry *fn) {
    return fn->result_count == 0 || (fn->result_count == 1 &&
        (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL || (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM) || fn->result_tag == TAG_FLOAT));
}

static int emit_scalar_return(Nvm2cBuf *b, Nvm2cStack *st,
                              const NvmFunctionEntry *fn, uint32_t idx) {
    if (st->sp != fn->result_count) {
        nvm2c_fail(b, "function %u: return leaves %d values, expected %u", idx, st->sp, fn->result_count);
        return 0;
    }
    if (fn->result_count && (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM)) {
        int slot = stack_pop_expect(b, st, NVM2C_VK_VALUE, "RET");
        if (b->failed) return 0;
        nvm2c_printf(b, "    if (v[%d].kind != %u) NVM2C_ABORT();\n    nresult = v[%d];\n", slot, fn->result_tag, slot);
    } else if (fn->result_count) {
        uint8_t kind = fn->result_tag == TAG_FLOAT ? NVM2C_VK_FLOAT :
                       fn->result_tag == TAG_BOOL ? NVM2C_VK_BOOL : NVM2C_VK_INT;
        int slot = stack_pop_expect(b, st, kind, "RET");
        if (b->failed) return 0;
        nvm2c_printf(b, "    nresult = %c[%d];\n", kind == NVM2C_VK_FLOAT ? 'f' : 't', slot);
    }
    nvm2c_puts(b, "    goto L_return;\n");
    return 1;
}

typedef struct {
    size_t offset;
    size_t length;
} Nvm2cLabelSpan;

static void emit_pc_label(Nvm2cBuf *b, size_t pc, Nvm2cLabelSpan *labels) {
    labels[pc].offset = b->len;
    nvm2c_printf(b, "L_%zu:", pc);
    labels[pc].length = b->len - labels[pc].offset;
    nvm2c_puts(b, " ;\n");
}

static void emit_function_body(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                               const uint8_t *kinds, const uint8_t *rec_fields,
                               const uint8_t *result_fields) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(b, fn, idx);
    if (!rt || b->failed) return;

    char name[64];
    uint16_t i;
    fn_c_name(mod, idx, name, sizeof name);
    nvm2c_printf(b, "static %s %s(", rt, name);
    if (fn->arity == 0) {
        nvm2c_puts(b, "void");
    } else {
        for (i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(b, kinds, idx, i)), (unsigned)i);
        }
    }
    nvm2c_puts(b, ") {\n    (void)ni64_from_bits;\n");

    unsigned record_locals = 0;
    for (i = 0; i < fn->local_count; i++)
        if (fn_local_kind(b, kinds, idx, i) == NVM2C_VK_REC) ++record_locals;
    if (record_locals) {
        nvm2c_printf(b, "    nrec_t *rl = calloc(%u, sizeof *rl);\n"
                       "    if (!rl) NVM2C_ABORT();\n", record_locals);
    }
    for (i = 0; i < fn->local_count; i++) {
        uint8_t lk = fn_local_kind(b, kinds, idx, i);
        if (lk == NVM2C_VK_REC) {
            if (i < fn->arity) {
                char local[32];
                local_operand(local, b, kinds, idx, i);
                nvm2c_printf(b, "    %s = a%u;\n", local, (unsigned)i);
            }
            continue;
        }
        if (i < fn->arity) {
            nvm2c_printf(b, "    %s l%u = a%u;\n", c_local_type(lk), (unsigned)i, (unsigned)i);
        } else if (lk == NVM2C_VK_STR) {
            nvm2c_printf(b, "    const char *l%u = \"\";\n", (unsigned)i);
        } else if (lk == NVM2C_VK_FLOAT) {
            nvm2c_printf(b, "    double l%u = 0.0;\n", (unsigned)i);
        } else if (word_array_storage(lk)) {
            nvm2c_printf(b, "    narr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_SARR) {
            nvm2c_printf(b, "    nsarr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_RARR) {
            nvm2c_printf(b, "    nrarr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_MAP) {
            nvm2c_printf(b, "    nmap_t l%u = NULL;\n", (unsigned)i);
        } else if (lk == NVM2C_VK_VALUE) {
            nvm2c_printf(b, "    nmap_value l%u = {0};\n", (unsigned)i);
        } else {
            nvm2c_printf(b, "    int64_t l%u = 0;\n", (unsigned)i);
        }
        nvm2c_printf(b, "    (void)l%u;\n", (unsigned)i);
    }
    size_t declarations_at = b->len;

    if (fn->code_offset > mod->code_size ||
        fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "function %u: code range is outside the module", idx);
        return;
    }

    const uint8_t *code = mod->code + fn->code_offset;
    size_t remaining = fn->code_length;
    if (remaining >= INT_MAX || remaining > SIZE_MAX / sizeof(Nvm2cStack) - 1 ||
        remaining > SIZE_MAX / sizeof(int) - 1 ||
        remaining > SIZE_MAX / sizeof(Nvm2cLabelSpan) - 1 ||
        remaining > SIZE_MAX / b->record_width - 1) {
        nvm2c_fail(b, "I cannot represent this function's emitter stack");
        return;
    }
    Nvm2cStack st = {0};
    st.capacity = remaining + 1;
    st.slots = malloc(st.capacity * sizeof *st.slots);
    st.kinds = malloc(st.capacity * sizeof *st.kinds);
    st.rec_capacity = st.capacity;
    st.rec_k = field_table_new(b, st.rec_capacity);
    st.rarr_k = field_table_new(b, st.rec_capacity);
    int *literal_elems = malloc(st.capacity * sizeof *literal_elems);
    uint8_t *aggregate_kinds = malloc(b->record_width);
    uint8_t *is_start = calloc(remaining + 1, 1);
    uint8_t *is_target = calloc(remaining + 1, 1);
    uint8_t *referenced = calloc(remaining + 1, 1);
    Nvm2cLabelSpan *labels = calloc(remaining + 1, sizeof *labels);
    Nvm2cStack *joins = NULL;
    uint8_t *join_set = NULL;
    if (!st.slots || !st.kinds || !st.rec_k || !st.rarr_k || !literal_elems || !aggregate_kinds || !is_start || !is_target || !referenced || !labels) {
        nvm2c_fail(b, "out of memory");
        goto done;
    }

    size_t scan = 0;
    int has_self_tail = 0;
    while (scan < remaining) {
        is_start[scan] = 1;
        DecodedInstruction look;
        uint32_t n = isa_decode(code + scan, remaining - scan, &look);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, scan);
            goto done;
        }
        if (look.opcode == OP_TAIL_CALL && look.operands[0].u32 == idx)
            has_self_tail = 1;
        if (look.opcode == OP_JMP || look.opcode == OP_JMP_FALSE || look.opcode == OP_JMP_TRUE) {
            size_t tgt = 0;
            if (!jump_target(b, idx, scan, look.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            is_target[tgt] = 1;
        }
        scan += n;
    }
    is_start[remaining] = 1;
    {
        size_t off;
        for (off = 0; off <= remaining; off++) {
            if (is_target[off] && !is_start[off]) {
                nvm2c_fail(b, "function %u: jump targets a non-instruction boundary at %zu",
                           idx, off);
                goto done;
            }
        }
    }

    joins = calloc(remaining + 1, sizeof(Nvm2cStack));
    join_set = calloc(remaining + 1, 1);
    if (!joins || !join_set) {
        nvm2c_fail(b, "out of memory");
        goto done;
    }

    size_t pc = 0;
    int terminated = 0;
    int previous_false_push = 0;
    /* A decoded self-tail instruction may be unreachable. Keep its label
     * syntactically referenced without executing an extra jump. */
    nvm2c_puts(b, "    if (0) goto L_return;\n");
    if (has_self_tail) nvm2c_puts(b, "    if (0) goto L_tco;\nL_tco: ;\n");
    if (b->has_maps && has_self_tail) {
        emit_map_roots(b, &st, fn, kinds, idx);
        nvm2c_puts(b, "    nmap_collect_if_needed();\n");
    }

    while (pc < remaining) {
        size_t start = pc;
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            goto done;
        }
        if (terminated && !join_set[start]) {
            previous_false_push = 0;
            pc += n;
            continue;
        }
        if (is_target[start]) {
            previous_false_push = 0;
            if (terminated) {
                if (!join_set[start]) {
                    nvm2c_fail(b, "function %u: label at %zu has no incoming stack",
                               idx, start);
                    goto done;
                }
                stack_restore_join(b, &st, &joins[start]);
                terminated = 0;
                emit_pc_label(b, start, labels);
            } else {
                if (!record_join(b, idx, joins, join_set, start, &st)) goto done;
                stack_restore_join(b, &st, &joins[start]);
                emit_pc_label(b, start, labels);
            }
        }
        pc += n;

        if (!prepare_typed_integer_pair(b, &st, ins.opcode)) goto done;
        switch (ins.opcode) {
        case OP_NOP:
            break;
        case OP_PUSH_I64: {
            char rhs[32];
            if (ins.operands[0].i64 == INT64_MIN)
                snprintf(rhs, sizeof rhs, "(-9223372036854775807LL - 1LL)");
            else snprintf(rhs, sizeof rhs, "%lldLL", (long long)ins.operands[0].i64);
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_PUSH_VOID:
            stack_push_value(b, &st, "(nmap_value){0, 0, NULL}");
            break;
        case OP_PUSH_F64: {
            char rhs[80];
            uint64_t bits;
            memcpy(&bits, &ins.operands[0].f64, sizeof bits);
            snprintf(rhs, sizeof rhs, "nf64_from_bits(UINT64_C(0x%016llx))",
                     (unsigned long long)bits);
            stack_push_float(b, &st, rhs);
            break;
        }
        case OP_ENUM_VAL: {
            char value[64];
            snprintf(value, sizeof value, "(nmap_value){9, %u, NULL}", ins.operands[1].u16);
            stack_push_value(b, &st, value);
            break;
        }
        case OP_PUSH_U8: {
            char value[64];
            snprintf(value, sizeof value, "(nmap_value){2, %u, NULL}", ins.operands[0].u8);
            stack_push_value(b, &st, value);
            break;
        }
        case OP_PUSH_BOOL: {
            char rhs[8];
            snprintf(rhs, sizeof rhs, "%dLL", ins.operands[0].u8 ? 1 : 0);
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_PUSH_STR: {
            uint32_t sidx = ins.operands[0].u32;
            const char *lit = nvm_get_string(mod, sidx);
            uint32_t slen = nvm_get_string_len(mod, sidx);
            if (!lit) {
                nvm2c_fail(b, "function %u: PUSH_STR string index %u is out of range", idx, sidx);
                goto done;
            }
            if ((size_t)st.sp >= st.capacity) {
                nvm2c_fail(b, "operand stack overflow");
                goto done;
            }
            if ((size_t)st.next_str >= st.capacity) {
                nvm2c_fail(b, "too many string temporaries");
                goto done;
            }
            {
                int slot = st.next_str++;
                nvm2c_printf(b, "    s[%d] = ", slot);
                emit_c_string_lit(b, lit, slen);
                if (b->failed) goto done;
                nvm2c_puts(b, ";\n");
                st.slots[st.sp] = slot;
                st.kinds[st.sp] = NVM2C_VK_STR;
                st.sp++;
            }
            break;
        }
        case OP_DUP: {
            if (st.sp <= 0) {
                nvm2c_fail(b, "function %u: DUP on empty stack", idx);
                goto done;
            }
            {
                int src = st.slots[st.sp - 1];
                uint8_t k = st.kinds[st.sp - 1];
                char rhs[32];
                if (k == NVM2C_VK_STR) {
                    snprintf(rhs, sizeof rhs, "s[%d]", src);
                    stack_push_str(b, &st, rhs);
                } else if (k == NVM2C_VK_FLOAT) {
                    snprintf(rhs, sizeof rhs, "f[%d]", src);
                    stack_push_float(b, &st, rhs);
                } else if (word_array_storage(k)) {
                    snprintf(rhs, sizeof rhs, "a[%d]", src);
                    stack_push_iarray(b, &st, rhs, k);
                } else if (k == NVM2C_VK_SARR) {
                    snprintf(rhs, sizeof rhs, "sa[%d]", src);
                    stack_push_sarr(b, &st, rhs);
                } else if (k == NVM2C_VK_REC) {
                    snprintf(rhs, sizeof rhs, "r[%d]", src);
                    {
                        int nr = stack_push_rec(b, &st, rhs);
                        if (nr >= 0) {
                            memcpy(st.rec_k[nr], st.rec_k[src], b->record_width);
                        }
                    }
                } else if (k == NVM2C_VK_RARR) {
                    snprintf(rhs, sizeof rhs, "ra[%d]", src);
                    {
                        int nr = stack_push_rarr(b, &st, rhs);
                        if (nr >= 0) memcpy(st.rarr_k[nr], st.rarr_k[src], b->record_width);
                    }
                } else if (k == NVM2C_VK_MAP) {
                    snprintf(rhs, sizeof rhs, "m[%d]", src);
                    stack_push_map(b, &st, rhs);
                } else if (k == NVM2C_VK_VALUE) {
                    snprintf(rhs, sizeof rhs, "v[%d]", src);
                    stack_push_value(b, &st, rhs);
                } else {
                    snprintf(rhs, sizeof rhs, "t[%d]", src);
                    if (k == NVM2C_VK_BOOL) stack_push_bool(b, &st, rhs);
                    else stack_push_temp(b, &st, rhs);
                }
            }
            break;
        }
        case OP_POP:
            (void)stack_pop(b, &st);
            break;
        case OP_PRINT:
        case OP_PRINTLN: {
            uint8_t k = NVM2C_VK_INT;
            int slot = stack_pop_kind(b, &st, &k);
            int nl = (ins.opcode == OP_PRINTLN);
            if (b->failed) goto done;
            if (k == NVM2C_VK_INT) {
                if (nl) {
                    nvm2c_printf(b, "    printf(\"%%lld\\n\", (long long)t[%d]);\n", slot);
                } else {
                    nvm2c_printf(b, "    printf(\"%%lld\", (long long)t[%d]);\n", slot);
                }
            } else if (k == NVM2C_VK_FLOAT) {
                nvm2c_printf(b, "    nf64_print(f[%d]);\n", slot);
                if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            } else if (k == NVM2C_VK_VALUE) {
                nvm2c_printf(b, "    if (v[%d].kind == 1 || v[%d].kind == 2) printf(\"%%lld\", (long long)v[%d].integer);\n", slot, slot, slot);
                nvm2c_printf(b, "    else if (v[%d].kind == 3) nf64_print(nvalue_require_float(v[%d]));\n", slot, slot);
                nvm2c_printf(b, "    else if (v[%d].kind == 4) fputs(v[%d].integer ? \"true\" : \"false\", stdout);\n", slot, slot);
                nvm2c_printf(b, "    else if (v[%d].kind == 5) fputs(v[%d].text, stdout);\n", slot, slot);
                nvm2c_printf(b, "    else if (v[%d].kind == 7) nvalue_array_print(v[%d]);\n", slot, slot);
                nvm2c_printf(b, "    else if (v[%d].kind == 9) printf(\"enum(%%lld)\", (long long)v[%d].integer);\n", slot, slot);
                nvm2c_puts(b, "    else fputs(\"void\", stdout);\n");
                if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            } else if (k == NVM2C_VK_BOOL) {
                nvm2c_printf(b, "    fputs(t[%d] ? \"true\" : \"false\", stdout);\n", slot);
                if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            } else if (k == NVM2C_VK_STR) {
                nvm2c_printf(b, "    fputs(s[%d] ? s[%d] : \"\", stdout);\n", slot, slot);
                if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            } else {
                nvm2c_fail(b, "function %u: PRINT of arrays and records is refused", idx);
                goto done;
            }
            nvm2c_puts(b, "    fflush(stdout);\n");
            break;
        }
        case OP_ASSERT: {
            int cond = stack_pop_condition(b, &st, "ASSERT");
            if (b->failed) goto done;
            nvm2c_printf(b, "    if (!t[%d]) NVM2C_ABORT();\n", cond);
            if (previous_false_push) terminated = 1;
            break;
        }
        case OP_SWAP: {
            uint8_t kx = 0, ky = 0;
            int x = stack_pop_kind(b, &st, &kx);
            int y = stack_pop_kind(b, &st, &ky);
            if (b->failed) goto done;
            st.slots[st.sp] = x;
            st.kinds[st.sp] = kx;
            st.sp++;
            st.slots[st.sp] = y;
            st.kinds[st.sp] = ky;
            st.sp++;
            break;
        }
        case OP_ROT3: {
            uint8_t kt = 0, km = 0, kb = 0;
            int top = stack_pop_kind(b, &st, &kt);
            int middle = stack_pop_kind(b, &st, &km);
            int bottom = stack_pop_kind(b, &st, &kb);
            if (b->failed) goto done;
            if ((kt != NVM2C_VK_INT && kt != NVM2C_VK_BOOL) ||
                (km != NVM2C_VK_INT && km != NVM2C_VK_BOOL) ||
                (kb != NVM2C_VK_INT && kb != NVM2C_VK_BOOL)) {
                nvm2c_fail(b, "I require exact int/bool operands for native ROT3");
                goto done;
            }
            st.slots[st.sp] = top; st.kinds[st.sp++] = kt;
            st.slots[st.sp] = bottom; st.kinds[st.sp++] = kb;
            st.slots[st.sp] = middle; st.kinds[st.sp++] = km;
            break;
        }
        case OP_LOAD_GLOBAL: {
            uint32_t slot = ins.operands[0].u32;
            uint8_t kind = resolved_shape_kind(b, b->shape_globals[slot]);
            if (kind == NVM2C_VK_RARR) {
                nvm2c_printf(b, "    if (nglobal[%u].kind != %u || nglobal[%u].integer != %u || !nglobal[%u].text) NVM2C_ABORT();\n",
                             slot, TAG_ARRAY, slot, NVM2C_VK_RARR, slot);
                char expression[64];
                snprintf(expression, sizeof expression, "(nrarr_t)nglobal[%u].text", slot);
                int array = stack_push_rarr(b, &st, expression);
                if (array >= 0) {
                    NvmShapeId element = nvm_shape_lookup(&b->shapes, b->shape_globals[slot], 0);
                    for (size_t f = 0; f < b->record_width; ++f) {
                        NvmShapeId field = element ? nvm_shape_lookup(&b->shapes, element, (uint32_t)f) : 0;
                        st.rarr_k[array][f] = resolved_shape_kind(b, field);
                    }
                    if (!shape_ok(b)) goto done;
                }
            } else {
                char expression[64];
                snprintf(expression, sizeof expression, "nglobal[%u]", slot);
                stack_push_value(b, &st, expression);
            }
            break;
        }
        case OP_STORE_GLOBAL: {
            uint8_t kind;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            unsigned slot = ins.operands[0].u32;
            if (kind == NVM2C_VK_VALUE)
                nvm2c_printf(b, "    nglobal[%u] = v[%d];\n", slot, value);
            else if (kind == NVM2C_VK_FLOAT)
                nvm2c_printf(b, "    nglobal[%u] = nvalue_from_float(f[%d]);\n", slot, value);
            else if (kind == NVM2C_VK_MAP)
                nvm2c_printf(b, "    nglobal[%u] = (nmap_value){13, 0, (char *)m[%d]};\n", slot, value);
            else if (kind == NVM2C_VK_STR)
                nvm2c_printf(b, "    nglobal[%u] = (nmap_value){5, 0, (char *)s[%d]};\n", slot, value);
            else if (kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL)
                nvm2c_printf(b, "    nglobal[%u] = (nmap_value){%u, t[%d], NULL};\n", slot,
                             kind == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT, value);
            else if (word_array_storage(kind) || kind == NVM2C_VK_SARR)
                nvm2c_printf(b, "    nglobal[%u] = (nmap_value){7, %u, (char *)%s[%d]};\n",
                             slot, kind, stack_array_name(kind), value);
            else if (kind == NVM2C_VK_RARR)
                nvm2c_printf(b, "    nglobal[%u] = (nmap_value){%u, %u, (char *)ra[%d]};\n",
                             slot, TAG_ARRAY, NVM2C_VK_RARR, value);
            else { nvm2c_fail(b, "I cannot yet emit an aggregate global store"); goto done; }
            break;
        }
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                goto done;
            }
            char rhs[32];
            local_operand(rhs, b, kinds, idx, slot);
            if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_STR) {
                stack_push_str(b, &st, rhs);
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_FLOAT) {
                stack_push_float(b, &st, rhs);
            } else if (word_array_storage(fn_local_kind(b, kinds, idx, slot))) {
                stack_push_iarray(b, &st, rhs, fn_local_kind(b, kinds, idx, slot));
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_SARR) {
                stack_push_sarr(b, &st, rhs);
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_REC) {
                int r = stack_push_rec(b, &st, rhs);
                if (r >= 0) {
                    memcpy(st.rec_k[r], fn_rec_k_const(b, rec_fields, idx, slot),
                           b->record_width);
                }
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_MAP) {
                stack_push_map(b, &st, rhs);
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_VALUE) {
                stack_push_value(b, &st, rhs);
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_BOOL) {
                stack_push_bool(b, &st, rhs);
            } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_RARR) {
                int a = stack_push_rarr(b, &st, rhs);
                if (a >= 0) {
                    memcpy(st.rarr_k[a], fn_rec_k_const(b, rec_fields, idx, slot),
                           b->record_width);
                }
            } else {
                stack_push_temp(b, &st, rhs);
            }
            break;
        }
        case OP_STORE_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: STORE_LOCAL %u out of range", idx, slot);
                goto done;
            }
            {
                uint8_t expect = fn_local_kind(b, kinds, idx, slot);
                int t = stack_pop_expect(b, &st, expect, "STORE_LOCAL");
                if (b->failed) goto done;
                if (expect == NVM2C_VK_STR) {
                    nvm2c_printf(b, "    l%u = s[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_FLOAT) {
                    nvm2c_printf(b, "    l%u = f[%d];\n", (unsigned)slot, t);
                } else if (word_array_storage(expect)) {
                    nvm2c_printf(b, "    l%u = a[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    l%u = sa[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_REC) {
                    char local[32];
                    local_operand(local, b, kinds, idx, slot);
                    nvm2c_printf(b, "    %s = r[%d];\n", local, t);
                } else if (expect == NVM2C_VK_RARR) {
                    nvm2c_printf(b, "    l%u = ra[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_MAP) {
                    nvm2c_printf(b, "    l%u = m[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_VALUE) {
                    nvm2c_printf(b, "    l%u = v[%d];\n", (unsigned)slot, t);
                } else {
                    nvm2c_printf(b, "    l%u = t[%d];\n", (unsigned)slot, t);
                }
            }
            break;
        }
        case OP_F64_FROM_BITS: case OP_F64_TO_BITS: {
            int from = ins.opcode == OP_F64_FROM_BITS;
            int value = stack_pop_expect(b, &st, from ? NVM2C_VK_INT : NVM2C_VK_FLOAT, "binary64 bit transport");
            if (b->failed) goto done;
            char expression[80];
            snprintf(expression, sizeof expression, "nl_float_%s_bits(%s[%d])", from ? "from" : "to", from ? "t" : "f", value);
            if (from) stack_push_float(b, &st, expression);
            else stack_push_temp(b, &st, expression);
            break;
        }
        case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
        case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE:
        case OP_F64_GT: case OP_F64_GE: case OP_F64_NEG: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_FLOAT, "typed F64 operand");
            int lhs = ins.opcode == OP_F64_NEG ? -1 :
                stack_pop_expect(b, &st, NVM2C_VK_FLOAT, "typed F64 operand");
            if (b->failed) goto done;
            char expression[128];
            if (ins.opcode == OP_F64_NEG)
                snprintf(expression, sizeof expression, "-f[%d]", rhs);
            else if (ins.opcode == OP_F64_ADD || ins.opcode == OP_F64_SUB ||
                     ins.opcode == OP_F64_MUL || ins.opcode == OP_F64_DIV)
                snprintf(expression, sizeof expression, "nano_rt_f64_%s(f[%d], f[%d])",
                         ins.opcode == OP_F64_ADD ? "add" : ins.opcode == OP_F64_SUB ? "sub" :
                         ins.opcode == OP_F64_MUL ? "mul" : "div", lhs, rhs);
            else {
                const char *op = ins.opcode == OP_F64_ADD ? "+" : ins.opcode == OP_F64_SUB ? "-" :
                    ins.opcode == OP_F64_MUL ? "*" : ins.opcode == OP_F64_EQ ? "==" :
                    ins.opcode == OP_F64_NE ? "!=" : ins.opcode == OP_F64_LT ? "<" :
                    ins.opcode == OP_F64_LE ? "<=" : ins.opcode == OP_F64_GT ? ">" : ">=";
                snprintf(expression, sizeof expression, "f[%d] %s f[%d]", lhs, op, rhs);
            }
            if (boolean_result(ins.opcode)) stack_push_bool(b, &st, expression);
            else stack_push_float(b, &st, expression);
            break;
        }
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: {
            if (st.sp >= 2 && (st.kinds[st.sp - 1] == NVM2C_VK_VALUE ||
                              st.kinds[st.sp - 2] == NVM2C_VK_VALUE)) {
                uint8_t rk, lk;
                int rhs = stack_pop_kind(b, &st, &rk);
                int lhs = stack_pop_kind(b, &st, &lk);
                if ((lk != NVM2C_VK_VALUE && lk != NVM2C_VK_INT && lk != NVM2C_VK_FLOAT) ||
                    (rk != NVM2C_VK_VALUE && rk != NVM2C_VK_INT && rk != NVM2C_VK_FLOAT)) {
                    nvm2c_fail(b, "I require numeric or tagged operands for generic arithmetic");
                    goto done;
                }
                char left[96], right[96], expression[256];
                scalar_value_expression(b, left, sizeof left, lk, lhs);
                scalar_value_expression(b, right, sizeof right, rk, rhs);
                snprintf(expression, sizeof expression, "nvalue_numeric(%s, %s, '%c')", left, right,
                         ins.opcode == OP_ADD ? '+' : ins.opcode == OP_SUB ? '-' : ins.opcode == OP_MUL ? '*' : '/');
                stack_push_value(b, &st, expression);
                break;
            }
            if (st.sp >= 2 && (st.kinds[st.sp - 1] == NVM2C_VK_FLOAT ||
                              st.kinds[st.sp - 2] == NVM2C_VK_FLOAT)) {
                uint8_t rk, lk;
                int rhs = stack_pop_kind(b, &st, &rk);
                int lhs = stack_pop_kind(b, &st, &lk);
                if ((lk != NVM2C_VK_INT && lk != NVM2C_VK_FLOAT) ||
                    (rk != NVM2C_VK_INT && rk != NVM2C_VK_FLOAT)) {
                    nvm2c_fail(b, "I require known int or float operands for generic numeric promotion");
                    goto done;
                }
                char left[48], right[48], expression[192];
                snprintf(left, sizeof left, lk == NVM2C_VK_FLOAT ? "f[%d]" : "(double)t[%d]", lhs);
                snprintf(right, sizeof right, rk == NVM2C_VK_FLOAT ? "f[%d]" : "(double)t[%d]", rhs);
                snprintf(expression, sizeof expression, "nano_rt_f64_%s(%s, %s)",
                    ins.opcode == OP_ADD ? "add" : ins.opcode == OP_SUB ? "sub" :
                    ins.opcode == OP_MUL ? "mul" : "div", left, right);
                stack_push_float(b, &st, expression);
                break;
            }
            /* I retain the existing checked-int route for tagged operands. */
            if (ins.opcode == OP_ADD) emit_binop(b, &st, "+");
            else if (ins.opcode == OP_SUB) emit_binop(b, &st, "-");
            else if (ins.opcode == OP_MUL) emit_binop(b, &st, "*");
            else goto emit_integer_division;
            break;
        }
        case OP_I64_ADD_CARRY: case OP_I64_SUB_BORROW:
        case OP_I64_MUL_WIDE_S: case OP_I64_MUL_WIDE_U: {
            int carry = -1;
            if (ins.opcode == OP_I64_ADD_CARRY || ins.opcode == OP_I64_SUB_BORROW)
                carry = stack_pop_expect(b, &st, NVM2C_VK_INT, "integer pair carry");
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "integer pair rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "integer pair lhs");
            if (b->failed) goto done;
            unsigned operation = ins.opcode == OP_I64_ADD_CARRY ? 0 :
                ins.opcode == OP_I64_SUB_BORROW ? 1 : ins.opcode == OP_I64_MUL_WIDE_S ? 2 : 3;
            char carry_value[48];
            if (carry >= 0) snprintf(carry_value, sizeof carry_value, "t[%d]", carry);
            else snprintf(carry_value, sizeof carry_value, "INT64_C(0)");
            nvm2c_printf(b, "    { ni64_pair pair = ni64_pair_compute(t[%d], t[%d], %s, %u);\n",
                         lhs, rhs, carry_value, operation);
            stack_push_temp(b, &st, "pair.low");
            stack_push_temp(b, &st, "pair.high");
            nvm2c_puts(b, "    }\n");
            break;
        }
        case OP_I64_ADD:
            emit_binop(b, &st, "+");
            break;
        case OP_I64_SUB:
            emit_binop(b, &st, "-");
            break;
        case OP_I64_MUL:
            emit_binop(b, &st, "*");
            break;
        case OP_I64_DIV_S:
        emit_integer_division: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div lhs");
            if (b->failed) goto done;
            char expr[192];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? INT64_C(0) : (t[%d] == INT64_MIN && t[%d] == -1) ? INT64_MIN : t[%d] / t[%d])",
                     rhs, lhs, rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_MOD:
        case OP_I64_REM_S: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod lhs");
            if (b->failed) goto done;
            char expr[192];
            snprintf(expr, sizeof expr, "(t[%d] == 0 || (t[%d] == INT64_MIN && t[%d] == -1) ? INT64_C(0) : t[%d] %% t[%d])",
                     rhs, lhs, rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_NEG:
            if (st.sp > 0 && st.kinds[st.sp - 1] == NVM2C_VK_VALUE) {
                int value = stack_pop(b, &st);
                char expression[96];
                snprintf(expression, sizeof expression, "nvalue_numeric(v[%d], (nmap_value){0}, '~')", value);
                stack_push_value(b, &st, expression);
                break;
            }
            if (st.sp > 0 && st.kinds[st.sp - 1] == NVM2C_VK_FLOAT) {
                int value = stack_pop_expect(b, &st, NVM2C_VK_FLOAT, "generic NEG");
                char expression[48];
                snprintf(expression, sizeof expression, "-f[%d]", value);
                stack_push_float(b, &st, expression);
                break;
            }
            emit_unop(b, &st, "-");
            break;
        case OP_I64_NEG:
            emit_unop(b, &st, "-");
            break;
        case OP_BOOL_NOT:
            emit_unop(b, &st, "!");
            break;
        case OP_I64_EQ:
            emit_binop(b, &st, "==");
            break;
        case OP_EQ:
        case OP_NE: {
            uint8_t rk = NVM2C_VK_INT;
            uint8_t lk = NVM2C_VK_INT;
            int rhs = stack_pop_kind(b, &st, &rk);
            int lhs = stack_pop_kind(b, &st, &lk);
            if (b->failed) goto done;
            const char *op = ins.opcode == OP_EQ ? "==" :
                ins.opcode == OP_NE ? "!=" : ins.opcode == OP_LT ? "<" :
                ins.opcode == OP_LE ? "<=" : ins.opcode == OP_GT ? ">" : ">=";
            if ((lk == NVM2C_VK_VALUE || rk == NVM2C_VK_VALUE) &&
                (ins.opcode == OP_EQ || ins.opcode == OP_NE)) {
                char left[96], right[96], expression[256];
                uint8_t kinds_pair[2] = {lk, rk};
                int slots_pair[2] = {lhs, rhs};
                char *expressions[2] = {left, right};
                for (int i = 0; i < 2; ++i) {
                    if (kinds_pair[i] == NVM2C_VK_VALUE)
                        snprintf(expressions[i], 96, "v[%d]", slots_pair[i]);
                    else if (kinds_pair[i] == NVM2C_VK_FLOAT)
                        snprintf(expressions[i], 96, "nvalue_from_float(f[%d])", slots_pair[i]);
                    else if (kinds_pair[i] == NVM2C_VK_STR)
                        snprintf(expressions[i], 96, "(nmap_value){5, 0, (char *)s[%d]}", slots_pair[i]);
                    else if (kinds_pair[i] == NVM2C_VK_INT || kinds_pair[i] == NVM2C_VK_BOOL)
                        snprintf(expressions[i], 96, "(nmap_value){%u, t[%d], NULL}",
                                 kinds_pair[i] == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT, slots_pair[i]);
                    else { nvm2c_fail(b, "I require preserved runtime tags to compare this value with a tagged lookup"); goto done; }
                }
                snprintf(expression, sizeof expression, "%snvalue_equal(%s, %s)",
                          ins.opcode == OP_NE ? "!" : "", left, right);
                stack_push_temp(b, &st, expression);
            } else if ((lk == NVM2C_VK_INT || lk == NVM2C_VK_BOOL) && lk == rk) {
                char expr[64];
                snprintf(expr, sizeof expr, "t[%d] %s t[%d]", lhs, op, rhs);
                stack_push_temp(b, &st, expr);
            } else if (lk == NVM2C_VK_FLOAT && rk == NVM2C_VK_FLOAT) {
                char expr[64];
                snprintf(expr, sizeof expr, "f[%d] %s f[%d]", lhs, op, rhs);
                stack_push_temp(b, &st, expr);
            } else if (lk == NVM2C_VK_INT && rk == NVM2C_VK_FLOAT) {
                char expr[80];
                snprintf(expr, sizeof expr, "(double)t[%d] %s f[%d]", lhs, op, rhs);
                stack_push_temp(b, &st, expr);
            } else if (lk == NVM2C_VK_FLOAT && rk == NVM2C_VK_INT) {
                char expr[80];
                snprintf(expr, sizeof expr, "f[%d] %s (double)t[%d]", lhs, op, rhs);
                stack_push_temp(b, &st, expr);
            } else if ((lk == NVM2C_VK_BOOL && rk == NVM2C_VK_FLOAT) ||
                       (lk == NVM2C_VK_FLOAT && rk == NVM2C_VK_BOOL)) {
                /* I preserve unequal scalar tags, without numeric promotion. */
                stack_push_temp(b, &st, ins.opcode == OP_EQ ? "0" : "1");
            } else if (lk == NVM2C_VK_STR && rk == NVM2C_VK_STR) {
                char expr[192];
                snprintf(expr, sizeof expr,
                         "(int64_t)(strcmp(s[%d] ? s[%d] : \"\", s[%d] ? s[%d] : \"\") %s 0)",
                         lhs, lhs, rhs, rhs, op);
                stack_push_temp(b, &st, expr);
            } else if ((ins.opcode == OP_EQ || ins.opcode == OP_NE) &&
                       (lk == NVM2C_VK_INT || lk == NVM2C_VK_BOOL || lk == NVM2C_VK_STR) &&
                       (rk == NVM2C_VK_INT || rk == NVM2C_VK_BOOL || rk == NVM2C_VK_STR)) {
                stack_push_temp(b, &st, ins.opcode == OP_EQ ? "0" : "1");
            } else {
                nvm2c_fail(b, "function %u: %s has incompatible operand types",
                           idx, isa_get_info(ins.opcode)->name);
                goto done;
            }
            break;
        }
        case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            uint8_t rk, lk;
            int rhs = stack_pop_kind(b, &st, &rk);
            int lhs = stack_pop_kind(b, &st, &lk);
            if (b->failed) goto done;
            if ((lk == NVM2C_VK_FLOAT || rk == NVM2C_VK_FLOAT) &&
                (lk == NVM2C_VK_FLOAT || lk == NVM2C_VK_INT) &&
                (rk == NVM2C_VK_FLOAT || rk == NVM2C_VK_INT)) {
                char left[48], right[48], expression[256];
                snprintf(left, sizeof left, "(double)%s[%d]", stack_array_name(lk), lhs);
                snprintf(right, sizeof right, "(double)%s[%d]", stack_array_name(rk), rhs);
                const char *op = ins.opcode == OP_LT ? "<" : ins.opcode == OP_LE ? "<=" :
                                 ins.opcode == OP_GT ? ">" : ">=";
                /* I preserve val_compare's unordered-NaN result of zero. */
                snprintf(expression, sizeof expression, "(((%s > %s) - (%s < %s)) %s 0)",
                         left, right, left, right, op);
                stack_push_bool(b, &st, expression);
                break;
            }
            char left[96], right[96], expression[320];
            uint8_t pair[2] = {lk, rk};
            int slots[2] = {lhs, rhs};
            char *boxed[2] = {left, right};
            for (int side = 0; side < 2; ++side) {
                uint8_t kind = pair[side];
                if (word_array_storage(kind) || kind == NVM2C_VK_SARR || kind == NVM2C_VK_RARR)
                    snprintf(boxed[side], 96, "(nmap_value){%u, 0, NULL}", TAG_ARRAY);
                else if (kind == NVM2C_VK_MAP)
                    snprintf(boxed[side], 96, "(nmap_value){%u, 0, NULL}", TAG_HASHMAP);
                else if (kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL || kind == NVM2C_VK_FLOAT || kind == NVM2C_VK_STR || kind == NVM2C_VK_VALUE)
                    scalar_value_expression(b, boxed[side], 96, kind, slots[side]);
                else { nvm2c_fail(b, "I require preserved operand tags for generic comparison"); goto done; }
            }
            const char *op = ins.opcode == OP_LT ? "<" : ins.opcode == OP_LE ? "<=" :
                             ins.opcode == OP_GT ? ">" : ">=";
            snprintf(expression, sizeof expression, "(nvalue_compare(%s, %s) %s 0)", left, right, op);
            stack_push_bool(b, &st, expression);
            break;
        }
        case OP_I64_NE:
            emit_binop(b, &st, "!=");
            break;
        case OP_I64_LT_S:
            emit_binop(b, &st, "<");
            break;
        case OP_I64_LE_S:
            emit_binop(b, &st, "<=");
            break;
        case OP_I64_GT_S:
            emit_binop(b, &st, ">");
            break;
        case OP_I64_GE_S:
            emit_binop(b, &st, ">=");
            break;
        case OP_BOOL_AND:
            emit_binop(b, &st, "&&");
            break;
        case OP_BOOL_OR:
            emit_binop(b, &st, "||");
            break;
        case OP_STR_LEN: {
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_LEN");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "(int64_t)strlen(s[%d] ? s[%d] : \"\")", s, s);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_CONCAT: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONCAT rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONCAT lhs");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_concat(s[%d], s[%d])", lhs, rhs);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_TRIM: {
            int value = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_TRIM");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_trim(s[%d])", value);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_SUBSTR: {
            int len = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_SUBSTR length");
            int start = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_SUBSTR start");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_SUBSTR");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "nstr_substr(s[%d], t[%d], t[%d])", s, start, len);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_CONTAINS: {
            int needle = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONTAINS needle");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONTAINS haystack");
            if (b->failed) goto done;
            char expr[160];
            snprintf(expr, sizeof expr,
                     "(int64_t)(strstr(s[%d] ? s[%d] : \"\", s[%d] ? s[%d] : \"\") != NULL)",
                     hay, hay, needle, needle);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_STARTS_WITH: {
            int pre = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_STARTS_WITH prefix");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_STARTS_WITH");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_starts_with(s[%d], s[%d])", hay, pre);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_ENDS_WITH: {
            int suf = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_ENDS_WITH suffix");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_ENDS_WITH");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_ends_with(s[%d], s[%d])", hay, suf);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_CHAR_AT: {
            int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_CHAR_AT index");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CHAR_AT");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_char_at(s[%d], t[%d])", s, ix);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_CAST_STRING: {
            if (st.sp && st.kinds[st.sp - 1] == NVM2C_VK_STR) {
                int value = stack_pop(b, &st);
                char expression[48];
                snprintf(expression, sizeof expression, "s[%d]", value);
                stack_push_str(b, &st, expression);
                break;
            }
            if (st.sp && st.kinds[st.sp - 1] == NVM2C_VK_FLOAT) {
                int value = stack_pop(b, &st);
                char expression[64];
                snprintf(expression, sizeof expression, "nstr_from_f64(f[%d])", value);
                stack_push_str(b, &st, expression);
                break;
            }
            if (st.sp && st.kinds[st.sp - 1] == NVM2C_VK_BOOL) {
                int value = stack_pop(b, &st);
                char expression[64];
                snprintf(expression, sizeof expression, "(t[%d] ? \"true\" : \"false\")", value);
                stack_push_str(b, &st, expression);
                break;
            }
            if (st.sp && st.kinds[st.sp - 1] == NVM2C_VK_VALUE) {
                int value = stack_pop(b, &st);
                char expression[360];
                snprintf(expression, sizeof expression,
                    "(v[%d].kind == 3 ? nstr_from_f64(nvalue_require_float(v[%d])) : v[%d].kind == 5 ? v[%d].text : (v[%d].kind == 1 || v[%d].kind == 2) ? nstr_from_i64(v[%d].integer) : v[%d].kind == 4 ? (v[%d].integer ? \"true\" : \"false\") : \"\")",
                    value, value, value, value, value, value, value, value, value);
                stack_push_str(b, &st, expression);
                break;
            }
            int v = stack_pop_expect(b, &st, NVM2C_VK_INT, "CAST_STRING");
            if (b->failed) goto done;
            char expr[48];
            snprintf(expr, sizeof expr, "nstr_from_i64(t[%d])", v);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_CAST_BOOL: case OP_AND: case OP_OR: case OP_NOT: {
            int binary = ins.opcode == OP_AND || ins.opcode == OP_OR;
            int rhs = binary ? stack_pop_scalar_condition(b, &st) : -1;
            int lhs = stack_pop_scalar_condition(b, &st);
            if (b->failed) goto done;
            char expression[96];
            if (binary)
                snprintf(expression, sizeof expression, "((t[%d] != 0) %s (t[%d] != 0))",
                         lhs, ins.opcode == OP_AND ? "&" : "|", rhs);
            else
                snprintf(expression, sizeof expression, "(t[%d] %s 0)", lhs,
                         ins.opcode == OP_NOT ? "==" : "!=");
            stack_push_bool(b, &st, expression);
            break;
        }
        case OP_CAST_FLOAT: {
            uint8_t kind;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            char expression[128];
            if (kind == NVM2C_VK_VALUE)
                snprintf(expression, sizeof expression, "nvalue_cast_float(v[%d])", value);
            else if (kind == NVM2C_VK_FLOAT)
                snprintf(expression, sizeof expression, "f[%d]", value);
            else if (kind == NVM2C_VK_INT)
                snprintf(expression, sizeof expression, "(double)t[%d]", value);
            else if (kind == NVM2C_VK_BOOL)
                snprintf(expression, sizeof expression, "(t[%d] ? 1.0 : 0.0)", value);
            else if (kind == NVM2C_VK_STR)
                snprintf(expression, sizeof expression, "nparse_binary64(s[%d])", value);
            else if (kind == NVM2C_VK_REC || kind == NVM2C_VK_MAP ||
                     word_array_storage(kind) || kind == NVM2C_VK_SARR || kind == NVM2C_VK_RARR)
                snprintf(expression, sizeof expression, "0.0");
            else {
                nvm2c_fail(b, "I cannot emit CAST_FLOAT with an unresolved representation");
                goto done;
            }
            stack_push_float(b, &st, expression);
            break;
        }
        case OP_CAST_INT: {
            uint8_t kind;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            char expression[96];
            if (kind == NVM2C_VK_VALUE)
                snprintf(expression, sizeof expression, "nvalue_cast_int(v[%d])", value);
            else if (kind == NVM2C_VK_FLOAT)
                snprintf(expression, sizeof expression, "nf64_to_i64(f[%d])", value);
            else if (kind == NVM2C_VK_STR)
                snprintf(expression, sizeof expression, "(int64_t)strtoll(s[%d] ? s[%d] : \"\", NULL, 10)", value, value);
            else if (kind == NVM2C_VK_INT || kind == NVM2C_VK_BOOL)
                snprintf(expression, sizeof expression, "t[%d]", value);
            else if (kind == NVM2C_VK_REC || word_array_storage(kind) ||
                     kind == NVM2C_VK_SARR || kind == NVM2C_VK_RARR)
                snprintf(expression, sizeof expression, "0");
            else {
                nvm2c_fail(b, "I cannot emit CAST_INT with an unresolved representation");
                goto done;
            }
            stack_push_temp(b, &st, expression);
            break;
        }
        case OP_HM_NEW: {
            char expression[48];
            snprintf(expression, sizeof expression, "nmap_owned_new(%u)", (unsigned)ins.operands[1].u8);
            stack_push_map(b, &st, expression);
            break;
        }
        case OP_TYPE_CHECK: {
            if (st.sp && (st.kinds[st.sp - 1] == NVM2C_VK_INT || st.kinds[st.sp - 1] == NVM2C_VK_BOOL ||
                          st.kinds[st.sp - 1] == NVM2C_VK_STR || st.kinds[st.sp - 1] == NVM2C_VK_FLOAT)) {
                uint8_t kind;
                (void)stack_pop_kind(b, &st, &kind);
                uint8_t tag = kind == NVM2C_VK_INT ? TAG_INT : kind == NVM2C_VK_BOOL ? TAG_BOOL : kind == NVM2C_VK_FLOAT ? TAG_FLOAT : TAG_STRING;
                stack_push_bool(b, &st, tag == ins.operands[0].u8 ? "1" : "0");
                break;
            }
            if (st.sp && st.kinds[st.sp - 1] != NVM2C_VK_VALUE) {
                nvm2c_fail(b, "I require a resolved tagged representation for TYPE_CHECK"); goto done;
            }
            int value = stack_pop_expect(b, &st, NVM2C_VK_VALUE, "TYPE_CHECK");
            if (b->failed) goto done;
            char expression[64];
            snprintf(expression, sizeof expression, "(v[%d].kind == %u)", value, (unsigned)ins.operands[0].u8);
            stack_push_temp(b, &st, expression);
            break;
        }
        case OP_HM_SET:
        case OP_HM_GET:
        case OP_HM_HAS:
        case OP_HM_DELETE:
        case OP_HM_LEN: {
            uint8_t value_kind = NVM2C_VK_UNK;
            int value = -1, key = -1;
            if (ins.opcode == OP_HM_SET) value = stack_pop_kind(b, &st, &value_kind);
            if (ins.opcode != OP_HM_LEN) key = stack_pop_expect(b, &st, NVM2C_VK_STR, "hashmap key");
            int map = stack_pop_expect(b, &st, NVM2C_VK_MAP, "hashmap operation");
            if (b->failed) goto done;
            char expression[192];
            if (ins.opcode == OP_HM_GET) {
                snprintf(expression, sizeof expression, "nmap_owned_get(m[%d], s[%d])", map, key);
                stack_push_value(b, &st, expression);
            } else if (ins.opcode == OP_HM_SET) {
                if (value_kind == NVM2C_VK_INT)
                    snprintf(expression, sizeof expression, "nmap_set(m[%d], s[%d], (nmap_value){1, t[%d], NULL})", map, key, value);
                else if (value_kind == NVM2C_VK_STR)
                    snprintf(expression, sizeof expression, "nmap_set(m[%d], s[%d], (nmap_value){5, 0, (char *)s[%d]})", map, key, value);
                else { nvm2c_fail(b, "I cannot emit an unresolved or unsupported hashmap value"); goto done; }
                stack_push_map(b, &st, expression);
            } else if (ins.opcode == OP_HM_DELETE) {
                snprintf(expression, sizeof expression, "nmap_delete(m[%d], s[%d])", map, key);
                stack_push_map(b, &st, expression);
            } else {
                if (ins.opcode == OP_HM_HAS)
                    snprintf(expression, sizeof expression, "nmap_has(m[%d], s[%d])", map, key);
                else snprintf(expression, sizeof expression, "(int64_t)nmap_len(m[%d])", map);
                stack_push_temp(b, &st, expression);
            }
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            int as_sarr = (tag == TAG_STRING);
            if (tag == TAG_STRUCT) {
                int array = stack_push_rarr(b, &st, "nrarr_new()");
                if (array >= 0) {
                    memset(st.rarr_k[array], NVM2C_VK_UNK, b->record_width);
                    DecodedInstruction next;
                    if (isa_decode(code + pc, remaining - pc, &next) && next.opcode == OP_STORE_LOCAL) {
                        uint16_t slot = next.operands[0].u16;
                        if (slot < fn->local_count)
                            memcpy(st.rarr_k[array], fn_rec_k_const(b, rec_fields, idx, slot), b->record_width);
                    }
                }
                break;
            }
            if (tag != TAG_INT && tag != TAG_BOOL && tag != TAG_FLOAT && tag != TAG_STRING) {
                nvm2c_fail(b, "function %u: I support int, bool, float, string or struct elements in ARR_NEW", idx);
                goto done;
            }
            if (tag == TAG_INT) {
                DecodedInstruction nxt;
                uint32_t nn = isa_decode(code + pc, remaining - pc, &nxt);
                if (nn != 0 && nxt.opcode == OP_STORE_LOCAL) {
                    uint16_t slot = nxt.operands[0].u16;
                    if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_SARR) {
                        as_sarr = 1;
                    } else if (fn_local_kind(b, kinds, idx, slot) == NVM2C_VK_RARR) {
                        int a = stack_push_rarr(b, &st, "nrarr_new()");
                        if (a >= 0) {
                            memcpy(st.rarr_k[a], fn_rec_k_const(b, rec_fields, idx, slot),
                                   b->record_width);
                        }
                        break;
                    }
                }
            }
            if (as_sarr) {
                stack_push_sarr(b, &st, "nsarr_new()");
            } else {
                stack_push_iarray(b, &st, "narr_new()", tag == TAG_FLOAT ? NVM2C_VK_FARR : tag == TAG_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR);
            }
            break;
        }
        case OP_ARR_LITERAL: {
            uint8_t tag = ins.operands[0].u8;
            uint16_t count = ins.operands[1].u16;
            int *elems = literal_elems;
            int ei;
            uint8_t ekind = NVM2C_VK_INT;
            if (tag == TAG_INT) {
                ekind = NVM2C_VK_INT;
            } else if (tag == TAG_BOOL) {
                ekind = NVM2C_VK_BOOL;
            } else if (tag == TAG_FLOAT) {
                ekind = NVM2C_VK_FLOAT;
            } else if (tag == TAG_STRING) {
                ekind = NVM2C_VK_STR;
            } else if (tag == TAG_STRUCT) {
                ekind = NVM2C_VK_REC;
            } else {
                nvm2c_fail(b, "function %u: I support int, bool, float, string or record elements in ARR_LITERAL", idx);
                goto done;
            }
            if ((size_t)count > st.capacity) {
                nvm2c_fail(b, "function %u: ARR_LITERAL is too large", idx);
                goto done;
            }
            for (ei = (int)count - 1; ei >= 0; ei--) {
                elems[ei] = stack_pop_expect(b, &st, ekind, "ARR_LITERAL");
                if (b->failed) goto done;
            }
            if (tag == TAG_STRUCT) {
                int result = stack_push_rarr(b, &st, "nrarr_new()");
                if (result < 0 || b->failed) goto done;
                memset(st.rarr_k[result], NVM2C_VK_UNK, b->record_width);
                for (ei = 0; ei < (int)count; ++ei) {
                    nvm2c_printf(b, "    ra[%d] = nrarr_push(ra[%d], r[%d]);\n", result, result, elems[ei]);
                    for (size_t f = 0; f < b->record_width; ++f)
                        if (st.rec_k[elems[ei]][f] != NVM2C_VK_UNK)
                            st.rarr_k[result][f] = st.rec_k[elems[ei]][f];
                }
                break;
            }
            int result = tag == TAG_STRING ? stack_push_sarr(b, &st, "0")
                                           : stack_push_iarray(b, &st, "0", tag == TAG_FLOAT ? NVM2C_VK_FARR : tag == TAG_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR);
            if (b->failed) goto done;
            nvm2c_printf(b, "    %s[%d] = %s(", tag == TAG_STRING ? "sa" : "a",
                         result, tag == TAG_STRING ? "nsarr_lit" : "narr_lit");
            if (count) {
                nvm2c_puts(b, tag == TAG_STRING ? "(const char *[]){" : "(int64_t[]){");
                for (ei = 0; ei < (int)count; ++ei) {
                    if (tag == TAG_FLOAT) nvm2c_printf(b, "%snvalue_from_float(f[%d]).integer", ei ? ", " : "", elems[ei]);
                    else nvm2c_printf(b, "%s%s[%d]", ei ? ", " : "",
                                 tag == TAG_STRING ? "s" : "t", elems[ei]);
                }
                nvm2c_puts(b, "}");
            } else nvm2c_puts(b, "0");
            nvm2c_printf(b, ", %u);\n", (unsigned)count);
            break;
        }
        case OP_ARR_LEN: {
            uint8_t ak = NVM2C_VK_INT;
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_VALUE) {
                char expr[64];
                snprintf(expr, sizeof expr, "nvalue_array_len(v[%d])", arr);
                stack_push_temp(b, &st, expr);
            } else if (word_array_storage(ak)) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(a[%d] ? a[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(sa[%d] ? sa[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(ra[%d] ? ra[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "function %u: ARR_LEN expected an array", idx);
                goto done;
            }
            break;
        }
        case OP_ARR_GET: {
            uint8_t ak = NVM2C_VK_INT;
            int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "ARR_GET index");
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_VALUE) {
                char expr[80];
                snprintf(expr, sizeof expr, "nvalue_array_get(v[%d], t[%d])", arr, ix);
                stack_push_value(b, &st, expr);
            } else if (word_array_storage(ak)) {
                char expr[128];
                snprintf(expr, sizeof expr, "nvalue_array_get((nmap_value){7, %u, (char *)a[%d]}, t[%d])", ak, arr, ix);
                stack_push_value(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[128];
                snprintf(expr, sizeof expr, "nvalue_array_get((nmap_value){7, %u, (char *)sa[%d]}, t[%d])", ak, arr, ix);
                stack_push_value(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nrarr_get(ra[%d], t[%d])", arr, ix);
                int r = stack_push_rec(b, &st, expr);
                if (r >= 0) memcpy(st.rec_k[r], st.rarr_k[arr], b->record_width);
            } else {
                nvm2c_fail(b, "function %u: ARR_GET expected an array", idx);
                goto done;
            }
            break;
        }
        case OP_ARR_SET: {
            uint8_t vk, ak;
            int val = stack_pop_kind(b, &st, &vk);
            int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "ARR_SET index");
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_VALUE) {
                char boxed[96], expr[192];
                scalar_value_expression(b, boxed, sizeof boxed, vk, val);
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nvalue_array_set(v[%d], t[%d], %s)", arr, ix, boxed);
                stack_push_value(b, &st, expr);
                break;
            }
            const char *array = NULL, *value = NULL;
            if ((ak == NVM2C_VK_ARR && (vk == NVM2C_VK_INT || vk == NVM2C_VK_VALUE)) ||
                (ak == NVM2C_VK_BARR && (vk == NVM2C_VK_BOOL || vk == NVM2C_VK_VALUE))) { array = "a"; value = "t"; }
            else if (ak == NVM2C_VK_FARR && (vk == NVM2C_VK_FLOAT || vk == NVM2C_VK_VALUE)) { array = "a"; value = "f"; }
            else if (ak == NVM2C_VK_SARR && (vk == NVM2C_VK_STR || vk == NVM2C_VK_VALUE)) { array = "sa"; value = "s"; }
            else if (ak == NVM2C_VK_RARR && vk == NVM2C_VK_REC) { array = "ra"; value = "r"; }
            else {
                nvm2c_fail(b, "ARR_SET element representation mismatch");
                goto done;
            }
            nvm2c_printf(b,
                "    if (!%s[%d] || t[%d] < 0 || (uint64_t)t[%d] >= %s[%d]->len) NVM2C_ABORT();\n",
                array, arr, ix, ix, array, arr);
            if (ak == NVM2C_VK_RARR) {
                nvm2c_printf(b, "    if (!ra[%d]->data) NVM2C_ABORT();\n", arr);
                /* Classifier field kinds do not encode the runtime width. */
                nvm2c_printf(b,
                    "    if (ra[%d]->data[t[%d]].n != r[%d].n || ra[%d]->data[t[%d]].kind != r[%d].kind) NVM2C_ABORT();\n",
                    arr, ix, val, arr, ix, val);
                nvm2c_printf(b,
                    "    for (size_t f = 0; f < r[%d].n; ++f) if (!nrec_field_storage_matches(&ra[%d]->data[t[%d]], &r[%d], f)) NVM2C_ABORT();\n",
                    val, arr, ix, val);
            }
            if (ak == NVM2C_VK_FARR)
                nvm2c_printf(b, vk == NVM2C_VK_VALUE
                    ? "    a[%d]->data[t[%d]] = nvalue_from_float(nvalue_require_float(v[%d])).integer;\n"
                    : "    a[%d]->data[t[%d]] = nvalue_from_float(f[%d]).integer;\n", arr, ix, val);
            else if (ak == NVM2C_VK_SARR && vk == NVM2C_VK_VALUE)
                nvm2c_printf(b, "    sa[%d]->data[t[%d]] = nvalue_require_string(v[%d]);\n", arr, ix, val);
            else if (word_array_storage(ak) && vk == NVM2C_VK_VALUE)
                nvm2c_printf(b, "    a[%d]->data[t[%d]] = nvalue_require_%s(v[%d]);\n",
                             arr, ix, ak == NVM2C_VK_BARR ? "bool" : "int", val);
            else nvm2c_printf(b, "    %s[%d]->data[t[%d]] = %s[%d];\n", array, arr, ix, value, val);
            /* The result is the same handle, not a copy: aliases see the write. */
            st.slots[st.sp] = arr;
            st.kinds[st.sp++] = ak;
            break;
        }
        case OP_ARR_PUSH: {
            uint8_t vk = NVM2C_VK_INT;
            uint8_t ak = NVM2C_VK_INT;
            int val = stack_pop_kind(b, &st, &vk);
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_VALUE) {
                char boxed[96], expr[192];
                scalar_value_expression(b, boxed, sizeof boxed, vk, val);
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nvalue_array_push(v[%d], %s)", arr, boxed);
                stack_push_value(b, &st, expr);
            } else if (ak == NVM2C_VK_FARR && (vk == NVM2C_VK_FLOAT || vk == NVM2C_VK_VALUE)) {
                char expr[128];
                snprintf(expr, sizeof expr, vk == NVM2C_VK_VALUE
                    ? "narr_push(a[%d], nvalue_from_float(nvalue_require_float(v[%d])).integer)"
                    : "narr_push(a[%d], nvalue_from_float(f[%d]).integer)", arr, val);
                stack_push_iarray(b, &st, expr, ak);
            } else if (word_array_storage(ak) && vk == NVM2C_VK_VALUE) {
                char expr[96];
                snprintf(expr, sizeof expr, "narr_push(a[%d], nvalue_require_%s(v[%d]))",
                         arr, ak == NVM2C_VK_BARR ? "bool" : "int", val);
                stack_push_iarray(b, &st, expr, ak);
            } else if ((ak == NVM2C_VK_ARR && vk == NVM2C_VK_INT) ||
                       (ak == NVM2C_VK_BARR && vk == NVM2C_VK_BOOL)) {
                char expr[80];
                snprintf(expr, sizeof expr, "narr_push(a[%d], t[%d])", arr, val);
                stack_push_iarray(b, &st, expr, ak);
            } else if (ak == NVM2C_VK_SARR && (vk == NVM2C_VK_STR || vk == NVM2C_VK_VALUE)) {
                char expr[80];
                snprintf(expr, sizeof expr, vk == NVM2C_VK_VALUE
                         ? "nsarr_push(sa[%d], nvalue_require_string(v[%d]))"
                         : "nsarr_push(sa[%d], s[%d])", arr, val);
                stack_push_sarr(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR && vk == NVM2C_VK_REC) {
                char expr[80];
                snprintf(expr, sizeof expr, "nrarr_push(ra[%d], r[%d])", arr, val);
                int a = stack_push_rarr(b, &st, expr);
                if (a >= 0) memcpy(st.rarr_k[a], st.rec_k[val], b->record_width);
            } else {
                nvm2c_fail(b, "function %u: ARR_PUSH type mismatch", idx);
                goto done;
            }
            break;
        }
        case OP_AGG_PACK: {
            uint8_t kind = ins.operands[0].u8;
            uint16_t count = ins.operands[3].u16;
            int *elems = literal_elems;
            uint8_t *fkind = aggregate_kinds;
            int ei;
            if (kind != AGG_RECORD && kind != AGG_VARIANT && kind != AGG_TUPLE) {
                nvm2c_fail(b, "I support record, variant or tuple packing, not aggregate kind %u", kind);
                goto done;
            }
            if (count > b->record_width) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                goto done;
            }
            for (ei = (int)count - 1; ei >= 0; ei--) {
                uint8_t vk = NVM2C_VK_INT;
                elems[ei] = stack_pop_kind(b, &st, &vk);
                if (b->failed) goto done;
                if (vk != NVM2C_VK_INT && vk != NVM2C_VK_STR &&
                    !word_array_storage(vk) && vk != NVM2C_VK_SARR &&
                    vk != NVM2C_VK_RARR && vk != NVM2C_VK_REC && vk != NVM2C_VK_VALUE &&
                    vk != NVM2C_VK_BOOL && vk != NVM2C_VK_FLOAT && vk != NVM2C_VK_MAP) {
                    nvm2c_fail(b, "function %u: AGG_PACK field requires unsupported nested aggregate shape facts", idx);
                    goto done;
                }
                fkind[ei] = vk;
            }
            if ((size_t)st.next_rec >= st.rec_capacity) {
                nvm2c_fail(b, "too many record temporaries");
                goto done;
            }
            if ((size_t)st.sp >= st.capacity) {
                nvm2c_fail(b, "operand stack overflow");
                goto done;
            }
            {
                int r = st.next_rec++;
                nvm2c_printf(b, "    r[%d].n = %u;\n", r, (unsigned)count);
                nvm2c_printf(b, "    r[%d].kind = %u; r[%d].tag = %u;\n",
                             r, (unsigned)kind, r, (unsigned)ins.operands[2].u16);
                for (ei = 0; ei < (int)count; ei++) {
                    int boxed_scalar = kind == AGG_VARIANT && (variant_scalar_kind(fkind[ei]) || fkind[ei] == NVM2C_VK_ARR);
                    st.rec_k[r][ei] = boxed_scalar ? NVM2C_VK_VALUE : fkind[ei];
                    nvm2c_printf(b, "    r[%d].k[%d] = %u;\n", r, ei, (unsigned)st.rec_k[r][ei]);
                    if (boxed_scalar) {
                        unsigned tag = fkind[ei] == NVM2C_VK_STR ? TAG_STRING :
                                       fkind[ei] == NVM2C_VK_BOOL ? TAG_BOOL :
                                       fkind[ei] == NVM2C_VK_FLOAT ? TAG_FLOAT :
                                       fkind[ei] == NVM2C_VK_ARR ? TAG_ARRAY : TAG_INT;
                        nvm2c_printf(b, "    r[%d].vk[%d] = %u;\n", r, ei, tag);
                    }
                    if (fkind[ei] == NVM2C_VK_STR) {
                        nvm2c_printf(b, "    r[%d].s[%d] = s[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_FLOAT) {
                        nvm2c_printf(b, "    memcpy(&r[%d].f[%d], &f[%d], sizeof f[%d]);\n",
                                     r, ei, elems[ei], elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_VALUE) {
                        nvm2c_printf(b, "    r[%d].vk[%d] = v[%d].kind;\n"
                                         "    r[%d].f[%d] = v[%d].integer;\n"
                                         "    r[%d].s[%d] = v[%d].text;\n",
                                     r, ei, elems[ei], r, ei, elems[ei], r, ei, elems[ei]);
                    } else if (boxed_scalar && fkind[ei] == NVM2C_VK_ARR) {
                        nvm2c_printf(b, "    r[%d].f[%d] = %u; r[%d].s[%d] = (char *)a[%d];\n",
                                     r, ei, NVM2C_VK_ARR, r, ei, elems[ei]);
                    } else if (word_array_storage(fkind[ei])) {
                        nvm2c_printf(b, "    r[%d].a[%d] = a[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_SARR) {
                        nvm2c_printf(b, "    r[%d].sa[%d] = sa[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_RARR) {
                        nvm2c_printf(b, "    r[%d].ra[%d] = ra[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_REC) {
                        nvm2c_printf(b, "    r[%d].rec[%d] = nrec_snapshot(r[%d]);\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_MAP) {
                        nvm2c_printf(b, "    r[%d].m[%d] = m[%d];\n", r, ei, elems[ei]);
                    } else {
                        nvm2c_printf(b, "    r[%d].f[%d] = t[%d];\n", r, ei, elems[ei]);
                    }
                }
                st.slots[st.sp] = r;
                st.kinds[st.sp] = NVM2C_VK_REC;
                st.sp++;
            }
            break;
        }
        case OP_AGG_TAG: {
            int aggregate = stack_pop_expect(b, &st, NVM2C_VK_REC, "AGG_TAG");
            if (b->failed) goto done;
            nvm2c_printf(b, "    if (r[%d].kind != %u) NVM2C_ABORT();\n", aggregate, AGG_VARIANT);
            char expression[64];
            snprintf(expression, sizeof expression, "r[%d].tag", aggregate);
            stack_push_temp(b, &st, expression);
            break;
        }
        case OP_AGG_GET: {
            uint16_t fi = ins.operands[0].u16;
            int rec = stack_pop_expect(b, &st, NVM2C_VK_REC, "AGG_GET");
            if (b->failed) goto done;
            if (fi >= b->record_width) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                goto done;
            }
            uint8_t resolved = resolved_shape_kind(b, b->shape_outputs[idx][start]);
            if (!shape_ok(b)) goto done;
            if (resolved == NVM2C_VK_UNK &&
                nvm_shape_kind(&b->shapes, b->shape_outputs[idx][start]) == NVM_SHAPE_ARRAY) {
                /* I retain the record's runtime array storage tag instead of
                 * defaulting an unresolved element shape to integer storage. */
                nvm2c_printf(b, "    if (%u >= r[%d].n) NVM2C_ABORT();\n", (unsigned)fi, rec);
                nvm2c_printf(b, "    if (r[%d].k[%u] != 3 && r[%d].k[%u] != 10 && r[%d].k[%u] != 12 && r[%d].k[%u] != 5 && r[%d].k[%u] != 6) NVM2C_ABORT();\n",
                             rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi);
                char expr[384];
                snprintf(expr, sizeof expr,
                         "(nmap_value){7, r[%d].k[%u], (char *)(r[%d].k[%u] == 5 ? (void *)r[%d].sa[%u] : r[%d].k[%u] == 6 ? (void *)r[%d].ra[%u] : (void *)r[%d].a[%u])}",
                         rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi,
                         rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi);
                stack_push_value(b, &st, expr);
                break;
            }
            if (resolved != NVM2C_VK_UNK) st.rec_k[rec][fi] = resolved;
            if (st.rec_k[rec][fi] == NVM2C_VK_UNK) {
                /* I preserve an unconstrained scalar's runtime tag. Unknown
                 * is an inference marker, never a valid record storage tag. */
                nvm2c_printf(b, "    if (%u >= r[%d].n) NVM2C_ABORT();\n", (unsigned)fi, rec);
                nvm2c_printf(b, "    if (r[%d].k[%u] != 0 && r[%d].k[%u] != 9 && r[%d].k[%u] != 1 && r[%d].k[%u] != 11 && r[%d].k[%u] != 8) NVM2C_ABORT();\n",
                             rec, fi, rec, fi, rec, fi, rec, fi, rec, fi);
                nvm2c_printf(b, "    if (r[%d].k[%u] == 8 && r[%d].vk[%u] != 0 && r[%d].vk[%u] != 1 && r[%d].vk[%u] != 3 && r[%d].vk[%u] != 4 && r[%d].vk[%u] != 5) NVM2C_ABORT();\n",
                             rec, fi, rec, fi, rec, fi, rec, fi, rec, fi, rec, fi);
                nvm2c_printf(b, "    if ((r[%d].k[%u] == 1 || (r[%d].k[%u] == 8 && r[%d].vk[%u] == 5)) && !r[%d].s[%u]) NVM2C_ABORT();\n",
                             rec, fi, rec, fi, rec, fi, rec, fi);
                char expr[384];
                snprintf(expr, sizeof expr,
                         "(nmap_value){r[%d].k[%u] == 0 ? 1 : r[%d].k[%u] == 9 ? 4 : r[%d].k[%u] == 1 ? 5 : r[%d].k[%u] == 11 ? 3 : r[%d].vk[%u], r[%d].f[%u], (char *)r[%d].s[%u]}",
                         rec, fi, rec, fi, rec, fi, rec, fi, rec, fi, rec, fi, rec, fi);
                stack_push_value(b, &st, expr);
                break;
            }
            nvm2c_printf(b, "    if (%u >= r[%d].n) NVM2C_ABORT();\n", (unsigned)fi, rec);
            if (st.rec_k[rec][fi] == NVM2C_VK_VALUE)
                nvm2c_printf(b, "    if (r[%d].k[%u] != %u && r[%d].k[%u] != %u && r[%d].k[%u] != %u && r[%d].k[%u] != %u) NVM2C_ABORT();\n",
                             rec, (unsigned)fi, NVM2C_VK_VALUE, rec, (unsigned)fi, NVM2C_VK_STR,
                             rec, (unsigned)fi, NVM2C_VK_INT, rec, (unsigned)fi, NVM2C_VK_BOOL);
            else if (st.rec_k[rec][fi] == NVM2C_VK_STR ||
                     st.rec_k[rec][fi] == NVM2C_VK_INT ||
                     st.rec_k[rec][fi] == NVM2C_VK_BOOL) {
                /* Inferred storage can retain a boxed present scalar even
                 * when this projection has an exact scalar consumer. I check
                 * its payload tag before reading the same record slot. */
                unsigned tag = st.rec_k[rec][fi] == NVM2C_VK_STR ? TAG_STRING :
                               st.rec_k[rec][fi] == NVM2C_VK_BOOL ? TAG_BOOL : TAG_INT;
                nvm2c_printf(b, "    if (r[%d].k[%u] != %u && !(r[%d].k[%u] == %u && r[%d].vk[%u] == %u)) NVM2C_ABORT();\n",
                             rec, (unsigned)fi, (unsigned)st.rec_k[rec][fi],
                             rec, (unsigned)fi, NVM2C_VK_VALUE, rec, (unsigned)fi, tag);
                if (st.rec_k[rec][fi] == NVM2C_VK_STR)
                    nvm2c_printf(b, "    if (!r[%d].s[%u]) NVM2C_ABORT();\n", rec, (unsigned)fi);
            } else nvm2c_printf(b, "    if (r[%d].k[%u] != %u) NVM2C_ABORT();\n", rec, (unsigned)fi,
                               (unsigned)st.rec_k[rec][fi]);
            {
                char expr[256];
                if (st.rec_k[rec][fi] == NVM2C_VK_STR) {
                    snprintf(expr, sizeof expr, "r[%d].s[%u]", rec, (unsigned)fi);
                    stack_push_str(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_FLOAT) {
                    snprintf(expr, sizeof expr, "nrec_f64(r[%d].f[%u])", rec, (unsigned)fi);
                    stack_push_float(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_VALUE) {
                    snprintf(expr, sizeof expr,
                             "(nmap_value){r[%d].k[%u] == %u ? 5 : r[%d].k[%u] == %u ? 1 : r[%d].k[%u] == %u ? 4 : r[%d].vk[%u], r[%d].f[%u], (char *)r[%d].s[%u]}",
                             rec, (unsigned)fi, NVM2C_VK_STR,
                             rec, (unsigned)fi, NVM2C_VK_INT, rec, (unsigned)fi, NVM2C_VK_BOOL,
                             rec, (unsigned)fi, rec, (unsigned)fi, rec, (unsigned)fi);
                    stack_push_value(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_MAP) {
                    snprintf(expr, sizeof expr, "r[%d].m[%u]", rec, (unsigned)fi);
                    stack_push_map(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    if (!r[%d].rec[%u]) NVM2C_ABORT();\n", rec, (unsigned)fi);
                    snprintf(expr, sizeof expr, "*r[%d].rec[%u]", rec, (unsigned)fi);
                    int nested = stack_push_rec(b, &st, expr);
                    if (nested >= 0) {
                        NvmShapeId shape = b->shape_outputs[idx][start];
                        for (size_t f = 0; f < b->record_width; ++f)
                            st.rec_k[nested][f] = resolved_shape_kind(b,
                                nvm_shape_lookup(&b->shapes, shape, (uint32_t)f));
                        if (!shape_ok(b)) goto done;
                    }
                } else if (word_array_storage(st.rec_k[rec][fi])) {
                    snprintf(expr, sizeof expr, "r[%d].a[%u]", rec, (unsigned)fi);
                    stack_push_iarray(b, &st, expr, st.rec_k[rec][fi]);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_SARR) {
                    snprintf(expr, sizeof expr, "r[%d].sa[%u]", rec, (unsigned)fi);
                    stack_push_sarr(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_RARR) {
                    snprintf(expr, sizeof expr, "r[%d].ra[%u]", rec, (unsigned)fi);
                    int array = stack_push_rarr(b, &st, expr);
                    if (array >= 0) {
                        NvmShapeId element = nvm_shape_lookup(&b->shapes, b->shape_outputs[idx][start], 0);
                        for (size_t f = 0; f < b->record_width; ++f) {
                            NvmShapeId child = element ? nvm_shape_lookup(&b->shapes, element, (uint32_t)f) : 0;
                            st.rarr_k[array][f] = resolved_shape_kind(b, child);
                        }
                        if (!shape_ok(b)) goto done;
                    }
                } else {
                    snprintf(expr, sizeof expr, "r[%d].f[%u]", rec, (unsigned)fi);
                    if (st.rec_k[rec][fi] == NVM2C_VK_BOOL) stack_push_bool(b, &st, expr);
                    else stack_push_temp(b, &st, expr);
                }
            }
            break;
        }
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            emit_map_roots(b, &st, fn, kinds, idx);
            if (b->has_owned_strings || b->has_owned_aggregates)
                nvm2c_puts(b, "    nmap_collect_if_needed();\n");
            char call[NVM2C_CALL_SIZE];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, call, sizeof call)) {
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (result_is_i64(cf)) {
                if (cf->result_tag == TAG_BOOL) stack_push_bool(b, &st, call);
                else stack_push_temp(b, &st, call);
            } else if (cf->result_count == 1 && (cf->result_tag == TAG_U8 || cf->result_tag == TAG_ENUM)) {
                stack_push_value(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_FLOAT) {
                stack_push_float(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                stack_push_str(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_HASHMAP) {
                stack_push_map(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                if (word_array_storage(b->array_results[callee])) stack_push_iarray(b, &st, call, b->array_results[callee]);
                else if (b->array_results[callee] == NVM2C_VK_SARR) stack_push_sarr(b, &st, call);
                else {
                    int result = stack_push_rarr(b, &st, call);
                    if (result >= 0) memcpy(st.rarr_k[result], result_fields + (size_t)callee * b->record_width,
                                           b->record_width);
                }
            } else if (cf->result_count == 1 && aggregate_value_tag(cf->result_tag)) {
                int result = stack_push_rec(b, &st, call);
                if (result >= 0) memcpy(st.rec_k[result], result_fields + (size_t)callee * b->record_width,
                                        b->record_width);
            } else {
                nvm2c_printf(b, "    %s;\n", call);
            }
            break;
        }
        case OP_TAIL_CALL: {
            uint32_t callee = ins.operands[0].u32;
            emit_map_roots(b, &st, fn, kinds, idx);
            if (b->has_owned_strings || b->has_owned_aggregates)
                nvm2c_puts(b, "    nmap_collect_if_needed();\n");
            if (callee >= mod->function_count) {
                nvm2c_fail(b, "function %u: TAIL_CALL target %u is out of range", idx, callee);
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (cf->result_count != fn->result_count || cf->result_tag != fn->result_tag) {
                nvm2c_fail(b, "function %u: TAIL_CALL result signature mismatch", idx);
                goto done;
            }
            if (callee == idx) {
                if (!emit_self_tail_restart(b, &st, idx, fn, kinds)) goto done;
                terminated = 1;
                break;
            }
            char call[NVM2C_CALL_SIZE];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, call, sizeof call)) {
                goto done;
            }
            if (st.sp != 0) {
                nvm2c_fail(b, "function %u: TAIL_CALL leaves extra stack values", idx);
                goto done;
            }
            if (fn->result_count == 1 &&
                (result_is_i64(fn) || fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM || fn->result_tag == TAG_FLOAT || fn->result_tag == TAG_STRING ||
                 fn->result_tag == TAG_ARRAY || aggregate_value_tag(fn->result_tag) || fn->result_tag == TAG_HASHMAP)) {
                nvm2c_printf(b, "    nresult = %s;\n    goto L_return;\n", call);
            } else {
                nvm2c_printf(b, "    %s;\n    goto L_return;\n", call);
            }
            terminated = 1;
            break;
        }
        case OP_JMP: {
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            if (b->has_maps && tgt <= start) {
                emit_map_roots(b, &st, fn, kinds, idx);
                nvm2c_puts(b, "    nmap_collect_if_needed();\n");
            }
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            referenced[tgt] = 1;
            nvm2c_printf(b, "    goto L_%zu;\n", tgt);
            terminated = 1;
            break;
        }
        case OP_JMP_TRUE:
        case OP_JMP_FALSE: {
            int cond = stack_pop_condition(b, &st, isa_get_info(ins.opcode)->name);
            if (b->failed) goto done;
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            nvm2c_printf(b, "    if (%st[%d]) {\n", ins.opcode == OP_JMP_FALSE ? "!" : "", cond);
            if (b->has_maps && tgt <= start) {
                emit_map_roots(b, &st, fn, kinds, idx);
                nvm2c_puts(b, "    nmap_collect_if_needed();\n");
            }
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            referenced[tgt] = 1;
            nvm2c_printf(b, "    goto L_%zu;\n    }\n", tgt);
            break;
        }
        case OP_RET:
            if (scalar_return_profile(fn)) {
                if (!emit_scalar_return(b, &st, fn, idx)) goto done;
            } else if (fn->result_count == 1 && fn->result_tag == TAG_STRING) {
                int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    nresult = s[%d];\n    goto L_return;\n", s);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_HASHMAP) {
                int map = stack_pop_expect(b, &st, NVM2C_VK_MAP, "RET");
                if (b->failed) goto done;
                if (st.sp) { nvm2c_fail(b, "I cannot return a map with extra stack values"); goto done; }
                nvm2c_printf(b, "    nresult = m[%d];\n    goto L_return;\n", map);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_ARRAY) {
                int a = stack_pop_expect(b, &st, b->array_results[idx], "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    nresult = %s[%d];\n    goto L_return;\n", stack_array_name(b->array_results[idx]), a);
            } else if (fn->result_count == 1 && aggregate_value_tag(fn->result_tag)) {
                int record = stack_pop_expect(b, &st, NVM2C_VK_REC, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "I cannot return an aggregate with extra stack values");
                    goto done;
                }
                nvm2c_printf(b, "    if (r[%d].kind != %u) NVM2C_ABORT();\n    nresult = r[%d];\n    goto L_return;\n",
                             record, aggregate_kind_for_tag(fn->result_tag), record);
            } else {
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: void RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_puts(b, "    goto L_return;\n");
            }
            terminated = 1;
            break;
        case OP_HALT:
            if (result_is_i64(fn) && st.sp == 1) {
                nvm2c_printf(b, "    nresult = t[%d];\n    goto L_return;\n",
                             stack_pop_expect(b, &st, fn->result_tag == TAG_BOOL ? NVM2C_VK_BOOL : NVM2C_VK_INT, "HALT"));
            } else if (st.sp == 0 && (fn->result_count == 0 || fn->result_tag == TAG_VOID)) {
                nvm2c_puts(b, "    goto L_return;\n");
            } else if (st.sp == 0 && result_is_i64(fn)) {
                nvm2c_puts(b, "    nresult = 0;\n    goto L_return;\n");
            } else {
                nvm2c_fail(b, "function %u: HALT with unexpected stack height %d", idx, st.sp);
                goto done;
            }
            terminated = 1;
            break;
        case OP_CALL_EXTERN: {
            /* A host call may re-enter generated code through a callback. I
             * publish its caller before consuming the host arguments. */
            emit_map_roots(b, &st, fn, kinds, idx);
            if (b->has_owned_strings || b->has_owned_aggregates)
                nvm2c_puts(b, "    nmap_collect_if_needed();\n");
            const Nvm2cHost *host = import_host(mod, ins.operands[0].u32);
            if (!host) {
                nvm2c_fail(b, "CALL_EXTERN has no exact builtin host ABI");
                goto done;
            }
            char expression[128];
            if (scalar_artifact_adapter(host)) {
                int args[2] = {0};
                for (uint8_t p = host->argc; p > 0; --p)
                    args[p - 1] = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                if (b->failed) goto done;
                if (host->argc == 2)
                    snprintf(expression, sizeof expression, "nhost_artifact_%u(s[%d], s[%d])",
                             ins.operands[0].u32, args[0], args[1]);
                else if (host->argc) snprintf(expression, sizeof expression, "nhost_artifact_%u(s[%d])",
                                             ins.operands[0].u32, args[0]);
                else snprintf(expression, sizeof expression, "nhost_artifact_%u()", ins.operands[0].u32);
            } else if (host->argc == 2) {
                int right = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                int left = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                if (b->failed) goto done;
                snprintf(expression, sizeof expression, "%s(s[%d], s[%d])", host->c_name, left, right);
                if (host->result == TAG_ARRAY)
                    snprintf(expression, sizeof expression, "nhost_walk_%u(s[%d], s[%d])",
                             ins.operands[0].u32, left, right);
            } else if (host->argc) {
                uint8_t kind = host->parameter == TAG_STRING ? NVM2C_VK_STR :
                                   host->parameter == TAG_FLOAT ? NVM2C_VK_FLOAT : NVM2C_VK_INT;
                int arg = stack_pop_expect(b, &st, kind, "CALL_EXTERN");
                if (b->failed) goto done;
                snprintf(expression, sizeof expression, "%s(%c[%d])", host->c_name,
                         kind == NVM2C_VK_STR ? 's' : kind == NVM2C_VK_FLOAT ? 'f' : 't', arg);
                if (host->result == TAG_ARRAY)
                    snprintf(expression, sizeof expression, "nhost_walk_%u(s[%d])",
                             ins.operands[0].u32, arg);
            } else {
                snprintf(expression, sizeof expression, "%s()", host->c_name);
            }
            if (host->result == TAG_ARRAY) stack_push_sarr(b, &st, expression);
            else if (host->result == TAG_STRING) stack_push_str(b, &st, expression);
            else if (host->result == TAG_BOOL) stack_push_bool(b, &st, expression);
            else if (host->result == TAG_FLOAT) stack_push_float(b, &st, expression);
            else stack_push_temp(b, &st, expression);
            break;
        }
        default: {
            const InstructionInfo *info = isa_get_info(ins.opcode);
            nvm2c_fail(b, "unsupported opcode %s (0x%02X) in the nvm2c subset",
                       info ? info->name : "UNKNOWN", ins.opcode);
            goto done;
        }
        }
        if (b->failed) goto done;
        if (boolean_result(ins.opcode) && st.sp > 0) st.kinds[st.sp - 1] = NVM2C_VK_BOOL;
        previous_false_push = ins.opcode == OP_PUSH_BOOL && ins.operands[0].u8 == 0;
    }

    if (is_target[remaining] && join_set[remaining]) {
        if (!terminated && !record_join(b, idx, joins, join_set, remaining, &st)) goto done;
        stack_restore_join(b, &st, &joins[remaining]);
        terminated = 0;
        emit_pc_label(b, remaining, labels);
    }
    if (!terminated) {
        if (!scalar_return_profile(fn)) {
            nvm2c_fail(b, "function %u: I support implicit returns only for zero results or one int/u8/bool/float", idx);
            goto done;
        }
        if (!emit_scalar_return(b, &st, fn, idx)) goto done;
    }
    nvm2c_puts(b, "L_return:\n");
    if (b->has_maps) nvm2c_puts(b, "    nroot_head = nroots.prev; nroot_destroy(&nroots.live);\n");
    nvm2c_puts(b, "    free(r);\n");
    if (record_locals) nvm2c_puts(b, "    free(rl);\n");
    nvm2c_puts(b, strcmp(rt, "void") ? "    return nresult;\n}\n\n" : "    return;\n}\n\n");

    if (b->failed) goto done;
    /* Prescan targets also include skipped jumps. Keep every join decision,
     * but retain only labels referenced by actual emitted gotos. Stored byte
     * offsets survive buffer growth; patch before declaration insertion. */
    for (size_t off = 0; off <= remaining; ++off) {
        if (!referenced[off] && labels[off].length)
            memset(b->data + labels[off].offset, ' ', labels[off].length);
    }

    /* I emit the body once, then insert declarations using its actual
     * high-water counts. Record storage belongs to this invocation and is
     * released after a return snapshot. Self-tail restarts reuse the allocation. */
    {
        char declarations[1024];
        int count = snprintf(declarations, sizeof declarations,
            "    int64_t t[%d] = {0}; (void)t;\n"
            "    double f[%d] = {0}; (void)f;\n"
            "    const char *s[%d] = {0}; (void)s;\n"
            "    narr_t a[%d] = {0}; (void)a;\n"
            "    nsarr_t sa[%d] = {0}; (void)sa;\n"
            "    nrec_t *r = %d ? calloc(%d, sizeof *r) : NULL;\n"
            "    if (%d && !r) NVM2C_ABORT();\n"
            "    nrarr_t ra[%d] = {0}; (void)ra;\n"
            "    nmap_t m[%d] = {0}; (void)m;\n",
            st.next_temp ? st.next_temp : 1, st.next_float ? st.next_float : 1,
            st.next_str ? st.next_str : 1,
            st.next_arr ? st.next_arr : 1, st.next_sarr ? st.next_sarr : 1,
            st.next_rec, st.next_rec, st.next_rec, st.next_rarr ? st.next_rarr : 1,
            st.next_map ? st.next_map : 1);
        if (count < 0 || (size_t)count >= sizeof declarations) {
            nvm2c_fail(b, "I cannot format temporary declarations");
            goto done;
        }
        if (strcmp(rt, "void")) {
            int extra = snprintf(declarations + count, sizeof declarations - (size_t)count,
                                 "    %s nresult = {0};\n", rt);
            if (extra < 0 || (size_t)extra >= sizeof declarations - (size_t)count) {
                nvm2c_fail(b, "I cannot format the return snapshot"); goto done;
            }
            count += extra;
        }
        if (b->has_maps) {
            int extra = snprintf(declarations + count, sizeof declarations - (size_t)count,
                                 "    nmap_value v[%d] = {0}; (void)v;\n"
                                 "    nroot_frame nroots = {0}; nroots.prev = nroot_head; nroot_head = &nroots;\n",
                                 st.next_value ? st.next_value : 1);
            if (extra < 0 || (size_t)extra >= sizeof declarations - (size_t)count) {
                nvm2c_fail(b, "I cannot format tagged temporary declarations"); goto done;
            }
            count += extra;
        }
        if (!nvm2c_grow(b, (size_t)count)) goto done;
        memmove(b->data + declarations_at + count, b->data + declarations_at,
                b->len - declarations_at + 1);
        memcpy(b->data + declarations_at, declarations, (size_t)count);
        b->len += (size_t)count;
    }

done:
    if (joins) for (size_t off = 0; off <= remaining; ++off) {
        free(joins[off].slots);
        free(joins[off].kinds);
        field_table_free(joins[off].rec_k);
        field_table_free(joins[off].rarr_k);
    }
    free(st.slots);
    free(st.kinds);
    field_table_free(st.rec_k);
    field_table_free(st.rarr_k);
    free(literal_elems);
    free(aggregate_kinds);
    free(is_start);
    free(is_target);
    free(referenced);
    free(labels);
    free(joins);
    free(join_set);
}

static int module_has_opcode(const NvmModule *mod, uint8_t op) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) {
            continue;
        }
        const uint8_t *code = mod->code + fn->code_offset;
        size_t remaining = fn->code_length;
        size_t pc = 0;
        while (pc < remaining) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
            if (n == 0) break;
            if (ins.opcode == op) return 1;
            pc += n;
        }
    }
    return 0;
}

static int module_has_array_constructor(const Nvm2cBuf *b, const NvmModule *mod, const uint8_t *kinds,
                                        uint8_t wanted) {
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) continue;
        const uint8_t *code = mod->code + fn->code_offset;
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(code + pc, fn->code_length - pc, &ins);
            if (!n) break;
            pc += n;
            if (ins.opcode != OP_ARR_NEW) continue;
            uint8_t kind = ins.operands[0].u8 == TAG_STRING ? NVM2C_VK_SARR :
                           ins.operands[0].u8 == TAG_STRUCT ? NVM2C_VK_RARR :
                           ins.operands[0].u8 == TAG_FLOAT ? NVM2C_VK_FARR : ins.operands[0].u8 == TAG_BOOL ? NVM2C_VK_BARR : NVM2C_VK_ARR;
            if (ins.operands[0].u8 == TAG_INT) {
                DecodedInstruction next;
                if (isa_decode(code + pc, fn->code_length - pc, &next) &&
                    next.opcode == OP_STORE_LOCAL && next.operands[0].u16 < fn->local_count) {
                    uint8_t local = fn_local_kind(b, kinds, f, next.operands[0].u16);
                    if (local == NVM2C_VK_SARR || local == NVM2C_VK_RARR) kind = local;
                }
            }
            if (kind == wanted) return 1;
        }
    }
    return 0;
}

static int module_has_arr_op_tag(const NvmModule *mod, uint8_t op, uint8_t tag) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) {
            continue;
        }
        const uint8_t *code = mod->code + fn->code_offset;
        size_t remaining = fn->code_length;
        size_t pc = 0;
        while (pc < remaining) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
            if (n == 0) break;
            if (ins.opcode == op && ins.operands[0].u8 == tag) return 1;
            pc += n;
        }
    }
    return 0;
}

static int module_uses_host(const NvmModule *mod, const char *name) {
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size || fn->code_length > mod->code_size - fn->code_offset)
            continue;
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t size = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
            if (!size) break;
            if (ins.opcode == OP_CALL_EXTERN) {
                const Nvm2cHost *host = import_host(mod, ins.operands[0].u32);
                if (host && strcmp(host->c_name, name) == 0) return 1;
            }
            pc += size;
        }
    }
    return 0;
}

static int module_has_local_kind(const Nvm2cBuf *b, const uint8_t *kinds, uint32_t fn_count, uint8_t kind) {
    uint32_t i;
    uint16_t li;
    for (i = 0; i < fn_count; i++) {
        for (li = 0; li < b->local_width; li++) {
            if (fn_local_kind(b, kinds, i, li) == kind) return 1;
        }
    }
    return 0;
}

static void emit_nstr_storage(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "typedef struct nstr_owned { struct nstr_owned *next; size_t bytes; char data[]; } nstr_owned;\n"
        "static nstr_owned *nstr_owners;\n"
        "static size_t nstr_live_bytes, nstr_peak_bytes, nstr_allocation_debt;\n"
        "static size_t nstr_collection_budget = 65536;\n"
        "static char *nstr_try_allocate(size_t n) {\n"
        "    if (n > SIZE_MAX - sizeof(nstr_owned) - 1) return NULL;\n"
        "    size_t bytes = sizeof(nstr_owned) + n + 1;\n"
        "    if (bytes > SIZE_MAX - nstr_live_bytes || bytes > SIZE_MAX - nstr_allocation_debt) return NULL;\n"
        "    nstr_owned *owner = malloc(bytes);\n"
        "    if (!owner) { return NULL; } owner->next = nstr_owners; nstr_owners = owner;\n"
        "    owner->bytes = bytes; nstr_live_bytes += bytes; nstr_allocation_debt += bytes;\n"
        "    if (nstr_live_bytes > nstr_peak_bytes) nstr_peak_bytes = nstr_live_bytes;\n"
        "    owner->data[n] = 0; return owner->data;\n}\n"
        "static char *nstr_allocate(size_t n) {\n"
        "    char *value = nstr_try_allocate(n); if (!value) NVM2C_ABORT(); return value;\n}\n"
        "static inline const char *nstr_copy_release(const char *value, void (*release)(const char *)) {\n"
        "    size_t length = value ? strlen(value) : 0;\n"
        "    char *copy = value ? nstr_try_allocate(length) : NULL;\n"
        "    if (copy) memcpy(copy, value, length + 1);\n"
        "    release(value); if (!copy) NVM2C_ABORT(); return copy;\n}\n"
        "static inline const char *nstr_copy(const char *value) {\n"
        "    if (!value) value = \"\";\n"
        "    size_t length = strlen(value);\n"
        "    char *copy = nstr_allocate(length);\n"
        "    memcpy(copy, value, length + 1); return copy;\n}\n"
        "/* I consume only exact malloc-owned builtin temporaries. */\n"
        "static inline const char *nstr_take(char *value) {\n"
        "    if (!value) NVM2C_ABORT();\n"
        "    const char *copy = nstr_copy(value); free(value); return copy;\n}\n"
        "static void nstr_release_owned(void) {\n"
        "    while (nstr_owners) { nstr_owned *owner = nstr_owners;\n"
        "        nstr_owners = owner->next; free(owner); }\n"
        "    nstr_live_bytes = 0; nstr_allocation_debt = 0; nstr_collection_budget = 65536;\n}\n");
}

static void emit_nstr_sweep(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "/* I inspect known owners, never a header before an arbitrary borrowed pointer. */\n"
        "static void nstr_sweep(const nroot_list *roots) {\n"
        "    nstr_owned **link = &nstr_owners;\n"
        "    while (*link) {\n"
        "        nstr_owned *owner = *link;\n"
        "        if (nroot_contains(roots, 1, owner->data)) link = &owner->next;\n"
        "        else { *link = owner->next; nstr_live_bytes -= owner->bytes; free(owner); }\n"
        "    }\n"
        "    nstr_allocation_debt = 0;\n"
        "    nstr_collection_budget = nstr_live_bytes > 65536 ? nstr_live_bytes : 65536;\n"
        "}\n");
}

static void emit_nstr_concat(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_concat(const char *a, const char *b) {\n"
        "    size_t na = strlen(a ? a : \"\");\n"
        "    size_t nb = strlen(b ? b : \"\");\n"
        "    if (na > SIZE_MAX - nb) NVM2C_ABORT();\n"
        "    char *p = nstr_allocate(na + nb);\n"
        "    memcpy(p, a ? a : \"\", na);\n"
        "    memcpy(p + na, b ? b : \"\", nb);\n"
        "    p[na + nb] = 0;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_trim(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_trim(const char *s) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t start = 0, end = strlen(src);\n"
        "    while (start < end && (src[start] == ' ' || src[start] == '\\t' ||\n"
        "           src[start] == '\\n' || src[start] == '\\r')) start++;\n"
        "    while (end > start && (src[end-1] == ' ' || src[end-1] == '\\t' ||\n"
        "           src[end-1] == '\\n' || src[end-1] == '\\r')) end--;\n"
        "    char *result = nstr_allocate(end - start);\n"
        "    memcpy(result, src + start, end - start);\n"
        "    return result;\n}\n\n");
}

static void emit_nstr_substr(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_substr(const char *s, int64_t start, int64_t len) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t slen = strlen(src);\n"
        "    if (start < 0) start = 0;\n"
        "    if ((uint64_t)start >= slen || len <= 0) return \"\";\n"
        "    size_t count = (uint64_t)len > slen - (size_t)start ? slen - (size_t)start : (size_t)len;\n"
        "    char *p = nstr_allocate(count);\n"
        "    memcpy(p, src + (size_t)start, count);\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_char_at(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_char_at(const char *s, int64_t idx) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t n = strlen(src);\n"
        "    if (idx < 0 || (size_t)idx >= n) return -1;\n"
        "    return (int64_t)(unsigned char)src[idx];\n"
        "}\n\n");
}

static void emit_nstr_starts_with(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_starts_with(const char *s, const char *pre) {\n"
        "    const char *a = s ? s : \"\";\n"
        "    const char *b = pre ? pre : \"\";\n"
        "    size_t nb = strlen(b);\n"
        "    return (int64_t)(strncmp(a, b, nb) == 0);\n"
        "}\n\n");
}

static void emit_nstr_ends_with(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_ends_with(const char *s, const char *suf) {\n"
        "    const char *a = s ? s : \"\";\n"
        "    const char *b = suf ? suf : \"\";\n"
        "    size_t na = strlen(a);\n"
        "    size_t nb = strlen(b);\n"
        "    if (nb > na) return 0;\n"
        "    return (int64_t)(memcmp(a + (na - nb), b, nb) == 0);\n"
        "}\n\n");
}

static void emit_nstr_from_i64(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_from_i64(int64_t v) {\n"
        "    char tmp[32];\n"
        "    int n = snprintf(tmp, sizeof tmp, \"%lld\", (long long)v);\n"
        "    if (n < 0 || (size_t)n >= sizeof tmp) NVM2C_ABORT();\n"
        "    char *p = nstr_allocate((size_t)n);\n"
        "    memcpy(p, tmp, (size_t)n + 1);\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_from_f64(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_from_f64(double value) {\n"
        "    char tmp[64]; int n = nano_rt_f64_format(tmp, sizeof tmp, value);\n"
        "    if (n < 0 || (size_t)n >= sizeof tmp) NVM2C_ABORT();\n"
        "    char *text = nstr_allocate((size_t)n);\n"
        "    memcpy(text, tmp, (size_t)n + 1); return text;\n}\n");
}

static void emit_nagg_accounting(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static size_t nagg_live_bytes, nagg_peak_bytes, nagg_allocation_debt;\n"
        "static size_t nagg_collection_budget = 65536;\n"
        "static inline void nagg_add(size_t bytes) {\n"
        "    if (bytes > SIZE_MAX - nagg_live_bytes || bytes > SIZE_MAX - nagg_allocation_debt) NVM2C_ABORT();\n"
        "    nagg_live_bytes += bytes; nagg_allocation_debt += bytes;\n"
        "    if (nagg_live_bytes > nagg_peak_bytes) nagg_peak_bytes = nagg_live_bytes;\n}\n"
        "static inline void nagg_drop(size_t bytes) {\n"
        "    if (bytes > nagg_live_bytes) NVM2C_ABORT();\n"
        "    nagg_live_bytes -= bytes;\n}\n");
}

static void emit_nagg_sweep(Nvm2cBuf *b, int snapshots) {
    nvm2c_puts(b, "static void nagg_sweep(const nroot_list *roots) {\n    (void)roots;\n");
    if (snapshots) nvm2c_puts(b,
        "    nrec_owned **records = &nrec_owned_head;\n"
        "    while (*records) {\n"
        "        nrec_owned *owner = *records;\n"
        "        if (nroot_contains(roots, 4, &owner->value)) records = &owner->next;\n"
        "        else { *records = owner->next; nagg_drop(sizeof *owner); free(owner); }\n"
        "    }\n");
    const char *pools[] = {"narr", "nsarr", "nrarr"};
    int present[] = {b->has_integer_arrays, b->has_string_arrays, b->has_record_array_allocations};
    for (int i = 0; i < 3; ++i) {
        if (!present[i]) continue;
        const char *name = pools[i];
        nvm2c_printf(b,
            "    { struct %s_owner **link = &%s_owners;\n"
            "      while (*link) { struct %s_owner *owner = *link;\n"
            "        if (nroot_contains(roots, 128, owner)) link = &owner->next;\n"
            "        else { *link = owner->next; nagg_drop(owner->bytes);\n"
            "            free(owner->data); free(owner->handle); free(owner); }\n"
            "      } }\n", name, name, name);
    }
    if (b->has_string_arrays) nvm2c_puts(b,
        "    nsarr_string **strings = &nsarr_strings;\n"
        "    while (*strings) { nsarr_string *owner = *strings;\n"
        "        if (nroot_contains(roots, 1, owner->data)) strings = &owner->next;\n"
        "        else { *strings = owner->next; nagg_drop(owner->bytes); free(owner->data); free(owner); }\n"
        "    }\n");
    nvm2c_puts(b,
        "    nagg_allocation_debt = 0;\n"
        "    nagg_collection_budget = nagg_live_bytes > 65536 ? nagg_live_bytes : 65536;\n}\n");
}

static void emit_narr_storage(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "struct narr_owner { int64_t *data; size_t cap, bytes; narr_t handle; struct narr_owner *next; };\n"
        "static struct narr_owner *narr_owners;\n"
        "static inline struct narr_owner *narr_track(narr_t a, int own_handle) {\n"
        "    struct narr_owner *owner = calloc(1, sizeof *owner); if (!owner) NVM2C_ABORT();\n"
        "    owner->bytes = sizeof *owner + (own_handle ? sizeof *a : 0); nagg_add(owner->bytes);\n"
        "    owner->handle = own_handle ? a : NULL; owner->next = narr_owners;\n"
        "    narr_owners = owner; a->owner = owner; return owner;\n}\n"
        "static inline narr_t narr_new(void) {\n"
        "    narr_t a = calloc(1, sizeof *a); if (!a) NVM2C_ABORT();\n"
        "    narr_track(a, 1); return a;\n}\n"
        "static inline void narr_reserve(narr_t a, size_t n) {\n"
        "    size_t limit = SIZE_MAX / sizeof *a->data;\n"
        "    if (!a || n > limit || a->len > limit || (a->len && !a->data)) NVM2C_ABORT();\n"
        "    struct narr_owner *owner = a->owner;\n"
        "    if (owner && (owner->data != a->data || a->len > owner->cap)) NVM2C_ABORT();\n"
        "    if (owner && n <= owner->cap) return;\n"
        "    if (n < a->len) n = a->len;\n"
        "    size_t cap = owner && owner->cap ? owner->cap : 8;\n"
        "    while (cap < n) { if (cap > limit / 2) { cap = n; break; } cap *= 2; }\n"
        "    if (!owner) {\n"
        "        int64_t *data = malloc(cap * sizeof *data); if (!data) NVM2C_ABORT();\n"
        "        if (a->len) memcpy(data, a->data, a->len * sizeof *data);\n"
        "        owner = narr_track(a, 0); owner->data = data;\n"
        "    } else {\n"
        "        int64_t *data = realloc(owner->data, cap * sizeof *data);\n"
        "        if (!data) { NVM2C_ABORT(); } owner->data = data;\n"
        "    }\n"
        "    size_t growth = (cap - owner->cap) * sizeof *owner->data;\n"
        "    nagg_add(growth); owner->bytes += growth;\n"
        "    owner->cap = cap; a->data = owner->data;\n}\n"
        "static inline void narr_release_owned(void) {\n"
        "    while (narr_owners) { struct narr_owner *owner = narr_owners;\n"
        "        narr_owners = owner->next; nagg_drop(owner->bytes); free(owner->data); free(owner->handle); free(owner); }\n}\n");
}

static void emit_narr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_lit(const int64_t *elems, size_t n) {\n"
        "    if (n > 0 && !elems) NVM2C_ABORT();\n"
        "    narr_t a = narr_new(); narr_reserve(a, n);\n"
        "    if (n) memcpy(a->data, elems, n * sizeof *a->data);\n"
        "    a->len = n; return a;\n}\n");
}

static void emit_narr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t narr_get(narr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) NVM2C_ABORT();\n"
        "    return a->data[idx];\n}\n");
}

static void emit_narr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static inline narr_t narr_push(narr_t a, int64_t v) {\n"
        "    if (!a || a->len == SIZE_MAX) NVM2C_ABORT();\n"
        "    narr_reserve(a, a->len + 1); a->data[a->len++] = v; return a;\n}\n");
}

static void emit_nsarr_storage(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "#include <string.h>\n"
        "struct nsarr_owner { const char **data; size_t cap, bytes; nsarr_t handle; struct nsarr_owner *next; };\n"
        "static struct nsarr_owner *nsarr_owners;\n"
        "typedef struct nsarr_string { char *data; size_t bytes; struct nsarr_string *next; } nsarr_string;\n"
        "static nsarr_string *nsarr_strings;\n"
        "static inline struct nsarr_owner *nsarr_track(nsarr_t a, int own_handle) {\n"
        "    struct nsarr_owner *owner = calloc(1, sizeof *owner);\n"
        "    if (!owner) NVM2C_ABORT();\n"
        "    owner->bytes = sizeof *owner + (own_handle ? sizeof *a : 0); nagg_add(owner->bytes);\n"
        "    owner->handle = own_handle ? a : NULL; owner->next = nsarr_owners;\n"
        "    nsarr_owners = owner; a->owner = owner; return owner;\n}\n"
        "static inline nsarr_t nsarr_new(void) {\n"
        "    nsarr_t a = calloc(1, sizeof *a); if (!a) NVM2C_ABORT();\n"
        "    nsarr_track(a, 1); return a;\n}\n"
        "static inline void nsarr_reserve(nsarr_t a, size_t n) {\n"
        "    size_t limit = SIZE_MAX / sizeof *a->data;\n"
        "    if (!a || n > limit || a->len > limit || (a->len && !a->data)) NVM2C_ABORT();\n"
        "    struct nsarr_owner *owner = a->owner;\n"
        "    if (owner && (owner->data != a->data || a->len > owner->cap)) NVM2C_ABORT();\n"
        "    if (owner && n <= owner->cap) return;\n"
        "    if (n < a->len) n = a->len;\n"
        "    size_t cap = owner && owner->cap ? owner->cap : 8;\n"
        "    while (cap < n) { if (cap > limit / 2) { cap = n; break; } cap *= 2; }\n"
        "    if (!owner) {\n"
        "        const char **data = malloc(cap * sizeof *data); if (!data) NVM2C_ABORT();\n"
        "        if (a->len) memcpy(data, a->data, a->len * sizeof *data);\n"
        "        owner = nsarr_track(a, 0); owner->data = data;\n"
        "    } else {\n"
        "        const char **data = realloc(owner->data, cap * sizeof *data);\n"
        "        if (!data) { NVM2C_ABORT(); } owner->data = data;\n"
        "    }\n"
        "    size_t growth = (cap - owner->cap) * sizeof *owner->data;\n"
        "    nagg_add(growth); owner->bytes += growth;\n"
        "    owner->cap = cap; a->data = owner->data;\n}\n"
        "static inline const char *nsarr_copy_string(const char *value) {\n"
        "    if (!value) { NVM2C_ABORT(); } size_t n = strlen(value); if (n == SIZE_MAX) NVM2C_ABORT();\n"
        "    if (n > SIZE_MAX - sizeof(nsarr_string) - 1) NVM2C_ABORT();\n"
        "    nsarr_string *owner = malloc(sizeof *owner); if (!owner) NVM2C_ABORT();\n"
        "    owner->data = malloc(n + 1); if (!owner->data) NVM2C_ABORT();\n"
        "    owner->bytes = sizeof *owner + n + 1; nagg_add(owner->bytes);\n"
        "    memcpy(owner->data, value, n + 1); owner->next = nsarr_strings;\n"
        "    nsarr_strings = owner; return owner->data;\n}\n"
        "static inline void nsarr_release_owned(void) {\n"
        "    while (nsarr_owners) { struct nsarr_owner *owner = nsarr_owners;\n"
        "        nsarr_owners = owner->next; nagg_drop(owner->bytes); free(owner->data); free(owner->handle); free(owner); }\n"
        "    while (nsarr_strings) { nsarr_string *owner = nsarr_strings;\n"
        "        nsarr_strings = owner->next; nagg_drop(owner->bytes); free(owner->data); free(owner); }\n}\n");
}

static void emit_nsarr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_lit(const char *const *elems, size_t n) {\n"
        "    if (n > 0 && !elems) NVM2C_ABORT();\n"
        "    nsarr_t a = nsarr_new();\n"
        "    nsarr_reserve(a, n);\n"
        "    if (n) memcpy(a->data, elems, n * sizeof *a->data);\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nsarr_get(nsarr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) NVM2C_ABORT();\n"
        "    return a->data[idx] ? a->data[idx] : \"\";\n"
        "}\n\n");
}

static void emit_nsarr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static inline nsarr_t nsarr_push(nsarr_t a, const char *v) {\n"
        "    if (!a || a->len == SIZE_MAX) NVM2C_ABORT();\n"
        "    size_t n = a->len + 1;\n"
        "    nsarr_reserve(a, n);\n"
        "    a->data[a->len] = v ? v : \"\";\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_tagged_array_helpers(Nvm2cBuf *b, int int_push, int string_push,
                                     int int_get, int string_get, int printing) {
    nvm2c_puts(b,
        "static inline narr_t nvalue_require_int_array(nmap_value a) {\n"
        "    if (a.kind != 7 || a.integer != 3 || !a.text) NVM2C_ABORT();\n"
        "    return (narr_t)a.text;\n}\n"
        "static inline int64_t nvalue_array_len(nmap_value a) {\n"
        "    if (a.kind != 7 || !a.text) NVM2C_ABORT();\n"
        "    if (a.integer == 3 || a.integer == 10 || a.integer == 12) return (int64_t)((narr_t)a.text)->len;\n"
        "    if (a.integer == 5) return (int64_t)((nsarr_t)a.text)->len;\n"
        "    if (a.integer == 6) return (int64_t)((nrarr_t)a.text)->len;\n"
        "    NVM2C_ABORT();\n}\n"
        "static inline nmap_value nvalue_array_get(nmap_value a, int64_t index) {\n"
        "    if (a.integer == 6) NVM2C_ABORT();\n"
        "    int64_t length = nvalue_array_len(a);\n"
        "    if (index < 0 || (uint64_t)index >= (uint64_t)length) return (nmap_value){0, 0, NULL};\n"
        "    size_t at = (size_t)index;\n");
    nvm2c_printf(b,
        "    if (a.integer == 3 || a.integer == 10 || a.integer == 12) return (nmap_value){a.integer == 12 ? 3 : a.integer == 10 ? 4 : 1, %s, NULL};\n"
        "    return (nmap_value){5, 0, (char *)%s};\n}\n",
        int_get ? "narr_get((narr_t)a.text, at)" : "((narr_t)a.text)->data[at]",
        string_get ? "nsarr_get((nsarr_t)a.text, at)" : "((nsarr_t)a.text)->data[at]");
    nvm2c_puts(b,
        "static inline nmap_value nvalue_array_set(nmap_value a, int64_t index, nmap_value value) {\n"
        "    if (a.integer == 6) NVM2C_ABORT();\n"
        "    if (index < 0 || (uint64_t)index >= (uint64_t)nvalue_array_len(a)) NVM2C_ABORT();\n"
        "    size_t at = (size_t)index;\n"
        "    if (a.integer == 3) ((narr_t)a.text)->data[at] = nvalue_require_int(value);\n"
        "    else if (a.integer == 10) ((narr_t)a.text)->data[at] = nvalue_require_bool(value);\n"
        "    else if (a.integer == 12) ((narr_t)a.text)->data[at] = nvalue_from_float(nvalue_require_float(value)).integer;\n"
        "    else ((nsarr_t)a.text)->data[at] = nvalue_require_string(value);\n"
        "    return a;\n}\n"
        "static inline nmap_value nvalue_array_push(nmap_value a, nmap_value value) {\n"
        "    (void)nvalue_array_len(a); (void)value;\n");
    if (int_push) nvm2c_puts(b,
        "    if (a.integer == 3) { narr_push((narr_t)a.text, nvalue_require_int(value)); return a; }\n"
        "    if (a.integer == 10) { narr_push((narr_t)a.text, nvalue_require_bool(value)); return a; }\n"
        "    if (a.integer == 12) { narr_push((narr_t)a.text, nvalue_from_float(nvalue_require_float(value)).integer); return a; }\n");
    if (string_push) nvm2c_puts(b,
        "    if (a.integer == 5) { nsarr_push((nsarr_t)a.text, nvalue_require_string(value)); return a; }\n");
    nvm2c_puts(b, "    NVM2C_ABORT();\n}\n");
    if (printing) nvm2c_puts(b,
        "static inline void nvalue_array_print(nmap_value a) {\n"
        "    if (a.integer == 6) NVM2C_ABORT();\n"
        "    int64_t length = nvalue_array_len(a); fputc('[', stdout);\n"
        "    for (int64_t i = 0; i < length; ++i) {\n"
        "        if (i) fputs(\", \", stdout);\n"
        "        nmap_value v = nvalue_array_get(a, i);\n"
        "        if (v.kind == 1) printf(\"%lld\", (long long)v.integer);\n"
        "        else if (v.kind == 3) nf64_print(nvalue_require_float(v));\n"
        "        else if (v.kind == 4) fputs(v.integer ? \"true\" : \"false\", stdout);\n"
        "        else fputs(v.text, stdout);\n"
        "    }\n    fputc(']', stdout);\n}\n");
}

static void emit_host_normalize(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static inline const char *nhost_normalize(const char *path) {\n"
        "    if (!path) path = \"\";\n"
        "    size_t length = strlen(path), slots = length / 2 + 1;\n"
        "    if (length > SIZE_MAX - 2 || slots > SIZE_MAX / sizeof(size_t)) NVM2C_ABORT();\n"
        "    char *out = nstr_allocate(length + 1);\n"
        "    size_t *bases = malloc(slots * sizeof *bases);\n"
        "    if (!out || !bases) NVM2C_ABORT();\n"
        "    int absolute = path[0] == '/';\n"
        "    size_t used = 0, count = 0, cursor = 0;\n"
        "    if (absolute) out[used++] = '/';\n"
        "    while (cursor < length) {\n"
        "        if (path[cursor] == '/') { ++cursor; continue; }\n"
        "        size_t start = cursor;\n"
        "        while (cursor < length && path[cursor] != '/') ++cursor;\n"
        "        size_t size = cursor - start;\n"
        "        if (size == 1 && path[start] == '.') continue;\n"
        "        if (size == 2 && path[start] == '.' && path[start + 1] == '.') {\n"
        "            if (count) {\n"
        "                size_t previous = bases[count - 1];\n"
        "                if (out[previous] == '/') ++previous;\n"
        "                if (!(used - previous == 2 && out[previous] == '.' && out[previous + 1] == '.')) {\n"
        "                    used = bases[--count]; continue;\n"
        "                }\n"
        "            }\n"
        "            if (absolute) continue;\n"
        "        }\n"
        "        if (count >= slots) NVM2C_ABORT();\n"
        "        bases[count++] = used;\n"
        "        if (used && out[used - 1] != '/') out[used++] = '/';\n"
        "        memcpy(out + used, path + start, size); used += size;\n"
        "    }\n"
        "    if (!used) out[used++] = '.';\n"
        "    out[used] = 0; free(bases); return out;\n}\n");
}

static void emit_host_identity(Nvm2cBuf *b, int identity, int destinations) {
    nvm2c_puts(b, "#include <sys/stat.h>\n#include <errno.h>\n");
    if (identity) nvm2c_puts(b,
        "static inline int64_t nhost_identity(const char *source, const char *candidate) {\n"
        "    struct stat a, z;\n"
        "    if (!source || !*source || !candidate || !*candidate) return -1;\n"
        "    if (stat(source, &a) != 0) return -1;\n"
        "    if (stat(candidate, &z) != 0) return errno == ENOENT || errno == ENOTDIR ? 0 : -1;\n"
        "    return a.st_dev == z.st_dev && a.st_ino == z.st_ino ? 1 : 0;\n}\n");
    if (destinations) nvm2c_puts(b,
        "static int nhost_destination_stat(const char *path, struct stat *info) {\n"
        "    if (stat(path, info) == 0) return 1;\n"
        "    if (errno != ENOENT) return -1;\n"
        "    if (lstat(path, info) == 0 || errno != ENOENT) return -1;\n"
        "    return 0;\n}\n"
        "static inline int64_t nhost_destinations(const char *first, const char *second) {\n"
        "    if (!first || !*first || !second || !*second) return -1;\n"
        "    struct stat a, z;\n"
        "    int ae = nhost_destination_stat(first, &a);\n"
        "    int ze = nhost_destination_stat(second, &z);\n"
        "    if (ae < 0 || ze < 0) return -1;\n"
        "    if (ae && ze) return a.st_dev == z.st_dev && a.st_ino == z.st_ino;\n"
        "    if (ae || ze) return 0;\n"
        "    if (mkdir(first, 0700) != 0) return -1;\n"
        "    int64_t result = -1;\n"
        "    if (stat(first, &a) == 0) {\n"
        "        ze = nhost_destination_stat(second, &z);\n"
        "        if (ze == 0) result = 0;\n"
        "        else if (ze > 0) result = a.st_dev == z.st_dev && a.st_ino == z.st_ino;\n"
        "    }\n"
        "    if (rmdir(first) != 0) return -1;\n"
        "    return result;\n}\n");
}

static void emit_host_file_write(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "#include <stdio.h>\n"
        "static inline int64_t nhost_file_write(const char *path, const char *content) {\n"
        "    if (!path || !content) return -1;\n"
        "    FILE *file = fopen(path, \"w\");\n"
        "    if (!file) return -1;\n"
        "    size_t length = strlen(content);\n"
        "    size_t written = fwrite(content, 1, length, file);\n"
        "    int closed = fclose(file);\n"
        "    return written == length && closed == 0 ? 0 : -1;\n}\n");
}

static void emit_host_file_read(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "#include <stdio.h>\n"
        "static inline const char *nhost_file_read(const char *path) {\n"
        "    FILE *file = path ? fopen(path, \"rb\") : NULL;\n"
        "    size_t used = 0, capacity = 1;\n"
        "    char *text = malloc(capacity);\n"
        "    if (!text) NVM2C_ABORT();\n"
        "    int invalid = 0;\n"
        "    if (file) {\n"
        "        char chunk[4096];\n"
        "        size_t count;\n"
        "        while ((count = fread(chunk, 1, sizeof chunk, file)) != 0) {\n"
        "            if (memchr(chunk, 0, count)) invalid = 1;\n"
        "            if (invalid) continue;\n"
        "            if (used > SIZE_MAX - count - 1) NVM2C_ABORT();\n"
        "            size_t needed = used + count + 1;\n"
        "            if (needed > capacity) {\n"
        "                capacity = needed > SIZE_MAX / 2 ? needed : needed * 2;\n"
        "                char *grown = realloc(text, capacity);\n"
        "                if (!grown) NVM2C_ABORT();\n"
        "                text = grown;\n"
        "            }\n"
        "            memcpy(text + used, chunk, count); used += count;\n"
        "        }\n"
        "        if (ferror(file)) invalid = 1;\n"
        "        if (fclose(file) != 0) invalid = 1;\n"
        "    }\n"
        "    text[invalid ? 0 : used] = 0;\n"
        "    return nstr_take(text);\n}\n");
}

static void emit_scalar_artifact_adapters(Nvm2cBuf *b, const NvmModule *mod) {
    for (uint32_t i = 0; i < mod->import_count; ++i) {
        const Nvm2cHost *host = import_host(mod, i);
        if (!scalar_artifact_adapter(host)) continue;
        const char *result = host->result == TAG_STRING ? "const char *" :
                             host->result == TAG_BOOL ? "bool" : "int64_t";
        const char *parameters = host->argc == 2 ? "const char *a, const char *z" :
                                 host->argc == 1 ? "const char *a" : "void";
        const char *types = host->argc == 2 ? "const char *, const char *" :
                            host->argc == 1 ? "const char *" : "void";
        nvm2c_puts(b, "#include <dlfcn.h>\n#include <stdbool.h>\n");
        nvm2c_printf(b, "static inline %s nhost_artifact_%u(%s) {\n", result, i, parameters);
        nvm2c_printf(b, "    static void *library;\n    static %s (*function)(%s);\n", result, types);
        if (host->result == TAG_STRING)
            nvm2c_puts(b, "    static void (*release)(const char *);\n");
        nvm2c_puts(b, "    if (!library) {\n        library = dlopen(");
        const NvmImportEntry *imp = &mod->imports[i];
        emit_c_string_lit(b, mod->strings[imp->module_name_idx], mod->string_lengths[imp->module_name_idx]);
        nvm2c_puts(b, ", RTLD_NOW | RTLD_LOCAL);\n        if (!library) NVM2C_ABORT();\n");
        nvm2c_printf(b, "        function = (%s (*)(%s))dlsym(library, \"%s\");\n", result, types, host->name);
        nvm2c_puts(b, "        if (!function) NVM2C_ABORT();\n");
        if (host->result == TAG_STRING) {
            nvm2c_printf(b, "        void *cleanup = dlsym(library, \"%s__nano_string_release_v1\");\n", host->name);
            nvm2c_puts(b,
                "        if (cleanup) {\n            Dl_info origin, companion;\n"
                "            if (!dladdr((void *)function, &origin) || !dladdr(cleanup, &companion) ||\n"
                "                origin.dli_fbase != companion.dli_fbase) NVM2C_ABORT();\n"
                "            release = (void (*)(const char *))cleanup;\n        }\n");
        }
        nvm2c_puts(b, "    }\n");
        nvm2c_printf(b, "    %s value = function(%s);\n", result,
                      host->argc == 2 ? "a, z" : host->argc == 1 ? "a" : "");
        if (host->result == TAG_STRING) nvm2c_puts(b,
            "    if (release) return nstr_copy_release(value, release);\n"
            "    if (!value) NVM2C_ABORT();\n");
        if (!strcmp(host->c_name, "nhost_snapshot")) {
            nvm2c_puts(b, "    return nstr_copy(value);\n}\n");
        } else nvm2c_puts(b, "    return value;\n}\n");
    }
}

static void emit_walk_adapters(Nvm2cBuf *b, const NvmModule *mod) {
    int emitted = 0;
    for (uint32_t i = 0; i < mod->import_count; ++i) {
        const Nvm2cHost *host = import_host(mod, i);
        if (!host || host->result != TAG_ARRAY) continue;
        if (!emitted++) nvm2c_puts(b,
            "#include <dlfcn.h>\n#include <stdbool.h>\n"
            "typedef enum { nh_int=1, nh_float=2, nh_string=3, nh_bool=4,\n"
            "    nh_array=5, nh_struct=6, nh_pointer=7, nh_u8=8 } nh_element;\n"
            "typedef struct { int64_t length, capacity; nh_element type;\n"
            "    uint8_t width; void *data; } nh_array_value;\n");
        const char *parameters = host->argc == 2 ? "const char *root, const char *extension" : "const char *root";
        const char *types = host->argc == 2 ? "const char *, const char *" : "const char *";
        const char *release_name = !strcmp(host->name, "fs_walkdir") ? "fs_walkdir_release" : "nl_fs_list_release";
        nvm2c_printf(b, "static inline nsarr_t nhost_walk_%u(%s) {\n", i, parameters);
        nvm2c_puts(b, "    static void *library;\n");
        nvm2c_printf(b, "    static nh_array_value *(*walk)(%s);\n", types);
        nvm2c_puts(b,
            "    static bool (*release)(nh_array_value *);\n"
            "    if (!library) {\n"
            "        library = dlopen(");
        const NvmImportEntry *imp = &mod->imports[i];
        emit_c_string_lit(b, mod->strings[imp->module_name_idx],
                          mod->string_lengths[imp->module_name_idx]);
        nvm2c_puts(b,
            ", RTLD_NOW | RTLD_LOCAL);\n"
            "        if (!library) NVM2C_ABORT();\n");
        nvm2c_printf(b,
            "        const uint32_t *abi = (const uint32_t *)dlsym(library, \"%s__nano_array_abi\");\n"
            "        if (!abi || *abi != 1) NVM2C_ABORT();\n"
            "        walk = (nh_array_value *(*)(%s))dlsym(library, \"%s\");\n"
            "        release = (bool (*)(nh_array_value *))dlsym(library, \"%s\");\n",
            host->name, types, host->name, release_name);
        nvm2c_puts(b,
            "        if (!walk || !release) NVM2C_ABORT();\n"
            "        Dl_info producer, marker, companion;\n"
            "        if (!dladdr((void *)walk, &producer) || !dladdr((void *)abi, &marker) ||\n"
            "            !dladdr((void *)release, &companion) ||\n"
            "            producer.dli_fbase != marker.dli_fbase ||\n"
            "            producer.dli_fbase != companion.dli_fbase) NVM2C_ABORT();\n"
            "    }\n");
        nvm2c_printf(b, "    nh_array_value *foreign = walk(%s);\n", host->argc == 2 ? "root, extension" : "root");
        nvm2c_puts(b,
            "    if (!foreign || !foreign->data || foreign->type != nh_string ||\n"
            "        foreign->width != sizeof(char *) || foreign->length < 0 ||\n"
            "        foreign->capacity < foreign->length ||\n"
            "        (uint64_t)foreign->length > SIZE_MAX / sizeof(char *)) NVM2C_ABORT();\n"
            "    nsarr_t result = nsarr_new();\n"
            "    nsarr_reserve(result, (size_t)foreign->length);\n"
            "    result->len = (size_t)foreign->length;\n"
            "    for (size_t j = 0; j < result->len; ++j) {\n"
            "        const char *value = ((const char **)foreign->data)[j];\n"
            "        result->data[j] = nsarr_copy_string(value);\n"
            "    }\n"
            "    if (!release(foreign)) NVM2C_ABORT();\n"
            "    return result;\n}\n");
    }
}

static void emit_nrarr_helpers(Nvm2cBuf *b, int need_new, int need_push, int need_get) {
    if (need_new || need_push) {
        b->has_record_array_allocations = 1;
        nvm2c_puts(b,
            "#include <string.h>\n"
            "struct nrarr_owner { nrec_t *data; size_t cap, bytes; nrarr_t handle; struct nrarr_owner *next; };\n"
            "static struct nrarr_owner *nrarr_owners;\n"
            "static inline struct nrarr_owner *nrarr_track(nrarr_t a, int own_handle) {\n"
            "    struct nrarr_owner *owner = calloc(1, sizeof *owner);\n"
            "    if (!owner) NVM2C_ABORT();\n"
            "    owner->bytes = sizeof *owner + (own_handle ? sizeof *a : 0); nagg_add(owner->bytes);\n"
        "    owner->handle = own_handle ? a : NULL; owner->next = nrarr_owners;\n"
            "    nrarr_owners = owner; a->owner = owner; return owner;\n}\n"
            "static inline void nrarr_release_owned(void) {\n"
            "    while (nrarr_owners) { struct nrarr_owner *owner = nrarr_owners;\n"
            "        nrarr_owners = owner->next; nagg_drop(owner->bytes); free(owner->data); free(owner->handle); free(owner); }\n}\n"
            "static inline nrarr_t nrarr_new(void) {\n"
            "    nrarr_t a = calloc(1, sizeof *a); if (!a) NVM2C_ABORT();\n"
            "    nrarr_track(a, 1); return a;\n}\n"
            "static inline void nrarr_reserve(nrarr_t a, size_t n) {\n"
            "    size_t limit = SIZE_MAX / sizeof(nrec_t);\n"
            "    if (!a || n > limit || a->len > limit || (a->len && !a->data)) NVM2C_ABORT();\n"
            "    struct nrarr_owner *owner = a->owner;\n"
            "    if (owner && (owner->data != a->data || a->len > owner->cap)) NVM2C_ABORT();\n"
            "    if (owner && n <= owner->cap) return;\n"
            "    if (n < a->len) n = a->len;\n"
            "    size_t cap = owner && owner->cap ? owner->cap : 8;\n"
            "    while (cap < n) { if (cap > limit / 2) { cap = n; break; } cap *= 2; }\n"
            "    if (!owner) {\n"
            "        nrec_t *data = malloc(cap * sizeof *data); if (!data) NVM2C_ABORT();\n"
            "        if (a->len) memcpy(data, a->data, a->len * sizeof *data);\n"
            "        owner = nrarr_track(a, 0); owner->data = data;\n"
            "    } else {\n"
            "        nrec_t *data = realloc(owner->data, cap * sizeof *data);\n"
            "        if (!data) NVM2C_ABORT();\n"
            "        owner->data = data;\n"
            "    }\n"
            "    size_t growth = (cap - owner->cap) * sizeof *owner->data;\n"
        "    nagg_add(growth); owner->bytes += growth;\n"
        "    owner->cap = cap; a->data = owner->data;\n}\n\n");
    }
    if (need_push) nvm2c_puts(b,
        "static nrarr_t nrarr_push(nrarr_t a, nrec_t v) {\n"
        "    if (!a || a->len >= SIZE_MAX / sizeof(nrec_t)) NVM2C_ABORT();\n"
        "    nrarr_reserve(a, a->len + 1);\n"
        "    a->data[a->len++] = v;\n"
        "    return a;\n"
        "}\n\n");
    if (need_get) {
        b->has_record_array_getter = 1;
        nvm2c_puts(b,
        "static nrec_t nrarr_get(nrarr_t a, int64_t idx) {\n"
        "    if (!a || idx < 0 || (uint64_t)idx >= a->len || !a->data) NVM2C_ABORT();\n"
        "    return a->data[idx];\n"
        "}\n\n");
    }
}

/* The legacy module has nominal type IDs but no field-layout table. I can
 * recover an unresolved scalar field when constructors of that same declared
 * type, variant and width agree. Conflicting or aggregate evidence is not a
 * scalar layout; I leave it unresolved rather than choose a constructor. */
static int infer_nominal_scalar_fields(Nvm2cBuf *b, const NvmModule *mod) {
    typedef struct {
        uint8_t kind;
        uint16_t type, variant, width;
        NvmShapeId shape;
    } Pack;
    Pack *packs = NULL;
    uint8_t *inferred = NULL;
    size_t count = 0, capacity = 0;
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
            if (!n) { nvm2c_fail(b, "I cannot decode nominal field evidence"); goto done; }
            if (ins.opcode == OP_AGG_PACK && b->shape_outputs[f][pc]) {
                uint8_t kind = ins.operands[0].u8;
                uint16_t type = ins.operands[1].u16;
                uint32_t declared = kind == AGG_RECORD ? mod->struct_count :
                    kind == AGG_VARIANT ? mod->union_count : 0;
                if (type < declared) {
                    if (count == capacity) {
                        size_t next = capacity ? capacity * 2 : 64;
                        if (next < capacity || next > SIZE_MAX / sizeof *packs) {
                            nvm2c_fail(b, "I cannot represent nominal field evidence"); goto done;
                        }
                        Pack *grown = realloc(packs, next * sizeof *packs);
                        if (!grown) { nvm2c_fail(b, "I cannot allocate nominal field evidence"); goto done; }
                        packs = grown; capacity = next;
                    }
                    packs[count++] = (Pack){kind, type, ins.operands[2].u16,
                                            ins.operands[3].u16, b->shape_outputs[f][pc]};
                }
            }
            pc += n;
        }
    }
    if (count && b->record_width > SIZE_MAX / count) {
        nvm2c_fail(b, "I cannot represent inferred nominal fields"); goto done;
    }
    inferred = calloc(count ? count * b->record_width : 1, 1);
    if (!inferred) { nvm2c_fail(b, "I cannot allocate inferred nominal fields"); goto done; }
    int changed;
    do {
        changed = 0;
        for (size_t i = 0; i < count; ++i) {
            for (uint16_t field = 0; field < packs[i].width; ++field) {
                NvmShapeId target = nvm_shape_lookup(&b->shapes, packs[i].shape, field);
                if (!target || nvm_shape_kind(&b->shapes, target) != NVM_SHAPE_UNKNOWN) continue;
                NvmShapeKind agreed = NVM_SHAPE_UNKNOWN;
                int conflict = 0;
                for (size_t j = 0; j < count; ++j) {
                    if (packs[i].kind != packs[j].kind || packs[i].type != packs[j].type ||
                        packs[i].variant != packs[j].variant || packs[i].width != packs[j].width) continue;
                    NvmShapeId candidate = nvm_shape_lookup(&b->shapes, packs[j].shape, field);
                    if (!candidate) continue;
                    NvmShapeKind kind = nvm_shape_kind(&b->shapes, candidate);
                    if (kind == NVM_SHAPE_UNKNOWN) continue;
                    if ((kind != NVM_SHAPE_INT && kind != NVM_SHAPE_BOOL &&
                         kind != NVM_SHAPE_FLOAT && kind != NVM_SHAPE_STRING) ||
                        (agreed != NVM_SHAPE_UNKNOWN && agreed != kind)) { conflict = 1; break; }
                    agreed = kind;
                }
                if (!conflict && agreed != NVM_SHAPE_UNKNOWN) {
                    NvmShapeId root = nvm_shape_root(&b->shapes, target);
                    for (size_t j = 0; j < count; ++j)
                        for (uint16_t k = 0; k < packs[j].width; ++k)
                            if (nvm_shape_lookup(&b->shapes, packs[j].shape, k) == root)
                                inferred[j * b->record_width + k] = 1;
                    if (!shape_type(b, target, agreed)) goto done;
                    changed = 1;
                }
            }
        }
        if (!nvm_shape_solve_conversions(&b->shapes) || !shape_ok(b)) goto done;
    } while (changed);
    /* Later inference can expose another constructor's conflicting field.
     * Recheck every hint, so scan order cannot select one nominal layout. */
    for (size_t i = 0; i < count; ++i) {
        for (uint16_t field = 0; field < packs[i].width; ++field) {
            if (!inferred[i * b->record_width + field]) continue;
            NvmShapeKind expected = nvm_shape_kind(&b->shapes,
                nvm_shape_lookup(&b->shapes, packs[i].shape, field));
            for (size_t j = 0; j < count; ++j) {
                if (packs[i].kind != packs[j].kind || packs[i].type != packs[j].type ||
                    packs[i].variant != packs[j].variant || packs[i].width != packs[j].width) continue;
                NvmShapeKind kind = nvm_shape_kind(&b->shapes,
                    nvm_shape_lookup(&b->shapes, packs[j].shape, field));
                if (kind != NVM_SHAPE_UNKNOWN && kind != expected) {
                    nvm2c_fail(b, "I found conflicting nominal scalar field evidence"); goto done;
                }
            }
        }
    }
done:
    free(inferred);
    free(packs);
    return !b->failed && shape_ok(b);
}

/* I need executable layouts for both VM roots, not for uncalled parameters.
 * The existing whole-module validation still rejects unsupported instructions. */
static int mark_required_functions(Nvm2cBuf *b, const NvmModule *mod) {
    if (mod->header.entry_point >= mod->function_count) {
        nvm2c_fail(b, "entry_point %u is not a function", mod->header.entry_point);
        return 0;
    }
    b->required_functions[mod->header.entry_point] = 1;
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        const char *name = nvm_get_string(mod, mod->functions[i].name_idx);
        if (name && strcmp(name, "__init__") == 0) {
            b->required_functions[i] = 1;
            break;
        }
    }
    int changed;
    do {
        changed = 0;
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            if (!b->required_functions[i]) continue;
            const NvmFunctionEntry *fn = &mod->functions[i];
            for (size_t pc = 0; pc < fn->code_length;) {
                DecodedInstruction ins;
                uint32_t size = isa_decode(mod->code + fn->code_offset + pc,
                                           fn->code_length - pc, &ins);
                if (!size) { nvm2c_fail(b, "I cannot decode function reachability"); return 0; }
                if (ins.opcode == OP_CALL || ins.opcode == OP_TAIL_CALL) {
                    uint32_t callee = ins.operands[0].u32;
                    if (callee >= mod->function_count) {
                        nvm2c_fail(b, "function %u: CALL target %u is out of range", i, callee);
                        return 0;
                    }
                    if (!b->required_functions[callee]) {
                        b->required_functions[callee] = 1;
                        changed = 1;
                    }
                }
                pc += size;
            }
        }
    } while (changed);
    return 1;
}

/* An omitted callee also makes its uncalled callers unemittable. I retain
 * representable functions for native boundary probes and embedding. */
static int prune_unemittable_callers(Nvm2cBuf *b, const NvmModule *mod) {
    int changed;
    do {
        changed = 0;
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            if (!b->emitted_functions[i]) continue;
            const NvmFunctionEntry *fn = &mod->functions[i];
            for (size_t pc = 0; pc < fn->code_length;) {
                DecodedInstruction ins;
                uint32_t size = isa_decode(mod->code + fn->code_offset + pc,
                                           fn->code_length - pc, &ins);
                if (!size) { nvm2c_fail(b, "I cannot decode uncalled function dependencies"); return 0; }
                if (ins.opcode == OP_CALL || ins.opcode == OP_TAIL_CALL) {
                    uint32_t callee = ins.operands[0].u32;
                    if (callee >= mod->function_count) {
                        nvm2c_fail(b, "function %u: CALL target %u is out of range", i, callee);
                        return 0;
                    }
                    if (!b->emitted_functions[callee]) {
                        if (b->required_functions[i]) {
                            nvm2c_fail(b, "I cannot omit required function %u", i);
                            return 0;
                        }
                        b->emitted_functions[i] = 0;
                        changed = 1;
                        break;
                    }
                }
                pc += size;
            }
        }
    } while (changed);
    return 1;
}

#include "nvm2c_owned.h"

char *nvm2c_emit(const NvmModule *mod, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!mod) {
        if (err && err_len) snprintf(err, err_len, "module is null");
        return NULL;
    }
    bool needs_ownership = false;
    if (nvm_ownership_contracts_validate(mod, &needs_ownership) != NVM_V2_OK || needs_ownership ||
        nvm_uses_owned_transfers(mod)) {
        if (nvm_verify_owned_module(mod).ok) return emit_owned_module(mod, err, err_len);
        if (err && err_len) snprintf(err, err_len,
            "I require valid reference lifetime and ownership instruction verification before translation");
        return NULL;
    }
    if (mod->function_count == 0) {
        if (err && err_len) snprintf(err, err_len, "module has no functions");
        return NULL;
    }
    for (uint32_t i = 0; i < mod->import_count; ++i) {
        if (!import_host(mod, i)) {
            const char *name = mod->imports ? nvm_get_string(mod, mod->imports[i].function_name_idx) : NULL;
            if (err && err_len) {
                if (mod->imports && mod->imports[i].kind == NVM_IMPORT_ARTIFACT)
                    snprintf(err, err_len, "import %u (%s) is artifact-backed and requires an exact library binding and typed value adapter; nvm2c refuses CALL_EXTERN",
                             i, name ? name : "invalid name");
                else
                    snprintf(err, err_len, "import %u (%s) requires an exact builtin host ABI; nvm2c refuses CALL_EXTERN",
                             i, name ? name : "invalid name");
            }
            return NULL;
        }
    }

    Nvm2cBuf b;
    memset(&b, 0, sizeof b);
    b.err = err;
    b.err_len = err_len;
    b.record_width = 1;
    b.local_width = 1;
    b.has_owned_strings = module_has_opcode(mod, OP_STR_CONCAT) ||
        module_has_opcode(mod, OP_STR_SUBSTR) || module_has_opcode(mod, OP_STR_TRIM) || module_has_opcode(mod, OP_CAST_STRING) ||
        module_uses_host(mod, "nhost_from_char");
    /* I reserve string roots for explicit artifact cleanup companions too;
     * artifacts without a companion retain their existing borrowed contract. */
    for (uint32_t i = 0; i < mod->import_count; ++i) {
        const Nvm2cHost *host = import_host(mod, i);
        if (host && host->result == TAG_STRING)
            b.has_owned_strings = 1;
    }
    int has_global_store = 0, has_record_array_constructor = 0;
    /* The tagged map runtime also provides shared frame/aggregate root tracing.
     * Owned strings/aggregates need it even without map instructions. */
    if (b.has_owned_strings) b.has_maps = 1;
    b.has_owned_aggregates = module_has_opcode(mod, OP_AGG_PACK) ||
        module_has_opcode(mod, OP_ARR_NEW) || module_has_opcode(mod, OP_ARR_LITERAL) ||
        module_has_opcode(mod, OP_ARR_PUSH) || module_has_opcode(mod, OP_ARR_GET) ||
        module_has_opcode(mod, OP_ARR_SET) || module_has_opcode(mod, OP_ARR_LEN) ||
        module_uses_host(mod, "nhost_walk");
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        if (mod->functions[f].result_tag == TAG_ARRAY) b.has_owned_aggregates = 1;
        if (mod->function_param_types && mod->function_param_types[f])
            for (uint16_t p = 0; p < mod->functions[f].arity; ++p)
                if (mod->function_param_types[f][p] == TAG_ARRAY) b.has_owned_aggregates = 1;
    }
    if (b.has_owned_aggregates) b.has_maps = 1;
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
        if (fn->local_count > NVM2C_MAX_LOCALS) {
            nvm2c_fail(&b, "function %u: too many locals", f);
            return NULL;
        }
        if (fn->arity > fn->local_count) {
            nvm2c_fail(&b, "function %u: arity exceeds local_count", f);
            return NULL;
        }
        if (fn->result_tag == TAG_U8 || fn->result_tag == TAG_ENUM) b.has_maps = 1;
        if (mod->function_param_types && mod->function_param_types[f])
            for (uint16_t p = 0; p < fn->arity; ++p)
                if (mod->function_param_types[f][p] == TAG_U8 || mod->function_param_types[f][p] == TAG_ENUM) b.has_maps = 1;
        if (fn->local_count > b.local_width) b.local_width = fn->local_count;
        if (fn->code_offset > mod->code_size || fn->code_length > mod->code_size - fn->code_offset) {
            nvm2c_fail(&b, "I cannot inspect aggregate widths outside function %u", f);
            return NULL;
        }
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
            if (!n) {
                nvm2c_fail(&b, "I cannot decode aggregate widths in function %u", f);
                return NULL;
            }
            if (ins.opcode == OP_AGG_PACK && ins.operands[3].u16 > b.record_width)
                b.record_width = ins.operands[3].u16;
            if (ins.opcode == OP_HM_NEW || ins.opcode == OP_HM_SET || ins.opcode == OP_HM_HAS ||
                ins.opcode == OP_HM_DELETE || ins.opcode == OP_HM_LEN || ins.opcode == OP_HM_GET) b.has_maps = 1;
            if (ins.opcode == OP_PUSH_VOID || ins.opcode == OP_PUSH_U8 || ins.opcode == OP_ENUM_VAL) b.has_maps = 1; /* Tagged scalar storage. */
            if (ins.opcode == OP_LT || ins.opcode == OP_LE || ins.opcode == OP_GT || ins.opcode == OP_GE)
                b.has_maps = 1; /* I retain runtime tags for generic ordering. */
            if (ins.opcode == OP_LOAD_GLOBAL || ins.opcode == OP_STORE_GLOBAL) {
                size_t slot = ins.operands[0].u32;
                if (slot >= NVM_MAX_GLOBALS) {
                    nvm2c_fail(&b, "I cannot access global %zu in function %u at offset %zu: limit %u",
                               slot, f, pc, NVM_MAX_GLOBALS);
                    return NULL;
                }
                if (b.global_count <= slot) b.global_count = slot + 1;
                b.has_maps = 1; /* Globals share the tagged scalar runtime. */
                if (ins.opcode == OP_STORE_GLOBAL) has_global_store = 1;
            }
            if ((ins.opcode == OP_ARR_NEW || ins.opcode == OP_ARR_LITERAL) &&
                ins.operands[0].u8 == TAG_STRUCT) has_record_array_constructor = 1;
            pc += n;
        }
    }

    if (b.has_maps && b.record_width < 2) b.record_width = 2;
    size_t per_function = b.local_width * (1 + b.record_width) + b.record_width + 1;
    if (mod->function_count > SIZE_MAX / per_function) {
        if (err && err_len) snprintf(err, err_len, "I cannot allocate this many function facts");
        return NULL;
    }
    size_t fact_size = (size_t)mod->function_count * per_function;
    size_t per_global = 1 + b.record_width;
    if (b.global_count > (SIZE_MAX - fact_size) / per_global) {
        nvm2c_fail(&b, "I cannot allocate this many global facts");
        return NULL;
    }
    fact_size += b.global_count * per_global;
    size_t shape_code_count = mod->code_size;
    size_t shape_local_count = (size_t)mod->function_count * b.local_width;
    if (shape_code_count > SIZE_MAX / sizeof(NvmShapeId) || shape_local_count > SIZE_MAX / sizeof(NvmShapeId)) {
        nvm2c_fail(&b, "I cannot allocate this many instruction shapes");
        return NULL;
    }
    b.shape_locals = calloc((size_t)mod->function_count * b.local_width, sizeof(NvmShapeId));
    b.shape_results = calloc(mod->function_count, sizeof(NvmShapeId));
    b.shape_globals = calloc(b.global_count ? b.global_count : 1, sizeof(NvmShapeId));
    b.shape_outputs = calloc(mod->function_count, sizeof *b.shape_outputs);
    b.join_shapes = calloc(mod->function_count, sizeof *b.join_shapes);
    b.emitted_functions = calloc(mod->function_count, 1);
    b.required_functions = calloc(mod->function_count, 1);
    b.tagged_locals = calloc((size_t)mod->function_count * b.local_width, 1);
    b.local_scalar_tags = calloc(shape_local_count, sizeof *b.local_scalar_tags);
    uint8_t *inference = malloc(fact_size);
    uint8_t *global_stored = calloc(b.global_count ? b.global_count : 1, 1);
    uint8_t *kinds = calloc((size_t)mod->function_count * b.local_width, 1);
    uint8_t *rec_fields = calloc((size_t)mod->function_count * b.local_width
                                 * b.record_width, 1);
    if (!kinds || !rec_fields || !inference || !global_stored || !b.shape_locals || !b.shape_results ||
        !b.shape_globals || !b.shape_outputs || !b.join_shapes || !b.emitted_functions ||
        !b.required_functions || !b.tagged_locals || !b.local_scalar_tags) {
        free(inference);
        free(global_stored);
        free(kinds);
        free(rec_fields);
        free(b.shape_locals);
        free(b.shape_results);
        free(b.shape_globals);
        free(b.shape_outputs);
        free(b.join_shapes);
        free(b.emitted_functions);
        free(b.required_functions);
        free(b.tagged_locals);
        free(b.local_scalar_tags);
        if (err && err_len) snprintf(err, err_len, "out of memory");
        return NULL;
    }

    for (uint32_t f = 0; f < mod->function_count; ++f)
        for (uint16_t local = 0; local < mod->functions[f].local_count; ++local)
            b.local_scalar_tags[(size_t)f * b.local_width + local] =
                local < mod->functions[f].arity ? 0 : 1u << TAG_VOID;
    if (!mark_required_functions(&b, mod)) goto fail;
    for (uint32_t f = 0; f < mod->function_count; ++f)
        if (!mark_uninitialized_locals(&b, mod, f)) goto fail;
    memset(b.emitted_functions, 1, mod->function_count);
    memset(inference, NVM2C_VK_UNK, fact_size);
    Nvm2cFacts facts = {0};
    facts.parameters = inference;
    facts.fields = inference + (size_t)mod->function_count * b.local_width;
    facts.results = facts.fields + (size_t)mod->function_count * b.local_width * b.record_width;
    b.array_results = facts.results + (size_t)mod->function_count * b.record_width;
    facts.global_kinds = b.array_results + mod->function_count;
    facts.global_fields = facts.global_kinds + b.global_count;
    facts.global_stored = global_stored;
    /* I discover exact record-array globals before ordinary inference. A
     * load cannot publish the old tagged fallback into a local or parameter
     * before a later function reveals the global's exact element shape. */
    if (has_global_store && has_record_array_constructor) {
        facts.discover_globals = 1;
        for (size_t pass = 0; ; ++pass) {
            if (pass / 2 > fact_size && pass / 2 - fact_size > mod->code_size) {
                nvm2c_fail(&b, "I could not converge global representation facts");
                goto fail;
            }
            facts.changed = 0;
            for (uint32_t i = 0; i < mod->function_count; ++i) {
                if (!classify_function(&b, mod, i, kinds + (size_t)i * b.local_width,
                                       rec_fields + (size_t)i * b.local_width * b.record_width,
                                       &facts)) goto fail;
            }
            if (!facts.changed) break;
        }
        facts.discover_globals = 0;
        memset(inference, NVM2C_VK_UNK,
               (size_t)(facts.global_kinds - inference));
    }
    /* I add known facts and widen string parameters to optional storage when
     * needed. Payload and aggregate compatibility remain graph constraints. */
    for (size_t pass = 0; ; pass++) {
        if (pass / 2 > fact_size && pass / 2 - fact_size > mod->code_size) {
            nvm2c_fail(&b, "I could not converge function type facts");
            goto fail;
        }
        facts.changed = 0;
        for (uint32_t i = 0; i < mod->function_count; i++) {
            if (!classify_function(&b, mod, i, kinds + (size_t)i * b.local_width,
                                   rec_fields + (size_t)i * b.local_width * b.record_width,
                                   &facts)) {
                goto fail;
            }
        }
        if (facts.final) break;
        if (!facts.changed) {
            /* I use declarations only after caller facts converge. An
             * observed tagged argument keeps its checked representation;
             * an unused parameter still has its declared scalar/record type.
             * An array tag alone does not determine its element storage. */
            for (uint32_t f = 0; f < mod->function_count; ++f) {
                const uint8_t *tags = mod->function_param_types ? mod->function_param_types[f] : NULL;
                if (!tags) continue;
                for (uint16_t p = 0; p < mod->functions[f].arity; ++p) {
                    uint8_t *kind = &facts.parameters[(size_t)f * b.local_width + p];
                    if (*kind != NVM2C_VK_UNK) continue;
                    uint8_t declared = tags[p] == TAG_INT ? NVM2C_VK_INT :
                        (tags[p] == TAG_U8 || tags[p] == TAG_ENUM) ? NVM2C_VK_VALUE :
                        tags[p] == TAG_BOOL ? NVM2C_VK_BOOL :
                        tags[p] == TAG_FLOAT ? NVM2C_VK_FLOAT :
                        tags[p] == TAG_STRING ? NVM2C_VK_STR :
                        aggregate_value_tag(tags[p]) ? NVM2C_VK_REC : NVM2C_VK_UNK;
                    if (declared != NVM2C_VK_UNK) { *kind = declared; facts.changed = 1; }
                }
            }
            if (!facts.changed) facts.final = 1;
        }
    }

    if (!nvm_shape_solve_conversions(&b.shapes)) {
        nvm2c_fail(&b, "I cannot solve aggregate storage shape conversions: %s", b.shapes.error);
        goto fail;
    }
    for (size_t slot = 0; slot < b.global_count; ++slot) {
        uint8_t resolved = resolved_shape_kind(&b, b.shape_globals[slot]);
        if (facts.global_kinds[slot] == NVM2C_VK_UNK && resolved == NVM2C_VK_RARR) {
            facts.global_kinds[slot] = NVM2C_VK_RARR;
            NvmShapeId element = nvm_shape_lookup(&b.shapes, b.shape_globals[slot], 0);
            for (size_t field = 0; field < b.record_width; ++field) {
                NvmShapeId shape = element ? nvm_shape_lookup(&b.shapes, element, (uint32_t)field) : 0;
                facts.global_fields[slot * b.record_width + field] = resolved_shape_kind(&b, shape);
            }
            if (!shape_ok(&b)) goto fail;
        }
        if (facts.global_kinds[slot] != NVM2C_VK_RARR) continue;
        if (resolved != NVM2C_VK_RARR) {
            nvm2c_fail(&b, "I cannot resolve the record element shape of global %zu", slot);
            goto fail;
        }
        b.array_shape_kinds |= (uint16_t)(1u << NVM2C_VK_RARR);
    }
    if (!infer_nominal_scalar_fields(&b, mod)) goto fail;

    /* Nested projections may acquire their representation from a later
     * function's constraints. Resolve local storage after all final passes. */
    uint8_t *tagged_projections = calloc(b.shapes.count + 1, 1);
    if (!tagged_projections) { nvm2c_fail(&b, "I cannot allocate projected storage facts"); goto fail; }
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t size = isa_decode(mod->code + fn->code_offset + pc, fn->code_length - pc, &ins);
            if (!size) { free(tagged_projections); nvm2c_fail(&b, "I cannot decode projected storage facts"); goto fail; }
            NvmShapeId shape = b.shape_outputs[f][pc];
            if (ins.opcode == OP_AGG_GET && shape &&
                nvm_shape_kind(&b.shapes, shape) == NVM_SHAPE_UNKNOWN) {
                tagged_projections[nvm_shape_root(&b.shapes, shape)] = 1;
                b.has_maps = 1;
            }
            pc += size;
        }
    }
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        if (mod->functions[f].result_tag == TAG_ARRAY) {
            uint8_t resolved = resolved_shape_kind(&b, b.shape_results[f]);
            if (resolved != NVM2C_VK_UNK) b.array_results[f] = resolved;
        }
        for (uint16_t l = 0; l < mod->functions[f].local_count; ++l) {
            size_t at = (size_t)f * b.local_width + l;
            uint8_t resolved = resolved_shape_kind(&b, b.shape_locals[at]);
            if (resolved != NVM2C_VK_UNK) kinds[at] = resolved;
            else if (kinds[at] == NVM2C_VK_UNK && b.shape_locals[at] &&
                     tagged_projections[nvm_shape_root(&b.shapes, b.shape_locals[at])])
                kinds[at] = NVM2C_VK_VALUE;
            else if (kinds[at] == NVM2C_VK_UNK && b.shape_locals[at] &&
                     nvm_shape_kind(&b.shapes, b.shape_locals[at]) == NVM_SHAPE_ARRAY) {
                /* An array container with unresolved element storage uses the
                 * existing tagged handle ABI, including parameters and copies. */
                kinds[at] = NVM2C_VK_VALUE;
                b.has_maps = 1;
            }
            /* I choose the scalar fallback only after aggregate shapes have
             * had their chance to determine the local representation. */
            if (kinds[at] == NVM2C_VK_UNK) kinds[at] = NVM2C_VK_INT;
        }
    }
    free(tagged_projections);
    if (!shape_ok(&b)) goto fail;

    /* I finish graph construction before requiring packed field kinds.
     * A later caller can supply facts absent from the flat field vectors. */
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
        for (size_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction ins;
            uint32_t size = isa_decode(mod->code + fn->code_offset + pc,
                                       fn->code_length - pc, &ins);
            if (!size) { nvm2c_fail(&b, "I cannot decode packed field shapes"); goto fail; }
            NvmShapeId shape = b.shape_outputs[f][pc];
            uint8_t resolved = resolved_shape_kind(&b, shape);
            if (word_array_storage(resolved) || resolved == NVM2C_VK_SARR || resolved == NVM2C_VK_RARR)
                b.array_shape_kinds |= (uint16_t)(1u << resolved);
            if (ins.opcode == OP_AGG_GET && shape &&
                nvm_shape_kind(&b.shapes, shape) == NVM_SHAPE_ARRAY &&
                resolved_shape_kind(&b, shape) == NVM2C_VK_UNK) b.has_maps = 1;
            if (ins.opcode == OP_AGG_PACK && shape) {
                for (uint16_t field = 0; field < ins.operands[3].u16; ++field) {
                    if (resolved_shape_kind(&b, nvm_shape_lookup(&b.shapes, shape, field)) == NVM2C_VK_UNK) {
                        if (!b.required_functions[f]) {
                            b.emitted_functions[f] = 0;
                            break;
                        }
                        nvm2c_fail(&b, "function %u at offset %zu: I cannot resolve AGG_PACK field %u",
                                   f, pc, (unsigned)field);
                        goto fail;
                    }
                }
            }
            pc += size;
        }
    }

    {
        if (!prune_unemittable_callers(&b, mod)) goto fail;
        int need_concat = module_has_opcode(mod, OP_STR_CONCAT);
        int need_cast = module_has_opcode(mod, OP_CAST_STRING);
        int need_contains = module_has_opcode(mod, OP_STR_CONTAINS);
        int need_starts = module_has_opcode(mod, OP_STR_STARTS_WITH);
        int need_ends = module_has_opcode(mod, OP_STR_ENDS_WITH);
        int need_substr = module_has_opcode(mod, OP_STR_SUBSTR);
        int need_trim = module_has_opcode(mod, OP_STR_TRIM);
        int need_char_at = module_has_opcode(mod, OP_STR_CHAR_AT);
        int need_string = need_concat || need_cast || need_contains || need_substr || need_trim ||
            need_starts || need_ends ||
            need_char_at ||
            module_has_opcode(mod, OP_PUSH_STR) ||
            module_has_opcode(mod, OP_STR_LEN);
        int need_arr_lit = module_has_opcode(mod, OP_ARR_LITERAL);
        int need_arr_get = module_has_opcode(mod, OP_ARR_GET);
        int need_arr_push = module_has_opcode(mod, OP_ARR_PUSH);
        int need_iarr_new = module_has_array_constructor(&b, mod, kinds, NVM2C_VK_ARR) ||
            module_has_array_constructor(&b, mod, kinds, NVM2C_VK_BARR) ||
            module_has_array_constructor(&b, mod, kinds, NVM2C_VK_FARR);
        int need_sarr_new = module_has_array_constructor(&b, mod, kinds, NVM2C_VK_SARR);
        int need_iarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_INT) ||
            module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_BOOL) ||
            module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_FLOAT);
        int need_sarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRING);
        int need_iarr = need_iarr_new || need_iarr_lit ||
            (b.array_shape_kinds & ((1u << NVM2C_VK_ARR) | (1u << NVM2C_VK_BARR) | (1u << NVM2C_VK_FARR))) ||
            module_has_local_kind(&b, kinds, mod->function_count, NVM2C_VK_ARR) ||
            module_has_local_kind(&b, kinds, mod->function_count, NVM2C_VK_BARR) ||
            module_has_local_kind(&b, kinds, mod->function_count, NVM2C_VK_FARR);
        int need_sarr = need_sarr_new || need_sarr_lit || module_uses_host(mod, "nhost_walk") ||
            (b.array_shape_kinds & (1u << NVM2C_VK_SARR)) ||
            module_has_local_kind(&b, kinds, mod->function_count, NVM2C_VK_SARR);
        /* I emit retained adapters even when no instruction calls them. */
        for (uint32_t import = 0; import < mod->import_count; ++import) {
            const Nvm2cHost *host = import_host(mod, import);
            if (host && host->result == TAG_ARRAY) need_sarr = 1;
        }
        int need_rarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRUCT);
        int need_rarr = need_rarr_lit || module_has_array_constructor(&b, mod, kinds, NVM2C_VK_RARR) ||
            (b.array_shape_kinds & (1u << NVM2C_VK_RARR)) ||
            module_has_local_kind(&b, kinds, mod->function_count, NVM2C_VK_RARR);
        int need_iarr_get = need_arr_get && need_iarr;
        int need_sarr_get = need_arr_get && need_sarr;
        int need_iarr_push = need_arr_push && need_iarr;
        int need_sarr_push = need_arr_push && need_sarr;
        int need_agg_get = module_has_opcode(mod, OP_AGG_GET) || module_has_opcode(mod, OP_AGG_TAG);
        int need_print = module_has_opcode(mod, OP_PRINT) ||
            module_has_opcode(mod, OP_PRINTLN);
        int need_assert = module_has_opcode(mod, OP_ASSERT);
        uint32_t i;
        for (i = 0; i < mod->function_count && !need_string; i++) {
            const NvmFunctionEntry *fn = &mod->functions[i];
            uint16_t li;
            if (fn->result_tag == TAG_STRING) need_string = 1;
            for (li = 0; li < fn->local_count; li++) {
                if (fn_local_kind(&b, kinds, i, li) == NVM2C_VK_STR) need_string = 1;
            }
        }

        if (module_uses_host(mod, "nhost_destinations") || module_uses_host(mod, "nhost_capture") ||
            module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
            "#ifndef _POSIX_C_SOURCE\n#define _POSIX_C_SOURCE 200809L\n#endif\n");
        if (module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
            "#ifdef __APPLE__\n#ifndef _DARWIN_C_SOURCE\n#define _DARWIN_C_SOURCE\n#endif\n#endif\n");
        nvm2c_puts(&b,
            "#ifndef _GNU_SOURCE\n#define _GNU_SOURCE 1\n#endif\n"
            "#ifndef _DARWIN_C_SOURCE\n#define _DARWIN_C_SOURCE 1\n#endif\n");
        nvm2c_puts(&b,
            "/* Generated by nvm2c from NanoISA. Not a VM wrapper. */\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n#include <stdlib.h>\n#include <stdio.h>\n"
            "/* I preserve invariant termination while naming its generated source. */\n"
            "#define NVM2C_ABORT() do { fprintf(stderr, \"I stopped at a native invariant in %s at generated C line %d.\\n\", __func__, __LINE__); abort(); } while (0)\n");
        nvm2c_puts(&b,
            "/* I reconstruct wrapped bits without an out-of-range signed cast. */\n"
            "static inline int64_t ni64_from_bits(uint64_t bits) {\n"
            "    return bits <= INT64_MAX ? (int64_t)bits : -INT64_C(1) - (int64_t)(UINT64_MAX - bits);\n}\n");
        if (module_has_opcode(mod, OP_I64_ADD_CARRY) || module_has_opcode(mod, OP_I64_SUB_BORROW) ||
            module_has_opcode(mod, OP_I64_MUL_WIDE_S) || module_has_opcode(mod, OP_I64_MUL_WIDE_U))
            nvm2c_puts(&b,
                "typedef struct { int64_t low, high; } ni64_pair;\n"
                "static inline ni64_pair ni64_pair_compute(int64_t a, int64_t b, int64_t carry, unsigned op) {\n"
                "    uint64_t ua = (uint64_t)a, ub = (uint64_t)b, uc = (uint64_t)carry & UINT64_C(1);\n"
                "    uint64_t low, high;\n"
                "    if (op == 0) {\n"
                "        uint64_t partial = ua + ub; low = partial + uc;\n"
                "        high = (partial < ua) | (low < partial);\n"
                "    } else if (op == 1) {\n"
                "        uint64_t partial = ua - ub; low = partial - uc;\n"
                "        high = (ua < ub) | (partial < uc);\n"
                "    } else {\n"
                "        uint64_t mask = UINT64_C(0xffffffff);\n"
                "        uint64_t a0 = ua & mask, a1 = ua >> 32, b0 = ub & mask, b1 = ub >> 32;\n"
                "        uint64_t middle1 = a1 * b0 + ((a0 * b0) >> 32);\n"
                "        uint64_t middle2 = a0 * b1 + (middle1 & mask);\n"
                "        low = ua * ub; high = a1 * b1 + (middle1 >> 32) + (middle2 >> 32);\n"
                "        if (op == 2) { if (a < 0) high -= ub; if (b < 0) high -= ua; }\n"
                "    }\n"
                "    return (ni64_pair){ni64_from_bits(low), ni64_from_bits(high)};\n}\n");
        if (module_has_opcode(mod, OP_AGG_PACK) || module_has_opcode(mod, OP_AGG_GET))
            nvm2c_puts(&b, "#include <string.h>\n");
        if (module_has_opcode(mod, OP_AGG_GET)) nvm2c_puts(&b,
            "/* I retain binary64 bits in my existing 64-bit aggregate cells. */\n"
            "static inline double nrec_f64(int64_t cell) {\n"
            "    double value; memcpy(&value, &cell, sizeof value); return value;\n}\n");
        nvm2c_puts(&b,
            "#include <stdio.h>\n"
            "static inline int64_t nf64_to_i64(double value) {\n"
            "    if (!(value >= -0x1p63 && value < 0x1p63)) {\n"
            "        fputs(\"I cannot convert this float to int: I require a finite value in [-2^63, 2^63).\\n\", stderr);\n"
            "        exit(EXIT_FAILURE);\n    }\n"
            "    return (int64_t)value;\n}\n");
        if (b.has_owned_strings) {
            nvm2c_puts(&b, "#include <string.h>\n");
            emit_nstr_storage(&b);
        }
        if (module_has_opcode(mod, OP_PUSH_F64)) nvm2c_puts(&b,
            "#include <string.h>\n"
            "static inline double nf64_from_bits(uint64_t bits) {\n"
            "    double value; memcpy(&value, &bits, sizeof value); return value;\n}\n");
        if (mod->import_count) {
            nvm2c_puts(&b,
                "#include <stdlib.h>\n#include <string.h>\n#include <unistd.h>\n"
                "static int nhost_arg_count;\nstatic char **nhost_args;\n");
            int argv_used = module_uses_host(mod, "nhost_argv");
            int env_used = module_uses_host(mod, "nhost_getenv");
            int tmp_used = module_uses_host(mod, "nhost_tmp_dir");
            int cwd_used = module_uses_host(mod, "nhost_getcwd");
            if (module_uses_host(mod, "nhost_file_read")) emit_host_file_read(&b);
            if (module_uses_host(mod, "nhost_file_write")) emit_host_file_write(&b);
            if (module_uses_host(mod, "nhost_normalize")) emit_host_normalize(&b);
            /* I preserve the VM builtin contract, including its explicit int narrowing. */
            if (module_uses_host(mod, "nhost_is_digit")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_digit(int64_t code) { int c = (int)code; return c >= '0' && c <= '9'; }\n");
            if (module_uses_host(mod, "nhost_is_alpha")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_alpha(int64_t code) { int c = (int)code; return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }\n");
            if (module_uses_host(mod, "atan")) nvm2c_puts(&b, "#include <math.h>\n");
            if (module_uses_host(mod, "nhost_strlen")) nvm2c_puts(&b,
                "#include <string.h>\nstatic inline int64_t nhost_strlen(const char *value) { return (int64_t)strlen(value ? value : \"\"); }\n");
            if (module_uses_host(mod, "nhost_is_alnum")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_alnum(int64_t code) { int c = (int)code; return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }\n");
            if (module_uses_host(mod, "nhost_is_space")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_space(int64_t code) { int c = (int)code; unsigned char u = (unsigned char)c; return c >= 0 && (u == ' ' || (u >= 9 && u <= 13)); }\n");
            if (module_uses_host(mod, "nhost_is_upper")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_upper(int64_t code) { int c = (int)code; return c >= 'A' && c <= 'Z'; }\n");
            if (module_uses_host(mod, "nhost_is_lower")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_lower(int64_t code) { int c = (int)code; return c >= 'a' && c <= 'z'; }\n");
            if (module_uses_host(mod, "nhost_is_whitespace")) nvm2c_puts(&b,
                "static inline int64_t nhost_is_whitespace(int64_t code) { return code == ' ' || code == 9 || code == 10 || code == 13; }\n");
            if (module_uses_host(mod, "nhost_digit_value")) nvm2c_puts(&b,
                "static inline int64_t nhost_digit_value(int64_t code) { return code >= '0' && code <= '9' ? code - '0' : -1; }\n");
            if (module_uses_host(mod, "nhost_from_char")) nvm2c_puts(&b,
                "static char *nstr_allocate(size_t n);\n"
                "static inline const char *nhost_from_char(int64_t code) {\n"
                "    char *text = nstr_allocate(1);\n"
                "    text[0] = (char)code; text[1] = 0;\n"
                "    return text;\n}\n");
            if (module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
                "static inline const char *nhost_mktemp_dir(const char *prefix) {\n"
                "    const char *root = getenv(\"TMPDIR\");\n"
                "    if (!root || !*root) root = \"/tmp\";\n"
                "    if (!prefix) prefix = \"nano_\";\n"
                "    size_t a = strlen(root), z = strlen(prefix);\n"
                "    if (z > SIZE_MAX - 8 || a > SIZE_MAX - z - 8) NVM2C_ABORT();\n"
                "    char *path = nstr_allocate(a + z + 7);\n"
                "    if (!path) NVM2C_ABORT();\n"
                "    memcpy(path, root, a); path[a] = '/';\n"
                "    memcpy(path + a + 1, prefix, z);\n"
                "    memcpy(path + a + z + 1, \"XXXXXX\", 7);\n"
                "    if (!mkdtemp(path)) path[0] = 0;\n"
                "    return path;\n}\n");
            if (module_uses_host(mod, "nhost_shell")) nvm2c_puts(&b,
                "static inline int64_t nhost_shell(const char *command) {\n"
                "    return (int64_t)system(command);\n}\n");
            if (module_uses_host(mod, "nhost_capture")) nvm2c_puts(&b,
                "#include <stdio.h>\n"
                "static inline const char *nhost_capture(const char *command) {\n"
                "    char *output = malloc(65536);\n"
                "    if (!output) NVM2C_ABORT();\n"
                "    output[0] = 0;\n"
                "    FILE *pipe = popen(command, \"r\");\n"
                "    if (!pipe) return nstr_take(output);\n"
                "    size_t used = fread(output, 1, 65535, pipe);\n"
                "    output[used] = 0;\n"
                "    char discard[4096];\n"
                "    while (fread(discard, 1, sizeof discard, pipe)) {}\n"
                "    pclose(pipe);\n"
                "    return nstr_take(output);\n}\n");
            int identity_used = module_uses_host(mod, "nhost_identity");
            int destinations_used = module_uses_host(mod, "nhost_destinations");
            if (identity_used || destinations_used) emit_host_identity(&b, identity_used, destinations_used);
            int file_exists_used = module_uses_host(mod, "nhost_file_exists");
            int dir_exists_used = module_uses_host(mod, "nhost_dir_exists");
            if (file_exists_used || dir_exists_used) nvm2c_puts(&b, "#include <sys/stat.h>\n");
            if (file_exists_used) nvm2c_puts(&b,
                "static inline int64_t nhost_file_exists(const char *path) {\n"
                "    struct stat info;\n"
                "    return path && stat(path, &info) == 0 ? 1 : 0;\n}\n");
            if (dir_exists_used) nvm2c_puts(&b,
                "static inline int64_t nhost_dir_exists(const char *path) {\n"
                "    struct stat info;\n"
                "    return path && stat(path, &info) == 0 && S_ISDIR(info.st_mode) ? 1 : 0;\n}\n");
            int remove_used = module_uses_host(mod, "nhost_remove");
            int rename_used = module_uses_host(mod, "nhost_rename");
            if (remove_used || rename_used) nvm2c_puts(&b, "#include <stdio.h>\n");
            if (remove_used) nvm2c_puts(&b,
                "static inline int64_t nhost_remove(const char *path) {\n"
                "    return path && remove(path) == 0 ? 0 : -1;\n}\n");
            if (rename_used) nvm2c_puts(&b,
                "static inline int64_t nhost_rename(const char *from, const char *to) {\n"
                "    return from && to && rename(from, to) == 0 ? 0 : -1;\n}\n");
            if (argv_used || env_used || tmp_used || cwd_used) nvm2c_puts(&b,
                "static inline const char *nhost_copy(const char *value) {\n"
                "    return nstr_copy(value);\n}\n");
            if (module_uses_host(mod, "nhost_argc")) nvm2c_puts(&b,
                "static inline int64_t nhost_argc(void) { return nhost_arg_count; }\n");
            if (argv_used) nvm2c_puts(&b,
                "static inline const char *nhost_argv(int64_t i) {\n"
                "    return i < 0 || i >= nhost_arg_count ? \"\" : nhost_copy(nhost_args[i]);\n}\n");
            if (env_used) nvm2c_puts(&b,
                "static inline const char *nhost_getenv(const char *name) {\n"
                "    return nhost_copy(getenv(name ? name : \"\"));\n}\n");
            if (tmp_used) nvm2c_puts(&b,
                "static inline const char *nhost_tmp_dir(void) {\n"
                "    const char *value = getenv(\"TMPDIR\");\n"
                "    return nhost_copy(value && value[0] ? value : \"/tmp\");\n}\n");
            if (cwd_used) nvm2c_puts(&b,
                "static inline const char *nhost_getcwd(void) {\n"
                "    char value[1024];\n"
                "    return nhost_copy(getcwd(value, sizeof value) ? value : \"\");\n}\n");
        }
        if (need_print || need_cast) {
            nvm2c_puts(&b, "#include <stdio.h>\n");
        }
        if (module_has_opcode(mod, OP_F64_FROM_BITS) || module_has_opcode(mod, OP_F64_TO_BITS))
            nvm2c_puts(&b, NL_BINARY64_BITS_SOURCE);
        if (need_print || need_cast) nvm2c_puts(&b, NL_BINARY64_FORMAT_SOURCE);
        if (need_print) nvm2c_puts(&b,
            "static inline void nf64_print(double value) {\n"
            "    const char *special = nano_rt_f64_nonfinite(value);\n"
            "    if (special) fputs(special, stdout);\n"
            "    else if (value >= -1e15 && value <= 1e15 && value == (int64_t)value) printf(\"%.1f\", value);\n"
            "    else printf(\"%g\", value);\n}\n");
        if (need_concat || need_cast || need_substr || need_trim || need_arr_lit || need_arr_get ||
            need_arr_push || need_iarr_new || need_sarr_new || need_agg_get ||
            need_assert || need_rarr || b.has_maps || module_has_opcode(mod, OP_AGG_PACK) ||
            module_has_opcode(mod, OP_CAST_INT) || module_has_opcode(mod, OP_CAST_FLOAT)) {
            nvm2c_puts(&b, "#include <stdlib.h>\n#include <string.h>\n");
        } else if (need_string) {
            nvm2c_puts(&b, "#include <string.h>\n");
        }
        nvm2c_puts(&b,
            "\ntypedef struct nmap_s *nmap_t;\n"
            "typedef struct { int64_t *data; size_t len; struct narr_owner *owner; } narr_s;\n"
            "typedef narr_s *narr_t;\n"
            "typedef struct { const char **data; size_t len; struct nsarr_owner *owner; } nsarr_s;\n"
            "typedef nsarr_s *nsarr_t;\n");
        if (need_iarr || need_sarr || need_rarr) b.has_owned_aggregates = 1;
        if (b.has_owned_aggregates) {
            b.has_maps = 1;
            emit_nagg_accounting(&b);
        }
        b.has_float_arithmetic = b.has_maps ||
            module_has_opcode(mod, OP_F64_ADD) || module_has_opcode(mod, OP_F64_SUB) ||
            module_has_opcode(mod, OP_F64_MUL) || module_has_opcode(mod, OP_F64_DIV) ||
            module_has_opcode(mod, OP_ADD) || module_has_opcode(mod, OP_SUB) ||
            module_has_opcode(mod, OP_MUL) || module_has_opcode(mod, OP_DIV);
        if (b.has_float_arithmetic) nvm2c_puts(&b, nl_binary64_arithmetic_source);
        if (b.has_maps || module_has_opcode(mod, OP_CAST_FLOAT)) {
            nvm2c_puts(&b, nbp_parser_source);
            nvm2c_puts(&b,
                "static inline double nparse_binary64(const char *text) {\n"
                "    size_t length = text ? strlen(text) : 0; uint64_t bits = 0;\n"
                "    if (length > UINT32_MAX || !nbp_parse((const unsigned char *)text, (uint32_t)length, &bits)) NVM2C_ABORT();\n"
                "    double result; memcpy(&result, &bits, sizeof result); return result;\n}\n");
        }
        if (need_sarr) { b.has_string_arrays = 1; emit_nsarr_storage(&b); }
        if (need_iarr) { b.has_integer_arrays = 1; emit_narr_storage(&b); }
        if (b.has_maps) {
            nvm2c_puts(&b,
#include "nvm2c_map_runtime.inc"
            );
            nvm2c_puts(&b,
                "static size_t nmap_collection_budget = 65536;\n"
                "typedef struct nmap_owned { nmap_t map; struct nmap_owned *next; unsigned marked; } nmap_owned;\n"
                "static nmap_owned *nmap_owned_head;\n"
                "typedef struct nvalue_owned { nmap_value value; struct nvalue_owned *next; unsigned marked; } nvalue_owned;\n"
                "static nvalue_owned *nvalue_owned_head;\n"
                "static size_t nmap_owned_live, nmap_owned_peak;\n"
                "static nmap_value nmap_owned_get(nmap_t map, const char *key) {\n"
                "    nmap_value value = nmap_get(map, key);\n"
                "    if (value.kind == 5) { nvalue_owned *owner = malloc(sizeof *owner);\n"
                "        if (!owner) { nmap_release_value(value); NVM2C_ABORT(); }\n"
                "        *owner = (nvalue_owned){value, nvalue_owned_head, 0}; nvalue_owned_head = owner;\n"
                "        nmap_bytes_add(sizeof *owner);\n"
                "        if (++nmap_owned_live > nmap_owned_peak) nmap_owned_peak = nmap_owned_live; }\n"
                "    return value;\n}\n"
                "static inline nmap_value nvalue_from_float(double value) {\n"
                "    nmap_value result = {3, 0, NULL}; memcpy(&result.integer, &value, sizeof value); return result;\n}\n"
                "static inline double nvalue_require_float(nmap_value value) {\n"
                "    if (value.kind != 3) NVM2C_ABORT();\n"
                "    double result; memcpy(&result, &value.integer, sizeof result); return result;\n}\n"
                "static inline nmap_value nvalue_numeric(nmap_value a, nmap_value b, char op) {\n"
                "    if (op != '~' && a.kind == 9) a.kind = 1;\n"
                "    if (a.kind != 1 && a.kind != 3) NVM2C_ABORT();\n"
                "    if (op == '~') return a.kind == 3 ? nvalue_from_float(-nvalue_require_float(a)) : (nmap_value){1, ni64_from_bits(UINT64_C(0) - (uint64_t)a.integer), NULL};\n"
                "    if (b.kind == 9) b.kind = 1;\n"
                "    if (b.kind != 1 && b.kind != 3) NVM2C_ABORT();\n"
                "    if (a.kind == 1 && b.kind == 1) {\n"
                "        int64_t x = a.integer, y = b.integer, value;\n"
                "        if (op == '+') value = ni64_from_bits((uint64_t)x + (uint64_t)y);\n"
                "        else if (op == '-') value = ni64_from_bits((uint64_t)x - (uint64_t)y);\n"
                "        else if (op == '*') value = ni64_from_bits((uint64_t)x * (uint64_t)y);\n"
                "        else if (op == '/') value = y == 0 ? 0 : (x == INT64_MIN && y == -1) ? INT64_MIN : x / y;\n"
                "        else NVM2C_ABORT();\n"
                "        return (nmap_value){1, value, NULL};\n"
                "    }\n"
                "    double x = a.kind == 3 ? nvalue_require_float(a) : (double)a.integer;\n"
                "    double y = b.kind == 3 ? nvalue_require_float(b) : (double)b.integer;\n"
                "    if (op == '+') return nvalue_from_float(nano_rt_f64_add(x, y));\n"
                "    if (op == '-') return nvalue_from_float(nano_rt_f64_sub(x, y));\n"
                "    if (op == '*') return nvalue_from_float(nano_rt_f64_mul(x, y));\n"
                "    if (op == '/') return nvalue_from_float(nano_rt_f64_div(x, y));\n"
                "    NVM2C_ABORT();\n}\n"
                "static inline int64_t nvalue_require_int(nmap_value value) {\n"
                "    if (value.kind != 1) NVM2C_ABORT();\n    return value.integer;\n}\n"
                "static inline int64_t nvalue_require_bool(nmap_value value) {\n"
                "    if (value.kind != 4) NVM2C_ABORT();\n    return value.integer;\n}\n"
                "static inline const char *nvalue_require_string(nmap_value value) {\n"
                "    if (value.kind != 5) NVM2C_ABORT();\n    return value.text;\n}\n"
                "static inline nmap_t nvalue_require_map(nmap_value value) {\n"
                "    if (value.kind != 13 || !value.text) NVM2C_ABORT();\n    return (nmap_t)value.text;\n}\n"
                "static inline double nvalue_cast_float(nmap_value value) {\n"
                "    if (value.kind == 3) return nvalue_require_float(value);\n"
                "    if (value.kind == 1 || value.kind == 2) return (double)value.integer;\n"
                "    if (value.kind == 4) return value.integer ? 1.0 : 0.0;\n"
                "    if (value.kind == 5) return nparse_binary64(value.text);\n"
                "    return 0.0;\n}\n"
                "static inline int64_t nvalue_cast_int(nmap_value value) {\n"
                "    if (value.kind == 3) return nf64_to_i64(nvalue_require_float(value));\n"
                "    return value.kind == 1 || value.kind == 2 || value.kind == 4 || value.kind == 9 ? value.integer : value.kind == 5 ? (int64_t)strtoll(value.text, NULL, 10) : 0;\n}\n"
                "static inline int nvalue_equal(nmap_value a, nmap_value b) {\n"
                "    if ((a.kind == 9 && b.kind == 1) || (a.kind == 1 && b.kind == 9)) return a.integer == b.integer;\n"
                "    if (a.kind == 1 && b.kind == 3) return (double)a.integer == nvalue_require_float(b);\n"
                "    if (a.kind == 3 && b.kind == 1) return nvalue_require_float(a) == (double)b.integer;\n"
                "    if (a.kind != b.kind) return 0;\n"
                "    if (a.kind == 7 || a.kind == 13) return a.text == b.text;\n"
                "    if (a.kind == 3) return nvalue_require_float(a) == nvalue_require_float(b);\n"
                "    if (a.kind == 0) return 1;\n"
                "    if (a.kind == 1 || a.kind == 2 || a.kind == 4 || a.kind == 9) return a.integer == b.integer;\n"
                "    if (a.text == b.text) return 1;\n"
                "    if (!a.text || !b.text) return 0;\n"
                "    return strcmp(a.text, b.text) == 0;\n}\n"
                "static inline int nvalue_compare(nmap_value a, nmap_value b) {\n"
                "    if ((a.kind == 9 && b.kind == 1) || (a.kind == 1 && b.kind == 9)) return (a.integer > b.integer) - (a.integer < b.integer);\n"
                "    if ((a.kind == 3 && (b.kind == 3 || b.kind == 1)) || (a.kind == 1 && b.kind == 3)) {\n"
                "        double x = a.kind == 3 ? nvalue_require_float(a) : (double)a.integer;\n"
                "        double y = b.kind == 3 ? nvalue_require_float(b) : (double)b.integer;\n"
                "        return (x > y) - (x < y);\n    }\n"
                "    if (a.kind != b.kind) return (int)a.kind - (int)b.kind;\n"
                "    if (a.kind == 1 || a.kind == 2 || a.kind == 4) return (a.integer > b.integer) - (a.integer < b.integer);\n"
                "    if (a.kind == 5) {\n"
                "        if (a.text == b.text) return 0;\n"
                "        if (!a.text) { return -1; } if (!b.text) { return 1; }\n"
                "        return strcmp(a.text, b.text);\n    }\n"
                "    return 0;\n}\n"
                "static nmap_t nmap_owned_new(uint8_t kind) {\n"
                "    nmap_t map = nmap_new(kind); nmap_owned *owner = malloc(sizeof *owner);\n"
                "    if (!owner) { nmap_destroy(map); NVM2C_ABORT(); }\n"
                "    *owner = (nmap_owned){map, nmap_owned_head, 0}; nmap_owned_head = owner;\n"
                "    nmap_bytes_add(sizeof *owner);\n"
                "    if (++nmap_owned_live > nmap_owned_peak) nmap_owned_peak = nmap_owned_live;\n"
                "    return map;\n}\n"
                "static void nmap_release_owned(void) {\n"
                "    while (nvalue_owned_head) { nvalue_owned *owner = nvalue_owned_head;\n"
                "        nvalue_owned_head = owner->next; nmap_release_value(owner->value); nmap_bytes_drop(sizeof *owner); free(owner); --nmap_owned_live; }\n"
                "    while (nmap_owned_head) { nmap_owned *owner = nmap_owned_head;\n"
                "        nmap_owned_head = owner->next; nmap_destroy(owner->map); nmap_bytes_drop(sizeof *owner); free(owner); --nmap_owned_live; }\n}\n");
        }
        if (b.global_count) nvm2c_printf(&b, "static nmap_value nglobal[%zu];\n", b.global_count);
        emit_walk_adapters(&b, mod);
        emit_scalar_artifact_adapters(&b, mod);
        nvm2c_puts(&b, "typedef struct nrarr_s nrarr_s;\ntypedef nrarr_s *nrarr_t;\n");
        nvm2c_puts(&b, "typedef struct nrec_s nrec_t;\n");
        nvm2c_printf(&b,
            "struct nrec_s { int64_t f[%zu]; const char *s[%zu]; narr_t a[%zu]; nsarr_t sa[%zu]; nrarr_t ra[%zu]; const nrec_t *rec[%zu]; uint8_t k[%zu], vk[%zu]; uint16_t n, tag; uint8_t kind;",
            b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width);
        if (b.has_maps) nvm2c_printf(&b, " nmap_t m[%zu];", b.record_width);
        nvm2c_puts(&b, " };\n");
        if (module_has_opcode(mod, OP_ARR_SET)) nvm2c_puts(&b,
            "static inline int nrec_field_storage_matches(const nrec_t *a, const nrec_t *b, size_t field) {\n"
            "    unsigned ak = a->k[field], bk = b->k[field];\n"
            "    if (ak == bk) return 1;\n"
            "    unsigned at = ak == 0 ? 1 : ak == 9 ? 4 : ak == 1 ? 5 : ak == 8 ? a->vk[field] : 255;\n"
            "    unsigned bt = bk == 0 ? 1 : bk == 9 ? 4 : bk == 1 ? 5 : bk == 8 ? b->vk[field] : 255;\n"
            "    return at == bt && (at == 1 || at == 4 || at == 5) &&\n"
            "           (at != 5 || (a->s[field] && b->s[field]));\n}\n");
        nvm2c_puts(&b,
            "struct nrarr_s { nrec_t *data; size_t len; struct nrarr_owner *owner; };\n\n");
        if (module_has_opcode(mod, OP_AGG_PACK)) nvm2c_puts(&b,
            "typedef struct nrec_owned { nrec_t value; struct nrec_owned *next; } nrec_owned;\n"
            "static nrec_owned *nrec_owned_head;\n"
            "static inline const nrec_t *nrec_snapshot(nrec_t value) {\n"
            "    nrec_owned *node = malloc(sizeof *node);\n"
            "    if (!node) NVM2C_ABORT();\n"
            "    nagg_add(sizeof *node);\n"
            "    node->value = value; node->next = nrec_owned_head; nrec_owned_head = node;\n"
            "    return &node->value;\n}\n"
            "static void nrec_release_snapshots(void) {\n"
            "    while (nrec_owned_head) { nrec_owned *node = nrec_owned_head;\n"
            "        nrec_owned_head = node->next; nagg_drop(sizeof *node); free(node); }\n}\n");
        if (need_rarr) emit_nrarr_helpers(&b, need_rarr_lit || module_has_opcode(mod, OP_ARR_NEW),
                                         need_rarr_lit || need_arr_push, need_arr_get);
        if (b.has_maps) {
            nvm2c_puts(&b,
#include "nvm2c_map_roots.inc"
            );
            if (b.has_owned_strings) emit_nstr_sweep(&b);
            if (b.has_owned_aggregates) emit_nagg_sweep(&b, module_has_opcode(mod, OP_AGG_PACK));
            nvm2c_puts(&b, "static void nmap_collect(void) {\n    nroot_list work = {0};\n"
                "    for (nroot_frame *f = nroot_head; f; f = f->prev)\n"
                "        for (size_t i = 0; i < f->live.count; ++i)\n"
                "            nroot_add(&work, f->live.items[i].kind, f->live.items[i].ptr);\n");
            if (b.global_count) nvm2c_printf(&b,
                "    for (size_t i = 0; i < %zu; ++i) nroot_value(&work, nglobal[i]);\n", b.global_count);
            nvm2c_puts(&b, "    nroot_trace(&work);\n");
            if (b.has_owned_strings) nvm2c_puts(&b, "    nstr_sweep(&work);\n");
            if (b.has_owned_aggregates) nvm2c_puts(&b, "    nagg_sweep(&work);\n");
            nvm2c_puts(&b, "    nroot_destroy(&work); nmap_sweep();\n"
                "    nmap_allocation_debt = 0;\n"
                "    nmap_collection_budget = nmap_live_bytes > 65536 ? nmap_live_bytes : 65536;\n}\n"
                "/* I trace fresh mutable edges for map debt or an owned byte budget.\n"
                " * Without allocation, dropped owners wait for the next allocating safepoint. */\n"
                "static inline void nmap_collect_if_needed(void) {\n"
                "    if (nmap_allocation_debt >= nmap_collection_budget");
            if (b.has_owned_strings)
                nvm2c_puts(&b, " || nstr_allocation_debt >= nstr_collection_budget");
            if (b.has_owned_aggregates)
                nvm2c_puts(&b, " || nagg_allocation_debt >= nagg_collection_budget");
            nvm2c_puts(&b, ") nmap_collect();\n}\n");
        }
        if (need_concat) emit_nstr_concat(&b);
        if (need_substr) emit_nstr_substr(&b);
        if (need_trim) emit_nstr_trim(&b);
        if (need_char_at) emit_nstr_char_at(&b);
        if (need_starts) emit_nstr_starts_with(&b);
        if (need_ends) emit_nstr_ends_with(&b);
        if (need_cast) { emit_nstr_from_i64(&b); emit_nstr_from_f64(&b); }
        if (need_iarr_lit) emit_narr_lit(&b);
        if (need_iarr_get) emit_narr_get(&b);
        if (need_iarr_push) emit_narr_push(&b);
        if (need_sarr_lit) emit_nsarr_lit(&b);
        if (need_sarr_get) emit_nsarr_get(&b);
        if (need_sarr_push) emit_nsarr_push(&b);
        if (b.has_maps) emit_tagged_array_helpers(&b, need_iarr_push, need_sarr_push,
                                                need_iarr_get, need_sarr_get, need_print);
    }

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            if (!b.emitted_functions[i]) continue;
            emit_prototype(&b, mod, i, kinds);
            if (b.failed) goto fail;
        }
    }
    nvm2c_puts(&b, "\n");

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            if (!b.emitted_functions[i]) continue;
            emit_function_body(&b, mod, i, kinds, rec_fields, facts.results);
            if (b.failed) goto fail;
        }
    }

    {
        uint32_t entry = mod->header.entry_point;
        if (entry >= mod->function_count) {
            nvm2c_fail(&b, "entry_point %u is not a function", entry);
            goto fail;
        }
        const NvmFunctionEntry *ef = &mod->functions[entry];
        if (ef->arity != 0) {
            nvm2c_fail(&b, "entry function must have arity 0 to become C main");
            goto fail;
        }
        if (!(ef->result_count == 1 && ef->result_tag == TAG_INT)) {
            nvm2c_fail(&b, "entry function must return a single int");
            goto fail;
        }
        char ename[64];
        fn_c_name(mod, entry, ename, sizeof ename);
        if (mod->import_count) nvm2c_puts(&b,
            "int main(int argc, char **argv) {\n"
            "    nhost_arg_count = argc; nhost_args = argv;\n");
        else nvm2c_puts(&b, "int main(void) {\n");
        nvm2c_puts(&b, "    (void)nf64_to_i64;\n");
        if (b.has_float_arithmetic) nvm2c_puts(&b,
            "    (void)nano_rt_f64_add; (void)nano_rt_f64_sub; (void)nano_rt_f64_mul; (void)nano_rt_f64_div;\n");
        if (module_has_opcode(mod, OP_F64_FROM_BITS) || module_has_opcode(mod, OP_F64_TO_BITS))
            nvm2c_puts(&b, "    (void)nl_float_from_bits; (void)nl_float_to_bits;\n");
        if (module_has_opcode(mod, OP_I64_ADD_CARRY) || module_has_opcode(mod, OP_I64_SUB_BORROW) ||
            module_has_opcode(mod, OP_I64_MUL_WIDE_S) || module_has_opcode(mod, OP_I64_MUL_WIDE_U))
            nvm2c_puts(&b, "    (void)ni64_pair_compute;\n");
        if (module_has_opcode(mod, OP_AGG_GET))
            nvm2c_puts(&b, "    (void)nrec_f64;\n");
        /* Standard C references keep strict unused-function warnings clean. */
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            if (!b.emitted_functions[i]) continue;
            char name[64];
            fn_c_name(mod, i, name, sizeof name);
            nvm2c_printf(&b, "    (void)%s;\n", name);
        }
        if (b.has_maps) nvm2c_puts(&b,
            "    (void)nmap_owned_new; (void)nmap_set; (void)nmap_get; (void)nmap_owned_get;\n"
            "    (void)nvalue_numeric; (void)nvalue_from_float; (void)nvalue_require_int; (void)nvalue_require_bool; (void)nvalue_require_string; (void)nvalue_require_map; (void)nvalue_cast_int; (void)nvalue_cast_float; (void)nvalue_equal;\n"
            "    (void)nvalue_compare;\n"
            "    (void)nvalue_require_int_array; (void)nvalue_array_len; (void)nvalue_array_get; (void)nvalue_array_set; (void)nvalue_array_push;\n"
            "    (void)nmap_has; (void)nmap_len; (void)nmap_delete; (void)nmap_collect;\n"
            "    (void)nroot_reset; (void)nmap_collect_if_needed;\n");
        if (b.has_owned_aggregates) nvm2c_puts(&b,
            "    (void)nagg_add; (void)nagg_drop;\n");
        if (module_has_opcode(mod, OP_PRINT) || module_has_opcode(mod, OP_PRINTLN))
            nvm2c_puts(&b, "    (void)nf64_print;\n");
        if (b.has_string_arrays) nvm2c_puts(&b,
            "    (void)nsarr_new; (void)nsarr_reserve; (void)nsarr_copy_string;\n");
        if (b.has_integer_arrays) nvm2c_puts(&b,
            "    (void)narr_new; (void)narr_reserve;\n");
        if (b.has_record_array_allocations) nvm2c_puts(&b,
            "    (void)nrarr_new; (void)nrarr_reserve;\n");
        if (b.has_record_array_getter) nvm2c_puts(&b, "    (void)nrarr_get;\n");
        if (module_has_opcode(mod, OP_ARR_PUSH) && b.has_string_arrays)
            nvm2c_puts(&b, "    (void)nsarr_push;\n");
        if (module_has_opcode(mod, OP_ARR_PUSH) && b.has_integer_arrays)
            nvm2c_puts(&b, "    (void)narr_push;\n");
        if (module_has_opcode(mod, OP_AGG_PACK)) nvm2c_puts(&b, "    (void)nrec_snapshot;\n");
        if (module_has_opcode(mod, OP_ARR_SET)) nvm2c_puts(&b, "    (void)nrec_field_storage_matches;\n");
        if (b.has_owned_strings) nvm2c_puts(&b, "    (void)nstr_copy; (void)nstr_take; (void)nstr_copy_release;\n");
        if (module_has_opcode(mod, OP_CAST_STRING)) nvm2c_puts(&b, "    (void)nstr_from_i64; (void)nstr_from_f64;\n");
        if (b.has_maps && (module_has_opcode(mod, OP_PRINT) || module_has_opcode(mod, OP_PRINTLN)))
            nvm2c_puts(&b, "    (void)nvalue_array_print;\n");
        if (module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRUCT)) nvm2c_puts(&b, "    (void)nrarr_push;\n");
        for (uint32_t i = 0; i < mod->import_count; ++i) {
            const Nvm2cHost *host = import_host(mod, i);
            if (host && host->result == TAG_ARRAY)
                nvm2c_printf(&b, "    (void)nhost_walk_%u;\n", i);
            if (scalar_artifact_adapter(host))
                nvm2c_printf(&b, "    (void)nhost_artifact_%u;\n", i);
        }
        /* I mirror vm_execute: the first named initializer runs before entry,
         * even when that same function is also the entry point. */
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            const char *name = nvm_get_string(mod, mod->functions[i].name_idx);
            if (name && strcmp(name, "__init__") == 0) {
                if (mod->functions[i].arity != 0) {
                    nvm2c_fail(&b, "I require a zero-argument module initializer"); goto fail;
                }
                char initializer[64];
                fn_c_name(mod, i, initializer, sizeof initializer);
                nvm2c_printf(&b, "    (void)%s();\n", initializer);
                break;
            }
        }
        nvm2c_printf(&b, "    int result = (int)%s();\n", ename);
        if (module_has_opcode(mod, OP_AGG_PACK)) nvm2c_puts(&b, "    nrec_release_snapshots();\n");
        if (b.has_maps) nvm2c_puts(&b, "    nmap_release_owned();\n");
        if (b.has_record_array_allocations) nvm2c_puts(&b, "    nrarr_release_owned();\n");
        if (b.has_string_arrays) nvm2c_puts(&b, "    nsarr_release_owned();\n");
        if (b.has_integer_arrays) nvm2c_puts(&b, "    narr_release_owned();\n");
        if (b.has_owned_strings) nvm2c_puts(&b, "    nstr_release_owned();\n");
        nvm2c_puts(&b, "    return result;\n}\n");
    }

    if (b.failed) goto fail;
    if (contains_vm_wrapper_code(b.data)) {
        nvm2c_fail(&b, "internal error: emitted a VM wrapper rather than structured C");
        goto fail;
    }
    free(kinds);
    free(rec_fields);
    free(inference);
    free(global_stored);
    nvm_shape_destroy(&b.shapes);
    free(b.shape_locals);
    free(b.shape_results);
    free(b.shape_globals);
    for (uint32_t i = 0; i < mod->function_count; ++i) free(b.shape_outputs[i]);
    free(b.shape_outputs);
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        while (b.join_shapes[i]) {
            Nvm2cJoinShape *next = b.join_shapes[i]->next;
            free(b.join_shapes[i]); b.join_shapes[i] = next;
        }
    }
    free(b.join_shapes);
    free(b.emitted_functions);
    free(b.required_functions);
    free(b.tagged_locals);
    free(b.local_scalar_tags);
    while (b.scalar_joins) {
        Nvm2cScalarJoin *next = b.scalar_joins->next;
        free(b.scalar_joins); b.scalar_joins = next;
    }
    return b.data;

fail:
    free(kinds);
    free(rec_fields);
    free(inference);
    free(global_stored);
    nvm_shape_destroy(&b.shapes);
    free(b.shape_locals);
    free(b.shape_results);
    free(b.shape_globals);
    for (uint32_t i = 0; i < mod->function_count; ++i) free(b.shape_outputs[i]);
    free(b.shape_outputs);
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        while (b.join_shapes[i]) {
            Nvm2cJoinShape *next = b.join_shapes[i]->next;
            free(b.join_shapes[i]); b.join_shapes[i] = next;
        }
    }
    free(b.join_shapes);
    free(b.emitted_functions);
    free(b.required_functions);
    free(b.tagged_locals);
    free(b.local_scalar_tags);
    while (b.scalar_joins) {
        Nvm2cScalarJoin *next = b.scalar_joins->next;
        free(b.scalar_joins); b.scalar_joins = next;
    }
    free(b.data);
    return NULL;
}

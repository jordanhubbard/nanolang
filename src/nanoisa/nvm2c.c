/*
 * Structured C11 from a closed NanoISA subset.
 *
 * Temps are C arrays so backward goto is valid C. I64_ADD becomes
 * `t[i] = a + b`. Strings live in a parallel `s[]` of C string pointers.
 * The operand stack exists only while translating.
 */

#include "nvm2c.h"
#include "isa.h"
#include "utf8.h"
#include "nvm2c_shape.h"

#include <stdarg.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NVM2C_MAX_LOCALS 256

#define NVM2C_VK_INT 0
#define NVM2C_VK_STR 1
#define NVM2C_VK_UNK 2
#define NVM2C_VK_ARR 3
#define NVM2C_VK_REC 4
#define NVM2C_VK_SARR 5
#define NVM2C_VK_RARR 6
#define NVM2C_VK_MAP 7
#define NVM2C_VK_VALUE 8

typedef struct Nvm2cFieldBlock {
    struct Nvm2cFieldBlock *next;
    uint8_t data[];
} Nvm2cFieldBlock;

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
    Nvm2cFieldBlock *field_blocks;
    uint8_t *default_fields;
    NvmShapeGraph shapes;
    NvmShapeId *shape_locals, *shape_results, **shape_outputs;
    NvmShapeId *shape_current;
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

static const char *c_result_type(const NvmFunctionEntry *fn) {
    if (fn->result_count == 0 || fn->result_tag == TAG_VOID) return "void";
    if (fn->result_count != 1) return NULL;
    if (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL) return "int64_t";
    if (fn->result_tag == TAG_STRING) return "const char *";
    if (fn->result_tag == TAG_ARRAY) return "nrarr_t";
    if (fn->result_tag == TAG_HASHMAP) return "nmap_t";
    if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION) return "nrec_t";
    return NULL;
}

static int result_is_i64(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 &&
           (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL);
}

static const char *c_local_type(uint8_t kind) {
    if (kind == NVM2C_VK_STR) return "const char *";
    if (kind == NVM2C_VK_ARR) return "narr_t";
    if (kind == NVM2C_VK_SARR) return "nsarr_t";
    if (kind == NVM2C_VK_REC) return "nrec_t";
    if (kind == NVM2C_VK_RARR) return "nrarr_t";
    if (kind == NVM2C_VK_MAP) return "nmap_t";
    if (kind == NVM2C_VK_VALUE) return "nmap_value";
    return "int64_t";
}

static uint8_t fn_local_kind(const uint8_t *kinds, uint32_t fn, uint16_t slot) {
    return kinds[(size_t)fn * NVM2C_MAX_LOCALS + slot];
}

typedef struct {
    const char *name;
    const char *c_name;
    uint8_t argc, parameter, result;
} Nvm2cHost;

/* I recognize my builtin namespace, not arbitrary libraries exporting a name. */
static const Nvm2cHost host_adapters[] = {
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
    {"vm_string_from_char", "nhost_from_char", 1, TAG_INT, TAG_STRING},
    {"string_from_char", "nhost_from_char", 1, TAG_INT, TAG_STRING},
    {"vm_mktemp_dir", "nhost_mktemp_dir", 1, TAG_STRING, TAG_STRING},
};

/* These native contracts have homogeneous string parameters. I do not infer
 * an arbitrary artifact's ABI from its coarse NanoISA return tag. */
static const Nvm2cHost artifact_adapters[] = {
    {"fs_walkdir", "nhost_walk", 1, TAG_STRING, TAG_ARRAY},
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
                imp->return_type != host->result || !mod->import_param_types ||
                !mod->import_param_types[index]) continue;
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
        if (c == '\\' || c == '"') {
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
} Nvm2cSimSlot;

typedef struct {
    uint8_t *parameters;
    uint8_t *fields;
    uint8_t *results;
    int changed;
    int final;
} Nvm2cFacts;

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
                     offset % NVM2C_MAX_LOCALS, offset / NVM2C_MAX_LOCALS);
        } else if (dest < facts->results) {
            size_t offset = (size_t)(dest - facts->fields);
            size_t parameter = offset / b->record_width;
            snprintf(target, sizeof target, "field %zu of parameter %zu of function %zu",
                     offset % b->record_width, parameter % NVM2C_MAX_LOCALS,
                     parameter / NVM2C_MAX_LOCALS);
        } else {
            size_t offset = (size_t)(dest - facts->results);
            snprintf(target, sizeof target, "result field %zu of function %zu",
                     offset % b->record_width, offset / b->record_width);
        }
        nvm2c_fail(b, "I cannot assign conflicting kinds to a function parameter or aggregate field "
                      "(function %u, offset %zu: %s versus %s at %s)",
                   b->classify_function_index, b->classify_offset,
                   c_local_type(*dest), c_local_type(kind), target);
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

/* Callers opt into present-string to optional storage widening. Ordinary
 * aggregate field merging stays exact; the graph checks optional payloads. */
static int merge_parameter(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, uint8_t kind) {
    if (*dest == NVM2C_VK_VALUE && kind == NVM2C_VK_STR) return 1;
    if (*dest == NVM2C_VK_STR && kind == NVM2C_VK_VALUE) {
        *dest = NVM2C_VK_VALUE;
        facts->changed = 1;
        return 1;
    }
    return merge_fact(b, facts, dest, kind);
}

static int merge_record_results(Nvm2cBuf *b, Nvm2cFacts *facts, uint8_t *dest, const uint8_t *fields) {
    for (size_t i = 0; i < b->record_width; ++i)
        if (!merge_parameter(b, facts, &dest[i], fields[i])) return 0;
    return 1;
}

static int shape_ok(Nvm2cBuf *b) {
    if (b->shapes.error) {
        const InstructionInfo *info = isa_get_info(b->shape_opcode);
        nvm2c_fail(b, "I found invalid aggregate shape constraints during %s: %s",
                   info ? info->name : "classification", b->shapes.error);
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
    if (kind == NVM2C_VK_STR) return shape_type(b, id, NVM_SHAPE_STRING);
    if (kind == NVM2C_VK_REC) return shape_type(b, id, NVM_SHAPE_RECORD);
    if (kind == NVM2C_VK_MAP) return shape_type(b, id, NVM_SHAPE_MAP);
    if (kind == NVM2C_VK_VALUE) return shape_type(b, id, NVM_SHAPE_OPTIONAL);
    if (!shape_type(b, id, NVM_SHAPE_ARRAY)) return 0;
    if (b->shape_generic_array) return 1;
    NvmShapeId element = nvm_shape_child(&b->shapes, id, 0);
    return shape_type(b, element, kind == NVM2C_VK_RARR ? NVM_SHAPE_RECORD :
                                kind == NVM2C_VK_SARR ? NVM_SHAPE_STRING : NVM_SHAPE_INT);
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

/* Returning a present string into optional record storage is a conversion,
 * not equality between the source string and an optional shape. */
static int shape_record_return(Nvm2cBuf *b, NvmShapeId source, NvmShapeId result,
                               const uint8_t *source_fields, const uint8_t *result_fields) {
    if (!b->track_shapes) return 1;
    int optional = 0;
    for (size_t i = 0; i < b->record_width; ++i)
        if (result_fields[i] == NVM2C_VK_VALUE) optional = 1;
    if (!optional) return shape_equal(b, source, result);
    if (!shape_type(b, source, NVM_SHAPE_RECORD) || !shape_type(b, result, NVM_SHAPE_RECORD)) return 0;
    for (size_t i = 0; i < b->record_width; ++i) {
        NvmShapeId from = shape_child(b, source, (uint32_t)i);
        NvmShapeId to = shape_child(b, result, (uint32_t)i);
        if (!shape_kind(b, from, source_fields[i]) || !shape_kind(b, to, result_fields[i])) return 0;
        if (source_fields[i] == NVM2C_VK_STR && result_fields[i] == NVM2C_VK_VALUE) {
            if (!shape_equal(b, from, shape_child(b, to, 0))) return 0;
        } else if (!shape_equal(b, from, to)) return 0;
    }
    return 1;
}

static int sim_push_slot(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                         Nvm2cSimSlot slot) {
    if ((size_t)*sp >= b->sim_stack_capacity) {
        nvm2c_fail(b, "function %u: operand stack overflow", idx);
        return 0;
    }
    if (!slot.shape) slot.shape = shape_variable(b, b->shape_current);
    if (!shape_kind(b, slot.shape, slot.kind)) return 0;
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
    case NVM_SHAPE_STRING: return NVM2C_VK_STR;
    case NVM_SHAPE_RECORD: return NVM2C_VK_REC;
    case NVM_SHAPE_MAP: return NVM2C_VK_MAP;
    case NVM_SHAPE_OPTIONAL: return NVM2C_VK_VALUE;
    case NVM_SHAPE_ARRAY: {
        NvmShapeId element = nvm_shape_lookup(&b->shapes, id, 0);
        if (!element) return NVM2C_VK_UNK;
        switch (nvm_shape_kind(&b->shapes, element)) {
        case NVM_SHAPE_INT: return NVM2C_VK_ARR;
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
    Nvm2cSimSlot slot;
    memset(&slot, 0, sizeof slot);
    slot.kind = kind;
    slot.origin = origin;
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

static void mark_str_origin(uint8_t *local_kind, uint16_t nloc, int origin) {
    mark_origin(local_kind, nloc, origin, NVM2C_VK_STR);
}

static const uint8_t *fn_rec_k_const(const Nvm2cBuf *b, const uint8_t *tab, uint32_t fn, uint16_t slot) {
    return tab + ((size_t)fn * NVM2C_MAX_LOCALS + slot) * b->record_width;
}

typedef struct {
    Nvm2cSimSlot *slots;
    int sp;
    int set;
} Nvm2cSimJoin;

static int sim_join(Nvm2cBuf *b, uint32_t idx, Nvm2cSimJoin *join,
                    const Nvm2cSimSlot *stack, int sp) {
    if (!join->set) {
        if (sp) {
            join->slots = malloc((size_t)sp * sizeof *stack);
            if (!join->slots) {
                nvm2c_fail(b, "I cannot allocate classifier branch stack");
                return 0;
            }
            memcpy(join->slots, stack, (size_t)sp * sizeof *stack);
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
        if (b->track_shapes && !nvm_shape_unify(&b->shapes, join->slots[i].shape, stack[i].shape)) {
            nvm2c_fail(b, "I found incompatible shapes at a join in function %u: %s", idx, b->shapes.error);
            return 0;
        }
        int origin = join->slots[i].origin == stack[i].origin ? stack[i].origin : -1;
        if (join->slots[i].kind == NVM2C_VK_UNK) {
            join->slots[i] = stack[i];
            join->slots[i].origin = origin;
            continue;
        }
        join->slots[i].origin = origin;
        if (stack[i].kind == NVM2C_VK_UNK) continue;
        if (join->slots[i].kind != stack[i].kind) {
            nvm2c_fail(b, "I found incompatible stack kinds at a join in function %u", idx);
            return 0;
        }
        if (stack[i].kind == NVM2C_VK_REC || stack[i].kind == NVM2C_VK_RARR || stack[i].kind == NVM2C_VK_MAP) {
            join->slots[i].rec_k = sim_fields(b, join->slots[i].rec_k, 0);
            if (!join->slots[i].rec_k) return 0;
            for (size_t field = 0; field < b->record_width; field++) {
                uint8_t incoming = stack[i].rec_k[field];
                uint8_t *current = &join->slots[i].rec_k[field];
                if (*current == NVM2C_VK_UNK) *current = incoming;
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
    memcpy(local_kind, facts->parameters + (size_t)idx * NVM2C_MAX_LOCALS, fn->arity);
    memcpy(rec_fields, facts->fields + (size_t)idx * NVM2C_MAX_LOCALS * b->record_width,
           (size_t)fn->arity * b->record_width);

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

    while (pc < remaining) {
        size_t start = pc;
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            return 0;
        }
        pc += n;
        if (terminated && !joins[start].set) continue;
        if (targets[start]) {
            if (!terminated && !sim_join(b, idx, &joins[start], stk, sp)) return 0;
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
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            Nvm2cSimSlot loaded;
            if (slot >= nloc) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                return 0;
            }
            memset(&loaded, 0, sizeof loaded);
            loaded.kind = local_kind[slot];
            loaded.origin = (int)slot;
            loaded.shape = shape_variable(b, &b->shape_locals[(size_t)idx * NVM2C_MAX_LOCALS + slot]);
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
            if (!shape_equal(b, v.shape, shape_variable(b, &b->shape_locals[(size_t)idx * NVM2C_MAX_LOCALS + slot]))) return 0;
            if (v.kind == NVM2C_VK_VALUE) {
                local_kind[slot] = NVM2C_VK_VALUE;
            } else if (v.kind == NVM2C_VK_STR) {
                local_kind[slot] = NVM2C_VK_STR;
            } else if (v.kind == NVM2C_VK_ARR) {
                local_kind[slot] = NVM2C_VK_ARR;
            } else if (v.kind == NVM2C_VK_SARR) {
                local_kind[slot] = NVM2C_VK_SARR;
            } else if (v.kind == NVM2C_VK_REC) {
                local_kind[slot] = NVM2C_VK_REC;
                memcpy(rec_fields + (size_t)slot * b->record_width,
                       v.rec_k, b->record_width);
            } else if (v.kind == NVM2C_VK_RARR || v.kind == NVM2C_VK_MAP) {
                local_kind[slot] = v.kind;
                memcpy(rec_fields + (size_t)slot * b->record_width,
                       v.rec_k, b->record_width);
            } else if (v.kind == NVM2C_VK_INT && local_kind[slot] != NVM2C_VK_STR
                       && local_kind[slot] != NVM2C_VK_ARR
                        && local_kind[slot] != NVM2C_VK_SARR
                        && local_kind[slot] != NVM2C_VK_REC
                        && local_kind[slot] != NVM2C_VK_RARR
                        && local_kind[slot] != NVM2C_VK_MAP
                        && local_kind[slot] != NVM2C_VK_VALUE) {
                local_kind[slot] = NVM2C_VK_INT;
            }
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
            (void)rhs;
            (void)lhs;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_NEG:
        case OP_I64_NEG:
        case OP_BOOL_NOT: {
            Nvm2cSimSlot x;
            if (!sim_pop(b, idx, stk, &sp, &x)) return 0;
            (void)x;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            mark_str_origin(local_kind, nloc, v.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CONCAT: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            mark_str_origin(local_kind, nloc, rhs.origin);
            mark_str_origin(local_kind, nloc, lhs.origin);
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
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_STARTS_WITH:
        case OP_STR_ENDS_WITH:
        case OP_STR_CONTAINS: {
            Nvm2cSimSlot needle, hay;
            if (!sim_pop(b, idx, stk, &sp, &needle)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &hay)) return 0;
            mark_str_origin(local_kind, nloc, needle.origin);
            mark_str_origin(local_kind, nloc, hay.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CHAR_AT: {
            Nvm2cSimSlot ix, s;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)ix;
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
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
            if (value.kind != NVM2C_VK_VALUE && value.kind != NVM2C_VK_UNK && facts->final) {
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
        case OP_NE: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (lhs.kind != NVM2C_VK_VALUE && rhs.kind != NVM2C_VK_VALUE &&
                (lhs.kind != NVM2C_VK_INT || rhs.kind != NVM2C_VK_INT)) {
                mark_str_origin(local_kind, nloc, lhs.origin);
                mark_str_origin(local_kind, nloc, rhs.origin);
            }
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
            if (map.kind != NVM2C_VK_MAP && map.kind != NVM2C_VK_UNK) {
                nvm2c_fail(b, "I require a map for this hashmap operation"); return 0;
            }
            if (!shape_type(b, map.shape, NVM_SHAPE_MAP)) return 0;
            mark_origin(local_kind, nloc, map.origin, NVM2C_VK_MAP);
            if (ins.opcode != OP_HM_LEN) {
                if (key.kind != NVM2C_VK_STR && key.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "I require a string hashmap key"); return 0;
                }
                mark_str_origin(local_kind, nloc, key.origin);
                if (!shape_type(b, key.shape, NVM_SHAPE_STRING) ||
                    !shape_equal(b, shape_child(b, map.shape, 0), key.shape)) return 0;
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
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            }
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            if (tag == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else if (tag == TAG_INT) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            } else if (tag == TAG_STRUCT) {
                Nvm2cSimSlot array;
                memset(&array, 0, sizeof array);
                array.kind = NVM2C_VK_RARR;
                array.origin = -1;
                array.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!array.rec_k) return 0;
                if (!sim_push_slot(b, idx, stk, &sp, array)) return 0;
            } else {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int, string or struct elements", idx);
                return 0;
            }
            break;
        }
        case OP_ARR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (v.kind == NVM2C_VK_RARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_RARR);
            } else if (v.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_SARR);
            } else if (v.kind != NVM2C_VK_UNK) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_ARR);
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ARR_GET: {
            Nvm2cSimSlot ix, arr;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            (void)ix;
            NvmShapeId element_shape = shape_child(b, arr.shape, 0);
            if (!shape_equal(b, shape_variable(b, b->shape_current), element_shape)) return 0;
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
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            } else if (arr.kind == NVM2C_VK_UNK) {
                Nvm2cSimSlot value = {0};
                value.kind = NVM2C_VK_UNK;
                value.origin = -1;
                value.rec_k = sim_fields(b, NULL, NVM2C_VK_UNK);
                if (!value.rec_k || !sim_push_slot(b, idx, stk, &sp, value)) return 0;
            } else {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
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
                if (index.kind != NVM2C_VK_INT && index.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "ARR_SET index must be an integer");
                    return 0;
                }
            }
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
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
                    val.kind == NVM2C_VK_STR ? NVM2C_VK_SARR : NVM2C_VK_ARR;
                if (arr.kind != expected) {
                    nvm2c_fail(b, "ARR_SET element representation mismatch");
                    return 0;
                }
                if (expected == NVM2C_VK_RARR) {
                    for (size_t f = 0; f < b->record_width; ++f) {
                        if ((facts->final ||
                            (arr.rec_k[f] != NVM2C_VK_UNK &&
                             val.rec_k[f] != NVM2C_VK_UNK)) &&
                            arr.rec_k[f] != val.rec_k[f]) {
                            nvm2c_fail(b, "ARR_SET record field representation mismatch");
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
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            }
            if (!shape_equal(b, shape_child(b, arr.shape, 0), val.shape) ||
                !shape_equal(b, stk[sp - 1].shape, arr.shape)) return 0;
            break;
        }
        case OP_AGG_PACK: {
            uint16_t count = ins.operands[3].u16;
            Nvm2cSimSlot packed;
            uint16_t ai;
            memset(&packed, 0, sizeof packed);
            packed.kind = NVM2C_VK_REC;
            packed.origin = -1;
            packed.shape = shape_variable(b, b->shape_current);
            if (!shape_type(b, packed.shape, NVM_SHAPE_RECORD)) return 0;
            packed.rec_k = sim_fields(b, NULL, NVM2C_VK_INT);
            if (!packed.rec_k) return 0;
            if (count > b->record_width) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                return 0;
            }
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (v.kind != NVM2C_VK_INT && v.kind != NVM2C_VK_STR &&
                    v.kind != NVM2C_VK_ARR && v.kind != NVM2C_VK_SARR &&
                    v.kind != NVM2C_VK_RARR && v.kind != NVM2C_VK_REC && v.kind != NVM2C_VK_VALUE &&
                    !(v.kind == NVM2C_VK_UNK && !facts->final)) {
                    nvm2c_fail(b, "function %u: AGG_PACK field requires unsupported nested aggregate shape facts", idx);
                    return 0;
                }
                packed.rec_k[count - 1 - ai] = v.kind;
                if (!shape_equal(b, shape_child(b, packed.shape, count - 1 - ai), v.shape)) return 0;
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
            mark_origin(local_kind, nloc, rec.origin, NVM2C_VK_REC);
            if (fi >= b->record_width) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                return 0;
            }
            fk = rec.rec_k[fi];
            if (!shape_equal(b, shape_variable(b, b->shape_current),
                             shape_child(b, rec.shape, fi))) return 0;
            Nvm2cSimSlot field = {0};
            field.kind = fk;
            field.origin = -1;
            if (fk == NVM2C_VK_RARR || fk == NVM2C_VK_REC || fk == NVM2C_VK_UNK) {
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
                size_t at = (size_t)callee * NVM2C_MAX_LOCALS + i - 1;
                if (!merge_parameter(b, facts, &facts->parameters[at], arg.kind)) return 0;
                NvmShapeId parameter = shape_variable(b, &b->shape_locals[at]);
                if (facts->parameters[at] == NVM2C_VK_VALUE && arg.kind == NVM2C_VK_STR) {
                    if (!shape_type(b, parameter, NVM_SHAPE_OPTIONAL) ||
                        !shape_equal(b, arg.shape, shape_child(b, parameter, 0))) return 0;
                } else if (!shape_equal(b, arg.shape, parameter)) return 0;
                if (arg.kind == NVM2C_VK_UNK) {
                    mark_origin(local_kind, nloc, arg.origin, facts->parameters[at]);
                }
                if (arg.kind == NVM2C_VK_REC || arg.kind == NVM2C_VK_RARR || arg.kind == NVM2C_VK_MAP) {
                    if (!merge_fields(b, facts, facts->fields + at * b->record_width, arg.rec_k)) return 0;
                }
            }
            if (ins.opcode == OP_CALL) {
                if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
                } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                    Nvm2cSimSlot result = {0};
                    result.kind = NVM2C_VK_RARR;
                    result.origin = -1;
                    result.rec_k = sim_fields(b, facts->results + (size_t)callee * b->record_width, 0);
                    if (!result.rec_k || !sim_push_slot(b, idx, stk, &sp, result)) return 0;
                } else if (result_is_i64(cf)) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
                } else if (cf->result_count == 1 &&
                           (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION || cf->result_tag == TAG_HASHMAP)) {
                    Nvm2cSimSlot result;
                    memset(&result, 0, sizeof result);
                    result.kind = cf->result_tag == TAG_HASHMAP ? NVM2C_VK_MAP : NVM2C_VK_REC;
                    result.origin = -1;
                    result.rec_k = sim_fields(b, facts->results + (size_t)callee * b->record_width, 0);
                    if (!result.rec_k) return 0;
                    if (!sim_push_slot(b, idx, stk, &sp, result)) return 0;
                }
            } else if (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION) {
                if (!merge_record_results(b, facts, facts->results + (size_t)idx * b->record_width,
                                          facts->results + (size_t)callee * b->record_width)) return 0;
            } else if (cf->result_tag == TAG_ARRAY || cf->result_tag == TAG_HASHMAP) {
                if (!merge_fields(b, facts, facts->results + (size_t)idx * b->record_width,
                                  facts->results + (size_t)callee * b->record_width)) return 0;
            }
            if (ins.opcode == OP_CALL && cf->result_count == 1 && sp > 0) {
                if (!shape_equal(b, stk[sp - 1].shape, shape_variable(b, &b->shape_results[callee]))) return 0;
            } else if (ins.opcode == OP_TAIL_CALL && cf->result_count == 1) {
                if (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION) {
                    if (!shape_record_return(b, shape_variable(b, &b->shape_results[callee]),
                                             shape_variable(b, &b->shape_results[idx]),
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
                uint8_t expected = host->parameter == TAG_STRING ? NVM2C_VK_STR : NVM2C_VK_INT;
                if (arg.kind != expected && arg.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "function %u: CALL_EXTERN argument kind mismatch", idx);
                    return 0;
                }
                mark_origin(local_kind, nloc, arg.origin, expected);
            }
            if (!sim_push(b, idx, stk, &sp,
                          host->result == TAG_ARRAY ? NVM2C_VK_SARR :
                          host->result == TAG_STRING ? NVM2C_VK_STR : NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_JMP_FALSE: {
            Nvm2cSimSlot cond;
            if (!sim_pop(b, idx, stk, &sp, &cond)) return 0;
            (void)cond;
            break;
        }
        case OP_RET:
        case OP_HALT: {
            if (fn->result_count == 1 &&
                (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL ||
                 fn->result_tag == TAG_STRING || fn->result_tag == TAG_ARRAY ||
                 fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION || fn->result_tag == TAG_HASHMAP) &&
                sp > 0) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (fn->result_tag == TAG_STRING) {
                    mark_str_origin(local_kind, nloc, v.origin);
                } else if (fn->result_tag == TAG_ARRAY) {
                    mark_origin(local_kind, nloc, v.origin, NVM2C_VK_RARR);
                    if (v.kind == NVM2C_VK_RARR &&
                        !merge_fields(b, facts, facts->results + (size_t)idx * b->record_width,
                                      v.rec_k)) return 0;
                } else if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION || fn->result_tag == TAG_HASHMAP) {
                    mark_origin(local_kind, nloc, v.origin, fn->result_tag == TAG_HASHMAP ? NVM2C_VK_MAP : NVM2C_VK_REC);
                    if (v.kind == NVM2C_VK_REC &&
                        !merge_record_results(b, facts, facts->results + (size_t)idx * b->record_width,
                                              v.rec_k)) return 0;
                    if (v.kind == NVM2C_VK_MAP &&
                        !merge_fields(b, facts, facts->results + (size_t)idx * b->record_width,
                                      v.rec_k)) return 0;
                }
                NvmShapeKind declared = fn->result_tag == TAG_STRING ? NVM_SHAPE_STRING :
                    fn->result_tag == TAG_HASHMAP ? NVM_SHAPE_MAP :
                    fn->result_tag == TAG_ARRAY ? NVM_SHAPE_ARRAY :
                    (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION) ? NVM_SHAPE_RECORD : NVM_SHAPE_INT;
                if (!shape_type(b, v.shape, declared)) return 0;
                if (declared == NVM_SHAPE_RECORD) {
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
        if (ins.opcode == OP_JMP || ins.opcode == OP_JMP_FALSE) {
            size_t target;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &target) ||
                !sim_join(b, idx, &joins[target], stk, sp)) return 0;
        }
        if (ins.opcode == OP_JMP || ins.opcode == OP_RET ||
            ins.opcode == OP_HALT || ins.opcode == OP_TAIL_CALL) terminated = 1;
        if (b->failed) return 0;
    }

    for (i = 0; i < nloc; i++) {
        if (i < fn->arity &&
            !merge_parameter(b, facts, &facts->parameters[(size_t)idx * NVM2C_MAX_LOCALS + i], local_kind[i])) return 0;
        if (facts->final && local_kind[i] == NVM2C_VK_UNK) local_kind[i] = NVM2C_VK_INT;
    }
    return 1;
}

static int classify_function(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                             uint8_t *local_kind, uint8_t *rec_fields, Nvm2cFacts *facts) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    if (fn->local_count > NVM2C_MAX_LOCALS || fn->arity > fn->local_count) {
        nvm2c_fail(b, "I cannot classify function %u with invalid local or parameter counts", idx);
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
        if (ins.opcode == OP_JMP || ins.opcode == OP_JMP_FALSE) {
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
    const char *rt = c_result_type(fn);
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
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i);
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

static int stack_pop_expect(Nvm2cBuf *b, Nvm2cStack *st, uint8_t kind, const char *what) {
    uint8_t got = NVM2C_VK_INT;
    int slot = stack_pop_kind(b, st, &got);
    if (b->failed) return -1;
    if (got == NVM2C_VK_STR && kind == NVM2C_VK_VALUE) {
        char expression[80];
        snprintf(expression, sizeof expression, "(nmap_value){5, 0, (char *)s[%d]}", slot);
        stack_push_value(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got == NVM2C_VK_VALUE && (kind == NVM2C_VK_INT || kind == NVM2C_VK_STR)) {
        char expression[64];
        snprintf(expression, sizeof expression, "nvalue_require_%s(v[%d])",
                 kind == NVM2C_VK_INT ? "int" : "string", slot);
        if (kind == NVM2C_VK_INT) stack_push_temp(b, st, expression);
        else stack_push_str(b, st, expression);
        return b->failed ? -1 : stack_pop_kind(b, st, NULL);
    }
    if (got != kind) {
        const char *want = "int";
        if (kind == NVM2C_VK_STR) want = "string";
        else if (kind == NVM2C_VK_ARR) want = "array";
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
        char expression[112];
        snprintf(expression, sizeof expression,
                 "(v[%d].kind == 5 || (v[%d].kind == 1 && v[%d].integer != 0))",
                 value, value, value);
        stack_push_temp(b, st, expression);
    }
    return stack_pop_expect(b, st, NVM2C_VK_INT, what);
}

static void emit_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    if ((strcmp(op, "&&") == 0 || strcmp(op, "||") == 0) && st->sp >= 2 &&
        (st->kinds[st->sp - 1] == NVM2C_VK_VALUE || st->kinds[st->sp - 2] == NVM2C_VK_VALUE)) {
        nvm2c_fail(b, "I cannot use an optional integer or string as a boolean"); return;
    }
    int rhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op rhs");
    int lhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op lhs");
    if (b->failed) return;
    char expr[80];
    snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    stack_push_temp(b, st, expr);
}

static void emit_unop(Nvm2cBuf *b, Nvm2cStack *st, const char *prefix) {
    if (strcmp(prefix, "!") == 0 && st->sp && st->kinds[st->sp - 1] == NVM2C_VK_VALUE) {
        nvm2c_fail(b, "I cannot use an optional integer or string as a boolean"); return;
    }
    int x = stack_pop_expect(b, st, NVM2C_VK_INT, "unary op");
    if (b->failed) return;
    char expr[64];
    snprintf(expr, sizeof expr, "(%s t[%d])", prefix, x);
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
}

static const char *stack_array_name(uint8_t kind) {
    switch (kind) {
    case NVM2C_VK_STR: return "s";
    case NVM2C_VK_ARR: return "a";
    case NVM2C_VK_SARR: return "sa";
    case NVM2C_VK_REC: return "r";
    case NVM2C_VK_RARR: return "ra";
    case NVM2C_VK_MAP: return "m";
    case NVM2C_VK_VALUE: return "v";
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

static int record_join(Nvm2cBuf *b, uint32_t idx, Nvm2cStack *joins, uint8_t *set,
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
        args[i] = stack_pop_expect(b, st, fn_local_kind(kinds, idx, (uint16_t)i),
                                  "self TAIL_CALL argument");
        if (b->failed) return 0;
    }
    if (st->sp != 0) {
        nvm2c_fail(b, "self TAIL_CALL leaves extra stack values");
        return 0;
    }
    nvm2c_puts(b, "    {\n");
    for (uint16_t i = 0; i < fn->arity; ++i) {
        uint8_t kind = fn_local_kind(kinds, idx, i);
        nvm2c_printf(b, "        %s tc%u = %s[%d];\n", c_local_type(kind),
                     (unsigned)i, stack_array_name(kind), args[i]);
    }
    for (uint16_t i = 0; i < fn->arity; ++i)
        nvm2c_printf(b, "        l%u = tc%u;\n", (unsigned)i, (unsigned)i);
    for (uint16_t i = fn->arity; i < fn->local_count; ++i) {
        uint8_t kind = fn_local_kind(kinds, idx, i);
        if (kind == NVM2C_VK_STR)
            nvm2c_printf(b, "        l%u = \"\";\n", (unsigned)i);
        else
            nvm2c_printf(b, "        l%u = (%s){0};\n", (unsigned)i, c_local_type(kind));
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
    if (c_result_type(cf) == NULL) {
        nvm2c_fail(b, "function %u: CALL target %u has an unsupported result", idx, callee);
        return 0;
    }
    int args[NVM2C_MAX_LOCALS];
    uint8_t argk[NVM2C_MAX_LOCALS];
    int i;
    for (i = (int)cf->arity - 1; i >= 0; i--) {
        uint8_t pk = fn_local_kind(kinds, callee, (uint16_t)i);
        argk[i] = pk;
        args[i] = stack_pop_expect(b, st, pk, "CALL argument");
        if (b->failed) return 0;
    }
    char cname[64];
    fn_c_name(mod, callee, cname, sizeof cname);
    size_t pos = 0;
    pos += (size_t)snprintf(call + pos, call_sz - pos, "%s(", cname);
    for (uint16_t a = 0; a < cf->arity; a++) {
        if (a) pos += (size_t)snprintf(call + pos, call_sz - pos, ", ");
        if (argk[a] == NVM2C_VK_STR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "s[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_ARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "a[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_SARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "sa[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_REC) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "r[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_RARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "ra[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_MAP) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "m[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_VALUE) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "v[%d]", args[a]);
        } else {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "t[%d]", args[a]);
        }
        if (pos >= call_sz) {
            nvm2c_fail(b, "function %u: CALL argument list overflow", idx);
            return 0;
        }
    }
    snprintf(call + pos, call_sz - pos, ")");
    return 1;
}

static void emit_function_body(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                               const uint8_t *kinds, const uint8_t *rec_fields,
                               const uint8_t *result_fields) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn);
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
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i);
        }
    }
    nvm2c_puts(b, ") {\n");

    for (i = 0; i < fn->local_count; i++) {
        uint8_t lk = fn_local_kind(kinds, idx, i);
        if (i < fn->arity) {
            nvm2c_printf(b, "    %s l%u = a%u;\n", c_local_type(lk), (unsigned)i, (unsigned)i);
        } else if (lk == NVM2C_VK_STR) {
            nvm2c_printf(b, "    const char *l%u = \"\";\n", (unsigned)i);
        } else if (lk == NVM2C_VK_ARR) {
            nvm2c_printf(b, "    narr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_SARR) {
            nvm2c_printf(b, "    nsarr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_REC) {
            nvm2c_printf(b, "    nrec_t l%u = {0};\n", (unsigned)i);
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
    Nvm2cStack *joins = NULL;
    uint8_t *join_set = NULL;
    if (!st.slots || !st.kinds || !st.rec_k || !st.rarr_k || !literal_elems || !aggregate_kinds || !is_start || !is_target) {
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
        if (look.opcode == OP_JMP || look.opcode == OP_JMP_FALSE) {
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
    /* A decoded self-tail instruction may be unreachable. Keep its label
     * syntactically referenced without executing an extra jump. */
    if (has_self_tail) nvm2c_puts(b, "    if (0) goto L_tco;\nL_tco: ;\n");

    while (pc < remaining) {
        size_t start = pc;
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            goto done;
        }
        if (terminated && !join_set[start]) {
            pc += n;
            continue;
        }
        if (is_target[start]) {
            if (terminated) {
                if (!join_set[start]) {
                    nvm2c_fail(b, "function %u: label at %zu has no incoming stack",
                               idx, start);
                    goto done;
                }
                stack_restore_join(b, &st, &joins[start]);
                terminated = 0;
                nvm2c_printf(b, "L_%zu: ;\n", start);
            } else {
                if (!record_join(b, idx, joins, join_set, start, &st)) goto done;
                stack_restore_join(b, &st, &joins[start]);
                nvm2c_printf(b, "L_%zu: ;\n", start);
            }
        }
        pc += n;

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
                } else if (k == NVM2C_VK_ARR) {
                    snprintf(rhs, sizeof rhs, "a[%d]", src);
                    stack_push_arr(b, &st, rhs);
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
                    stack_push_temp(b, &st, rhs);
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
            nvm2c_printf(b, "    if (!t[%d]) abort();\n", cond);
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
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                goto done;
            }
            char rhs[32];
            snprintf(rhs, sizeof rhs, "l%u", (unsigned)slot);
            if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_STR) {
                stack_push_str(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_ARR) {
                stack_push_arr(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_SARR) {
                stack_push_sarr(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_REC) {
                int r = stack_push_rec(b, &st, rhs);
                if (r >= 0) {
                    memcpy(st.rec_k[r], fn_rec_k_const(b, rec_fields, idx, slot),
                           b->record_width);
                }
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_MAP) {
                stack_push_map(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_VALUE) {
                stack_push_value(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_RARR) {
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
                uint8_t expect = fn_local_kind(kinds, idx, slot);
                int t = stack_pop_expect(b, &st, expect, "STORE_LOCAL");
                if (b->failed) goto done;
                if (expect == NVM2C_VK_STR) {
                    nvm2c_printf(b, "    l%u = s[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_ARR) {
                    nvm2c_printf(b, "    l%u = a[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    l%u = sa[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    l%u = r[%d];\n", (unsigned)slot, t);
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
        case OP_ADD:
        case OP_I64_ADD:
            emit_binop(b, &st, "+");
            break;
        case OP_SUB:
        case OP_I64_SUB:
            emit_binop(b, &st, "-");
            break;
        case OP_MUL:
        case OP_I64_MUL:
            emit_binop(b, &st, "*");
            break;
        case OP_DIV:
        case OP_I64_DIV_S: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div lhs");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? (int64_t)0 : t[%d] / t[%d])",
                     rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_MOD:
        case OP_I64_REM_S: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod lhs");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? (int64_t)0 : t[%d] %% t[%d])",
                     rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_NEG:
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
            if (lk == NVM2C_VK_VALUE || rk == NVM2C_VK_VALUE) {
                char left[96], right[96], expression[256];
                uint8_t kinds_pair[2] = {lk, rk};
                int slots_pair[2] = {lhs, rhs};
                char *expressions[2] = {left, right};
                for (int i = 0; i < 2; ++i) {
                    if (kinds_pair[i] == NVM2C_VK_VALUE)
                        snprintf(expressions[i], 96, "v[%d]", slots_pair[i]);
                    else if (kinds_pair[i] == NVM2C_VK_STR)
                        snprintf(expressions[i], 96, "(nmap_value){5, 0, (char *)s[%d]}", slots_pair[i]);
                    else { nvm2c_fail(b, "I require preserved runtime tags to compare this value with a tagged lookup"); goto done; }
                }
                snprintf(expression, sizeof expression, "%snvalue_equal(%s, %s)",
                         ins.opcode == OP_NE ? "!" : "", left, right);
                stack_push_temp(b, &st, expression);
            } else if (lk == NVM2C_VK_INT && rk == NVM2C_VK_INT) {
                char expr[64];
                snprintf(expr, sizeof expr, "t[%d] %s t[%d]",
                         lhs, ins.opcode == OP_EQ ? "==" : "!=", rhs);
                stack_push_temp(b, &st, expr);
            } else if (lk == NVM2C_VK_STR && rk == NVM2C_VK_STR) {
                char expr[192];
                snprintf(expr, sizeof expr,
                         "(int64_t)(strcmp(s[%d] ? s[%d] : \"\", s[%d] ? s[%d] : \"\") %s 0)",
                         lhs, lhs, rhs, rhs, ins.opcode == OP_EQ ? "==" : "!=");
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "function %u: EQ/NE of mixed or non-string values is refused", idx);
                goto done;
            }
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
            if (st.sp && st.kinds[st.sp - 1] == NVM2C_VK_VALUE) {
                int value = stack_pop(b, &st);
                char expression[160];
                snprintf(expression, sizeof expression,
                    "(v[%d].kind == 5 ? v[%d].text : v[%d].kind == 1 ? nstr_from_i64(v[%d].integer) : \"\")",
                    value, value, value, value);
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
        case OP_CAST_INT: {
            uint8_t kind;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            char expression[96];
            if (kind == NVM2C_VK_VALUE)
                snprintf(expression, sizeof expression, "nvalue_cast_int(v[%d])", value);
            else if (kind == NVM2C_VK_STR)
                snprintf(expression, sizeof expression, "(int64_t)strtoll(s[%d] ? s[%d] : \"\", NULL, 10)", value, value);
            else if (kind == NVM2C_VK_INT)
                snprintf(expression, sizeof expression, "t[%d]", value);
            else if (kind == NVM2C_VK_REC || kind == NVM2C_VK_ARR ||
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
            if (tag != TAG_INT && tag != TAG_STRING) {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int, string or struct elements", idx);
                goto done;
            }
            if (tag == TAG_INT) {
                DecodedInstruction nxt;
                uint32_t nn = isa_decode(code + pc, remaining - pc, &nxt);
                if (nn != 0 && nxt.opcode == OP_STORE_LOCAL) {
                    uint16_t slot = nxt.operands[0].u16;
                    if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_SARR) {
                        as_sarr = 1;
                    } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_RARR) {
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
                stack_push_arr(b, &st, "narr_new()");
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
            } else if (tag == TAG_STRING) {
                ekind = NVM2C_VK_STR;
            } else if (tag == TAG_STRUCT) {
                ekind = NVM2C_VK_REC;
            } else {
                nvm2c_fail(b, "function %u: ARR_LITERAL only supports int, string or record elements", idx);
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
                                           : stack_push_arr(b, &st, "0");
            if (b->failed) goto done;
            nvm2c_printf(b, "    %s[%d] = %s(", tag == TAG_STRING ? "sa" : "a",
                         result, tag == TAG_STRING ? "nsarr_lit" : "narr_lit");
            if (count) {
                nvm2c_puts(b, tag == TAG_STRING ? "(const char *[]){" : "(int64_t[]){");
                for (ei = 0; ei < (int)count; ++ei) {
                    nvm2c_printf(b, "%s%s[%d]", ei ? ", " : "",
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
            if (ak == NVM2C_VK_ARR) {
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
            if (ak == NVM2C_VK_ARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "narr_get(a[%d], t[%d])", arr, ix);
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nsarr_get(sa[%d], t[%d])", arr, ix);
                stack_push_str(b, &st, expr);
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
            const char *array = NULL, *value = NULL;
            if (ak == NVM2C_VK_ARR && vk == NVM2C_VK_INT) { array = "a"; value = "t"; }
            else if (ak == NVM2C_VK_SARR && vk == NVM2C_VK_STR) { array = "sa"; value = "s"; }
            else if (ak == NVM2C_VK_RARR && vk == NVM2C_VK_REC) { array = "ra"; value = "r"; }
            else {
                nvm2c_fail(b, "ARR_SET element representation mismatch");
                goto done;
            }
            nvm2c_printf(b,
                "    if (!%s[%d] || t[%d] < 0 || (uint64_t)t[%d] >= %s[%d]->len) abort();\n",
                array, arr, ix, ix, array, arr);
            if (ak == NVM2C_VK_RARR) {
                /* Classifier field kinds do not encode the runtime width. */
                nvm2c_printf(b,
                    "    if (ra[%d]->data[t[%d]].n != r[%d].n || ra[%d]->data[t[%d]].kind != r[%d].kind) abort();\n",
                    arr, ix, val, arr, ix, val);
                nvm2c_printf(b,
                    "    for (size_t f = 0; f < r[%d].n; ++f) if (ra[%d]->data[t[%d]].k[f] != r[%d].k[f]) abort();\n",
                    val, arr, ix, val);
            }
            nvm2c_printf(b, "    %s[%d]->data[t[%d]] = %s[%d];\n", array, arr, ix, value, val);
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
            if (ak == NVM2C_VK_ARR && vk == NVM2C_VK_INT) {
                char expr[80];
                snprintf(expr, sizeof expr, "narr_push(a[%d], t[%d])", arr, val);
                stack_push_arr(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR && vk == NVM2C_VK_STR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nsarr_push(sa[%d], s[%d])", arr, val);
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
            if (kind != AGG_RECORD && kind != AGG_VARIANT) {
                nvm2c_fail(b, "I support record and variant packing, not aggregate kind %u", kind);
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
                    vk != NVM2C_VK_ARR && vk != NVM2C_VK_SARR &&
                    vk != NVM2C_VK_RARR && vk != NVM2C_VK_REC && vk != NVM2C_VK_VALUE) {
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
                    st.rec_k[r][ei] = fkind[ei];
                    nvm2c_printf(b, "    r[%d].k[%d] = %u;\n", r, ei, (unsigned)fkind[ei]);
                    if (fkind[ei] == NVM2C_VK_STR) {
                        nvm2c_printf(b, "    r[%d].s[%d] = s[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_VALUE) {
                        nvm2c_printf(b, "    r[%d].vk[%d] = v[%d].kind;\n"
                                         "    r[%d].f[%d] = v[%d].integer;\n"
                                         "    r[%d].s[%d] = v[%d].text;\n",
                                     r, ei, elems[ei], r, ei, elems[ei], r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_ARR) {
                        nvm2c_printf(b, "    r[%d].a[%d] = a[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_SARR) {
                        nvm2c_printf(b, "    r[%d].sa[%d] = sa[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_RARR) {
                        nvm2c_printf(b, "    r[%d].ra[%d] = ra[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_REC) {
                        nvm2c_printf(b, "    r[%d].rec[%d] = nrec_snapshot(r[%d]);\n", r, ei, elems[ei]);
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
            nvm2c_printf(b, "    if (r[%d].kind != %u) abort();\n", aggregate, AGG_VARIANT);
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
            if (resolved != NVM2C_VK_UNK) st.rec_k[rec][fi] = resolved;
            nvm2c_printf(b, "    if (%u >= r[%d].n) abort();\n", (unsigned)fi, rec);
            if (st.rec_k[rec][fi] == NVM2C_VK_VALUE)
                nvm2c_printf(b, "    if (r[%d].k[%u] != %u && r[%d].k[%u] != %u) abort();\n",
                             rec, (unsigned)fi, NVM2C_VK_VALUE, rec, (unsigned)fi, NVM2C_VK_STR);
            else nvm2c_printf(b, "    if (r[%d].k[%u] != %u) abort();\n", rec, (unsigned)fi,
                              (unsigned)st.rec_k[rec][fi]);
            {
                char expr[256];
                if (st.rec_k[rec][fi] == NVM2C_VK_STR) {
                    snprintf(expr, sizeof expr, "r[%d].s[%u]", rec, (unsigned)fi);
                    stack_push_str(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_VALUE) {
                    snprintf(expr, sizeof expr,
                             "(nmap_value){r[%d].k[%u] == %u ? 5 : r[%d].vk[%u], r[%d].f[%u], (char *)r[%d].s[%u]}",
                             rec, (unsigned)fi, NVM2C_VK_STR, rec, (unsigned)fi,
                             rec, (unsigned)fi, rec, (unsigned)fi);
                    stack_push_value(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    if (!r[%d].rec[%u]) abort();\n", rec, (unsigned)fi);
                    snprintf(expr, sizeof expr, "*r[%d].rec[%u]", rec, (unsigned)fi);
                    int nested = stack_push_rec(b, &st, expr);
                    if (nested >= 0) {
                        NvmShapeId shape = b->shape_outputs[idx][start];
                        for (size_t f = 0; f < b->record_width; ++f)
                            st.rec_k[nested][f] = resolved_shape_kind(b,
                                nvm_shape_lookup(&b->shapes, shape, (uint32_t)f));
                        if (!shape_ok(b)) goto done;
                    }
                } else if (st.rec_k[rec][fi] == NVM2C_VK_ARR) {
                    snprintf(expr, sizeof expr, "r[%d].a[%u]", rec, (unsigned)fi);
                    stack_push_arr(b, &st, expr);
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
                    stack_push_temp(b, &st, expr);
                }
            }
            break;
        }
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            char call[768];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, call, sizeof call)) {
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (result_is_i64(cf)) {
                stack_push_temp(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                stack_push_str(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_HASHMAP) {
                stack_push_map(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                int result = stack_push_rarr(b, &st, call);
                if (result >= 0) memcpy(st.rarr_k[result], result_fields + (size_t)callee * b->record_width,
                                       b->record_width);
            } else if (cf->result_count == 1 &&
                       (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION)) {
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
            char call[768];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, call, sizeof call)) {
                goto done;
            }
            if (st.sp != 0) {
                nvm2c_fail(b, "function %u: TAIL_CALL leaves extra stack values", idx);
                goto done;
            }
            if (fn->result_count == 1 &&
                (result_is_i64(fn) || fn->result_tag == TAG_STRING ||
                 fn->result_tag == TAG_ARRAY || fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION || fn->result_tag == TAG_HASHMAP)) {
                nvm2c_printf(b, "    return %s;\n", call);
            } else {
                nvm2c_printf(b, "    %s;\n    return;\n", call);
            }
            terminated = 1;
            break;
        }
        case OP_JMP: {
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            nvm2c_printf(b, "    goto L_%zu;\n", tgt);
            terminated = 1;
            break;
        }
        case OP_JMP_FALSE: {
            int cond = stack_pop_condition(b, &st, "JMP_FALSE");
            if (b->failed) goto done;
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            nvm2c_printf(b, "    if (!t[%d]) {\n", cond);
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            nvm2c_printf(b, "    goto L_%zu;\n    }\n", tgt);
            break;
        }
        case OP_RET:
            if (result_is_i64(fn)) {
                int t = stack_pop_expect(b, &st, NVM2C_VK_INT, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return t[%d];\n", t);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_STRING) {
                int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return s[%d];\n", s);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_HASHMAP) {
                int map = stack_pop_expect(b, &st, NVM2C_VK_MAP, "RET");
                if (b->failed) goto done;
                if (st.sp) { nvm2c_fail(b, "I cannot return a map with extra stack values"); goto done; }
                nvm2c_printf(b, "    return m[%d];\n", map);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_ARRAY) {
                int a = stack_pop_expect(b, &st, NVM2C_VK_RARR, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return ra[%d];\n", a);
            } else if (fn->result_count == 1 &&
                       (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION)) {
                int record = stack_pop_expect(b, &st, NVM2C_VK_REC, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "I cannot return an aggregate with extra stack values");
                    goto done;
                }
                nvm2c_printf(b, "    if (r[%d].kind != %u) abort();\n    return r[%d];\n",
                             record, fn->result_tag == TAG_STRUCT ? AGG_RECORD : AGG_VARIANT, record);
            } else {
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: void RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_puts(b, "    return;\n");
            }
            terminated = 1;
            break;
        case OP_HALT:
            if (result_is_i64(fn) && st.sp == 1) {
                nvm2c_printf(b, "    return t[%d];\n",
                             stack_pop_expect(b, &st, NVM2C_VK_INT, "HALT"));
            } else if (st.sp == 0 && (fn->result_count == 0 || fn->result_tag == TAG_VOID)) {
                nvm2c_puts(b, "    return;\n");
            } else if (st.sp == 0 && result_is_i64(fn)) {
                nvm2c_puts(b, "    return 0;\n");
            } else {
                nvm2c_fail(b, "function %u: HALT with unexpected stack height %d", idx, st.sp);
                goto done;
            }
            terminated = 1;
            break;
        case OP_CALL_EXTERN: {
            const Nvm2cHost *host = import_host(mod, ins.operands[0].u32);
            if (!host) {
                nvm2c_fail(b, "CALL_EXTERN has no exact builtin host ABI");
                goto done;
            }
            char expression[128];
            if (strcmp(host->c_name, "nhost_artifact") == 0) {
                int args[2] = {0};
                for (uint8_t p = host->argc; p > 0; --p)
                    args[p - 1] = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                if (b->failed) goto done;
                if (host->argc == 2)
                    snprintf(expression, sizeof expression, "nhost_artifact_%u(s[%d], s[%d])",
                             ins.operands[0].u32, args[0], args[1]);
                else snprintf(expression, sizeof expression, "nhost_artifact_%u(s[%d])",
                              ins.operands[0].u32, args[0]);
            } else if (host->argc == 2) {
                int right = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                int left = stack_pop_expect(b, &st, NVM2C_VK_STR, "CALL_EXTERN");
                if (b->failed) goto done;
                snprintf(expression, sizeof expression, "%s(s[%d], s[%d])", host->c_name, left, right);
            } else if (host->argc) {
                uint8_t kind = host->parameter == TAG_STRING ? NVM2C_VK_STR : NVM2C_VK_INT;
                int arg = stack_pop_expect(b, &st, kind, "CALL_EXTERN");
                if (b->failed) goto done;
                snprintf(expression, sizeof expression, "%s(%c[%d])", host->c_name,
                         kind == NVM2C_VK_STR ? 's' : 't', arg);
                if (host->result == TAG_ARRAY)
                    snprintf(expression, sizeof expression, "nhost_walk_%u(s[%d])",
                             ins.operands[0].u32, arg);
            } else {
                snprintf(expression, sizeof expression, "%s()", host->c_name);
            }
            if (host->result == TAG_ARRAY) stack_push_sarr(b, &st, expression);
            else if (host->result == TAG_STRING) stack_push_str(b, &st, expression);
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
    }

    if (is_target[remaining] && join_set[remaining]) {
        nvm2c_printf(b, "L_%zu: ;\n", remaining);
    }
    if (!terminated) {
        nvm2c_fail(b, "function %u: falls off the end without RET or HALT", idx);
        goto done;
    }
    nvm2c_puts(b, "}\n\n");

    /* I emit the body once, then insert declarations using its actual
     * high-water counts. Unused kinds get one slot to remain valid C11. */
    {
        char declarations[1024];
        int count = snprintf(declarations, sizeof declarations,
            "    int64_t t[%d] = {0}; (void)t;\n"
            "    const char *s[%d] = {0}; (void)s;\n"
            "    narr_t a[%d] = {0}; (void)a;\n"
            "    nsarr_t sa[%d] = {0}; (void)sa;\n"
            "    nrec_t r[%d] = {0}; (void)r;\n"
            "    nrarr_t ra[%d] = {0}; (void)ra;\n"
            "    nmap_t m[%d] = {0}; (void)m;\n",
            st.next_temp ? st.next_temp : 1, st.next_str ? st.next_str : 1,
            st.next_arr ? st.next_arr : 1, st.next_sarr ? st.next_sarr : 1,
            st.next_rec ? st.next_rec : 1, st.next_rarr ? st.next_rarr : 1,
            st.next_map ? st.next_map : 1);
        if (count < 0 || (size_t)count >= sizeof declarations) {
            nvm2c_fail(b, "I cannot format temporary declarations");
            goto done;
        }
        if (b->has_maps) {
            int extra = snprintf(declarations + count, sizeof declarations - (size_t)count,
                                 "    nmap_value v[%d] = {0}; (void)v;\n",
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

static int module_has_array_constructor(const NvmModule *mod, const uint8_t *kinds,
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
                           ins.operands[0].u8 == TAG_STRUCT ? NVM2C_VK_RARR : NVM2C_VK_ARR;
            if (ins.operands[0].u8 == TAG_INT) {
                DecodedInstruction next;
                if (isa_decode(code + pc, fn->code_length - pc, &next) &&
                    next.opcode == OP_STORE_LOCAL && next.operands[0].u16 < fn->local_count) {
                    uint8_t local = fn_local_kind(kinds, f, next.operands[0].u16);
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

static int module_has_local_kind(const uint8_t *kinds, uint32_t fn_count, uint8_t kind) {
    uint32_t i;
    uint16_t li;
    for (i = 0; i < fn_count; i++) {
        for (li = 0; li < NVM2C_MAX_LOCALS; li++) {
            if (fn_local_kind(kinds, i, li) == kind) return 1;
        }
    }
    return 0;
}

static void emit_nstr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static char nstr_arena[65536];\n"
        "static size_t nstr_used;\n");
}

static void emit_nstr_concat(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_concat(const char *a, const char *b) {\n"
        "    size_t na = strlen(a ? a : \"\");\n"
        "    size_t nb = strlen(b ? b : \"\");\n"
        "    if (nstr_used + na + nb + 1 > sizeof nstr_arena) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, a ? a : \"\", na);\n"
        "    memcpy(p + na, b ? b : \"\", nb);\n"
        "    p[na + nb] = 0;\n"
        "    nstr_used += na + nb + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_substr(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_substr(const char *s, int64_t start, int64_t len) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    int64_t slen = (int64_t)strlen(src);\n"
        "    if (start < 0) start = 0;\n"
        "    if (start >= slen || len <= 0) return \"\";\n"
        "    if (len > slen - start) len = slen - start;\n"
        "    if (nstr_used + (size_t)len + 1 > sizeof nstr_arena) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, src + start, (size_t)len);\n"
        "    p[len] = 0;\n"
        "    nstr_used += (size_t)len + 1;\n"
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
        "    if (n < 0 || (size_t)n + 1 > sizeof nstr_arena - nstr_used) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, tmp, (size_t)n + 1);\n"
        "    nstr_used += (size_t)n + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_narr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t narr_arena[65536];\n"
        "static size_t narr_used;\n");
}

static void emit_narr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_new(void) {\n"
        "    narr_t a = (narr_t)calloc(1, sizeof(narr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_lit(const int64_t *elems, size_t n) {\n"
        "    if (n > 0 && !elems) abort();\n"
        "    if (narr_used + n > (sizeof narr_arena / sizeof narr_arena[0])) abort();\n"
        "    int64_t *p = narr_arena + narr_used;\n"
        "    if (n) memcpy(p, elems, n * sizeof(int64_t));\n"
        "    narr_used += n;\n"
        "    narr_t a = narr_new();\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t narr_get(narr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx];\n"
        "}\n\n");
}

static void emit_narr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_push(narr_t a, int64_t v) {\n"
        "    if (!a) abort();\n"
        "    size_t n = a->len + 1;\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (narr_used + n > (sizeof narr_arena / sizeof narr_arena[0])) abort();\n"
        "    int64_t *p = narr_arena + narr_used;\n"
        "    if (a->len) memcpy(p, a->data, a->len * sizeof(int64_t));\n"
        "    p[a->len] = v;\n"
        "    narr_used += n;\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nsarr_arena[65536];\n"
        "static size_t nsarr_used;\n");
}

static void emit_nsarr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_new(void) {\n"
        "    nsarr_t a = (nsarr_t)calloc(1, sizeof(nsarr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_lit(const char *const *elems, size_t n) {\n"
        "    if (n > 0 && !elems) abort();\n"
        "    if (nsarr_used + n > (sizeof nsarr_arena / sizeof nsarr_arena[0])) abort();\n"
        "    const char **p = nsarr_arena + nsarr_used;\n"
        "    if (n) memcpy(p, elems, n * sizeof(const char *));\n"
        "    nsarr_used += n;\n"
        "    nsarr_t a = nsarr_new();\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nsarr_get(nsarr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx] ? a->data[idx] : \"\";\n"
        "}\n\n");
}

static void emit_nsarr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_push(nsarr_t a, const char *v) {\n"
        "    if (!a) abort();\n"
        "    size_t n = a->len + 1;\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (nsarr_used + n > (sizeof nsarr_arena / sizeof nsarr_arena[0])) abort();\n"
        "    const char **p = nsarr_arena + nsarr_used;\n"
        "    if (a->len) memcpy(p, a->data, a->len * sizeof(const char *));\n"
        "    p[a->len] = v ? v : \"\";\n"
        "    nsarr_used += n;\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_host_normalize(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static inline const char *nhost_normalize(const char *path) {\n"
        "    if (!path) path = \"\";\n"
        "    size_t length = strlen(path), slots = length / 2 + 1;\n"
        "    if (length > SIZE_MAX - 2 || slots > SIZE_MAX / sizeof(size_t)) abort();\n"
        "    char *out = malloc(length + 2);\n"
        "    size_t *bases = malloc(slots * sizeof *bases);\n"
        "    if (!out || !bases) abort();\n"
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
        "        if (count >= slots) abort();\n"
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
        "    if (!text) abort();\n"
        "    int invalid = 0;\n"
        "    if (file) {\n"
        "        char chunk[4096];\n"
        "        size_t count;\n"
        "        while ((count = fread(chunk, 1, sizeof chunk, file)) != 0) {\n"
        "            if (memchr(chunk, 0, count)) invalid = 1;\n"
        "            if (invalid) continue;\n"
        "            if (used > SIZE_MAX - count - 1) abort();\n"
        "            size_t needed = used + count + 1;\n"
        "            if (needed > capacity) {\n"
        "                capacity = needed > SIZE_MAX / 2 ? needed : needed * 2;\n"
        "                char *grown = realloc(text, capacity);\n"
        "                if (!grown) abort();\n"
        "                text = grown;\n"
        "            }\n"
        "            memcpy(text + used, chunk, count); used += count;\n"
        "        }\n"
        "        if (ferror(file)) invalid = 1;\n"
        "        if (fclose(file) != 0) invalid = 1;\n"
        "    }\n"
        "    text[invalid ? 0 : used] = 0;\n"
        "    return text;\n}\n");
}

static void emit_scalar_artifact_adapters(Nvm2cBuf *b, const NvmModule *mod) {
    for (uint32_t i = 0; i < mod->import_count; ++i) {
        const Nvm2cHost *host = import_host(mod, i);
        if (!host || strcmp(host->c_name, "nhost_artifact")) continue;
        const char *result = host->result == TAG_STRING ? "const char *" :
                             host->result == TAG_BOOL ? "bool" : "int64_t";
        const char *parameters = host->argc == 2 ? "const char *a, const char *z" : "const char *a";
        const char *types = host->argc == 2 ? "const char *, const char *" : "const char *";
        nvm2c_puts(b, "#include <dlfcn.h>\n#include <stdbool.h>\n");
        nvm2c_printf(b, "static inline %s nhost_artifact_%u(%s) {\n", result, i, parameters);
        nvm2c_printf(b, "    static void *library;\n    static %s (*function)(%s);\n", result, types);
        nvm2c_puts(b, "    if (!library) {\n        library = dlopen(");
        const NvmImportEntry *imp = &mod->imports[i];
        emit_c_string_lit(b, mod->strings[imp->module_name_idx], mod->string_lengths[imp->module_name_idx]);
        nvm2c_puts(b, ", RTLD_NOW | RTLD_LOCAL);\n        if (!library) abort();\n");
        nvm2c_printf(b, "        function = (%s (*)(%s))dlsym(library, \"%s\");\n", result, types, host->name);
        nvm2c_puts(b, "        if (!function) abort();\n    }\n");
        nvm2c_printf(b, "    %s value = function(%s);\n", result, host->argc == 2 ? "a, z" : "a");
        if (host->result == TAG_STRING) nvm2c_puts(b, "    if (!value) abort();\n");
        nvm2c_puts(b, "    return value;\n}\n");
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
        nvm2c_printf(b, "static inline nsarr_t nhost_walk_%u(const char *root) {\n", i);
        nvm2c_puts(b,
            "    static void *library;\n"
            "    static nh_array_value *(*walk)(const char *);\n"
            "    static bool (*release)(nh_array_value *);\n"
            "    if (!library) {\n"
            "        library = dlopen(");
        const NvmImportEntry *imp = &mod->imports[i];
        emit_c_string_lit(b, mod->strings[imp->module_name_idx],
                          mod->string_lengths[imp->module_name_idx]);
        nvm2c_puts(b,
            ", RTLD_NOW | RTLD_LOCAL);\n"
            "        if (!library) abort();\n"
            "        const uint32_t *abi = (const uint32_t *)dlsym(library, \"fs_walkdir__nano_array_abi\");\n"
            "        if (!abi || *abi != 1) abort();\n"
            "        walk = (nh_array_value *(*)(const char *))dlsym(library, \"fs_walkdir\");\n"
            "        release = (bool (*)(nh_array_value *))dlsym(library, \"fs_walkdir_release\");\n"
            "        if (!walk || !release) abort();\n"
            "    }\n"
            "    nh_array_value *foreign = walk(root);\n"
            "    if (!foreign || !foreign->data || foreign->type != nh_string ||\n"
            "        foreign->width != sizeof(char *) || foreign->length < 0 ||\n"
            "        foreign->capacity < foreign->length ||\n"
            "        (uint64_t)foreign->length > SIZE_MAX / sizeof(char *)) abort();\n"
            "    nsarr_t result = calloc(1, sizeof *result);\n"
            "    if (!result) abort();\n"
            "    result->len = (size_t)foreign->length;\n"
            "    result->data = calloc(result->len ? result->len : 1, sizeof *result->data);\n"
            "    if (!result->data) abort();\n"
            "    for (size_t j = 0; j < result->len; ++j) {\n"
            "        const char *value = ((const char **)foreign->data)[j];\n"
            "        if (!value) abort();\n"
            "        size_t length = strlen(value);\n"
            "        if (length == SIZE_MAX) abort();\n"
            "        char *copy = malloc(length + 1);\n"
            "        if (!copy) abort();\n"
            "        memcpy(copy, value, length + 1); result->data[j] = copy;\n"
            "    }\n"
            "    if (!release(foreign)) abort();\n"
            "    return result;\n}\n");
    }
}

static void emit_nrarr_helpers(Nvm2cBuf *b, int need_new, int need_push, int need_get) {
    if (need_new) nvm2c_puts(b,
        "static nrarr_t nrarr_new(void) {\n"
        "    nrarr_t a = (nrarr_t)calloc(1, sizeof(nrarr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
    if (need_push) nvm2c_puts(b,
        "static nrarr_t nrarr_push(nrarr_t a, nrec_t v) {\n"
        "    if (!a || a->len >= NVM2C_RECORD_ARRAY_CAP) abort();\n"
        "    a->data[a->len++] = v;\n"
        "    return a;\n"
        "}\n\n");
    if (need_get) nvm2c_puts(b,
        "static nrec_t nrarr_get(nrarr_t a, int64_t idx) {\n"
        "    if (!a || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx];\n"
        "}\n\n");
}

char *nvm2c_emit(const NvmModule *mod, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!mod) {
        if (err && err_len) snprintf(err, err_len, "module is null");
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
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        const NvmFunctionEntry *fn = &mod->functions[f];
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
            pc += n;
        }
    }

    if (b.has_maps && b.record_width < 2) b.record_width = 2;
    size_t per_function = NVM2C_MAX_LOCALS * (1 + b.record_width) + b.record_width;
    if (mod->function_count > SIZE_MAX / per_function) {
        if (err && err_len) snprintf(err, err_len, "I cannot allocate this many function facts");
        return NULL;
    }
    size_t fact_size = (size_t)mod->function_count * per_function;
    size_t shape_code_count = mod->code_size;
    size_t shape_local_count = (size_t)mod->function_count * NVM2C_MAX_LOCALS;
    if (shape_code_count > SIZE_MAX / sizeof(NvmShapeId) || shape_local_count > SIZE_MAX / sizeof(NvmShapeId)) {
        nvm2c_fail(&b, "I cannot allocate this many instruction shapes");
        return NULL;
    }
    b.shape_locals = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS, sizeof(NvmShapeId));
    b.shape_results = calloc(mod->function_count, sizeof(NvmShapeId));
    b.shape_outputs = calloc(mod->function_count, sizeof *b.shape_outputs);
    uint8_t *inference = malloc(fact_size);
    uint8_t *kinds = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS, 1);
    uint8_t *rec_fields = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS
                                 * b.record_width, 1);
    if (!kinds || !rec_fields || !inference || !b.shape_locals || !b.shape_results || !b.shape_outputs) {
        free(inference);
        free(kinds);
        free(rec_fields);
        free(b.shape_locals);
        free(b.shape_results);
        free(b.shape_outputs);
        if (err && err_len) snprintf(err, err_len, "out of memory");
        return NULL;
    }

    memset(inference, NVM2C_VK_UNK, fact_size);
    Nvm2cFacts facts = {0};
    facts.parameters = inference;
    facts.fields = inference + (size_t)mod->function_count * NVM2C_MAX_LOCALS;
    facts.results = facts.fields + (size_t)mod->function_count * NVM2C_MAX_LOCALS * b.record_width;
    /* I add known facts and widen string parameters to optional storage when
     * needed. Payload and aggregate compatibility remain graph constraints. */
    for (size_t pass = 0; ; pass++) {
        if (pass / 2 > fact_size) {
            nvm2c_fail(&b, "I could not converge function type facts");
            goto fail;
        }
        facts.changed = 0;
        for (uint32_t i = 0; i < mod->function_count; i++) {
            if (!classify_function(&b, mod, i, kinds + (size_t)i * NVM2C_MAX_LOCALS,
                                   rec_fields + (size_t)i * NVM2C_MAX_LOCALS * b.record_width,
                                   &facts)) {
                goto fail;
            }
        }
        if (facts.final) break;
        if (!facts.changed) facts.final = 1;
    }

    /* Nested projections may acquire their representation from a later
     * function's constraints. Resolve local storage after all final passes. */
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        for (uint16_t l = 0; l < mod->functions[f].local_count; ++l) {
            size_t at = (size_t)f * NVM2C_MAX_LOCALS + l;
            uint8_t resolved = resolved_shape_kind(&b, b.shape_locals[at]);
            if (resolved != NVM2C_VK_UNK) kinds[at] = resolved;
        }
    }
    if (!shape_ok(&b)) goto fail;

    {
        int need_concat = module_has_opcode(mod, OP_STR_CONCAT);
        int need_cast = module_has_opcode(mod, OP_CAST_STRING);
        int need_contains = module_has_opcode(mod, OP_STR_CONTAINS);
        int need_starts = module_has_opcode(mod, OP_STR_STARTS_WITH);
        int need_ends = module_has_opcode(mod, OP_STR_ENDS_WITH);
        int need_substr = module_has_opcode(mod, OP_STR_SUBSTR);
        int need_char_at = module_has_opcode(mod, OP_STR_CHAR_AT);
        int need_string = need_concat || need_cast || need_contains || need_substr ||
            need_starts || need_ends ||
            need_char_at ||
            module_has_opcode(mod, OP_PUSH_STR) ||
            module_has_opcode(mod, OP_STR_LEN);
        int need_arr_lit = module_has_opcode(mod, OP_ARR_LITERAL);
        int need_arr_get = module_has_opcode(mod, OP_ARR_GET);
        int need_arr_push = module_has_opcode(mod, OP_ARR_PUSH);
        int need_iarr_new = module_has_array_constructor(mod, kinds, NVM2C_VK_ARR);
        int need_sarr_new = module_has_array_constructor(mod, kinds, NVM2C_VK_SARR);
        int need_iarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_INT);
        int need_sarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRING);
        int need_iarr = need_iarr_new || need_iarr_lit ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_ARR);
        int need_sarr = need_sarr_new || need_sarr_lit || module_uses_host(mod, "nhost_walk") ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_SARR);
        int need_rarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRUCT);
        int need_rarr = need_rarr_lit || module_has_array_constructor(mod, kinds, NVM2C_VK_RARR) ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_RARR);
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
                if (fn_local_kind(kinds, i, li) == NVM2C_VK_STR) need_string = 1;
            }
        }

        if (module_uses_host(mod, "nhost_destinations") || module_uses_host(mod, "nhost_capture") ||
            module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
            "#ifndef _POSIX_C_SOURCE\n#define _POSIX_C_SOURCE 200809L\n#endif\n");
        if (module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
            "#ifdef __APPLE__\n#ifndef _DARWIN_C_SOURCE\n#define _DARWIN_C_SOURCE\n#endif\n#endif\n");
        nvm2c_puts(&b,
            "/* Generated by nvm2c from NanoISA. Not a VM wrapper. */\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n");
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
            if (module_uses_host(mod, "nhost_from_char")) nvm2c_puts(&b,
                "static inline const char *nhost_from_char(int64_t code) {\n"
                "    char *text = malloc(2);\n"
                "    if (!text) abort();\n"
                "    text[0] = (char)code; text[1] = 0;\n"
                "    return text;\n}\n");
            if (module_uses_host(mod, "nhost_mktemp_dir")) nvm2c_puts(&b,
                "static inline const char *nhost_mktemp_dir(const char *prefix) {\n"
                "    const char *root = getenv(\"TMPDIR\");\n"
                "    if (!root || !*root) root = \"/tmp\";\n"
                "    if (!prefix) prefix = \"nano_\";\n"
                "    size_t a = strlen(root), z = strlen(prefix);\n"
                "    if (z > SIZE_MAX - 8 || a > SIZE_MAX - z - 8) abort();\n"
                "    char *path = malloc(a + z + 8);\n"
                "    if (!path) abort();\n"
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
                "    if (!output) abort();\n"
                "    output[0] = 0;\n"
                "    FILE *pipe = popen(command, \"r\");\n"
                "    if (!pipe) return output;\n"
                "    size_t used = fread(output, 1, 65535, pipe);\n"
                "    output[used] = 0;\n"
                "    char discard[4096];\n"
                "    while (fread(discard, 1, sizeof discard, pipe)) {}\n"
                "    pclose(pipe);\n"
                "    return output;\n}\n");
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
                "    if (!value) value = \"\";\n"
                "    size_t n = strlen(value);\n"
                "    if (n == SIZE_MAX) abort();\n"
                "    char *copy = malloc(n + 1);\n"
                "    if (!copy) abort();\n"
                "    memcpy(copy, value, n + 1); return copy;\n}\n");
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
        if (need_concat || need_cast || need_substr || need_arr_lit || need_arr_get ||
            need_arr_push || need_iarr_new || need_sarr_new || need_agg_get ||
            need_assert || need_rarr || b.has_maps || module_has_opcode(mod, OP_AGG_PACK) ||
            module_has_opcode(mod, OP_CAST_INT)) {
            nvm2c_puts(&b, "#include <stdlib.h>\n#include <string.h>\n");
        } else if (need_string) {
            nvm2c_puts(&b, "#include <string.h>\n");
        }
        nvm2c_puts(&b,
            "\ntypedef struct nmap_s *nmap_t;\n"
            "typedef struct { int64_t *data; size_t len; } narr_s;\n"
            "typedef narr_s *narr_t;\n"
            "typedef struct { const char **data; size_t len; } nsarr_s;\n"
            "typedef nsarr_s *nsarr_t;\n");
        if (b.has_maps) {
            nvm2c_puts(&b,
#include "nvm2c_map_runtime.inc"
            );
            nvm2c_puts(&b,
                "typedef struct nmap_owned { nmap_t map; struct nmap_owned *next; } nmap_owned;\n"
                "static nmap_owned *nmap_owned_head;\n"
                "typedef struct nvalue_owned { nmap_value value; struct nvalue_owned *next; } nvalue_owned;\n"
                "static nvalue_owned *nvalue_owned_head;\n"
                "static nmap_value nmap_owned_get(nmap_t map, const char *key) {\n"
                "    nmap_value value = nmap_get(map, key);\n"
                "    if (value.kind == 5) { nvalue_owned *owner = malloc(sizeof *owner);\n"
                "        if (!owner) { nmap_release_value(value); abort(); }\n"
                "        *owner = (nvalue_owned){value, nvalue_owned_head}; nvalue_owned_head = owner; }\n"
                "    return value;\n}\n"
                "static inline int64_t nvalue_require_int(nmap_value value) {\n"
                "    if (value.kind != 1) abort();\n    return value.integer;\n}\n"
                "static inline const char *nvalue_require_string(nmap_value value) {\n"
                "    if (value.kind != 5) abort();\n    return value.text;\n}\n"
                "static inline int64_t nvalue_cast_int(nmap_value value) {\n"
                "    return value.kind == 1 ? value.integer : value.kind == 5 ? (int64_t)strtoll(value.text, NULL, 10) : 0;\n}\n"
                "static inline int nvalue_equal(nmap_value a, nmap_value b) {\n"
                "    if (a.kind != b.kind) return 0;\n"
                "    return a.kind == 0 || (a.kind == 1 ? a.integer == b.integer : strcmp(a.text, b.text) == 0);\n}\n"
                "static nmap_t nmap_owned_new(uint8_t kind) {\n"
                "    nmap_t map = nmap_new(kind); nmap_owned *owner = malloc(sizeof *owner);\n"
                "    if (!owner) { nmap_destroy(map); abort(); }\n"
                "    *owner = (nmap_owned){map, nmap_owned_head}; nmap_owned_head = owner; return map;\n}\n"
                "static void nmap_release_owned(void) {\n"
                "    while (nvalue_owned_head) { nvalue_owned *owner = nvalue_owned_head;\n"
                "        nvalue_owned_head = owner->next; nmap_release_value(owner->value); free(owner); }\n"
                "    while (nmap_owned_head) { nmap_owned *owner = nmap_owned_head;\n"
                "        nmap_owned_head = owner->next; nmap_destroy(owner->map); free(owner); }\n}\n");
        }
        emit_walk_adapters(&b, mod);
        emit_scalar_artifact_adapters(&b, mod);
        nvm2c_puts(&b, "typedef struct nrarr_s nrarr_s;\ntypedef nrarr_s *nrarr_t;\n");
        nvm2c_puts(&b, "typedef struct nrec_s nrec_t;\n");
        nvm2c_printf(&b,
            "struct nrec_s { int64_t f[%zu]; const char *s[%zu]; narr_t a[%zu]; nsarr_t sa[%zu]; nrarr_t ra[%zu]; const nrec_t *rec[%zu]; uint8_t k[%zu], vk[%zu]; uint16_t n, tag; uint8_t kind; };\n",
            b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width, b.record_width);
        nvm2c_puts(&b,
            "enum { NVM2C_RECORD_ARRAY_CAP = 256 };\n"
            "struct nrarr_s { nrec_t data[NVM2C_RECORD_ARRAY_CAP]; size_t len; };\n\n");
        if (module_has_opcode(mod, OP_AGG_PACK)) nvm2c_puts(&b,
            "typedef struct nrec_owned { nrec_t value; struct nrec_owned *next; } nrec_owned;\n"
            "static nrec_owned *nrec_owned_head;\n"
            "static inline const nrec_t *nrec_snapshot(nrec_t value) {\n"
            "    nrec_owned *node = malloc(sizeof *node);\n"
            "    if (!node) abort();\n"
            "    node->value = value; node->next = nrec_owned_head; nrec_owned_head = node;\n"
            "    return &node->value;\n}\n"
            "static void nrec_release_snapshots(void) {\n"
            "    while (nrec_owned_head) { nrec_owned *node = nrec_owned_head;\n"
            "        nrec_owned_head = node->next; free(node); }\n}\n");
        if (need_concat || need_cast || need_substr) emit_nstr_arena(&b);
        if (need_concat) emit_nstr_concat(&b);
        if (need_substr) emit_nstr_substr(&b);
        if (need_char_at) emit_nstr_char_at(&b);
        if (need_starts) emit_nstr_starts_with(&b);
        if (need_ends) emit_nstr_ends_with(&b);
        if (need_cast) emit_nstr_from_i64(&b);
        if (need_iarr_lit || (need_iarr && need_iarr_new)) {
            emit_narr_new(&b);
        }
        if (need_iarr_lit || need_iarr_push) {
            emit_narr_arena(&b);
        }
        if (need_iarr_lit) emit_narr_lit(&b);
        if (need_iarr_get) emit_narr_get(&b);
        if (need_iarr_push) emit_narr_push(&b);
        if (need_sarr_lit || need_sarr_new) {
            emit_nsarr_new(&b);
        }
        if (need_sarr_lit || need_sarr_push) {
            emit_nsarr_arena(&b);
        }
        if (need_sarr_lit) emit_nsarr_lit(&b);
        if (need_sarr_get) emit_nsarr_get(&b);
        if (need_sarr_push) emit_nsarr_push(&b);
        if (need_rarr) emit_nrarr_helpers(&b, need_rarr_lit || module_has_opcode(mod, OP_ARR_NEW),
                                         need_rarr_lit || need_arr_push, need_arr_get);
    }

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            emit_prototype(&b, mod, i, kinds);
            if (b.failed) goto fail;
        }
    }
    nvm2c_puts(&b, "\n");

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
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
        if (b.has_maps) nvm2c_puts(&b,
            "    (void)nmap_owned_new; (void)nmap_set; (void)nmap_get; (void)nmap_owned_get;\n"
            "    (void)nvalue_require_int; (void)nvalue_require_string; (void)nvalue_cast_int; (void)nvalue_equal;\n"
            "    (void)nmap_has; (void)nmap_len; (void)nmap_delete;\n");
        if (module_has_opcode(mod, OP_AGG_PACK)) nvm2c_puts(&b, "    (void)nrec_snapshot;\n");
        if (module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRUCT)) nvm2c_puts(&b, "    (void)nrarr_push;\n");
        for (uint32_t i = 0; i < mod->import_count; ++i) {
            const Nvm2cHost *host = import_host(mod, i);
            if (host && host->result == TAG_ARRAY)
                nvm2c_printf(&b, "    (void)nhost_walk_%u;\n", i);
            if (host && strcmp(host->c_name, "nhost_artifact") == 0)
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
        nvm2c_puts(&b, "    return result;\n}\n");
    }

    if (b.failed) goto fail;
    if (strstr(b.data, "nano_vm") != NULL || strstr(b.data, "nvm_blob") != NULL) {
        nvm2c_fail(&b, "internal error: emitted a VM wrapper rather than structured C");
        goto fail;
    }
    free(kinds);
    free(rec_fields);
    free(inference);
    nvm_shape_destroy(&b.shapes);
    free(b.shape_locals);
    free(b.shape_results);
    for (uint32_t i = 0; i < mod->function_count; ++i) free(b.shape_outputs[i]);
    free(b.shape_outputs);
    return b.data;

fail:
    free(kinds);
    free(rec_fields);
    free(inference);
    nvm_shape_destroy(&b.shapes);
    free(b.shape_locals);
    free(b.shape_results);
    for (uint32_t i = 0; i < mod->function_count; ++i) free(b.shape_outputs[i]);
    free(b.shape_outputs);
    free(b.data);
    return NULL;
}

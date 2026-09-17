/*
 * NVM Binary Format - serialization and deserialization
 */

#include "nvm_format.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========================================================================
 * CRC32 (standard polynomial 0xEDB88320)
 * ======================================================================== */

static uint32_t crc32_table[256];
static bool crc32_initialized = false;

static void crc32_init(void) {
    if (crc32_initialized) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc32_initialized = true;
}

uint32_t nvm_crc32(const uint8_t *data, uint32_t size) {
    crc32_init();
    uint32_t crc = 0xFFFFFFFF;
    for (uint32_t i = 0; i < size; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/* ========================================================================
 * Little-endian helpers
 * ======================================================================== */

static void le_write_u16(uint8_t *buf, uint16_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
}

static void le_write_u32(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

static uint16_t le_read_u16(const uint8_t *buf) {
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

static uint32_t le_read_u32(const uint8_t *buf) {
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

/* ========================================================================
 * Module Management
 * ======================================================================== */

NvmModule *nvm_module_new(void) {
    NvmModule *mod = calloc(1, sizeof(NvmModule));
    if (!mod) return NULL;

    /* Initialize header magic */
    mod->header.magic[0] = NVM_MAGIC_0;
    mod->header.magic[1] = NVM_MAGIC_1;
    mod->header.magic[2] = NVM_MAGIC_2;
    mod->header.magic[3] = NVM_MAGIC_3;
    mod->header.format_version = NVM_FORMAT_VERSION;

    /* Initial capacities */
    mod->string_capacity = 64;
    mod->strings = calloc(mod->string_capacity, sizeof(char *));
    mod->string_lengths = calloc(mod->string_capacity, sizeof(uint32_t));

    mod->function_capacity = 32;
    mod->functions = calloc(mod->function_capacity, sizeof(NvmFunctionEntry));
    mod->function_param_types = calloc(mod->function_capacity, sizeof(uint8_t *));

    mod->code_capacity = 4096;
    mod->code = calloc(mod->code_capacity, sizeof(uint8_t));

    mod->debug_capacity = 256;
    mod->debug_entries = calloc(mod->debug_capacity, sizeof(NvmDebugEntry));

    mod->import_capacity = 32;
    mod->imports = calloc(mod->import_capacity, sizeof(NvmImportEntry));
    mod->import_param_types = calloc(mod->import_capacity, sizeof(uint8_t *));

    mod->module_ref_capacity = 16;
    mod->module_refs = calloc(mod->module_ref_capacity, sizeof(NvmModuleRefEntry));

    if (!mod->strings || !mod->string_lengths || !mod->functions || !mod->function_param_types ||
        !mod->code || !mod->debug_entries || !mod->imports ||
        !mod->import_param_types || !mod->module_refs) {
        nvm_module_free(mod);
        return NULL;
    }

    return mod;
}

void nvm_module_free(NvmModule *mod) {
    if (!mod) return;

    if (mod->strings) {
        for (uint32_t i = 0; i < mod->string_count; i++) {
            free(mod->strings[i]);
        }
        free(mod->strings);
    }
    free(mod->string_lengths);
    free(mod->functions);
    if (mod->function_param_types) {
        for (uint32_t i = 0; i < mod->function_count; i++)
            free(mod->function_param_types[i]);
        free(mod->function_param_types);
    }
    free(mod->code);
    free(mod->debug_entries);
    if (mod->import_param_types) {
        for (uint32_t i = 0; i < mod->import_count; i++) {
            free(mod->import_param_types[i]);
        }
        free(mod->import_param_types);
    }
    free(mod->imports);
    free(mod->callback_contracts);
    free(mod->passive_data);
    free(mod->layout_data);
    free(mod->ownership_data);
    free(mod->module_refs);    free(mod->call_descriptors);
    free(mod);
}

void nvm_call_descriptors_reset(NvmModule *mod) {
    if (!mod) return;
    free(mod->call_descriptors);
    mod->call_descriptors = NULL;
    mod->call_descriptor_count = 0;
}

/* ========================================================================
 * String Pool
 * ======================================================================== */

uint32_t nvm_add_string(NvmModule *mod, const char *str, uint32_t length) {
    if (!mod || (!str && length)) return UINT32_MAX;
    if (!str) str = "";
    /* Deduplicate */
    for (uint32_t i = 0; i < mod->string_count; i++) {
        if (mod->string_lengths[i] == length &&
            memcmp(mod->strings[i], str, length) == 0) {
            return i;
        }
    }

    /* Grow if needed */
    if (mod->string_count >= mod->string_capacity) {
        if (mod->string_capacity > UINT32_MAX / 2) return UINT32_MAX;
        uint32_t new_cap = mod->string_capacity ? mod->string_capacity * 2 : 16;
#if SIZE_MAX <= UINT32_MAX
        if (new_cap > SIZE_MAX / sizeof(char *) || new_cap > SIZE_MAX / sizeof(uint32_t)) return UINT32_MAX;
#endif
        char **new_strs = malloc((size_t)new_cap * sizeof(char *));
        uint32_t *new_lens = malloc((size_t)new_cap * sizeof(uint32_t));
        if (!new_strs || !new_lens) {
            free(new_strs);
            free(new_lens);
            return UINT32_MAX;
        }
        if (mod->string_count) {
            memcpy(new_strs, mod->strings, (size_t)mod->string_count * sizeof(char *));
            memcpy(new_lens, mod->string_lengths, (size_t)mod->string_count * sizeof(uint32_t));
        }
        free(mod->strings);
        free(mod->string_lengths);
        mod->strings = new_strs;
        mod->string_lengths = new_lens;
        mod->string_capacity = new_cap;
    }

    uint32_t idx = mod->string_count;
    if ((size_t)length + 1 < length) return UINT32_MAX;
    mod->strings[idx] = malloc((size_t)length + 1);
    if (!mod->strings[idx]) return UINT32_MAX;
    memcpy(mod->strings[idx], str, length);
    mod->strings[idx][length] = '\0';
    mod->string_lengths[idx] = length;
    mod->string_count++;

    return idx;
}

const char *nvm_get_string(const NvmModule *mod, uint32_t index) {
    if (index >= mod->string_count) return NULL;
    return mod->strings[index];
}

uint32_t nvm_get_string_len(const NvmModule *mod, uint32_t index) {    if (index >= mod->string_count) return 0;
    return mod->string_lengths[index];
}

/* ========================================================================
 * Function Table
 * ======================================================================== */

uint32_t nvm_add_function(NvmModule *mod, const NvmFunctionEntry *entry) {
    if (!mod || !entry) return UINT32_MAX;
    NvmFunctionEntry copied_entry = *entry;
    if (mod->function_count >= mod->function_capacity) {
        if (mod->function_capacity > UINT32_MAX / 2) return UINT32_MAX;
        uint32_t new_cap = mod->function_capacity ? mod->function_capacity * 2 : 32;
        NvmFunctionEntry *new_fns = calloc(new_cap, sizeof(NvmFunctionEntry));
        uint8_t **new_types = calloc(new_cap, sizeof(uint8_t *));
        if (!new_fns || !new_types) {
            free(new_fns);
            free(new_types);
            return UINT32_MAX;
        }
        if (mod->function_count) {
            memcpy(new_fns, mod->functions, mod->function_count * sizeof(*new_fns));
            if (mod->function_param_types)
                memcpy(new_types, mod->function_param_types,
                       mod->function_count * sizeof(*new_types));
        }
        free(mod->functions);
        free(mod->function_param_types);
        mod->functions = new_fns;
        mod->function_param_types = new_types;
        mod->function_capacity = new_cap;
    }

    uint32_t idx = mod->function_count;
    mod->functions[idx] = copied_entry;
    mod->function_count++;
    return idx;
}

bool nvm_set_function_param_types(NvmModule *mod, uint32_t index,
                                  const uint8_t *tags, uint16_t count) {
    if (!mod || index >= mod->function_count || !mod->function_param_types ||
        count != mod->functions[index].arity || (count && !tags)) return false;
    for (uint16_t i = 0; i < count; i++)
        if (tags[i] >= TAG_COUNT) return false;
    uint8_t *copy = count ? malloc(count) : NULL;
    if (count && !copy) return false;
    if (count) memcpy(copy, tags, count);
    free(mod->function_param_types[index]);
    mod->function_param_types[index] = copy;
    return true;
}

static bool callback_scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_FLOAT || tag == TAG_BOOL ||
           tag == TAG_U8 || tag == TAG_OPAQUE;
}

bool nvm_callback_shape_valid(const uint8_t *tags, uint16_t count, uint8_t result) {
    if (count > NANO_MAX_FFI_ARGS || (count && !tags) ||
        (result != TAG_VOID && !callback_scalar(result))) return false;
    for (uint16_t i = 0; i < count; i++)
        if (!callback_scalar(tags[i])) return false;
    return true;
}

bool nvm_add_callback_contract(NvmModule *mod, const NvmCallbackContract *contract) {
    if (!mod || !contract || contract->abi_version != NVM_CALLBACK_ABI_RETAINED_V1 ||
        contract->execution > NVM_FOREIGN_WORKER_THREAD ||
        !nvm_callback_shape_valid(contract->param_tags, contract->param_count,
                                  contract->return_tag)) return false;
    NvmCallbackContract copy = *contract;
    if (copy.parameter_idx == NVM_CALLBACK_NO_PARAMETER &&
        (copy.param_count || copy.return_tag != TAG_VOID)) return false;
    if (mod->callback_contract_count) {
        const NvmCallbackContract *last = &mod->callback_contracts[mod->callback_contract_count - 1];
        if (copy.import_idx < last->import_idx ||
            (copy.import_idx == last->import_idx && copy.parameter_idx <= last->parameter_idx))
            return false;
    }
    if (mod->callback_contract_count == mod->callback_contract_capacity) {
        if (mod->callback_contract_capacity > UINT32_MAX / 2) return false;
        uint32_t capacity = mod->callback_contract_capacity ? mod->callback_contract_capacity * 2 : 8;
        NvmCallbackContract *items = calloc(capacity, sizeof(*items));
        if (!items) return false;
        if (mod->callback_contract_count)
            memcpy(items, mod->callback_contracts, mod->callback_contract_count * sizeof(*items));
        free(mod->callback_contracts);
        mod->callback_contracts = items;
        mod->callback_contract_capacity = capacity;
    }
    mod->callback_contracts[mod->callback_contract_count++] = copy;
    return true;
}

bool nvm_callback_contracts_valid(const NvmModule *mod) {
    if (!mod || (mod->callback_contract_count && !mod->callback_contracts)) return false;
    uint32_t begin = 0;
    while (begin < mod->callback_contract_count) {
        const NvmCallbackContract *first = &mod->callback_contracts[begin];
        if (first->import_idx >= mod->import_count || !mod->imports) return false;
        const NvmImportEntry *import = &mod->imports[first->import_idx];
        if (import->param_count > NANO_MAX_FFI_ARGS ||
            (import->param_count && (!mod->import_param_types ||
                                     !mod->import_param_types[first->import_idx]))) return false;
        uint32_t expected = 0, actual = 0;
        for (uint16_t p = 0; p < import->param_count; p++) {
            uint8_t tag = mod->import_param_types[first->import_idx][p];
            if (tag == TAG_FUNCTION || tag == TAG_CLOSURE) expected |= 1u << p;
        }
        uint32_t end = begin;
        while (end < mod->callback_contract_count &&
               mod->callback_contracts[end].import_idx == first->import_idx) {
            const NvmCallbackContract *c = &mod->callback_contracts[end];
            if (c->abi_version != NVM_CALLBACK_ABI_RETAINED_V1 ||
                c->execution > NVM_FOREIGN_WORKER_THREAD ||
                c->execution != first->execution || c->adapter_name_idx != first->adapter_name_idx ||
                !nvm_callback_shape_valid(c->param_tags, c->param_count, c->return_tag)) return false;
            const char *adapter = nvm_get_string(mod, c->adapter_name_idx);
            if (!adapter || !adapter[0] ||
                strlen(adapter) != nvm_get_string_len(mod, c->adapter_name_idx)) return false;
            if (end > begin && c->parameter_idx <= mod->callback_contracts[end - 1].parameter_idx)
                return false;
            if (c->parameter_idx == NVM_CALLBACK_NO_PARAMETER) {
                if (expected || end != begin || c->param_count || c->return_tag != TAG_VOID)
                    return false;
            } else {
                if (c->parameter_idx >= import->param_count || !(expected & (1u << c->parameter_idx)))
                    return false;
                actual |= 1u << c->parameter_idx;
            }
            end++;
        }
        if (actual != expected ||
            (end < mod->callback_contract_count && mod->callback_contracts[end].import_idx < first->import_idx))
            return false;
        begin = end;
    }
    return true;
}

uint32_t nvm_find_function(const NvmModule *mod, const char *name) {
    if (!mod || !name) return UINT32_MAX;
    for (uint32_t i = mod->function_count; i > 0; i--) {
        const char *candidate = nvm_get_string(mod, mod->functions[i - 1].name_idx);
        if (candidate && strcmp(candidate, name) == 0) return i - 1;
    }
    return UINT32_MAX;
}

/* ========================================================================
 * Code Section
 * ======================================================================== */

uint32_t nvm_append_code(NvmModule *mod, const uint8_t *code, uint32_t size) {
    while (mod->code_size + size > mod->code_capacity) {
        uint32_t new_cap = mod->code_capacity * 2;
        uint8_t *new_code = realloc(mod->code, new_cap);
        if (!new_code) return 0;
        mod->code = new_code;
        mod->code_capacity = new_cap;
    }

    uint32_t offset = mod->code_size;
    memcpy(mod->code + offset, code, size);
    mod->code_size += size;
    return offset;
}

/* ========================================================================
 * Debug Info
 * ======================================================================== */

void nvm_add_debug_entry(NvmModule *mod, uint32_t bytecode_offset,
                         uint32_t source_line, uint32_t source_col) {
    if (mod->debug_count >= mod->debug_capacity) {
        uint32_t new_cap = mod->debug_capacity * 2;
        NvmDebugEntry *new_entries = realloc(mod->debug_entries, new_cap * sizeof(NvmDebugEntry));
        if (!new_entries) return;
        mod->debug_entries = new_entries;
        mod->debug_capacity = new_cap;
    }

    mod->debug_entries[mod->debug_count].bytecode_offset = bytecode_offset;
    mod->debug_entries[mod->debug_count].source_line = source_line;
    mod->debug_entries[mod->debug_count].source_col  = source_col;
    mod->debug_count++;
}

void nvm_strip_debug_info(NvmModule *mod) {
    if (!mod) return;
    mod->debug_count = 0;
    mod->source_file_idx = 0;
    mod->header.flags &= ~NVM_FLAG_DEBUG_INFO;
}

/* ========================================================================
 * Import Table
 * ======================================================================== */

uint32_t nvm_add_import(NvmModule *mod, uint32_t module_name_idx,
                        uint32_t function_name_idx, uint16_t param_count,
                        uint8_t return_type, const uint8_t *param_types) {
    if (!mod) return UINT32_MAX;
    if (mod->import_count >= mod->import_capacity) {
        if (mod->import_capacity > UINT32_MAX / 2) return UINT32_MAX;
        uint32_t new_cap = mod->import_capacity ? mod->import_capacity * 2 : 16;
#if SIZE_MAX <= UINT32_MAX
        if (new_cap > SIZE_MAX / sizeof(NvmImportEntry) || new_cap > SIZE_MAX / sizeof(uint8_t *)) return UINT32_MAX;
#endif
        NvmImportEntry *new_imp = malloc((size_t)new_cap * sizeof(NvmImportEntry));
        uint8_t **new_pt = malloc((size_t)new_cap * sizeof(uint8_t *));
        if (!new_imp || !new_pt) {
            free(new_imp);
            free(new_pt);
            return UINT32_MAX;
        }
        if (mod->import_count) {
            memcpy(new_imp, mod->imports, (size_t)mod->import_count * sizeof(NvmImportEntry));
            memcpy(new_pt, mod->import_param_types, (size_t)mod->import_count * sizeof(uint8_t *));
        }
        free(mod->imports);
        free(mod->import_param_types);
        mod->imports = new_imp;
        mod->import_param_types = new_pt;
        mod->import_capacity = new_cap;
    }

    uint32_t idx = mod->import_count;
    mod->imports[idx].module_name_idx = module_name_idx;
    mod->imports[idx].function_name_idx = function_name_idx;
    mod->imports[idx].param_count = param_count;
    mod->imports[idx].return_type = return_type;
    mod->imports[idx].kind = NVM_IMPORT_FFI;

    if (param_count > 0 && param_types) {
        mod->import_param_types[idx] = malloc(param_count);
        if (!mod->import_param_types[idx]) return UINT32_MAX;
        memcpy(mod->import_param_types[idx], param_types, param_count);
    } else {
        mod->import_param_types[idx] = NULL;
    }

    mod->import_count++;
    return idx;
}

/* ========================================================================
 * Header Validation
 * ======================================================================== */

bool nvm_validate_header(const NvmHeader *header) {
    if (header->magic[0] != NVM_MAGIC_0 ||
        header->magic[1] != NVM_MAGIC_1 ||
        header->magic[2] != NVM_MAGIC_2 ||
        header->magic[3] != NVM_MAGIC_3) {
        return false;
    }
    if (header->format_version != NVM_FORMAT_VERSION) {
        return false;
    }
    if (header->section_count > NVM_MAX_SECTIONS) {
        return false;
    }
    return true;
}

/* ========================================================================
 * Serialization
 *
 * File layout:
 *   [Header: 32 bytes]
 *   [Section Directory: 12 * section_count bytes]
 *   [String Pool Section]
 *   [Code Section]
 *   [Function Table Section]
 *   [Debug Section (if present)]
 *   [Import Section (if present)]
 * ======================================================================== */

/* Helper: serialize string pool into a buffer. Returns size. */
static uint32_t serialize_string_pool(const NvmModule *mod, uint8_t *buf) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        le_write_u32(buf + pos, mod->string_lengths[i]);
        pos += 4;
        memcpy(buf + pos, mod->strings[i], mod->string_lengths[i]);
        pos += mod->string_lengths[i];
    }
    return pos;
}

/* Calculate string pool serialized size */
static uint32_t string_pool_size(const NvmModule *mod) {
    uint32_t size = 0;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        size += 4 + mod->string_lengths[i]; /* u32 length + bytes */
    }
    return size;
}

/* Helper: serialize function table */
static uint32_t serialize_functions(const NvmModule *mod, uint8_t *buf) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        le_write_u32(buf + pos, fn->name_idx);       pos += 4;
        le_write_u16(buf + pos, fn->arity);           pos += 2;
        le_write_u32(buf + pos, fn->code_offset);     pos += 4;
        le_write_u32(buf + pos, fn->code_length);     pos += 4;
        le_write_u16(buf + pos, fn->local_count);     pos += 2;
        le_write_u16(buf + pos, fn->upvalue_count);   pos += 2;
        buf[pos++] = fn->result_tag;
        buf[pos++] = fn->result_count;
    }
    return pos;
}

/* Helper: serialize debug info */
static uint32_t serialize_debug(const NvmModule *mod, uint8_t *buf) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < mod->debug_count; i++) {
        le_write_u32(buf + pos, mod->debug_entries[i].bytecode_offset);
        pos += 4;
        le_write_u32(buf + pos, mod->debug_entries[i].source_line);
        pos += 4;
        le_write_u32(buf + pos, mod->debug_entries[i].source_col);
        pos += 4;
    }
    return pos;
}

/* Helper: serialize import table */
static uint32_t serialize_imports(const NvmModule *mod, uint8_t *buf) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < mod->import_count; i++) {
        const NvmImportEntry *imp = &mod->imports[i];
        le_write_u32(buf + pos, imp->module_name_idx);   pos += 4;
        le_write_u32(buf + pos, imp->function_name_idx);  pos += 4;
        le_write_u16(buf + pos, imp->param_count);         pos += 2;
        buf[pos++] = imp->return_type;
        if (mod->import_param_types[i]) {
            memcpy(buf + pos, mod->import_param_types[i], imp->param_count);
        }
        pos += imp->param_count;
    }
    return pos;
}

static uint32_t import_section_size(const NvmModule *mod) {
    uint32_t size = 0;
    for (uint32_t i = 0; i < mod->import_count; i++) {
        size += NVM_IMPORT_ENTRY_BASE_SIZE + mod->imports[i].param_count;
    }
    return size;
}

uint32_t nvm_add_module_ref(NvmModule *mod, uint32_t module_name_idx) {
    if (!mod || module_name_idx >= mod->string_count) return UINT32_MAX;
    if (mod->module_ref_count >= mod->module_ref_capacity) {
        uint32_t new_cap = mod->module_ref_capacity * 2;
        NvmModuleRefEntry *grown = realloc(
            mod->module_refs, new_cap * sizeof(NvmModuleRefEntry));
        if (!grown) return UINT32_MAX;
        mod->module_refs = grown;
        mod->module_ref_capacity = new_cap;
    }
    uint32_t idx = mod->module_ref_count++;
    mod->module_refs[idx].module_name_idx = module_name_idx;
    return idx;
}

uint8_t *nvm_serialize(const NvmModule *mod, uint32_t *out_size) {
    if (mod->callback_contract_count || mod->passive_size || mod->layout_size || mod->ownership_size) {
        if (out_size) *out_size = 0;
        return NULL;
    }
    /* I cannot erase an exact binding or coprocess kind in legacy output. */
    for (uint32_t i = 0; i < mod->import_count; i++) {
        if (mod->imports[i].kind != NVM_IMPORT_FFI) {
            if (out_size) *out_size = 0;
            return NULL;
        }
    }
    /* Count sections we'll write */
    uint32_t nsections = 0;
    bool has_strings   = (mod->string_count > 0);
    bool has_code      = (mod->code_size > 0);
    bool has_functions  = (mod->function_count > 0);
    bool has_debug     = (mod->debug_count > 0);
    bool has_imports   = (mod->import_count > 0);
    bool has_module_refs = (mod->module_ref_count > 0);

    if (has_strings)   nsections++;
    if (has_code)      nsections++;
    if (has_functions) nsections++;
    if (has_debug)     nsections++;
    if (has_imports)   nsections++;
    if (has_module_refs) nsections++;

    /* Calculate sizes */
    uint32_t str_size = has_strings ? string_pool_size(mod) : 0;
    uint32_t code_size_bytes = has_code ? mod->code_size : 0;
    uint32_t fn_size = has_functions ? mod->function_count * NVM_FUNCTION_ENTRY_SIZE : 0;
    uint32_t dbg_size = has_debug ? mod->debug_count * NVM_DEBUG_ENTRY_SIZE : 0;
    uint32_t imp_size = has_imports ? import_section_size(mod) : 0;
    uint32_t ref_size = has_module_refs
        ? mod->module_ref_count * NVM_MODULE_REF_ENTRY_SIZE : 0;

    uint32_t dir_size = nsections * NVM_SECTION_ENTRY_SIZE;
    uint32_t data_size = str_size + code_size_bytes + fn_size + dbg_size + imp_size + ref_size;
    uint32_t total_size = NVM_HEADER_SIZE + dir_size + data_size;

    uint8_t *buf = calloc(1, total_size);
    if (!buf) return NULL;

    /* Build section directory and data */
    uint32_t data_offset = NVM_HEADER_SIZE + dir_size;
    uint32_t dir_pos = NVM_HEADER_SIZE;
    uint32_t data_pos = data_offset;
    uint32_t str_pool_offset = 0;
    uint32_t str_pool_length = 0;

    /* String pool section */
    if (has_strings) {
        le_write_u32(buf + dir_pos, NVM_SECTION_STRINGS);   dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);               dir_pos += 4;
        le_write_u32(buf + dir_pos, str_size);               dir_pos += 4;
        str_pool_offset = data_pos;
        str_pool_length = str_size;
        serialize_string_pool(mod, buf + data_pos);
        data_pos += str_size;
    }

    /* Code section */
    if (has_code) {
        le_write_u32(buf + dir_pos, NVM_SECTION_CODE);   dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);            dir_pos += 4;
        le_write_u32(buf + dir_pos, code_size_bytes);     dir_pos += 4;
        memcpy(buf + data_pos, mod->code, code_size_bytes);
        data_pos += code_size_bytes;
    }

    /* Function table section */
    if (has_functions) {
        le_write_u32(buf + dir_pos, NVM_SECTION_FUNCTIONS); dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);               dir_pos += 4;
        le_write_u32(buf + dir_pos, fn_size);                dir_pos += 4;
        serialize_functions(mod, buf + data_pos);
        data_pos += fn_size;
    }

    /* Debug section */
    if (has_debug) {
        le_write_u32(buf + dir_pos, NVM_SECTION_DEBUG);   dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);             dir_pos += 4;
        le_write_u32(buf + dir_pos, dbg_size);             dir_pos += 4;
        serialize_debug(mod, buf + data_pos);
        data_pos += dbg_size;
    }

    /* Import section */
    if (has_imports) {
        le_write_u32(buf + dir_pos, NVM_SECTION_IMPORTS); dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);             dir_pos += 4;
        le_write_u32(buf + dir_pos, imp_size);             dir_pos += 4;
        serialize_imports(mod, buf + data_pos);
        data_pos += imp_size;
    }

    if (has_module_refs) {
        le_write_u32(buf + dir_pos, NVM_SECTION_MODULE_REFS); dir_pos += 4;
        le_write_u32(buf + dir_pos, data_pos);                 dir_pos += 4;
        le_write_u32(buf + dir_pos, ref_size);                 dir_pos += 4;
        for (uint32_t i = 0; i < mod->module_ref_count; i++) {
            le_write_u32(buf + data_pos, mod->module_refs[i].module_name_idx);
            data_pos += NVM_MODULE_REF_ENTRY_SIZE;
        }
    }

    /* Write header */
    buf[0] = NVM_MAGIC_0;
    buf[1] = NVM_MAGIC_1;
    buf[2] = NVM_MAGIC_2;
    buf[3] = NVM_MAGIC_3;
    le_write_u32(buf + 4,  NVM_FORMAT_VERSION);
    le_write_u32(buf + 8,  mod->header.flags);
    le_write_u32(buf + 12, mod->header.entry_point);
    le_write_u32(buf + 16, nsections);
    le_write_u32(buf + 20, str_pool_offset);
    le_write_u32(buf + 24, str_pool_length);

    /* CRC32 over everything after the header */
    uint32_t crc = nvm_crc32(buf + NVM_HEADER_SIZE, total_size - NVM_HEADER_SIZE);
    le_write_u32(buf + 28, crc);

    *out_size = total_size;
    return buf;
}

/* ========================================================================
 * Deserialization
 * ======================================================================== */

/* Return the fixed record size for section types whose payload is an array of
 * fixed-width records, or 0 for variable-length / opaque sections. A section
 * whose declared size is not a whole multiple of a fixed record size contains
 * a partial record and is rejected. */
static uint32_t nvm_section_record_size(uint32_t sec_type) {
    switch (sec_type) {
        case NVM_SECTION_FUNCTIONS: return NVM_FUNCTION_ENTRY_SIZE;
        case NVM_SECTION_DEBUG:     return NVM_DEBUG_ENTRY_SIZE;
        case NVM_SECTION_MODULE_REFS: return NVM_MODULE_REF_ENTRY_SIZE;
        default:                    return 0;
    }
}

/* Return true if a section type may appear at most once in the directory. */
static bool nvm_section_is_singleton(uint32_t sec_type) {
    switch (sec_type) {
        case NVM_SECTION_CODE:
        case NVM_SECTION_STRINGS:
        case NVM_SECTION_FUNCTIONS:
        case NVM_SECTION_STRUCTS:
        case NVM_SECTION_ENUMS:
        case NVM_SECTION_UNIONS:
        case NVM_SECTION_GLOBALS:
        case NVM_SECTION_IMPORTS:
        case NVM_SECTION_DEBUG:
        case NVM_SECTION_METADATA:
        case NVM_SECTION_MODULE_REFS:
            return true;
        default:
            return false; /* unknown section types are not constrained */
    }
}

/* Structurally validate the section directory before any payload is parsed.
 * Rejects duplicate singleton sections, sections that overlap each other or the
 * header/directory, partial fixed-width records, and any trailing or interior
 * bytes not covered by a section. All arithmetic is overflow-safe: bounds are
 * compared with subtraction so no addition can wrap uint32.
 *
 * data_start is the first byte after the section directory; size is the total
 * buffer length. Returns true when the directory describes a gap-free,
 * non-overlapping, fully covering partition of [data_start, size). */
static bool nvm_validate_section_directory(const uint8_t *data, uint32_t size,
                                           uint32_t section_count,
                                           uint32_t data_start) {
    /* Track seen singleton section types to reject duplicates. */
    uint32_t seen_types[NVM_MAX_SECTIONS];
    uint32_t seen_count = 0;

    /* Collect (offset, size) pairs so we can verify a gap-free partition. */
    uint32_t offsets[NVM_MAX_SECTIONS];
    uint32_t sizes[NVM_MAX_SECTIONS];

    if (section_count > NVM_MAX_SECTIONS) return false;

    for (uint32_t i = 0; i < section_count; i++) {
        uint32_t dir_off = NVM_HEADER_SIZE + i * NVM_SECTION_ENTRY_SIZE;
        uint32_t sec_type   = le_read_u32(data + dir_off);
        uint32_t sec_offset = le_read_u32(data + dir_off + 4);
        uint32_t sec_size   = le_read_u32(data + dir_off + 8);

        /* Overflow-safe bounds: sec_offset + sec_size must stay within size. */
        if (sec_offset > size || sec_size > size - sec_offset) return false;

        /* A section must not overlap the header or the directory itself. */
        if (sec_offset < data_start) return false;

        /* Partial fixed-width record: size must be a whole record multiple. */
        uint32_t rec = nvm_section_record_size(sec_type);
        if (rec != 0 && (sec_size % rec) != 0) return false;

        /* Duplicate singleton section. */
        if (nvm_section_is_singleton(sec_type)) {
            for (uint32_t j = 0; j < seen_count; j++) {
                if (seen_types[j] == sec_type) return false;
            }
            seen_types[seen_count++] = sec_type;
        }

        offsets[i] = sec_offset;
        sizes[i]   = sec_size;
    }

    /* Verify the sections form a gap-free, non-overlapping cover of the data
     * region. Walk the region from data_start; at each step find the section
     * that begins exactly where the cursor is. Overlaps, gaps, and trailing
     * data all surface as a cursor that cannot advance or does not reach size.
     * Zero-length sections are permitted and consume no space. */
    uint32_t cursor = data_start;
    uint32_t consumed = 0;
    while (cursor < size && consumed < section_count) {
        bool advanced = false;
        for (uint32_t i = 0; i < section_count; i++) {
            if (sizes[i] != 0 && offsets[i] == cursor) {
                cursor += sizes[i]; /* overflow-safe: bounded by size above */
                consumed++;
                advanced = true;
                break;
            }
        }
        if (!advanced) return false; /* gap or overlap: no section starts here */
    }

    /* Trailing data: the covered region must reach exactly the end of file. */
    if (cursor != size) return false;

    /* Every non-empty section must have been consumed exactly once. Leftover
     * non-empty sections indicate two sections sharing a start offset (an
     * overlap the forward walk skipped). */
    uint32_t nonempty = 0;
    for (uint32_t i = 0; i < section_count; i++) {
        if (sizes[i] != 0) nonempty++;
    }
    if (consumed != nonempty) return false;

    return true;
}

NvmModule *nvm_deserialize(const uint8_t *data, uint32_t size) {
    if (size < NVM_HEADER_SIZE) return NULL;

    /* Parse header */
    NvmHeader header;
    header.magic[0] = data[0];
    header.magic[1] = data[1];
    header.magic[2] = data[2];
    header.magic[3] = data[3];
    header.format_version    = le_read_u32(data + 4);
    header.flags             = le_read_u32(data + 8);
    header.entry_point       = le_read_u32(data + 12);
    header.section_count     = le_read_u32(data + 16);
    header.string_pool_offset = le_read_u32(data + 20);
    header.string_pool_length = le_read_u32(data + 24);
    header.checksum          = le_read_u32(data + 28);

    if (!nvm_validate_header(&header)) return NULL;

    /* Verify CRC32 */
    uint32_t expected_crc = nvm_crc32(data + NVM_HEADER_SIZE, size - NVM_HEADER_SIZE);
    if (expected_crc != header.checksum) return NULL;

    /* Check section directory fits */
    uint32_t dir_end = NVM_HEADER_SIZE + header.section_count * NVM_SECTION_ENTRY_SIZE;
    if (dir_end > size) return NULL;

    /* Structurally validate the directory before parsing any payload: reject
     * duplicate singleton sections, overlaps, partial records, and trailing or
     * interior bytes not covered by a section. */
    if (!nvm_validate_section_directory(data, size, header.section_count, dir_end)) {
        return NULL;
    }

    NvmModule *mod = nvm_module_new();
    if (!mod) return NULL;

    mod->header = header;
    mod->section_count = header.section_count;

    /* Parse section directory */
    for (uint32_t i = 0; i < header.section_count; i++) {
        uint32_t dir_off = NVM_HEADER_SIZE + i * NVM_SECTION_ENTRY_SIZE;
        uint32_t sec_type   = le_read_u32(data + dir_off);
        uint32_t sec_offset = le_read_u32(data + dir_off + 4);
        uint32_t sec_size   = le_read_u32(data + dir_off + 8);

        /* Overflow-safe bounds check: sec_offset + sec_size can wrap around
         * uint32, letting a crafted offset (e.g. 0xFFFFFF00) slip past a naive
         * `sec_offset + sec_size > size` test and point sec_data far outside
         * the buffer. Compare via subtraction so no addition can overflow. */
        if (sec_offset > size || sec_size > size - sec_offset) {
            nvm_module_free(mod);
            return NULL;
        }

        mod->sections[i].type   = sec_type;
        mod->sections[i].offset = sec_offset;
        mod->sections[i].size   = sec_size;

        const uint8_t *sec_data = data + sec_offset;

        switch (sec_type) {
            case NVM_SECTION_STRINGS: {
                uint32_t pos = 0;
                while (pos + 4 <= sec_size) {
                    uint32_t slen = le_read_u32(sec_data + pos);
                    pos += 4;
                    /* Subtraction form: pos <= sec_size here, so no overflow. */
                    if (slen > sec_size - pos) break;
                    nvm_add_string(mod, (const char *)(sec_data + pos), slen);
                    pos += slen;
                }
                break;
            }

            case NVM_SECTION_CODE: {
                nvm_append_code(mod, sec_data, sec_size);
                break;
            }

            case NVM_SECTION_FUNCTIONS: {
                uint32_t pos = 0;
                while (pos + NVM_FUNCTION_ENTRY_SIZE <= sec_size) {
                    NvmFunctionEntry fn;
                    fn.name_idx      = le_read_u32(sec_data + pos);     pos += 4;
                    fn.arity         = le_read_u16(sec_data + pos);     pos += 2;
                    fn.code_offset   = le_read_u32(sec_data + pos);     pos += 4;
                    fn.code_length   = le_read_u32(sec_data + pos);     pos += 4;
                    fn.local_count   = le_read_u16(sec_data + pos);     pos += 2;
                    fn.upvalue_count = le_read_u16(sec_data + pos);     pos += 2;
                    fn.result_tag    = sec_data[pos++];
                    fn.result_count  = sec_data[pos++];
                    if (nvm_add_function(mod, &fn) == UINT32_MAX) {
                        nvm_module_free(mod);
                        return NULL;
                    }
                }
                break;
            }

            case NVM_SECTION_DEBUG: {
                uint32_t pos = 0;
                while (pos + NVM_DEBUG_ENTRY_SIZE <= sec_size) {
                    uint32_t bc_off = le_read_u32(sec_data + pos); pos += 4;
                    uint32_t line   = le_read_u32(sec_data + pos); pos += 4;
                    uint32_t col    = le_read_u32(sec_data + pos); pos += 4;
                    nvm_add_debug_entry(mod, bc_off, line, col);
                }
                break;
            }

            case NVM_SECTION_IMPORTS: {
                uint32_t pos = 0;
                while (pos + NVM_IMPORT_ENTRY_BASE_SIZE <= sec_size) {
                    if (mod->import_count >= mod->import_capacity) {
                        uint32_t new_cap = mod->import_capacity * 2;
                        NvmImportEntry *new_imp = realloc(mod->imports, new_cap * sizeof(NvmImportEntry));
                        uint8_t **new_pt = realloc(mod->import_param_types, new_cap * sizeof(uint8_t *));
                        if (!new_imp || !new_pt) break;
                        mod->imports = new_imp;
                        mod->import_param_types = new_pt;
                        mod->import_capacity = new_cap;
                    }

                    uint32_t idx = mod->import_count;
                    mod->imports[idx].module_name_idx   = le_read_u32(sec_data + pos); pos += 4;
                    mod->imports[idx].function_name_idx  = le_read_u32(sec_data + pos); pos += 4;
                    mod->imports[idx].param_count        = le_read_u16(sec_data + pos); pos += 2;
                    mod->imports[idx].return_type        = sec_data[pos++];
                    mod->imports[idx].kind               = NVM_IMPORT_FFI;

                    if (pos + mod->imports[idx].param_count > sec_size) break;

                    if (mod->imports[idx].param_count > 0) {
                        mod->import_param_types[idx] = malloc(mod->imports[idx].param_count);
                        if (mod->import_param_types[idx]) {
                            memcpy(mod->import_param_types[idx], sec_data + pos,
                                   mod->imports[idx].param_count);
                        }
                    } else {
                        mod->import_param_types[idx] = NULL;
                    }
                    pos += mod->imports[idx].param_count;
                    mod->import_count++;
                }
                break;
            }

            case NVM_SECTION_MODULE_REFS: {
                for (uint32_t pos = 0; pos < sec_size;
                     pos += NVM_MODULE_REF_ENTRY_SIZE) {
                    if (nvm_add_module_ref(mod, le_read_u32(sec_data + pos)) == UINT32_MAX) {
                        nvm_module_free(mod);
                        return NULL;
                    }
                }
                break;
            }

            default:
                /* Unknown section type - skip */
                break;
        }
    }

    return mod;
}

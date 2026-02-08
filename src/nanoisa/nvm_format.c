/*
 * NVM Binary Format - serialization and deserialization
 */

#include "nvm_format.h"
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

    mod->code_capacity = 4096;
    mod->code = calloc(mod->code_capacity, sizeof(uint8_t));

    mod->debug_capacity = 256;
    mod->debug_entries = calloc(mod->debug_capacity, sizeof(NvmDebugEntry));

    mod->import_capacity = 32;
    mod->imports = calloc(mod->import_capacity, sizeof(NvmImportEntry));
    mod->import_param_types = calloc(mod->import_capacity, sizeof(uint8_t *));

    if (!mod->strings || !mod->string_lengths || !mod->functions ||
        !mod->code || !mod->debug_entries || !mod->imports || !mod->import_param_types) {
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
    free(mod->code);
    free(mod->debug_entries);
    if (mod->import_param_types) {
        for (uint32_t i = 0; i < mod->import_count; i++) {
            free(mod->import_param_types[i]);
        }
        free(mod->import_param_types);
    }
    free(mod->imports);
    free(mod);
}

/* ========================================================================
 * String Pool
 * ======================================================================== */

uint32_t nvm_add_string(NvmModule *mod, const char *str, uint32_t length) {
    /* Deduplicate */
    for (uint32_t i = 0; i < mod->string_count; i++) {
        if (mod->string_lengths[i] == length &&
            memcmp(mod->strings[i], str, length) == 0) {
            return i;
        }
    }

    /* Grow if needed */
    if (mod->string_count >= mod->string_capacity) {
        uint32_t new_cap = mod->string_capacity * 2;
        char **new_strs = realloc(mod->strings, new_cap * sizeof(char *));
        uint32_t *new_lens = realloc(mod->string_lengths, new_cap * sizeof(uint32_t));
        if (!new_strs || !new_lens) {
            free(new_strs);
            free(new_lens);
            return 0; /* error */
        }
        mod->strings = new_strs;
        mod->string_lengths = new_lens;
        mod->string_capacity = new_cap;
    }

    uint32_t idx = mod->string_count;
    mod->strings[idx] = malloc(length + 1);
    if (!mod->strings[idx]) return 0;
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

/* ========================================================================
 * Function Table
 * ======================================================================== */

uint32_t nvm_add_function(NvmModule *mod, const NvmFunctionEntry *entry) {
    if (mod->function_count >= mod->function_capacity) {
        uint32_t new_cap = mod->function_capacity * 2;
        NvmFunctionEntry *new_fns = realloc(mod->functions, new_cap * sizeof(NvmFunctionEntry));
        if (!new_fns) return 0;
        mod->functions = new_fns;
        mod->function_capacity = new_cap;
    }

    uint32_t idx = mod->function_count;
    mod->functions[idx] = *entry;
    mod->function_count++;
    return idx;
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

void nvm_add_debug_entry(NvmModule *mod, uint32_t bytecode_offset, uint32_t source_line) {
    if (mod->debug_count >= mod->debug_capacity) {
        uint32_t new_cap = mod->debug_capacity * 2;
        NvmDebugEntry *new_entries = realloc(mod->debug_entries, new_cap * sizeof(NvmDebugEntry));
        if (!new_entries) return;
        mod->debug_entries = new_entries;
        mod->debug_capacity = new_cap;
    }

    mod->debug_entries[mod->debug_count].bytecode_offset = bytecode_offset;
    mod->debug_entries[mod->debug_count].source_line = source_line;
    mod->debug_count++;
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

uint8_t *nvm_serialize(const NvmModule *mod, uint32_t *out_size) {
    /* Count sections we'll write */
    uint32_t nsections = 0;
    bool has_strings   = (mod->string_count > 0);
    bool has_code      = (mod->code_size > 0);
    bool has_functions  = (mod->function_count > 0);
    bool has_debug     = (mod->debug_count > 0);
    bool has_imports   = (mod->import_count > 0);

    if (has_strings)   nsections++;
    if (has_code)      nsections++;
    if (has_functions) nsections++;
    if (has_debug)     nsections++;
    if (has_imports)   nsections++;

    /* Calculate sizes */
    uint32_t str_size = has_strings ? string_pool_size(mod) : 0;
    uint32_t code_size_bytes = has_code ? mod->code_size : 0;
    uint32_t fn_size = has_functions ? mod->function_count * NVM_FUNCTION_ENTRY_SIZE : 0;
    uint32_t dbg_size = has_debug ? mod->debug_count * NVM_DEBUG_ENTRY_SIZE : 0;
    uint32_t imp_size = has_imports ? import_section_size(mod) : 0;

    uint32_t dir_size = nsections * NVM_SECTION_ENTRY_SIZE;
    uint32_t data_size = str_size + code_size_bytes + fn_size + dbg_size + imp_size;
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

        if (sec_offset + sec_size > size) {
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
                    if (pos + slen > sec_size) break;
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
                    nvm_add_function(mod, &fn);
                }
                break;
            }

            case NVM_SECTION_DEBUG: {
                uint32_t pos = 0;
                while (pos + NVM_DEBUG_ENTRY_SIZE <= sec_size) {
                    uint32_t bc_off = le_read_u32(sec_data + pos); pos += 4;
                    uint32_t line   = le_read_u32(sec_data + pos); pos += 4;
                    nvm_add_debug_entry(mod, bc_off, line);
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

            default:
                /* Unknown section type - skip */
                break;
        }
    }

    return mod;
}

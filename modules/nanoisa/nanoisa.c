#define _POSIX_C_SOURCE 200809L

#include "nanoisa.h"

#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanoisa/isa.h"
#include "../../src/nanoisa/nvm_v2_sections.h"
#include "../../src/nanoisa/verifier.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NANOISA_MAX_FILE_SIZE (100U * 1024U * 1024U)

static char last_error[256];
static char *last_output;

static void clear_error(NanoisaErr *err) {
    if (err) {
        memset(err, 0, sizeof(*err));
    }
}

static void set_error(NanoisaErr *err, NanoisaErrorCode code, uint32_t line,
                      const char *format, ...) {
    va_list args;

    if (!err) {
        return;
    }
    err->code = code;
    err->line = line;
    va_start(args, format);
    vsnprintf(err->message, sizeof(err->message), format, args);
    va_end(args);
}

static const char *section_name(uint32_t type) {
    switch (type) {
        case NVM_SECTION_CODE: return "code";
        case NVM_SECTION_STRINGS: return "strings";
        case NVM_SECTION_FUNCTIONS: return "functions";
        case NVM_SECTION_STRUCTS: return "structs";
        case NVM_SECTION_ENUMS: return "enums";
        case NVM_SECTION_UNIONS: return "unions";
        case NVM_SECTION_GLOBALS: return "globals";
        case NVM_SECTION_IMPORTS: return "imports";
        case NVM_SECTION_DEBUG: return "debug";
        case NVM_SECTION_METADATA: return "metadata";
        case NVM_SECTION_MODULE_REFS: return "module_refs";
        default: return "unknown";
    }
}

static uint32_t logical_string_size(const NvmModule *mod) {
    uint32_t size = 0;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        size += 4 + mod->string_lengths[i];
    }
    return size;
}

static uint32_t logical_import_size(const NvmModule *mod) {
    uint32_t size = 0;
    for (uint32_t i = 0; i < mod->import_count; i++) {
        size += NVM_IMPORT_ENTRY_BASE_SIZE + mod->imports[i].param_count;
    }
    return size;
}

static void print_logical_section(FILE *stream, const char *name,
                                  uint32_t size) {
    if (size > 0) {
        fprintf(stream, "  [logical] %-12s offset=unserialized size=%u\n",
                name, size);
    }
}

NvmModule *nanoisa_load_bytes(const uint8_t *data, uint32_t size,
                              NanoisaErr *err) {
    clear_error(err);
    if (!data) {
        set_error(err, NANOISA_ERR_ARGUMENT, 0, "NVM data is null");
        return NULL;
    }
    if (size < NVM_HEADER_SIZE) {
        set_error(err, NANOISA_ERR_FORMAT, 0,
                  "NVM data is truncated: %u bytes", size);
        return NULL;
    }
    if (data[0] != NVM_MAGIC_0 || data[1] != NVM_MAGIC_1
            || data[2] != NVM_MAGIC_2) {
        set_error(err, NANOISA_ERR_FORMAT, 0, "Invalid NVM magic");
        return NULL;
    }

    /* The fourth magic byte is the container version, and this function is the
     * single funnel every consumer goes through -- the VM, the co-process, the
     * daemon, generated wrappers -- so dispatching here is the whole loader
     * change. A byte belonging to neither format is still rejected rather than
     * guessed at. */
    if (data[3] == NVM_V2_MAGIC_3) {
        NvmV2Module v2;
        NvmV2Result r = nvm_v2_module_deserialize(data, size, &v2);
        if (r != NVM_V2_OK) {
            set_error(err, NANOISA_ERR_FORMAT, 0,
                      "Invalid NVM v2 module: %s", nvm_v2_result_name(r));
            return NULL;
        }
        NvmModule *mod = NULL;
        r = nvm_v2_to_nvm_module(&v2, &mod);
        if (r != NVM_V2_OK || !mod) {
            nvm_v2_module_free(&v2);
            set_error(err, NANOISA_ERR_FORMAT, 0,
                      "NVM v2 module is not expressible as v1: %s",
                      nvm_v2_result_name(r));
            return NULL;
        }

        /* Confirm the declared operand depth rather than recompute it. A
         * declared 0 means the producer had nothing to declare; any other
         * value must be one the verifier agrees with, because a disagreement
         * between producer and verifier is exactly the kind of thing that
         * otherwise shows up as a stack overflow at run time. */
        for (uint32_t i = 0; i < v2.functions.count; i++) {
            uint16_t declared = v2.functions.items[i].max_stack;
            if (declared == 0) continue;
            uint16_t computed = 0;
            NvmVerifyResult vr = nvm_verify_function_max_stack(mod, i, &computed);
            if (!vr.ok) {
                nvm_v2_module_free(&v2);
                nvm_module_free(mod);
                set_error(err, NANOISA_ERR_FORMAT, 0,
                          "NVM v2 function %u does not verify: %s",
                          i, vr.error_msg);
                return NULL;
            }
            if (declared < computed) {
                nvm_v2_module_free(&v2);
                nvm_module_free(mod);
                set_error(err, NANOISA_ERR_FORMAT, 0,
                          "NVM v2 function %u declares max_stack %u but reaches %u",
                          i, (unsigned)declared, (unsigned)computed);
                return NULL;
            }
        }

        nvm_v2_module_free(&v2);
        return mod;
    }

    if (data[3] == NVM_MAGIC_3) {
        /* v1 is retired as of 4.0. .nvm files are build artifacts rather than
         * distributed packages, so the fix is to rebuild rather than to keep a
         * compatibility path that would have to stay correct forever. */
        set_error(err, NANOISA_ERR_FORMAT, 0,
                  "module was built for NanoISA v1 (NVM\\x01); "
                  "rebuild it with nanoc 4.0 or later");
        return NULL;
    }

    set_error(err, NANOISA_ERR_FORMAT, 0,
              "Unknown NVM container version %u", (unsigned)data[3]);
    return NULL;
}

uint8_t *nanoisa_save_bytes(const NvmModule *mod, uint32_t *out_size,
                            NanoisaErr *err) {
    clear_error(err);
    if (!mod || !out_size) {
        set_error(err, NANOISA_ERR_ARGUMENT, 0,
                  "Module and output size are required");
        return NULL;
    }
    *out_size = 0;

    NvmV2Module v2;
    NvmV2Result r = nvm_v2_from_nvm_module(mod, &v2);
    if (r != NVM_V2_OK) {
        set_error(err, NANOISA_ERR_MEMORY, 0,
                  "Could not convert module to v2: %s", nvm_v2_result_name(r));
        return NULL;
    }

    size_t need = 0;
    r = nvm_v2_module_serialize(&v2, NULL, 0, &need);
    if (r != NVM_V2_OK || need > UINT32_MAX) {
        nvm_v2_module_free(&v2);
        set_error(err, NANOISA_ERR_FORMAT, 0, "Could not size v2 module");
        return NULL;
    }

    uint8_t *buf = malloc(need);
    if (!buf) {
        nvm_v2_module_free(&v2);
        set_error(err, NANOISA_ERR_MEMORY, 0, "Out of memory");
        return NULL;
    }

    size_t written = 0;
    r = nvm_v2_module_serialize(&v2, buf, need, &written);
    nvm_v2_module_free(&v2);
    if (r != NVM_V2_OK) {
        free(buf);
        set_error(err, NANOISA_ERR_FORMAT, 0,
                  "Could not serialize v2 module: %s", nvm_v2_result_name(r));
        return NULL;
    }
    *out_size = (uint32_t)written;
    return buf;
}

NvmModule *nanoisa_load_file(const char *path, NanoisaErr *err) {
    clear_error(err);
    if (!path || path[0] == '\0') {
        set_error(err, NANOISA_ERR_ARGUMENT, 0, "NVM path is empty");
        return NULL;
    }

    FILE *file = fopen(path, "rb");
    if (!file) {
        set_error(err, NANOISA_ERR_IO, 0, "Cannot open '%s'", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        set_error(err, NANOISA_ERR_IO, 0, "Cannot seek '%s'", path);
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size <= 0 || (unsigned long)file_size > NANOISA_MAX_FILE_SIZE) {
        fclose(file);
        set_error(err, NANOISA_ERR_IO, 0,
                  "Invalid NVM file size for '%s'", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        set_error(err, NANOISA_ERR_IO, 0, "Cannot seek '%s'", path);
        return NULL;
    }

    uint8_t *data = malloc((size_t)file_size);
    if (!data) {
        fclose(file);
        set_error(err, NANOISA_ERR_MEMORY, 0,
                  "Out of memory reading '%s'", path);
        return NULL;
    }
    size_t bytes_read = fread(data, 1, (size_t)file_size, file);
    fclose(file);
    if (bytes_read != (size_t)file_size) {
        free(data);
        set_error(err, NANOISA_ERR_IO, 0, "Short read from '%s'", path);
        return NULL;
    }

    NvmModule *mod = nanoisa_load_bytes(data, (uint32_t)file_size, err);
    free(data);
    return mod;
}

uint8_t *nanoisa_save_bytes_v1(const NvmModule *mod, uint32_t *out_size,
                               NanoisaErr *err) {
    clear_error(err);
    if (!mod || !out_size) {
        set_error(err, NANOISA_ERR_ARGUMENT, 0,
                  "Module and output size are required");
        return NULL;
    }
    *out_size = 0;
    uint8_t *data = nvm_serialize(mod, out_size);
    if (!data) {
        set_error(err, NANOISA_ERR_MEMORY, 0,
                  "Could not serialize NVM module");
    }
    return data;
}

int nanoisa_save_file(const NvmModule *mod, const char *path, NanoisaErr *err) {
    clear_error(err);
    if (!path || path[0] == '\0') {
        set_error(err, NANOISA_ERR_ARGUMENT, 0, "NVM path is empty");
        return NANOISA_ERR_ARGUMENT;
    }

    uint32_t size = 0;
    uint8_t *data = nanoisa_save_bytes(mod, &size, err);
    if (!data) {
        return err ? err->code : NANOISA_ERR_MEMORY;
    }

    FILE *file = fopen(path, "wb");
    if (!file) {
        free(data);
        set_error(err, NANOISA_ERR_IO, 0, "Cannot write '%s'", path);
        return NANOISA_ERR_IO;
    }
    size_t bytes_written = fwrite(data, 1, size, file);
    int close_result = fclose(file);
    free(data);
    if (bytes_written != size || close_result != 0) {
        set_error(err, NANOISA_ERR_IO, 0, "Short write to '%s'", path);
        return NANOISA_ERR_IO;
    }
    return NANOISA_OK;
}

static NvmModule *map_assembly_result(NvmModule *mod,
                                      const AsmResult *assembly,
                                      NanoisaErr *err) {
    if (!mod) {
        NanoisaErrorCode code = assembly->error == ASM_ERR_MEMORY
            ? NANOISA_ERR_MEMORY : NANOISA_ERR_ASSEMBLY;
        set_error(err, code, assembly->line, "%s",
                  assembly->message[0] ? assembly->message
                                       : "NanoISA assembly failed");
    }
    return mod;
}

NvmModule *nanoisa_assemble_text(const char *source, NanoisaErr *err) {
    clear_error(err);
    if (!source) {
        set_error(err, NANOISA_ERR_ARGUMENT, 0,
                  "Assembly source is null");
        return NULL;
    }
    AsmResult assembly = {0};
    return map_assembly_result(asm_assemble(source, &assembly),
                               &assembly, err);
}

NvmModule *nanoisa_assemble_file(const char *path, NanoisaErr *err) {
    clear_error(err);
    if (!path || path[0] == '\0') {
        set_error(err, NANOISA_ERR_ARGUMENT, 0,
                  "Assembly path is empty");
        return NULL;
    }
    AsmResult assembly = {0};
    return map_assembly_result(asm_assemble_file(path, &assembly),
                               &assembly, err);
}

char *nanoisa_print(const NvmModule *mod) {
    if (!mod) {
        return NULL;
    }
    return disasm_module_styled(mod, DISASM_STYLE_CANONICAL);
}

char *nanoisa_pretty_print(const NvmModule *mod) {
    if (!mod) {
        return NULL;
    }

    char *output = NULL;
    size_t output_size = 0;
    FILE *stream = open_memstream(&output, &output_size);
    if (!stream) {
        return NULL;
    }

    fprintf(stream, "NVM module\n");
    fprintf(stream, "  magic: NVM\\x01\n");
    fprintf(stream, "  version: %u\n", mod->header.format_version);
    fprintf(stream, "  flags: 0x%08x\n", mod->header.flags);
    fprintf(stream, "  entry: %u\n", mod->header.entry_point);
    fprintf(stream, "  strings: %u\n", mod->string_count);
    fprintf(stream, "  debug entries: %u\n\n", mod->debug_count);

    fprintf(stream, "Sections\n");
    if (mod->section_count > 0) {
        for (uint32_t i = 0; i < mod->section_count; i++) {
            const NvmSectionEntry *section = &mod->sections[i];
            fprintf(stream, "  [%u] %-12s offset=0x%08x size=%u\n",
                    i, section_name(section->type), section->offset,
                    section->size);
        }
    } else {
        print_logical_section(stream, "strings", logical_string_size(mod));
        print_logical_section(stream, "code", mod->code_size);
        print_logical_section(stream, "functions",
                              mod->function_count * NVM_FUNCTION_ENTRY_SIZE);
        print_logical_section(stream, "debug",
                              mod->debug_count * NVM_DEBUG_ENTRY_SIZE);
        print_logical_section(stream, "imports", logical_import_size(mod));
    }

    fprintf(stream, "\nFunctions\n");
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *function = &mod->functions[i];
        const char *name = nvm_get_string(mod, function->name_idx);
        fprintf(stream,
                "  [%u] %s arity=%u locals=%u upvalues=%u result=%s/%u "
                "code=0x%08x+%u\n",
                i, name ? name : "???", function->arity,
                function->local_count, function->upvalue_count,
                isa_tag_name(function->result_tag), function->result_count,
                function->code_offset, function->code_length);
    }

    fprintf(stream, "\nImports\n");
    for (uint32_t i = 0; i < mod->import_count; i++) {
        const NvmImportEntry *import = &mod->imports[i];
        const char *module_name = nvm_get_string(mod,
                                                  import->module_name_idx);
        const char *function_name = nvm_get_string(
            mod, import->function_name_idx);
        fprintf(stream, "  [%u] %s.%s params=%u return=%u\n",
                i, module_name ? module_name : "",
                function_name ? function_name : "???",
                import->param_count, import->return_type);
    }

    fprintf(stream, "\nAssembly\n");
    disasm_module_to_file(mod, stream);
    if (fclose(stream) != 0) {
        free(output);
        return NULL;
    }
    return output;
}

static const char *store_output(char *output, const NanoisaErr *err) {
    free(last_output);
    last_output = output;
    if (output) {
        last_error[0] = '\0';
        return last_output;
    }
    snprintf(last_error, sizeof(last_error), "%s",
             err && err->message[0] ? err->message : "NanoISA operation failed");
    return "";
}

const char *nl_nanoisa_load_print(const char *path) {
    NanoisaErr err;
    NvmModule *mod = nanoisa_load_file(path, &err);
    if (!mod) {
        return store_output(NULL, &err);
    }
    char *output = nanoisa_print(mod);
    nvm_module_free(mod);
    return store_output(output, &err);
}

const char *nl_nanoisa_load_pretty(const char *path) {
    NanoisaErr err;
    NvmModule *mod = nanoisa_load_file(path, &err);
    if (!mod) {
        return store_output(NULL, &err);
    }
    char *output = nanoisa_pretty_print(mod);
    nvm_module_free(mod);
    return store_output(output, &err);
}

int64_t nl_nanoisa_assemble_save(const char *nasm_path,
                                 const char *nvm_path) {
    NanoisaErr err;
    NvmModule *mod = nanoisa_assemble_file(nasm_path, &err);
    if (!mod) {
        store_output(NULL, &err);
        return err.code;
    }
    int result = nanoisa_save_file(mod, nvm_path, &err);
    nvm_module_free(mod);
    if (result != NANOISA_OK) {
        store_output(NULL, &err);
    } else {
        last_error[0] = '\0';
    }
    return result;
}

const char *nl_nanoisa_last_error(void) {
    return last_error;
}

#ifndef NANOLANG_MODULE_NANOISA_H
#define NANOLANG_MODULE_NANOISA_H

#include "../../src/nanoisa/nvm_format.h"

#include <stdint.h>

typedef enum {
    NANOISA_OK = 0,
    NANOISA_ERR_ARGUMENT,
    NANOISA_ERR_IO,
    NANOISA_ERR_FORMAT,
    NANOISA_ERR_ASSEMBLY,
    NANOISA_ERR_MEMORY
} NanoisaErrorCode;

typedef struct {
    NanoisaErrorCode code;
    uint32_t line;
    char message[256];
} NanoisaErr;

NvmModule *nanoisa_load_file(const char *path, NanoisaErr *err);
NvmModule *nanoisa_load_bytes(const uint8_t *data, uint32_t size,
                              NanoisaErr *err);
int nanoisa_save_file(const NvmModule *mod, const char *path, NanoisaErr *err);
/* Serialize a module. Since 4.0 this writes the NanoISA v2 container: explicit
 * string lengths, a deduplicated signature table, 64-bit code offsets, and a
 * verifier-confirmed operand depth. Caller frees the buffer. */
uint8_t *nanoisa_save_bytes(const NvmModule *mod, uint32_t *out_size,
                            NanoisaErr *err);

/* Serialize in the retired v1 container. Kept so tests can build a v1 module
 * and confirm the loader refuses it; nothing in the toolchain writes v1. */
uint8_t *nanoisa_save_bytes_v1(const NvmModule *mod, uint32_t *out_size,
                               NanoisaErr *err);

NvmModule *nanoisa_assemble_text(const char *source, NanoisaErr *err);
NvmModule *nanoisa_assemble_file(const char *path, NanoisaErr *err);

char *nanoisa_print(const NvmModule *mod);
char *nanoisa_pretty_print(const NvmModule *mod);

const char *nl_nanoisa_load_print(const char *path);
const char *nl_nanoisa_load_pretty(const char *path);
int64_t nl_nanoisa_assemble_save(const char *nasm_path, const char *nvm_path);
const char *nl_nanoisa_last_error(void);

#endif

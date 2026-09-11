/*
 * Nano Object — 4.6 laboratory frontend.
 *
 * I compile methods to verified NanoISA and dispatch in a C host.
 * Inline caches live in the host. I do not add cache opcodes.
 * Authority: docs/OBJECT.md.
 */

#ifndef NANOLANG_OBJECT_H
#define NANOLANG_OBJECT_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_OBJ_ERR_SIZE 256
#define NL_OBJ_IMAGE_SIZE 512

NvmModule *nl_object_compile(const char *src, const char *path,
                             char *err, size_t errlen);

NlFrontendResult nl_object_accept(const NvmModule *mod, const char *path);

int nl_object_eval_i64(const char *src, int64_t *out,
                       char *err, size_t errlen);

int nl_object_last_ic_hits(void);
int nl_object_last_ic_misses(void);
int nl_object_last_image(char *out, size_t outlen);

#endif

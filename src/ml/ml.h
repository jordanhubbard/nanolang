/*
 * Nano ML — 4.6 laboratory frontend.
 *
 * I compile a bounded ML-family subset to verified NanoISA. This is a C
 * host compiler, like src/scheme/, not a Standard ML or OCaml. Refs,
 * exceptions, and functors stay out of scope. Authority: docs/ML.md.
 */

#ifndef NANOLANG_ML_H
#define NANOLANG_ML_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_ML_ERR_SIZE 256

NvmModule *nl_ml_compile(const char *src, const char *path,
                         char *err, size_t errlen);

NlFrontendResult nl_ml_accept(const NvmModule *mod, const char *path);

VmResult nl_ml_execute(const NvmModule *mod, NanoValue *out,
                       char *err, size_t errlen);

int nl_ml_eval_i64(const char *src, int64_t *out, char *err, size_t errlen);

/* Inferred type of a top-level name after compile, e.g. "a -> a". */
int nl_ml_type_of(const char *src, const char *name, char *out, size_t outlen,
                  char *err, size_t errlen);

uint32_t nl_ml_fun_index(const char *src, const char *name, char *err,
                         size_t errlen);

#endif

/*
 * Nano Logic — 4.6 laboratory frontend.
 *
 * I compile I64_EQ unification to verified NanoISA and evaluate a
 * bounded Datalog subset in the host. Authority: docs/LOGIC.md.
 */

#ifndef NANOLANG_LOGIC_H
#define NANOLANG_LOGIC_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_LG_ERR_SIZE 256

NvmModule *nl_logic_compile(const char *src, const char *path,
                            char *err, size_t errlen);

NlFrontendResult nl_logic_accept(const NvmModule *mod, const char *path);

int nl_logic_eval_i64(const char *src, int64_t *out,
                      char *err, size_t errlen);

#endif

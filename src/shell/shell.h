/*
 * Nano Shell — 4.6 laboratory frontend.
 *
 * I compile function bodies to verified NanoISA and run typed i64
 * pipelines in a C host. Host effects require explicit need-caps and
 * then fail closed in this subset. Authority: docs/SHELL.md.
 */

#ifndef NANOLANG_SHELL_H
#define NANOLANG_SHELL_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_SH_ERR_SIZE 256

NvmModule *nl_shell_compile(const char *src, const char *path,
                            char *err, size_t errlen);

NlFrontendResult nl_shell_accept(const NvmModule *mod, const char *path);

int nl_shell_eval_i64(const char *src, int64_t *out,
                      char *err, size_t errlen);

#endif

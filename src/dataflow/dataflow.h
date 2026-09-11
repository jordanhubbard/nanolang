/*
 * Nano Dataflow — 4.6 laboratory frontend.
 *
 * I compile typed node bodies to verified NanoISA and run a deterministic
 * DAG in one host process. This is a C host, like src/actor/, not a
 * workflow product. Remote placement stays out of scope.
 * Authority: docs/DATAFLOW.md.
 */

#ifndef NANOLANG_DATAFLOW_H
#define NANOLANG_DATAFLOW_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_DF_ERR_SIZE 256

NvmModule *nl_dataflow_compile(const char *src, const char *path,
                               char *err, size_t errlen);

NlFrontendResult nl_dataflow_accept(const NvmModule *mod, const char *path);

int nl_dataflow_eval_i64(const char *src, int64_t *out,
                         char *err, size_t errlen);

int nl_dataflow_eval_i64_sched(const char *src, int reverse, int64_t *out,
                               char *err, size_t errlen);

#endif

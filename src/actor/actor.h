/*
 * Nano Actor — 4.6 laboratory frontend.
 *
 * I compile receive handlers to verified NanoISA and run them in isolated
 * NanoVM contexts in one host process. This is a C host, like src/ml/,
 * not Erlang/OTP. Remote (Phase 18) spawn stays out of scope.
 * Authority: docs/ACTOR.md.
 */

#ifndef NANOLANG_ACTOR_H
#define NANOLANG_ACTOR_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_ACTOR_ERR_SIZE 256

NvmModule *nl_actor_compile(const char *src, const char *path,
                            char *err, size_t errlen);

NlFrontendResult nl_actor_accept(const NvmModule *mod, const char *path);

int nl_actor_eval_i64(const char *src, int64_t *out, char *err, size_t errlen);

#endif
